"""Upload application service using the same backend decoder and AI queues as cameras."""

import io
import math
import os
import sys
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps

from .config import ROOT, configure_runtime, runtime_root
from .evidence import TASKS
from .gemma import GemmaClient
from .service import ConfirmationQueue, EventStore, ScreeningQueue

LABELS = {
    "smoking": "Smoking",
    "littering": "Littering",
    "loitering": "Loitering",
    "rough_sleeping": "Apparent public sleeping",
    "unattended_items": "Unattended items",
}

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

VIDEO_TYPES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

MAX_UPLOAD_BYTES = 200 * 1024**2


def summarize(events):
    """Select strongest model signals without treating unsupported windows as normal."""
    output = {}
    for task in TASKS:
        values = [
            {**e["predictions"][task], "timestamp_s": e["timestamp_s"]}
            for e in events
            if task in e["predictions"]
        ]
        scored = [
            v
            for v in values
            if v.get("decision") in {"normal", "anomaly"}
            and math.isfinite(v.get("score", float("nan")))
            and math.isfinite(v.get("threshold", float("nan")))
        ]
        positive = [v for v in scored if v["decision"] == "anomaly"]
        pool = positive or scored
        chosen = (
            max(pool, key=lambda v: v["score"] - v["threshold"])
            if pool
            else next(
                (v for v in reversed(values) if v.get("decision") == "unknown"),
                {"decision": "unknown", "evidence": "No valid scored evidence"},
            )
            if values
            else {"decision": "unknown", "evidence": "No supported evidence window"}
        )
        output[task] = {
            **chosen,
            "label": LABELS[task],
            "scored_windows": len(scored),
            "positive_windows": len(positive),
            "unsupported_windows": len(values) - len(scored),
            "alert": False,
        }
    return output


def jpeg_preview(image):
    image = image.convert("RGB")
    image.thumbnail((960, 640))
    out = io.BytesIO()
    image.save(out, "JPEG", quality=85)
    return out.getvalue()


def event_report(report, events, gemma_enabled):
    """Use the same provisional/confirmation states for uploads and live video."""
    for event in events:
        event.setdefault(
            "predictions",
            {
                t: {
                    "decision": "unknown",
                    "evidence": "Screening not completed",
                    "alert": False,
                }
                for t in TASKS
            },
        )
    return {
        **report,
        "events": events,
        "summary": summarize(events),
        "gemma_enabled": gemma_enabled,
        "confirmation_pending": sum(
            e["status"] in {"confirmation_pending", "confirming"} for e in events
        ),
        "confirmation_failed": sum(
            e["status"] in {"confirmation_error", "confirmation_not_run"}
            for e in events
        ),
        "screening_failed": sum(
            e["status"] in {"not_run", "screening_error"} for e in events
        ),
    }


class LockedClassifier:
    def __init__(self, classifier):
        self.classifier = classifier
        self.metadata = classifier.metadata
        self.lock = threading.Lock()

    def classify(self, candidate):
        with self.lock:
            return self.classifier.classify(candidate)


class Reviewer:
    def __init__(self, classifier=None, gemma=None, model=None, device="cpu"):
        configure_runtime()
        if classifier is None:
            from .classifier import default_classifier, load_classifier

            model = model or os.environ.get("VISION_CLASSIFIER") or default_classifier()
            classifier = load_classifier(model, device)
        self.classifier = LockedClassifier(classifier)
        url = os.environ.get("GEMMA_BASE_URL")
        if gemma is None and url:
            gemma = GemmaClient(url, os.environ.get("GEMMA_MODEL", "gemma4-26b-a4b"))
        self.confirmation = ConfirmationQueue(gemma) if gemma else None
        self.analysis_lock = threading.Lock()
        self.runs_lock = threading.RLock()
        self.runs = {}

    def analyze(self, data, name, progress=lambda fraction, message: None):
        suffix = Path(name).suffix.lower()
        if (
            suffix not in IMAGE_TYPES | VIDEO_TYPES
            or not data
            or len(data) > MAX_UPLOAD_BYTES
        ):
            raise ValueError("Choose a supported, non-empty file up to 200 MB")
        progress(None, "Waiting for the screening model…")
        with self.analysis_lock:
            parent = runtime_root() / "uploads"
            parent.mkdir(parents=True, exist_ok=True)
            temp = tempfile.TemporaryDirectory(prefix="review-", dir=parent)
            folder = Path(temp.name)
            store = EventStore(folder, cleanup=temp.cleanup)
            # Uploads have no real-time deadline: wait for the classifier instead
            # of marking later windows of a long video as not run.
            worker = ScreeningQueue(
                self.classifier, store, self.confirmation, wait_for_space=True
            )
            run_id = uuid4().hex
            started = time.perf_counter()
            with self.runs_lock:
                self._prune()
                self.runs[run_id] = {"store": store, "created": time.monotonic()}
            try:
                source = folder / ("upload" + suffix)
                source.write_bytes(data)
                if suffix in IMAGE_TYPES:
                    previews, facts = self._image(source, folder, worker, progress)
                    kind = "image"
                else:
                    previews, facts = self._video(source, folder, worker, progress)
                    kind = "video"
                progress(0.99, "Finishing provisional screening…")
                worker.close()
                report = {
                    "filename": Path(name).name,
                    "kind": kind,
                    "model": self.classifier.metadata,
                    "processing_seconds": time.perf_counter() - started,
                    **facts,
                    "scope": "Scene/window screening. Proposed regions and ownership estimates are not verified facts. Scores are not probabilities; automatic alert delivery is disabled.",
                }
                result = {"run_id": run_id, "report": report, "preview_map": previews}
                result = self.refresh(result)
                progress(
                    1,
                    "Screening complete · background review continues"
                    if self.confirmation
                    else "Review complete",
                )
                return result
            except BaseException:
                worker.close()
                self.release(run_id)
                raise

    def _submit(self, candidate, worker, previews):
        # Evidence files must outlive any asynchronous review of this event.
        with Image.open(candidate["context_frame_path"]) as image:
            previews[candidate["event_id"]] = jpeg_preview(image)
        worker.submit(candidate)

    def _image(self, source, folder, worker, progress):
        progress(0.1, "Reviewing the image…")
        with Image.open(source) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
        path = folder / "frame.png"
        image.save(path)
        view = {"paths": [str(path)], "times_s": [0.0]}
        candidate = {
            "source_id": "upload",
            "event_id": "image_scene",
            "event_type": "image_review",
            "timestamp_s": 0.0,
            "tasks": list(TASKS),
            "views": {"short": view, "long": view},
            "context_frame_path": str(path),
        }
        previews = {}
        self._submit(candidate, worker, previews)
        return previews, {"selected_frames": 1, "observed_seconds": 0.0}

    def _video(self, source, folder, worker, progress):
        import av

        backend = str(ROOT / "data-pipeline")
        if backend not in sys.path:
            sys.path.insert(0, backend)
        from data_pipeline.detection.stage1 import Stage1Detector
        from data_pipeline.video_ingestion.decoding.decoder import decode_segment
        from data_pipeline.video_ingestion.decoding.selection import select_frames
        from data_pipeline.video_ingestion.scheduling.worker import SegmentJob

        with av.open(str(source)) as container:
            duration = container.duration / av.time_base if container.duration else None
        detector = Stage1Detector("upload", folder, fps=5, safety_scan_seconds=10)
        job = SegmentJob(0, "upload", uuid4(), source.name, 0, 0)
        previews, count, end = {}, 0, 0.0
        try:
            with closing(decode_segment(job, source.parent)) as frames:
                for frame in select_frames(frames, target_fps=5):
                    candidates = detector.process(frame)
                    count += 1
                    end = (frame.timestamp_ms - detector.start_ms) / 1000
                    if count % 5 == 1:
                        progress(
                            min(0.95, end / duration) if duration else None,
                            f"Tracking · {end:.1f}s observed · {len(previews)} windows queued",
                        )
                    for candidate in candidates:
                        self._submit(candidate, worker, previews)
            if not count:
                raise ValueError("The video contains no decodable frames")
            for candidate in detector.finish():
                self._submit(candidate, worker, previews)
        finally:
            detector._close_pool()
        return previews, {"selected_frames": count, "observed_seconds": end}

    def refresh(self, result):
        with self.runs_lock:
            run = self.runs.get(result["run_id"])
            if run is None:
                raise ValueError("This review has expired; upload the file again")
            events = run["store"].snapshot()
        report = event_report(result["report"], events, self.confirmation is not None)
        return {
            **result,
            "report": report,
            "previews": [result["preview_map"][e["event_id"]] for e in events],
        }

    def release(self, run_id):
        with self.runs_lock:
            run = self.runs.pop(run_id, None)
        if run:
            run["store"].release()

    def _prune(self):
        for ident, run in list(self.runs.items()):
            if time.monotonic() - run["created"] > 3600:
                self.release(ident)

    def close(self):
        if self.confirmation:
            self.confirmation.close()
        for ident in list(self.runs):
            self.release(ident)
