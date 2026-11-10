"""Local image/video screening with the deployed classifier and causal gate.

No Streamlit dependency: the UI and command-line checks use this same service.
Each upload has isolated tracking state and temporary evidence. A lock protects
the resident classifier's mutable, per-task interpolated backbone.
"""

from __future__ import annotations

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

from anomaly_eval.reasoning.evidence import TASKS

ROOT = Path(__file__).resolve().parents[1]
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


def configure_runtime():
    """Use packaged cache when present, otherwise the checkout's existing cache."""
    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        YOLO_AUTOINSTALL="false",
        YOLO_CONFIG_DIR=str(ROOT / ".local/yolo"),
    )
    cache = ROOT / "data/models/huggingface/hub"
    if cache.is_dir():
        os.environ["HF_HUB_CACHE"] = str(cache)


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


class Reviewer:
    def __init__(self, classifier=None):
        if classifier is None:
            configure_runtime()
            from anomaly_eval.reasoning.backends import ClipStrategy

            classifier = ClipStrategy("mobileclip2-s2", "cpu")
        self.classifier = classifier
        self.lock = threading.Lock()

    def analyze(self, data, name, progress=lambda fraction, message: None):
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_TYPES | VIDEO_TYPES:
            raise ValueError("Choose a supported image or video file")
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("Choose a non-empty file smaller than 200 MB")
        progress(None, "Waiting for the local model…")
        with self.lock, tempfile.TemporaryDirectory(prefix="scene-review-") as temp:
            started = time.perf_counter()
            folder = Path(temp)
            source = folder / ("upload" + suffix)
            source.write_bytes(data)
            if suffix in IMAGE_TYPES:
                events, previews, facts = self._image(source, folder, progress)
                kind = "image"
            else:
                events, previews, facts = self._video(source, folder, progress)
                kind = "video"
            report = {
                "filename": Path(name).name,
                "kind": kind,
                "model": self.classifier.metadata,
                "processing_seconds": time.perf_counter() - started,
                **facts,
                "summary": summarize(events),
                "events": events,
                "scope": "Scene/window screening only. No confirmed alerts, identity or ownership. Scores are uncalibrated classifier outputs, not probabilities.",
            }
            progress(1.0, "Review complete")
            return {"report": report, "previews": previews}

    def _review(self, candidate, events, previews, preview):
        result = self.classifier.classify(candidate)
        event = {
            "event_id": candidate["event_id"],
            "timestamp_s": candidate["timestamp_s"],
            "kind": candidate["event_type"],
            "entity": candidate.get("entity_id"),
            "predictions": result["predictions"],
            "inference_seconds": result["seconds"],
            "view_times_s": {k: v["times_s"] for k, v in candidate["views"].items()},
        }
        events.append(event)
        previews.append(preview)

    def _image(self, source, folder, progress):
        from PIL import Image, ImageOps

        progress(0.1, "Reviewing the image…")
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        path = folder / "frame.png"
        image.save(path)
        view = {"paths": [str(path)], "times_s": [0.0]}
        candidate = {
            "event_id": "image_scene",
            "event_type": "image_review",
            "timestamp_s": 0.0,
            "tasks": list(TASKS),
            "views": {"short": view, "long": view},
        }
        events, previews = [], []
        self._review(candidate, events, previews, jpeg_preview(image))
        return events, previews, {"selected_frames": 1, "observed_seconds": 0.0}

    def _video(self, source, folder, progress):
        import av
        from PIL import Image

        pipeline = str(ROOT / "data-pipeline")
        if pipeline not in sys.path:
            sys.path.insert(0, pipeline)
        from data_pipeline.detection.stage1 import Stage1Detector
        from data_pipeline.video_ingestion.decoding.decoder import decode_segment
        from data_pipeline.video_ingestion.decoding.selection import select_frames
        from data_pipeline.video_ingestion.scheduling.worker import SegmentJob

        with av.open(str(source)) as container:
            duration = (
                float(container.duration / av.time_base)
                if container.duration is not None
                else None
            )
        detector = Stage1Detector("upload", folder, fps=5, safety_scan_seconds=10)
        job = SegmentJob(0, "upload", uuid4(), source.name, 0, 0)
        events, previews = [], []
        count, last, end = 0, None, 0.0
        with closing(decode_segment(job, source.parent)) as decoded:
            for frame in select_frames(decoded, target_fps=5):
                candidates = detector.process(frame)
                count += 1
                last = frame
                end = (frame.timestamp_ms - detector.start_ms) / 1000
                fraction = min(0.95, end / duration) if duration else None
                if count % 5 == 1:
                    progress(
                        fraction,
                        f"Tracking · {end:.1f}s observed · {len(events)} windows reviewed",
                    )
                for candidate in candidates:
                    progress(fraction, f"Reviewing evidence at {end:.1f}s…")
                    with Image.open(candidate["context_frame_path"]) as image:
                        preview = jpeg_preview(image)
                    self._review(candidate, events, previews, preview)
        if last is None:
            raise ValueError("The video contains no decodable frames")
        # Always examine the endpoint, including clips shorter than the periodic
        # gate interval. Keep actual past timestamps; never repeat a still image.
        if not events or not any(
            abs(e["timestamp_s"] - end) < 0.005001
            and set(e["predictions"]) == set(TASKS)
            for e in events[-4:]
        ):
            progress(0.97, "Reviewing the final evidence window…")
            candidate = {
                "event_id": "final_scene",
                "event_type": "final_scene_review",
                "frame_index": detector.index - 1,
                "timestamp_s": end,
                "tasks": list(TASKS),
                "views": {},
            }
            for kind, span in [("short", 10), ("long", 20)]:
                paths, times = detector.writer.window_frames(
                    candidate,
                    last.image,
                    detector.stream,
                    end - span,
                    end,
                    tag="_final_" + kind,
                )
                candidate["views"][kind] = {"paths": paths, "times_s": times}
            preview = jpeg_preview(Image.fromarray(last.image[:, :, ::-1]))
            self._review(candidate, events, previews, preview)
        return events, previews, {"selected_frames": count, "observed_seconds": end}
