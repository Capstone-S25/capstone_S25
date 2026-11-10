"""Local built-in/USB camera capture with independent preview and AI workers.

The camera belongs to the Streamlit host, not a remote browser. No database,
recording service or new streaming dependency is required for this path.
"""

import atexit
import logging
import math
import sys
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

from .config import ROOT, runtime_root
from .review import event_report, summarize
from .service import EventStore, ScreeningQueue

logger = logging.getLogger(__name__)


def open_camera(index, width, height):
    import cv2

    # CAP_ANY uses the platform's available webcam backend (V4L2/MSMF/etc.).
    capture = cv2.VideoCapture(index)
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, 15)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def stage1(source_id, folder, fps, scan_seconds):
    backend = str(ROOT / "data-pipeline")
    if backend not in sys.path:
        sys.path.insert(0, backend)
    from data_pipeline.detection.stage1 import Stage1Detector

    detector = Stage1Detector(
        source_id, folder, fps=fps, safety_scan_seconds=scan_seconds
    )
    # Keep enough genuinely observed short-window views for littering checks.
    detector.writer.buffer_fps = fps
    return detector


class LiveSession:
    """One camera owner; latest-frame capture, rolling evidence, bounded AI work."""

    def __init__(
        self,
        reviewer,
        device=0,
        *,
        width=640,
        height=480,
        fps=5,
        scan_seconds=3,
        idle_timeout=20,
        keep_events=30,
        capture_factory=None,
        detector_factory=None,
    ):
        if not isinstance(device, int) or not 0 <= device <= 15:
            raise ValueError("Camera index must be an integer between 0 and 15")
        if not all(
            math.isfinite(v) and v > 0 for v in (fps, scan_seconds, idle_timeout)
        ):
            raise ValueError("Camera rates and timeouts must be positive and finite")
        if keep_events < 1 or width < 1 or height < 1:
            raise ValueError(
                "Camera dimensions and retained event count must be positive"
            )
        self.reviewer, self.device = reviewer, device
        self.width, self.height, self.fps = width, height, fps
        self.scan_seconds, self.idle_timeout = scan_seconds, idle_timeout
        self.keep_events = keep_events
        self.capture_factory = capture_factory or open_camera
        self.detector_factory = detector_factory or stage1
        self.ident = uuid4()
        self.source_id = f"webcam_{device}"
        parent = runtime_root() / "live"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="camera-", dir=parent)
        self.folder = Path(self.temporary.name)
        self.store = EventStore(self.folder, cleanup=self.temporary.cleanup)
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.done = threading.Event()
        self.started = self.last_poll = time.monotonic()
        self.latest = None
        self.captured = self.selected = self.retired = self.skipped = 0
        self.first_selected_ms = self.last_selected_ms = None
        self.frame_size = None
        self.ready = False
        self.error = self.stop_reason = None
        self.discarded = False
        self.previews, self.evidence = {}, {}
        self.capture_thread = threading.Thread(
            target=self._capture, daemon=True, name="webcam-capture"
        )
        self.process_thread = threading.Thread(
            target=self._process, daemon=True, name="webcam-tracking"
        )
        self.capture_thread.start()
        self.process_thread.start()

    @property
    def active(self):
        return self.capture_thread.is_alive() or self.process_thread.is_alive()

    def stop(self, reason="Stopped by user"):
        with self.changed:
            if self.stop_reason is None:
                self.stop_reason = reason
            self.stop_event.set()
            self.changed.notify_all()

    def release(self):
        """Discard UI evidence after producers and any active AI calls finish."""
        with self.lock:
            self.discarded = True
            self.stop()
            if self.done.is_set():
                self.store.release()

    def _expired(self):
        with self.lock:
            expired = time.monotonic() - self.last_poll > self.idle_timeout
        if expired:
            self.stop("Camera stopped because the live view disconnected")
            self.release()
        return expired

    def _fail(self, message):
        with self.lock:
            self.error = message
        self.stop("Camera session failed")

    def _capture(self):
        capture = None
        try:
            capture = self.capture_factory(self.device, self.width, self.height)
            if not capture.isOpened():
                raise RuntimeError(
                    f"Cannot open camera {self.device}. Try another index, close other "
                    "camera applications and check the operating system camera permission."
                )
            while not self.stop_event.is_set() and not self._expired():
                ok, image = capture.read()
                if not ok or image is None or not image.size:
                    if self.stop_event.is_set():
                        break
                    raise RuntimeError(
                        "The camera stopped delivering frames; reconnect it and restart."
                    )
                stamp = time.monotonic()
                with self.changed:
                    self.captured += 1
                    self.latest = (self.captured, stamp, image)
                    self.frame_size = (image.shape[1], image.shape[0])
                    self.changed.notify_all()
        except Exception as exc:
            logger.exception("Local camera capture failed")
            self._fail(str(exc))
        finally:
            # The capture owner alone releases the driver; never race read/release.
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    logger.exception("Could not release the camera driver")
            self.stop()

    def _submit(self, candidate, worker):
        import cv2

        paths = {
            candidate.get("crop_path"),
            candidate.get("context_frame_path"),
            *candidate.get("window_frame_paths", []),
            *(p for view in candidate["views"].values() for p in view["paths"]),
        }
        context = cv2.imread(candidate["context_frame_path"])
        if context is None:
            raise ValueError("The camera evidence frame could not be read")
        ok, preview = cv2.imencode(".jpg", context, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise ValueError("The camera evidence preview could not be encoded")
        with self.lock:
            self.evidence[candidate["event_id"]] = {Path(p) for p in paths if p}
            self.previews[candidate["event_id"]] = preview.tobytes()
            worker.submit(candidate)

    def _prune(self):
        with self.lock:
            for ident in self.store.prune(self.keep_events):
                self.retired += 1
                self.previews.pop(ident, None)
                for path in self.evidence.pop(ident, set()):
                    # Only event-owned files inside this session can be retired.
                    if path.resolve().is_relative_to(self.folder):
                        path.unlink(missing_ok=True)

    def _process(self):
        detector = worker = None
        last_sequence, next_frame_at = 0, 0.0
        try:
            with self.changed:
                while self.latest is None and not self.stop_event.is_set():
                    self.changed.wait(0.2)
                    self._expired()
            if self.stop_event.is_set():
                return
            detector = self.detector_factory(
                self.source_id, self.folder, self.fps, self.scan_seconds
            )
            from data_pipeline.video_ingestion.decoding.decoder import DecodedFrame

            worker = ScreeningQueue(
                self.reviewer.classifier,
                self.store,
                self.reviewer.confirmation,
                capacity=1,
                latest_only=True,
            )
            with self.lock:
                self.ready = True
            while not self.stop_event.is_set() and not self._expired():
                with self.changed:
                    frame = self.latest
                    if (
                        frame is None
                        or frame[0] == last_sequence
                        or frame[1] < next_frame_at
                    ):
                        self.changed.wait(0.02)
                        continue
                sequence, captured_at, image = frame
                # Timestamp actual capture, not frame-count / requested FPS.
                stamp_ms = round((captured_at - self.started) * 1000)
                if (
                    self.last_selected_ms is not None
                    and stamp_ms <= self.last_selected_ms
                ):
                    continue
                with self.lock:
                    self.skipped += max(0, sequence - last_sequence - 1)
                    self.first_selected_ms = (
                        stamp_ms
                        if self.first_selected_ms is None
                        else self.first_selected_ms
                    )
                candidates = detector.process(
                    DecodedFrame(
                        0, self.source_id, self.ident, sequence, stamp_ms, image
                    )
                )
                for candidate in candidates:
                    candidate["timestamp_origin"] = (
                        "host monotonic camera-read time relative to this session"
                    )
                    self._submit(candidate, worker)
                with self.lock:
                    self.selected += 1
                    self.last_selected_ms = stamp_ms
                last_sequence = sequence
                next_frame_at = captured_at + 1 / self.fps
                self._prune()
            if self.selected:
                for candidate in detector.finish():
                    self._submit(candidate, worker)
        except Exception as exc:
            logger.exception("Live camera analysis failed")
            self._fail(str(exc))
        finally:
            try:
                if detector is not None:
                    detector._close_pool()
                if worker is not None:
                    worker.close()
                self._prune()
            except Exception as exc:
                self._fail(str(exc))
            with self.lock:
                self.done.set()
                if self.discarded:
                    self.store.release()

    def snapshot(self):
        import cv2

        with self.lock:
            self.last_poll = time.monotonic()
            latest = self.latest
            start_ms = self.first_selected_ms
            observed = (
                max(0, (self.last_selected_ms - start_ms) / 1000)
                if start_ms is not None and self.last_selected_ms is not None
                else 0
            )
            events = self.store.snapshot()
            report = event_report(
                {
                    "kind": "live",
                    "filename": f"Camera {self.device}",
                    "model": self.reviewer.classifier.metadata,
                    "processing_seconds": round(time.monotonic() - self.started, 2),
                    "observed_seconds": observed,
                    "selected_frames": self.selected,
                    "captured_frames": self.captured,
                    "skipped_capture_frames": self.skipped,
                    "retired_windows": self.retired,
                    "retention": f"Newest {self.keep_events} windows plus in-flight work; no full-video recording",
                    "scope": "Provisional live scene review; scores are not probabilities; no external alerts.",
                },
                events,
                self.reviewer.confirmation is not None,
            )
            # Live signals describe the most recent completed window, not the
            # strongest historical positive from the entire session.
            recent = {}
            for event in sorted(events, key=lambda e: e["timestamp_s"], reverse=True):
                if event["status"] in {"queued", "not_run"}:
                    continue
                for task, prediction in event["predictions"].items():
                    recent.setdefault(
                        task, {**event, "predictions": {task: prediction}}
                    )
            report["summary"] = summarize(list(recent.values()))
            end = max((e["timestamp_s"] for e in recent.values()), default=None)
            report["latest_screened_timestamp_s"] = end
            camera_t = (
                (latest[1] - self.started) - (start_ms or 0) / 1000 if latest else 0
            )
            report["result_age_seconds"] = (
                max(0, camera_t - end) if end is not None else None
            )
            report["tracking_lag_seconds"] = max(0, camera_t - observed)
            state = (
                "expired"
                if self.discarded
                else "error"
                if self.error
                else "stopped"
                if not self.active
                else "stopping"
                if self.stop_event.is_set()
                else "running"
                if self.ready
                else "starting"
            )
            result = {
                "run_id": self.ident.hex,
                "report": report,
                "previews": [self.previews.get(e["event_id"]) for e in events],
                "state": state,
                "error": self.error,
                "stop_reason": self.stop_reason,
                "frame_size": self.frame_size,
            }
        # Encode the latest raw frame outside the state lock; model work cannot
        # hold up capture or the browser preview.
        result["frame"] = None
        if latest:
            ok, jpg = cv2.imencode(".jpg", latest[2], [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                result["frame"] = jpg.tobytes()
        return result


class CameraManager:
    """One active local camera session per application process, across browsers."""

    def __init__(self, reviewer, **session_options):
        self.reviewer, self.options = reviewer, session_options
        self.lock = threading.Lock()
        self.current = None
        atexit.register(self.close)

    def start(self, device, **options):
        with self.lock:
            if self.current is not None:
                if self.current.active:
                    raise RuntimeError(
                        "A camera session is still active or finishing. Stop it and wait before starting another."
                    )
                self.current.release()
            self.current = LiveSession(
                self.reviewer, device, **(self.options | options)
            )
            return self.current

    def close(self):
        if self.current is not None:
            self.current.release()
