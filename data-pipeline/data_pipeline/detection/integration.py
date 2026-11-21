"""Connect the backend's live frames to persistent tracking and the AI queues."""

import threading
from pathlib import Path
from uuid import uuid4

from vision_runtime.service import EventStore, ScreeningQueue

from .stage1 import Stage1Detector, Stage1Frame, default_config, load_gate


class SerializedClassifier:
    """One classifier shared by several cameras' screening threads.

    Calls take turns, so caches inside the classifier are never updated from
    two threads at once.
    """

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()

    def __getattr__(self, name):
        value = getattr(self._inner, name)
        if not callable(value):
            return value

        def call(*args, **kwargs):
            with self._lock:
                return value(*args, **kwargs)

        return call


class VisionPipeline:
    def __init__(
        self,
        source_id,
        out_dir,
        classifier,
        confirmation=None,
        fps=5,
        device="cpu",
        on_event=None,
    ):
        self.store = EventStore(Path(out_dir), on_write=on_event)
        gate = load_gate()
        self.detector = Stage1Detector(
            source_id,
            Path(out_dir),
            fps=fps,
            cfg=default_config(gate, device),
            gate=gate,
            safety_scan_seconds=10,
        )
        self.worker = ScreeningQueue(
            classifier, self.store, confirmation, publish_task_results=True
        )
        # One tracking session per pipeline run; the streaming worker's frames
        # carry no recording session or segment.
        self.session_id = uuid4()

    def on_frame(self, frame):
        """Track one StreamFrame (or Stage1Frame) and queue its candidates."""
        if not isinstance(frame, Stage1Frame):
            frame = Stage1Frame(
                None,
                frame.source_id,
                self.session_id,
                frame.frame_index,
                frame.timestamp_ms,
                frame.image,
            )
        for candidate in self.detector.process(frame):
            self.worker.submit(candidate)

    def close(self):
        try:
            for candidate in self.detector.finish():
                self.worker.submit(candidate)
        finally:
            self.worker.close()


class CameraRouter:
    """The single frame handler for run_pipeline: sends each camera's frames
    to that camera's own VisionPipeline."""

    def __init__(self, pipelines):
        self.pipelines = dict(pipelines)

    def __call__(self, frame):
        self.pipelines[frame.source_id].on_frame(frame)

    def close(self):
        errors = []
        for pipeline in self.pipelines.values():
            try:
                pipeline.close()
            except Exception as error:  # close every camera before reporting
                errors.append(error)
        if errors:
            raise errors[0]
