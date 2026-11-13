"""Connect ordered backend frames to persistent tracking and the AI queues."""

from pathlib import Path

from vision_runtime.service import EventStore, ScreeningQueue

from .stage1 import Stage1Detector, default_config, load_gate


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
        self.worker = ScreeningQueue(classifier, self.store, confirmation)

    def on_frame(self, frame):
        for candidate in self.detector.process(frame):
            self.worker.submit(candidate)

    def close(self):
        try:
            for candidate in self.detector.finish():
                self.worker.submit(candidate)
        finally:
            self.worker.close()
