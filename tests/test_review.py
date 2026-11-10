"""Upload isolation and honest temporal/result semantics for the local UI."""

import importlib.util
import io
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from anomaly_eval.reasoning.evidence import TASKS, history_problem, validated_views
from anomaly_eval.review import ROOT, Reviewer, summarize


def picture(color=(90, 110, 100)):
    buffer = io.BytesIO()
    Image.new("RGB", (80, 60), color).save(buffer, "PNG")
    return buffer.getvalue()


class FakeClassifier:
    def __init__(self):
        self.metadata = {"model": "fixture", "device": "cpu"}
        self.paths = []
        self.active = 0
        self.maximum = 0
        self.guard = threading.Lock()

    def classify(self, candidate):
        views = validated_views(candidate)
        self.paths.extend(p for v in views.values() for p in v["paths"])
        with self.guard:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            time.sleep(0.01)
            output = {}
            for task in candidate["tasks"]:
                view = views["short" if task == "littering" else "long"]
                reason = history_problem(task, view, clip=True)
                output[task] = (
                    {"decision": "unknown", "evidence": reason, "alert": False}
                    if reason
                    else {
                        "decision": "normal",
                        "score": -1.0,
                        "threshold": 0.0,
                        "alert": False,
                    }
                )
            return {"predictions": output, "seconds": 0.01}
        finally:
            with self.guard:
                self.active -= 1


class ReviewTests(unittest.TestCase):
    def test_single_image_does_not_fabricate_temporal_evidence(self):
        classifier = FakeClassifier()
        result = Reviewer(classifier).analyze(picture(), "../../image.png")
        report = result["report"]
        self.assertEqual(report["filename"], "image.png")
        for task in ["smoking", "rough_sleeping"]:
            self.assertEqual(report["summary"][task]["decision"], "normal")
        for task in ["loitering", "littering", "unattended_items"]:
            self.assertEqual(report["summary"][task]["decision"], "unknown")
        self.assertEqual(
            report["events"][0]["view_times_s"], {"short": [0.0], "long": [0.0]}
        )
        self.assertEqual(set(report["summary"]), set(TASKS))
        self.assertEqual(len(result["previews"]), 1)
        self.assertTrue(all(not Path(p).exists() for p in classifier.paths))

    def test_invalid_and_empty_uploads_never_reach_classifier(self):
        classifier = FakeClassifier()
        reviewer = Reviewer(classifier)
        for data, name in [
            (b"", "image.png"),
            (b"bad", "file.exe"),
            (b"invalid image", "image.png"),
        ]:
            with self.subTest(name=name), self.assertRaises((ValueError, OSError)):
                reviewer.analyze(data, name)
        self.assertFalse(classifier.paths)

    def test_classifier_state_is_serialized_between_sessions(self):
        classifier = FakeClassifier()
        reviewer = Reviewer(classifier)
        with ThreadPoolExecutor(2) as pool:
            futures = [
                pool.submit(reviewer.analyze, picture(), f"{i}.png") for i in range(2)
            ]
            results = [f.result() for f in futures]
        self.assertEqual(classifier.maximum, 1)
        self.assertEqual([r["report"]["filename"] for r in results], ["0.png", "1.png"])
        self.assertEqual(len(set(classifier.paths)), 2)

    def test_temporary_evidence_is_removed_on_model_failure(self):
        class Broken(FakeClassifier):
            def classify(self, candidate):
                super().classify(candidate)
                raise RuntimeError("model failed")

        classifier = Broken()
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            Reviewer(classifier).analyze(picture(), "image.png")
        self.assertTrue(classifier.paths)
        self.assertTrue(all(not Path(p).exists() for p in classifier.paths))

    def test_summary_keeps_positive_signals_and_unknown_window_counts(self):
        events = [
            {
                "timestamp_s": 1,
                "predictions": {
                    "smoking": {"decision": "unknown", "evidence": "too little"}
                },
            },
            {
                "timestamp_s": 2,
                "predictions": {
                    "smoking": {"decision": "anomaly", "score": -2, "threshold": -3}
                },
            },
            {
                "timestamp_s": 3,
                "predictions": {
                    "smoking": {"decision": "normal", "score": 2, "threshold": 3}
                },
            },
        ]
        result = summarize(events)
        self.assertEqual(result["smoking"]["decision"], "anomaly")
        self.assertEqual(result["smoking"]["timestamp_s"], 2)
        self.assertEqual(result["smoking"]["unsupported_windows"], 1)
        self.assertFalse(result["smoking"]["alert"])
        self.assertEqual(result["loitering"]["decision"], "unknown")

    def test_nonfinite_scores_are_not_summarized_as_normal(self):
        result = summarize(
            [
                {
                    "timestamp_s": 0,
                    "predictions": {
                        "smoking": {
                            "decision": "normal",
                            "score": float("nan"),
                            "threshold": 0,
                        }
                    },
                }
            ]
        )
        self.assertEqual(result["smoking"]["decision"], "unknown")

    @unittest.skipUnless(importlib.util.find_spec("av"), "Optional CPU video runtime")
    def test_short_video_is_reviewed_even_when_no_rule_fires(self):
        result = self.video(2)
        report = result["report"]
        self.assertEqual(len(report["events"]), 1)
        self.assertEqual(report["events"][0]["kind"], "final_scene_review")
        self.assertEqual(report["summary"]["smoking"]["decision"], "normal")
        self.assertEqual(report["summary"]["loitering"]["decision"], "unknown")

    @unittest.skipUnless(importlib.util.find_spec("av"), "Optional CPU video runtime")
    def test_long_video_uses_distinct_actual_past_frames(self):
        result = self.video(21)
        report = result["report"]
        for times in report["events"][0]["view_times_s"].values():
            self.assertEqual(len(times), len(set(times)))
            self.assertLessEqual(max(times), report["observed_seconds"] + 0.005001)
        self.assertEqual(report["selected_frames"], 105)
        self.assertTrue(all(v["scored_windows"] for v in report["summary"].values()))

    def video(self, seconds):
        import av
        import numpy as np

        sys.path.insert(0, str(ROOT / "data-pipeline"))
        from data_pipeline.detection.stage1 import load_gate

        writer_type = load_gate().HandoffWriter

        class SilentGate:
            def __init__(self, source, folder, **kwargs):
                self.start_ms, self.index = None, 0
                self.writer = writer_type(folder / source, buffer_s=30)
                self.stream = SimpleNamespace(paths=None)
                self.writer.start(self.stream)

            def process(self, frame):
                if self.start_ms is None:
                    self.start_ms = frame.timestamp_ms
                self.writer.observe(
                    frame.image, (frame.timestamp_ms - self.start_ms) / 1000, self.index
                )
                self.index += 1
                return []

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "video.mp4"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("mpeg4", rate=5)
                stream.width, stream.height, stream.pix_fmt = 80, 60, "yuv420p"
                for i in range(seconds * 5):
                    frame = av.VideoFrame.from_ndarray(
                        np.full((60, 80, 3), i % 255, np.uint8), format="rgb24"
                    )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            with patch("data_pipeline.detection.stage1.Stage1Detector", SilentGate):
                return Reviewer(FakeClassifier()).analyze(path.read_bytes(), path.name)


if __name__ == "__main__":
    unittest.main()
