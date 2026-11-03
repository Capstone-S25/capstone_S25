from types import SimpleNamespace
from uuid import UUID

import numpy as np

from data_pipeline.detection.stage1 import Stage1Detector
from data_pipeline.video_ingestion.decoding.decoder import DecodedFrame


class FakeEngine:
    """Fires one reach_down event when it sees a frame at t >= 1 s."""

    created = 0

    def __init__(self, cfg, fps):
        FakeEngine.created += 1
        self.times = []

    def step(self, obs):
        self.times.append(obs.t)
        if obs.t >= 1 and len(self.times) == 3:
            return [
                {
                    "event_id": f"reach_down_person1_f{obs.index}",
                    "event_type": "reach_down",
                    "timestamp_s": round(obs.t, 2),
                    "roi_box": [0, 0, 2, 2],
                }
            ]
        return []


class FakePerception:
    def __init__(self, cfg):
        self.resets = 0

    def reset(self, fps):
        self.resets += 1

    def observe(self, image, index, t):
        return SimpleNamespace(index=index, t=t)


class FakeWriter:
    def __init__(self, out_dir, buffer_s):
        self.manifest = []
        self.windows = []

    def start(self, stream):
        pass

    def observe(self, image, t, index):
        pass

    def save(self, event, image, stream, t):
        event["crop_path"] = "crop.jpg"
        self.manifest.append(event)

    def window_frames(self, event, image, stream, start, t, tag=""):
        self.windows.append((tag, start, t))
        return [f"{tag}.jpg"], [t]


def fake_gate():
    return SimpleNamespace(
        Perception=FakePerception, RuleEngine=FakeEngine, HandoffWriter=FakeWriter
    )


def frame(offset_ms, session=1, segment=1):
    return DecodedFrame(
        segment_id=segment,
        source_id="camera_01",
        session_id=UUID(int=session),
        frame_index=0,
        timestamp_ms=1_700_000_000_000 + offset_ms,
        image=np.zeros((4, 6, 3), dtype=np.uint8),
    )


def make_detector(tmp_path):
    return Stage1Detector("camera_01", tmp_path, fps=2, cfg=object(), gate=fake_gate())


def test_state_and_time_continue_across_segments(tmp_path):
    FakeEngine.created = 0
    detector = make_detector(tmp_path)

    out = [detector.process(frame(ms, segment=ms // 1000 + 1)) for ms in (0, 500, 1000)]

    assert FakeEngine.created == 1
    assert detector.engine.times == [0.0, 0.5, 1.0]
    assert out[:2] == [[], []]
    [candidate] = out[2]
    assert candidate["segment_id"] == 2
    assert candidate["timestamp_ms"] == 1_700_000_001_000


def test_candidate_has_tasks_views_and_source_scoped_id(tmp_path):
    detector = make_detector(tmp_path)

    [candidate] = [c for ms in (0, 500, 1000) for c in detector.process(frame(ms))]

    assert (
        candidate["event_id"]
        == f"camera_01_{UUID(int=1).hex[:8]}_reach_down_person1_f2"
    )
    assert candidate["tasks"] == ["littering"]
    assert candidate["views"] == {
        "short": {"paths": ["_short.jpg"], "times_s": [1.0]},
        "long": {"paths": ["_long.jpg"], "times_s": [1.0]},
    }
    assert detector.writer.windows == [("_short", -3.0, 1.0), ("_long", -19.0, 1.0)]
    assert detector.writer.manifest == []


def test_new_session_resets_tracking_and_clock(tmp_path):
    FakeEngine.created = 0
    detector = make_detector(tmp_path)
    detector.process(frame(0))
    detector.process(frame(500))

    detector.process(frame(60_000, session=2))

    assert FakeEngine.created == 2
    assert detector.perception.resets == 2
    assert detector.engine.times == [0.0]
