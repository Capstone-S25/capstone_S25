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
    assert candidate["tasks"] == ["littering", "unattended_items"]
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


def test_duplicate_and_decreasing_timestamps(tmp_path):
    import pytest

    detector = make_detector(tmp_path)
    detector.process(frame(0))
    detector.process(frame(500))
    assert detector.process(frame(500)) == []
    assert detector.engine.times == [0.0, 0.5]
    with pytest.raises(ValueError, match="decreased"):
        detector.process(frame(400))


def test_gap_resets_tracker_and_history_without_rewinding_session_time(tmp_path):
    detector = make_detector(tmp_path)
    detector.process(frame(0))
    detector.process(frame(500))
    detector.process(frame(10_000))
    assert detector.engine.times == [10.0]
    assert detector.perception.resets == 2
    assert detector.continuity_epoch == 2


def test_periodic_scene_review_reaches_all_tasks_without_a_rule_trigger(tmp_path):
    detector = Stage1Detector(
        "camera_01",
        tmp_path,
        fps=2,
        cfg=object(),
        gate=fake_gate(),
        safety_scan_seconds=5,
    )
    events = [
        c
        for ms in range(0, 6500, 500)
        for c in detector.process(frame(ms, segment=ms // 2000 + 1))
    ]
    periodic = [e for e in events if e["event_type"] == "periodic_review"]
    assert len(periodic) == 1
    assert periodic[0]["timestamp_s"] == 5
    assert set(periodic[0]["tasks"]) == {
        "smoking",
        "littering",
        "loitering",
        "rough_sleeping",
        "unattended_items",
    }
    assert periodic[0]["roi_box"] == [0, 0, 6, 4]
    assert periodic[0]["candidate_origin"] == "scheduled_scene_review"
    # A discontinuity starts a fresh review period and cannot invent intervening history.
    assert detector.process(frame(20000)) == []


def test_periodic_action_window_covers_the_scan_interval(tmp_path):
    detector = Stage1Detector(
        "camera_01",
        tmp_path,
        fps=2,
        cfg=object(),
        gate=fake_gate(),
        safety_scan_seconds=10,
    )
    events = [c for ms in range(0, 10500, 500) for c in detector.process(frame(ms))]
    periodic = next(e for e in events if e["event_type"] == "periodic_review")
    assert periodic["timestamp_s"] == 10
    assert detector.writer.windows[-2:] == [("_short", 0, 10), ("_long", -10, 10)]
