"""Application integration contracts: queues, evidence lifetime and safe states."""

import io
import threading
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from PIL import Image

from vision_runtime.evidence import history_problem, validated_views
from vision_runtime.review import Reviewer
from vision_runtime.service import ConfirmationQueue, EventStore, ScreeningQueue


class Classifier:
    metadata = {"model": "fixture", "device": "cpu"}

    def classify(self, candidate):
        views = validated_views(candidate)
        predictions = {}
        for task in candidate["tasks"]:
            view = views["short" if task == "littering" else "long"]
            problem = history_problem(task, view, clip=True)
            predictions[task] = (
                {"decision": "unknown", "evidence": problem, "alert": False}
                if problem
                else {
                    "decision": "normal",
                    "score": -1.0,
                    "threshold": 0.0,
                    "alert": False,
                }
            )
        return {"predictions": predictions, "seconds": 0.001}


def candidate(tmp_path, ident="one"):
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (32, 32), (100, 120, 110)).save(path)
    view = {"paths": [str(path)], "times_s": [0.0]}
    return {
        "event_id": ident,
        "source_id": "camera",
        "event_type": "image_review",
        "timestamp_s": 0.0,
        "tasks": ["smoking"],
        "views": {"long": view},
    }


def test_confirmation_does_not_block_second_screening(tmp_path):
    entered, finish = threading.Event(), threading.Event()

    class SlowGemma:
        def classify(self, event):
            entered.set()
            assert finish.wait(5)
            return {
                "predictions": {
                    "smoking": {
                        "decision": "normal",
                        "frame_refs": [1],
                        "evidence": "Ordinary visible activity",
                        "alert": False,
                    }
                }
            }

    review = ConfirmationQueue(SlowGemma())
    store = EventStore(tmp_path)
    screening = ScreeningQueue(Classifier(), store, review, confirm="all")
    try:
        screening.submit(candidate(tmp_path))
        assert entered.wait(2)
        screening.submit(candidate(tmp_path, "two"))
        screening.close()
        assert len(store.snapshot()) == 2
        assert all(
            row["predictions"]["smoking"]["decision"] == "normal"
            for row in store.snapshot()
        )
        assert any(row["status"] == "confirming" for row in store.snapshot())
    finally:
        finish.set()
        screening.close()
        review.close()
    assert all(row["status"] == "reviewed" for row in store.snapshot())


def test_confirmation_overflow_is_explicit_and_preserves_provisional(tmp_path):
    entered, finish = threading.Event(), threading.Event()

    class Slow:
        def classify(self, event):
            entered.set()
            assert finish.wait(5)
            return {
                "predictions": {
                    "smoking": {
                        "decision": "unknown",
                        "evidence": "Not visible",
                        "alert": False,
                    }
                }
            }

    review = ConfirmationQueue(Slow(), capacity=1)
    store = EventStore(tmp_path)
    result = Classifier().classify(candidate(tmp_path))
    try:
        assert review.submit(candidate(tmp_path), result, store)
        assert entered.wait(2)
        assert review.submit(candidate(tmp_path, "two"), result, store)
        assert not review.submit(candidate(tmp_path, "three"), result, store)
        row = store.snapshot()[-1]
        assert row["status"] == "confirmation_not_run"
        assert row["reason"] == "confirmation_queue_full"
        assert row["predictions"]["smoking"]["decision"] == "normal"
    finally:
        finish.set()
        review.close()


def test_evidence_survives_clear_until_active_review_finishes(tmp_path):
    folder = tmp_path / "upload"
    folder.mkdir()
    path = folder / "frame.jpg"
    entered, finish = threading.Event(), threading.Event()

    class Slow:
        def classify(self, event):
            entered.set()
            assert finish.wait(5)
            assert path.exists()
            return {
                "predictions": {
                    "smoking": {
                        "decision": "normal",
                        "evidence": "Ordinary activity",
                        "alert": False,
                    }
                }
            }

    def cleanup():
        path.unlink()
        (folder / "events.jsonl").unlink()
        folder.rmdir()

    store = EventStore(folder, cleanup=cleanup)
    event = candidate(folder)
    review = ConfirmationQueue(Slow())
    try:
        review.submit(event, Classifier().classify(event), store)
        assert entered.wait(2)
        store.release()
        assert path.exists()
    finally:
        finish.set()
        review.close()
    assert not folder.exists()


def test_malformed_confirmation_is_unknown_and_cannot_alert(tmp_path):
    class Broken:
        def classify(self, event):
            return {"predictions": {}}

    store = EventStore(tmp_path)
    review = ConfirmationQueue(Broken())
    event = candidate(tmp_path)
    review.submit(event, Classifier().classify(event), store)
    review.close()
    result = store.snapshot()[0]
    assert result["status"] == "confirmation_error"
    assert result["confirmation"]["smoking"]["decision"] == "unknown"
    assert not result["confirmation"]["smoking"]["alert"]


def test_image_has_no_invented_temporal_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("GEMMA_BASE_URL", raising=False)
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, "PNG")
    reviewer = Reviewer(Classifier())
    result = reviewer.analyze(buffer.getvalue(), "image.png")
    assert all(
        result["report"]["summary"][task]["decision"] == "unknown"
        for task in ["littering", "loitering", "unattended_items"]
    )
    assert result["report"]["summary"]["smoking"]["decision"] == "normal"
    reviewer.release(result["run_id"])
    reviewer.close()
    assert not list((tmp_path / "uploads").iterdir())


def test_crop_delays_keep_trigger_frame_and_decision_availability(tmp_path):
    from data_pipeline.detection.stage1 import Stage1Detector
    from data_pipeline.video_ingestion.decoding.decoder import DecodedFrame

    from vision_runtime.tracking.gate import HandoffWriter

    class Perception:
        def __init__(self, cfg):
            pass

        def reset(self, fps):
            pass

        def observe(self, frame, index, t):
            return SimpleNamespace(t=t, index=index)

    class Engine:
        crop_check = None

        def __init__(self, cfg, fps):
            self.first = None

        def step(self, obs, frame):
            if self.first is None:
                self.first = frame.copy()
                return []
            return [
                {
                    "event_id": "delayed",
                    "event_type": "reach_down",
                    "timestamp_s": 0.0,
                    "frame_index": 0,
                    "roi_box": [0, 0, 8, 8],
                    "_frame": self.first,
                    "_t": 0.0,
                }
            ]

        def flush(self):
            return []

    gate = SimpleNamespace(
        Perception=Perception, RuleEngine=Engine, HandoffWriter=HandoffWriter
    )
    detector = Stage1Detector("camera", tmp_path, fps=5, cfg=object(), gate=gate)
    session = uuid4()
    first = DecodedFrame(
        0, "camera", session, 0, 0, np.full((8, 8, 3), (0, 0, 240), np.uint8)
    )
    second = DecodedFrame(
        0, "camera", session, 1, 1000, np.full((8, 8, 3), (240, 0, 0), np.uint8)
    )
    assert detector.process(first) == []
    event = detector.process(second)[0]
    assert event["timestamp_s"] == 0
    assert event["available_at_s"] == 1
    assert max(event["views"]["short"]["times_s"]) == 0
    with Image.open(event["context_frame_path"]) as image:
        assert image.getpixel((4, 4))[0] > 200
    # The session's first frame is the camera reference for every candidate.
    with Image.open(event["reference_frame_path"]) as image:
        assert image.getpixel((4, 4))[0] > 200
    assert detector.finish()[-1]["event_type"] == "final_scene_review"


def test_release_and_journaling_failure_do_not_hang_worker(tmp_path, monkeypatch):
    store = EventStore(tmp_path)
    original = store.write

    def fail_later(event, status, **fields):
        if status != "queued":
            raise OSError("disk full")
        return original(event, status, **fields)

    monkeypatch.setattr(store, "write", fail_later)
    worker = ScreeningQueue(Classifier(), store)
    worker.submit(candidate(tmp_path))
    import pytest

    with pytest.raises(RuntimeError, match="journal failed"):
        worker.close()


def test_upload_queue_waits_for_space_instead_of_dropping(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        ScreeningQueue(
            Classifier(), EventStore(tmp_path), latest_only=True, wait_for_space=True
        )
    entered, finish = threading.Event(), threading.Event()

    class Slow(Classifier):
        def classify(self, event):
            entered.set()
            assert finish.wait(5)
            return super().classify(event)

    store = EventStore(tmp_path)
    worker = ScreeningQueue(Slow(), store, capacity=1, wait_for_space=True)
    accepted = []

    def produce():
        for i in range(4):
            accepted.append(worker.submit(candidate(tmp_path, f"window{i}")))

    producer = threading.Thread(target=produce)
    try:
        producer.start()
        assert entered.wait(2)
        producer.join(0.3)
        # One window is screening and one waits in the queue; the producer waits
        # for space rather than recording later windows as not run.
        assert producer.is_alive()
        assert not any(row["status"] == "not_run" for row in store.snapshot())
    finally:
        finish.set()
        producer.join(5)
        worker.close()
    assert accepted == [True] * 4
    assert [row["status"] for row in store.snapshot()] == ["screened"] * 4
    assert store.pending == 0 and not store.active


def test_uploads_wait_for_screening_space(monkeypatch, tmp_path):
    import vision_runtime.review as review

    monkeypatch.setenv("VISION_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("GEMMA_BASE_URL", raising=False)
    options = []

    class Recording(ScreeningQueue):
        def __init__(self, *args, **kwargs):
            options.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(review, "ScreeningQueue", Recording)
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buffer, "PNG")
    reviewer = Reviewer(Classifier())
    result = reviewer.analyze(buffer.getvalue(), "image.png")
    reviewer.release(result["run_id"])
    reviewer.close()
    assert options == [{"wait_for_space": True}]


def test_long_view_spans_twenty_seconds_with_two_fps_buffer(tmp_path):
    """Drifting ~29.97 fps timestamps at a 2 fps buffer still give a full long view."""
    from data_pipeline.detection.stage1 import Stage1Detector
    from data_pipeline.video_ingestion.decoding.decoder import DecodedFrame

    from vision_runtime.features import temporal_support
    from vision_runtime.tracking.gate import HandoffWriter

    class Perception:
        def __init__(self, cfg):
            pass

        def reset(self, fps):
            pass

        def observe(self, frame, index, t):
            return SimpleNamespace(t=t, index=index)

    class Engine:
        crop_check = None

        def __init__(self, cfg, fps):
            pass

        def step(self, obs, frame):
            return []

        def flush(self):
            return []

    gate = SimpleNamespace(
        Perception=Perception, RuleEngine=Engine, HandoffWriter=HandoffWriter
    )
    detector = Stage1Detector(
        "camera", tmp_path, fps=5, cfg=object(), gate=gate, safety_scan_seconds=10
    )
    session, events = uuid4(), []
    image = np.zeros((8, 8, 3), np.uint8)
    for index in range(160):  # 32 s at 5 fps, 200.2 ms apart (29.97 fps source)
        frame = DecodedFrame(0, "camera", session, index, round(index * 200.2), image)
        events += detector.process(frame)
    reviews = [e for e in events if e["timestamp_s"] >= 20]
    assert reviews
    for event in reviews:
        long = event["views"]["long"]
        assert long["times_s"][-1] - long["times_s"][0] >= 19.5
        assert temporal_support("loitering", long["times_s"], len(long["paths"]))


def test_only_flagged_tasks_are_confirmed_and_at_most_once_per_interval(tmp_path):
    asked = []

    class Flags:
        def classify(self, event):
            return {
                "predictions": {
                    "smoking": {"decision": "anomaly", "score": 1.0, "alert": False},
                    "littering": {"decision": "normal", "score": -1.0, "alert": False},
                },
                "seconds": 0.0,
            }

    class Gemma:
        def classify(self, event):
            asked.append((event["timestamp_s"], tuple(event["tasks"])))
            return {
                "predictions": {
                    "smoking": {
                        "decision": "anomaly",
                        "frame_refs": [1],
                        "evidence": "Cigarette at mouth",
                        "alert": True,
                    }
                },
                "summary": "One person smokes by the door.",
            }

    review = ConfirmationQueue(Gemma())
    store = EventStore(tmp_path)
    screening = ScreeningQueue(Flags(), store, review, confirm_interval=60)
    try:
        for name, at in [("a", 10.0), ("b", 20.0), ("c", 75.0)]:
            event = candidate(tmp_path, name)
            event.update(tasks=["smoking", "littering"], timestamp_s=at)
            screening.submit(event)
            # Let each review finish, so only the interval decides.
            screening.queue.join()
            review.queue.join()
    finally:
        screening.close()
        review.close()
    # 20 s is inside the 60 s interval of the 10 s review; 75 s is outside it.
    assert asked == [(10.0, ("smoking",)), (75.0, ("smoking",))]
    rows = {row["timestamp_s"]: row for row in store.snapshot()}
    assert rows[10.0]["status"] == "reviewed" and rows[20.0]["status"] == "screened"
    # The journal keeps screening results for the task that was not reviewed.
    assert set(rows[10.0]["predictions"]) == {"smoking", "littering"}
    assert set(rows[10.0]["confirmation"]) == {"smoking"}
    assert rows[10.0]["confirmation_details"]["summary"].startswith("One person")


def test_gemma_answer_carries_a_scene_summary():
    from vision_runtime.evidence import (
        headline,
        output_schema,
        parse_answer,
        parse_summary,
        prompt_style,
    )

    text = (
        '{"smoking": {"decision": "anomaly", "frame_refs": [1], "evidence": "Cigarette lit"},'
        ' "summary": "A man   smokes beside a door."}'
    )
    parsed = parse_answer(text, ["smoking"], 2)
    assert parsed["smoking"]["decision"] == "anomaly"
    assert parse_summary(text) == "A man smokes beside a door."
    assert parse_summary("not json") is None
    # The summary is written first for long-view tasks, last for littering.
    assert output_schema(["smoking"], 2)["required"] == ["summary", "smoking"]
    assert output_schema(["littering"], 8, prompt_style(view="short"))["required"] == [
        "littering",
        "summary",
    ]
    guarded = {"decision": "unknown", "model_decision_before_guard": "anomaly"}
    assert headline(
        {"unattended_items": guarded, "smoking": {"decision": "normal"}}
    ) == ("confirmed unattended items; did not see smoking")


def test_no_second_review_while_the_first_is_still_running(tmp_path):
    entered, finish, asked = threading.Event(), threading.Event(), []

    class Flags:
        def classify(self, event):
            return {
                "predictions": {"smoking": {"decision": "anomaly", "alert": False}},
                "seconds": 0.0,
            }

    class SlowGemma:
        def classify(self, event):
            asked.append(event["timestamp_s"])
            entered.set()
            assert finish.wait(5)
            return {
                "predictions": {
                    "smoking": {
                        "decision": "anomaly",
                        "frame_refs": [1],
                        "evidence": "Cigarette at mouth",
                        "alert": True,
                    }
                }
            }

    review = ConfirmationQueue(SlowGemma())
    store = EventStore(tmp_path)
    screening = ScreeningQueue(Flags(), store, review, confirm_interval=1)
    try:
        for name, at in [("a", 10.0), ("b", 50.0)]:
            event = candidate(tmp_path, name)
            event["timestamp_s"] = at
            screening.submit(event)
            screening.queue.join()
            assert entered.wait(2)
        assert review.busy("camera", "smoking")
    finally:
        finish.set()
        screening.close()
        review.close()
    # 50 s is past the interval, but the 10 s review had not finished.
    assert asked == [10.0]
    assert not review.busy("camera", "smoking")
