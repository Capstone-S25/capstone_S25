"""Live camera lifecycle, bounded work, timestamps and evidence retention."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vision_runtime.live import CameraManager, LiveSession
from vision_runtime.review import LockedClassifier
from vision_runtime.service import EventStore, ScreeningQueue


def wait_until(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for a camera state")


class Capture:
    def __init__(self, *, opened=True, fail_after=None):
        self.opened, self.fail_after = opened, fail_after
        self.count = 0
        self.released = threading.Event()

    def isOpened(self):
        return self.opened

    def read(self):
        time.sleep(0.01)
        self.count += 1
        if self.fail_after is not None and self.count > self.fail_after:
            return False, None
        return True, np.full((24, 32, 3), self.count % 255, np.uint8)

    def release(self):
        self.released.set()


class Classifier:
    metadata = {"model": "fixture", "device": "cpu"}

    def classify(self, event):
        return {
            "predictions": {
                t: {"decision": "normal", "score": -1, "threshold": 0, "alert": False}
                for t in event["tasks"]
            },
            "seconds": 0.001,
        }


class Detector:
    def __init__(self, source_id, folder, fps, scan_seconds):
        self.source = source_id
        self.folder = folder
        self.frames = []
        self.closed = False

    def process(self, frame):
        self.frames.append(frame)
        path = self.folder / f"frame_{frame.frame_index}.jpg"
        Image.fromarray(frame.image).save(path)
        view = {"paths": [str(path)], "times_s": [frame.timestamp_ms / 1000]}
        return [
            {
                "event_id": f"camera_{frame.frame_index}",
                "source_id": self.source,
                "timestamp_s": (frame.timestamp_ms - self.frames[0].timestamp_ms)
                / 1000,
                "tasks": ["smoking"],
                "event_type": "periodic_review",
                "context_frame_path": str(path),
                "views": {"long": view},
            }
        ]

    def finish(self):
        return []

    def _close_pool(self):
        self.closed = True


@pytest.fixture
def camera_setup(monkeypatch, tmp_path):
    monkeypatch.setenv("VISION_RUNTIME_DIR", str(tmp_path))
    reviewer = SimpleNamespace(
        classifier=LockedClassifier(Classifier()), confirmation=None
    )
    capture = Capture()
    detectors = []

    def detector(*args):
        detectors.append(Detector(*args))
        return detectors[-1]

    options = {"capture_factory": lambda *args: capture, "detector_factory": detector}
    return reviewer, capture, detectors, options


def test_capture_runs_while_classifier_is_blocked_and_stop_releases_device(
    camera_setup,
):
    reviewer, capture, detectors, options = camera_setup
    entered, finish = threading.Event(), threading.Event()

    class Slow(Classifier):
        def classify(self, event):
            entered.set()
            assert finish.wait(5)
            return super().classify(event)

    reviewer.classifier = LockedClassifier(Slow())
    session = LiveSession(reviewer, **options)
    try:
        assert entered.wait(2)
        first = session.snapshot()["report"]["captured_frames"]
        wait_until(
            lambda: session.snapshot()["report"]["captured_frames"] >= first + 10
        )
        assert session.snapshot()["frame"]
        session.stop()
        assert capture.released.wait(1)
        assert not session.done.is_set()  # AI drains after the camera closes.
    finally:
        finish.set()
        session.stop()
        wait_until(lambda: not session.active)
        session.release()
    assert detectors[0].closed
    assert not session.folder.exists()


def test_unavailable_camera_is_an_error_and_never_starts_tracking(camera_setup):
    reviewer, capture, detectors, options = camera_setup
    capture.opened = False
    session = LiveSession(reviewer, **options)
    try:
        wait_until(lambda: not session.active)
        result = session.snapshot()
        assert result["state"] == "error"
        assert "Cannot open camera 0" in result["error"]
        assert not detectors
        assert capture.released.is_set()
    finally:
        session.release()


def test_tracking_initialization_failure_closes_camera(camera_setup):
    reviewer, capture, _, options = camera_setup

    def fail(*args):
        raise RuntimeError("Missing tracking weights")

    options["detector_factory"] = fail
    session = LiveSession(reviewer, **options)
    try:
        wait_until(lambda: not session.active)
        assert session.snapshot()["error"] == "Missing tracking weights"
        assert capture.released.is_set()
    finally:
        session.release()


def test_live_summary_uses_latest_completed_signal_instead_of_past_positive(
    camera_setup,
):
    reviewer, capture, _, options = camera_setup
    capture.opened = False
    session = LiveSession(reviewer, **options)
    try:
        wait_until(lambda: not session.active)
        for ident, stamp, decision, score in [
            ("past_positive", 0, "anomaly", 10),
            ("recent_negative", 5, "normal", -1),
        ]:
            session.store.write(
                {
                    "event_id": ident,
                    "timestamp_s": stamp,
                    "event_type": "periodic_review",
                },
                "screened",
                predictions={
                    "smoking": {
                        "decision": decision,
                        "score": score,
                        "threshold": 0,
                        "alert": False,
                    }
                },
            )
        session.store.write(
            {
                "event_id": "still_waiting",
                "timestamp_s": 6,
                "event_type": "periodic_review",
            },
            "queued",
        )
        report = session.snapshot()["report"]
        assert report["summary"]["smoking"]["decision"] == "normal"
        assert report["summary"]["smoking"]["timestamp_s"] == 5
        assert report["latest_screened_timestamp_s"] == 5
    finally:
        session.release()


def test_unplug_and_idle_disconnect_stop_the_device(camera_setup):
    reviewer, capture, _, options = camera_setup
    capture.fail_after = 3
    session = LiveSession(reviewer, **options)
    try:
        wait_until(lambda: not session.active)
        assert "stopped delivering frames" in session.snapshot()["error"]
        assert capture.released.is_set()
    finally:
        session.release()
    capture = Capture()
    options["capture_factory"] = lambda *args: capture
    session = LiveSession(reviewer, idle_timeout=0.1, **options)
    wait_until(lambda: not session.active)
    assert capture.released.is_set()
    assert session.discarded
    assert not session.folder.exists()


def test_rolling_evidence_is_bounded_and_camera_indices_are_exclusive(camera_setup):
    reviewer, capture, detectors, options = camera_setup
    manager = CameraManager(reviewer, fps=40, keep_events=3, **options)
    session = manager.start(0)
    try:
        with pytest.raises(RuntimeError, match="still active"):
            manager.start(1)
        wait_until(lambda: session.snapshot()["report"]["retired_windows"] >= 4)
        session.stop()
        wait_until(lambda: not session.active)
        report = session.snapshot()["report"]
        assert len(report["events"]) <= 3
        assert len(list(session.folder.glob("frame_*.jpg"))) <= 3
        assert report["retired_windows"] >= 4
        stamps = [f.timestamp_ms for f in detectors[0].frames]
        assert all(b > a for a, b in zip(stamps, stamps[1:]))
        # 100 FPS capture, 40 FPS tracking: indices skip while time stays real.
        assert session.skipped > 0
        assert stamps[-1] - stamps[0] >= 100
        assert report["summary"]["smoking"]["decision"] == "normal"
        assert report["summary"]["loitering"]["decision"] == "unknown"
    finally:
        manager.close()
        wait_until(lambda: not session.active)
    assert capture.released.is_set()
    assert not session.folder.exists()


def test_prune_keeps_evidence_until_all_screening_and_confirmation_refs_finish(
    tmp_path,
):
    store = EventStore(tmp_path)
    old = {"event_id": "old", "timestamp_s": 0, "event_type": "periodic_review"}
    new = {**old, "event_id": "new", "timestamp_s": 1}
    store.retain("old")
    store.retain("old")
    store.write(old, "reviewed")
    store.write(new, "reviewed")
    assert store.prune(1) == []
    store.done("old")
    assert store.prune(1) == []
    store.done("old")
    assert store.prune(1) == ["old"]
    assert store.pending == 0 and not store.active
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 1


def test_live_queue_replaces_waiting_work_with_newest_evidence(tmp_path):
    entered, finish = threading.Event(), threading.Event()
    seen = []

    class Slow(Classifier):
        def classify(self, event):
            seen.append(event["event_id"])
            if event["event_id"] == "first":
                entered.set()
                assert finish.wait(5)
            return super().classify(event)

    store = EventStore(tmp_path)
    worker = ScreeningQueue(Slow(), store, capacity=1, latest_only=True)

    def event(ident):
        return {
            "event_id": ident,
            "timestamp_s": 0,
            "event_type": "periodic_review",
            "tasks": ["smoking"],
            "views": {},
        }

    try:
        worker.submit(event("first"))
        assert entered.wait(2)
        worker.submit(event("old"))
        worker.submit(event("newest"))
        old = next(e for e in store.snapshot() if e["event_id"] == "old")
        assert old["status"] == "not_run"
        assert old["reason"] == "newer_live_evidence_available"
    finally:
        finish.set()
        worker.close()
    assert seen == ["first", "newest"]
    assert store.pending == 0 and not store.active


def test_discard_waits_for_active_confirmation_before_removing_files(
    camera_setup, monkeypatch
):
    from vision_runtime.service import ConfirmationQueue

    # This fixture's windows screen as normal; review them all to hold a job.
    monkeypatch.setenv("GEMMA_CONFIRM", "all")

    reviewer, _, _, options = camera_setup
    entered, finish = threading.Event(), threading.Event()

    class Slow:
        def classify(self, event):
            entered.set()
            assert finish.wait(5)
            assert all(
                Path(p).exists() for v in event["views"].values() for p in v["paths"]
            )
            return {"predictions": {"smoking": {"decision": "normal", "alert": False}}}

    reviewer.confirmation = ConfirmationQueue(Slow())
    session = LiveSession(reviewer, **options)
    try:
        assert entered.wait(2)
        session.release()
        wait_until(lambda: not session.active)
        assert session.folder.exists()
    finally:
        finish.set()
        reviewer.confirmation.close()
        session.release()
    assert not session.folder.exists()
