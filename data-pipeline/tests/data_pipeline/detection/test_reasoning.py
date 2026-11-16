import json
import sys
import threading

import pytest

from data_pipeline.detection.reasoning import (
    CandidateService,
    CascadeClient,
    GroupedCandidateService,
    ResidentClient,
    add_arguments,
    build_service,
)


class Client:
    def __init__(self):
        self.started = 0
        self.closed = 0
        self.calls = []

    def start(self):
        self.started += 1

    def close(self):
        self.closed += 1

    def classify(self, candidate):
        self.calls.append(candidate["event_id"])
        return {
            "predictions": {
                t: {"decision": "normal", "alert": False} for t in candidate["tasks"]
            }
        }


def candidate(i):
    return {"event_id": str(i), "source_id": "camera", "tasks": ["smoking"]}


def test_fifo_overflow_and_drain_keep_one_terminal_record(tmp_path):
    active, release = threading.Event(), threading.Event()

    class Slow(Client):
        def classify(self, c):
            active.set()
            assert release.wait(3)
            return super().classify(c)

    client = Slow()
    service = CandidateService(tmp_path, client, max_pending=1)
    assert service.submit(candidate(1))
    assert active.wait(2)
    assert service.submit(candidate(2))
    assert not service.submit(candidate(3))
    release.set()
    service.close()
    assert client.calls == ["1", "2"] and client.started == 1 and client.closed == 1
    assert service.counts == {
        "submitted": 2,
        "completed": 2,
        "failed": 0,
        "rejected": 1,
    }
    rows = [
        json.loads(s)
        for s in (tmp_path / "camera/candidates.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert (
        next(r for r in rows if r["event_id"] == "3")["vlm"]["smoking"]["decision"]
        == "not_run"
    )
    assert all(
        r["response_seconds"] >= r["queue_seconds"]
        for r in rows
        if r["status"] == "completed"
    )
    assert (
        len((tmp_path / "camera/candidate_journal.jsonl").read_text().splitlines()) == 6
    )


def test_backend_failure_is_unknown_and_does_not_kill_service(tmp_path):
    class Broken(Client):
        def classify(self, c):
            raise RuntimeError("test GPU failure")

    service = CandidateService(tmp_path, Broken())
    service.submit(candidate(1))
    service.close()
    row = json.loads((tmp_path / "camera/candidates.jsonl").read_text())
    assert row["status"] == "error" and row["vlm"]["smoking"]["decision"] == "unknown"
    assert not row["vlm"]["smoking"]["alert"]


def test_classifier_negative_never_prevents_reasoner_call():
    classifier, reasoner = Client(), Client()
    reasoner.classify = lambda c: {
        "predictions": {"smoking": {"decision": "anomaly", "alert": True}}
    }
    result = CascadeClient(classifier, reasoner).classify(candidate(1))
    assert result["predictions"]["smoking"]["decision"] == "anomaly"
    assert result["predictions"]["smoking"]["review_required"]
    assert not result["predictions"]["smoking"]["alert"]


def test_nonfinite_timeouts_fail_before_starting_any_process(tmp_path):
    for value in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            ResidentClient(
                [sys.executable], tmp_path, tmp_path / "worker.log", timeout=value
            )


def test_resident_process_is_loaded_once_and_response_ids_match(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(
        'import json,sys,os\nprint(json.dumps({"type":"ready","pid":os.getpid()}),flush=True)\n'
        'for line in sys.stdin:\n r=json.loads(line);print(json.dumps({"id":r["id"],"result":{"pid":os.getpid()}}),flush=True)\n'
    )
    client = ResidentClient(
        [sys.executable, "-u", script], tmp_path, tmp_path / "worker.log"
    )
    try:
        a = client.classify(candidate(1))
        b = client.classify(candidate(2))
        assert a["pid"] == b["pid"] == client.ready["pid"]
    finally:
        client.close()
    assert client.process is None


def test_worker_timeout_stops_owned_process(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(
        'import json,sys,time\nprint(json.dumps({"type":"ready"}),flush=True)\nfor line in sys.stdin:time.sleep(10)\n'
    )
    client = ResidentClient(
        [sys.executable, "-u", script], tmp_path, tmp_path / "worker.log", timeout=0.05
    )
    with pytest.raises(TimeoutError):
        client.classify(candidate(1))
    assert client.process is None


class Inner:
    """Stands in for CandidateService: records what reaches the model queue."""

    def __init__(self, admit=True):
        self.admit = admit
        self.sent = []
        self.events = []

    def start(self):
        self.events.append("start")

    def submit(self, candidate):
        self.sent.append(candidate)
        return self.admit

    def close(self):
        self.events.append(("close", len(self.sent)))


def alert(
    event_id, t, entity="person1", *, event_type="reach_down", tasks=None, **extra
):
    return {
        "event_id": event_id,
        "source_id": "cam",
        "session_id": "s1",
        "continuity_epoch": 1,
        "entity_id": entity,
        "event_type": event_type,
        "timestamp_s": t,
        "tasks": tasks or ["littering", "unattended_items"],
        "roi_box": [t, 0, t + 1, 1],
        "views": {
            "short": {
                "paths": [f"{event_id}_a.jpg", "shared.jpg"],
                "times_s": [t, t - 1],
            },
            "long": {"paths": [f"{event_id}_l.jpg"], "times_s": [t]},
        },
        **extra,
    }


def grouped(tmp_path, admit=True, window_s=30.0):
    inner = Inner(admit)
    return inner, GroupedCandidateService(inner, tmp_path, window_s=window_s)


def test_first_alert_goes_straight_through_unchanged(tmp_path):
    inner, service = grouped(tmp_path)
    first = alert("a", 5.0)
    assert service.submit(first)
    [sent] = inner.sent
    assert sent["group_role"] == "first"
    assert {k: v for k, v in sent.items() if not k.startswith("group_")} == first
    assert "group_role" not in first  # the caller's candidate is not modified


def test_later_alerts_in_window_become_one_follow_up_on_the_media_clock(tmp_path):
    inner, service = grouped(tmp_path)
    service.submit(alert("a", 0.0))
    service.submit(
        alert("b", 10.0, event_type="hand_to_face", tasks=["smoking", "littering"])
    )
    service.submit(
        alert(
            "c",
            30.0,
            event_type="sustained_presence",
            tasks=["loitering", "smoking", "rough_sleeping"],
        )
    )
    assert [c["event_id"] for c in inner.sent] == ["a"]
    # Another person's alert after the window advances the clock and closes person1's group.
    service.submit(alert("x", 30.5, entity="person2"))
    first, follow, other = inner.sent
    assert other["event_id"] == "x" and other["group_role"] == "first"
    assert follow["event_id"] == "a_followup"
    assert follow["group_role"] == "follow_up"
    assert follow["grouped_event_ids"] == ["b", "c"] and follow["group_size"] == 2
    assert follow["tasks"] == ["littering", "loitering", "rough_sleeping", "smoking"]
    assert follow["event_type"] == "sustained_presence"
    assert follow["grouped_event_types"] == ["hand_to_face", "sustained_presence"]
    assert follow["roi_box"] == [30.0, 0, 31.0, 1]
    assert follow["roi_boxes"] == [[10.0, 0, 11.0, 1], [30.0, 0, 31.0, 1]]
    assert follow["timestamp_s"] == 30.0  # everything else from the last member
    assert follow["views"]["short"] == {
        "paths": ["shared.jpg", "b_a.jpg", "c_a.jpg"],
        "times_s": [9.0, 10.0, 30.0],
    }
    assert follow["views"]["long"] == {
        "paths": ["b_l.jpg", "c_l.jpg"],
        "times_s": [10.0, 30.0],
    }
    assert follow["group_key"] == ["cam", "s1", 1, "person1"]


def test_alert_after_the_window_starts_a_new_group(tmp_path):
    inner, service = grouped(tmp_path)
    service.submit(alert("a", 0.0))
    service.submit(alert("b", 31.0))
    assert [(c["event_id"], c["group_role"]) for c in inner.sent] == [
        ("a", "first"),
        ("b", "first"),
    ]


def test_replay_speed_does_not_matter(tmp_path):
    """Submitted back to back (as --replay-candidates does), alerts minutes
    apart in media time are still separate groups."""
    inner, service = grouped(tmp_path)
    for i in range(5):
        service.submit(alert(f"e{i}", i * 60.0))
    service.close()
    assert [c["event_id"] for c in inner.sent] == ["e0", "e1", "e2", "e3", "e4"]


def test_scene_reviews_and_other_identities_are_never_merged(tmp_path):
    inner, service = grouped(tmp_path)
    review = alert("r", 1.0, event_type="periodic_review")
    del review["entity_id"]
    service.submit(alert("a", 0.0))
    service.submit(review)
    service.submit(
        alert("b", 2.0, continuity_epoch=2)
    )  # tracker restarted: a new person1
    service.submit(alert("c", 3.0, session_id="s2"))  # new session closes s1's groups
    service.submit(alert("d", 4.0, entity="backpack7", event_type="static_object_left"))
    service.close()
    assert [c["event_id"] for c in inner.sent] == ["a", "r", "b", "c", "d"]
    assert "group_role" not in inner.sent[1]


def test_close_flushes_held_follow_ups_before_closing_the_inner_service(tmp_path):
    inner, service = grouped(tmp_path)
    service.start()
    service.submit(alert("a", 0.0))
    service.submit(alert("b", 5.0))
    service.close()
    assert [c["event_id"] for c in inner.sent] == ["a", "a_followup"]
    assert inner.events == ["start", ("close", 2)]
    summary = json.loads((tmp_path / "grouping_summary.json").read_text())
    assert summary["counts"]["inner_calls"] == 2 and summary["calls_saved"] == 0


def test_a_rejected_first_alert_opens_no_group(tmp_path):
    inner, service = grouped(tmp_path, admit=False)
    assert not service.submit(alert("a", 0.0))
    assert not service.submit(alert("b", 1.0))
    assert [c["group_role"] for c in inner.sent] == ["first", "first"]


def test_build_service_wraps_only_when_asked(tmp_path):
    import argparse

    parser = argparse.ArgumentParser()
    add_arguments(parser, tmp_path)
    assert isinstance(
        build_service(parser.parse_args([]), tmp_path, tmp_path), CandidateService
    )
    args = parser.parse_args(["--group-alerts", "--group-window-seconds", "12"])
    service = build_service(args, tmp_path, tmp_path)
    assert isinstance(service, GroupedCandidateService) and service.window_s == 12.0
