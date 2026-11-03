import json
import sys
import threading

import pytest

from data_pipeline.detection.reasoning import (
    CandidateService,
    CascadeClient,
    ResidentClient,
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
