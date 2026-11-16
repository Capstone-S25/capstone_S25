import json

from data_pipeline.detection.vlm import CandidateLog, NoVLM, PredictClipVLM


def candidate(tasks):
    return {
        "source_id": "camera_01",
        "event_type": "sustained_presence",
        "tasks": tasks,
        "views": {
            "short": {"paths": ["s0.jpg", "s1.jpg"], "times_s": [3.5, 4.0]},
            "long": {"paths": ["l0.jpg", "l1.jpg"], "times_s": [0, 20.5]},
        },
    }


def fake_runner(calls):
    def run(command):
        calls.append(command)
        images = command[command.index("--images") + 1 : command.index("--timestamps")]
        output = command[command.index("--output") + 1]
        predictions = {
            task: {"decision": "anomaly", "images": images}
            for task in ("littering", "loitering", "smoking", "rough_sleeping")
        }
        with open(output, "w") as fh:
            json.dump({"predictions": predictions}, fh)

    return run


def test_predict_clip_uses_one_call_per_view_set():
    calls = []
    vlm = PredictClipVLM("py", ".", runner=fake_runner(calls))

    result = vlm.classify(candidate(["loitering", "smoking", "littering"]))

    assert len(calls) == 2
    assert calls[0][:5] == ["py", "-m", "anomaly_eval", "predict-clip", "--model"]
    assert result["loitering"]["images"] == ["l0.jpg", "l1.jpg"]
    assert result["smoking"]["view"] == "long"
    assert result["littering"]["images"] == ["s0.jpg", "s1.jpg"]
    assert set(result) == {"loitering", "smoking", "littering"}


def test_timestamps_are_passed_in_order():
    vlm = PredictClipVLM("py", ".")

    command = vlm.command(candidate([])["views"]["long"], "out.json")

    i = command.index("--timestamps")
    assert command[i + 1 : i + 3] == ["0", "20.5"]


def test_candidate_log_appends_jsonl(tmp_path):
    log = CandidateLog(tmp_path, NoVLM())

    log(candidate(["loitering"]))
    log(candidate(["littering"]))

    lines = (tmp_path / "camera_01" / "candidates.jsonl").read_text().splitlines()
    assert [json.loads(line)["vlm"] for line in lines] == [
        {"loitering": {"decision": "not_run"}},
        {"littering": {"decision": "not_run"}},
    ]
