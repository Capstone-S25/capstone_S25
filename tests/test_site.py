"""Per-camera calibration on a site's own ordinary footage."""

import json

import pytest

from vision_runtime import site
from vision_runtime.ensemble import EnsembleStrategy


def journal(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def screened(event, scores, status="screened", strategy="ensemble-20261114"):
    return {
        "event_id": event,
        "status": status,
        "predictions": {
            task: {"decision": "normal", "score": value, "strategy": strategy}
            for task, value in scores.items()
        },
    }


def test_journal_scores_take_each_screened_window_once(tmp_path):
    file = journal(
        tmp_path / "events.jsonl",
        [
            {"event_id": "a", "status": "candidate"},
            screened("a", {"littering": 0.2, "smoking": -1.0}),
            # Later transitions repeat the predictions; they are not new windows.
            screened("a", {"littering": 0.2}, status="confirmation_pending"),
            screened("b", {"littering": 0.9}),
            screened("c", {"littering": 5.0}, strategy="ensemble-20261113"),
            {
                "event_id": "d",
                "status": "screened",
                "predictions": {"loitering": {"decision": "unknown"}},
            },
        ],
    )
    assert site.journal_scores([file], "ensemble-20261114") == {
        "littering": [0.2, 0.9],
        "smoking": [-1.0],
    }


def test_calibration_only_raises_and_respects_the_share():
    scores = {"littering": [0.01 * k for k in range(100)], "smoking": [-2.0] * 100}
    result = site.calibrate(
        scores, {"littering": 0.5, "smoking": 0.0, "loitering": 0.4}
    )
    # 2% of 100 windows may stay flagged: the two highest.
    assert result["littering"]["flagged_before"] == pytest.approx(0.5)
    assert result["littering"]["flagged_after"] == pytest.approx(0.02)
    assert 0.97 < result["littering"]["threshold"] <= 0.98
    # Nothing was flagged: the model's own threshold stays.
    assert result["smoking"]["threshold"] == 0.0
    # No windows for loitering: not calibrated at all.
    assert "loitering" not in result
    strict = site.calibrate(scores, {"littering": 0.5}, share=0)
    assert strict["littering"]["flagged_after"] == 0
    assert site.calibrate({"littering": [9.0] * 5}, {"littering": 0.5}) == {}


def test_saved_calibration_applies_to_its_camera_model_and_point(tmp_path):
    path = tmp_path / "site.json"
    tasks = site.calibrate({"littering": [1.0] * 40}, {"littering": 0.5}, share=0)
    site.save("camera_01", "ensemble-20261114", "balanced", tasks, 0.0, path)
    table = site.load("ensemble-20261114", "balanced", path)
    assert table["camera_01"]["littering"] > 1.0
    # Another model's scores are on another scale; so are other thresholds.
    assert site.load("ensemble-20261113", None, path) == {}
    assert site.load("ensemble-20261114", "low_false_alarm", path) == {}

    strategy = object.__new__(EnsembleStrategy)
    strategy.tasks = {"littering": {"threshold": 0.5}, "smoking": {"threshold": 0.0}}
    strategy.site = table
    assert strategy.threshold("littering", "camera_01") == (
        table["camera_01"]["littering"],
        0.5,
    )
    assert strategy.threshold("littering", "camera_02") == (0.5, 0.5)
    assert strategy.threshold("smoking", "camera_01") == (0.0, 0.0)
    # A calibration below the model's threshold never lowers it.
    strategy.site = {"camera_01": {"littering": 0.1}}
    assert strategy.threshold("littering", "camera_01") == (0.5, 0.5)

    assert (
        site.clear("camera_01", path)
        and site.load("ensemble-20261114", "balanced", path) == {}
    )
    assert not site.clear("camera_01", path)
