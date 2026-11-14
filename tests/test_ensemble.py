"""Ensemble classifier: feature contracts and the installed bundle."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_runtime.classifier import ENSEMBLES, load_classifier
from vision_runtime.config import models_root
from vision_runtime.ensemble import (
    LRU,
    EnsembleStrategy,
    crop_box,
    dwell,
    operating_point,
    pool,
)


def person(x, y, w=0.1, h=0.3, label=1, confidence=0.9):
    return {"box": [x, y, x + w, y + h], "label": label, "confidence": confidence}


def test_dwell_separates_staying_from_passing():
    staying = [[person(0.4, 0.4)] for _ in range(4)]
    passing = [[person(0.1 + 0.25 * k, 0.4)] for k in range(4)]
    stay, walk = dwell(staying), dwell(passing)
    # Person part: minimum IoU over the window, then centre shift.
    assert stay[0] == pytest.approx(1.0) and walk[0] == 0.0
    assert stay[3] == pytest.approx(0.0) and walk[3] > 1.0
    # No bags: the bag part reports "nothing stayed" and no nearby person.
    assert list(stay[6:]) == [0.0, 0.0, 0.0, 2.0, 0.0, 1.5]


def test_unattended_bag_far_from_people_is_measured():
    bag = person(0.7, 0.7, 0.05, 0.05, label=31)
    found = [[bag, person(0.1, 0.1)] for _ in range(4)]
    values = dwell(found)
    assert values[6] == pytest.approx(1.0)  # the bag never moved
    assert values[11] > 0.5  # distance from the bag to the nearest person


def test_crop_box_unions_top_person_and_item():
    box = crop_box(
        [person(0.1, 0.1), person(0.6, 0.6, 0.1, 0.1, label=27, confidence=0.6)]
    )
    assert box[0] < 0.1 and box[2] > 0.7
    assert crop_box([]) is None


def test_pool_keeps_endpoint_mean_max_and_change():
    views = np.eye(3, dtype=np.float32)
    value = pool(views)
    assert value.shape == (12,)
    assert np.linalg.norm(value) == pytest.approx(1.0)


def test_offset_is_window_minus_reference(tmp_path):
    """Offset recipes score departure from the camera's own reference scene."""
    strategy = object.__new__(EnsembleStrategy)
    strategy.cache = LRU(64)
    rng = np.random.default_rng(0)
    vectors = {}

    def embed(backbone, stage, paths, keys):
        for p in paths:
            vectors.setdefault((stage, p), rng.normal(size=8).astype(np.float32))
        return {p: vectors[stage, p] / np.linalg.norm(vectors[stage, p]) for p in paths}

    strategy._embed = embed
    strategy._boxes = lambda source, paths, keys: {p: [person(0.4, 0.4)] for p in paths}
    views = [f"v{k}" for k in range(4)]
    recipe = {"backbone": "b", "geometry": True}
    plain = strategy._vector(recipe, views, "ref", {})
    reference = strategy._vector(recipe, ["ref"] * 4, "ref", {})
    shifted = strategy._vector({**recipe, "offset": True}, views, "ref", {})
    assert np.allclose(shifted, plain - reference)


def bare_strategy(tasks, vectors=None, smoking=0.0):
    """An EnsembleStrategy with stand-in encoders, for the scoring rules."""
    strategy = object.__new__(EnsembleStrategy)
    strategy.cache = LRU(64)
    strategy.tasks = tasks
    strategy.site = {}
    strategy._key = lambda path: path
    strategy._embed = lambda backbone, stage, paths, keys: {
        p: np.asarray(vectors[stage, p], np.float32) for p in paths
    }
    strategy._smoking = lambda paths, keys: np.array([[smoking, 0.0, 0.0]] * len(paths))
    return strategy


def text_member(**changes):
    return {
        "kind": "zeroshot",
        "backbone": "b",
        "stages": ["full"],
        "positive": np.array([[1.0, 0.0], [0.6, 0.8]], np.float32),
        "negative": np.array([[0.0, 1.0]], np.float32),
        "mean": 0.0,
        "std": 1.0,
    } | changes


def test_text_member_is_behaviour_minus_ordinary_similarity():
    vectors = {("full", "smoker"): [1.0, 0.0], ("full", "walker"): [0.0, 1.0]}
    strategy = bare_strategy({}, vectors)
    member = text_member()
    # Best behaviour description minus best ordinary description.
    assert strategy._member(member, ["smoker"], "smoker", {}) == pytest.approx(1.0)
    assert strategy._member(member, ["walker"], "walker", {}) == pytest.approx(-0.2)
    # Views are averaged, then the stored mean and spread standardize.
    both = strategy._member(
        text_member(mean=0.4, std=2.0), ["smoker", "walker"], "walker", {}
    )
    assert both == pytest.approx(0.0, abs=1e-6)


def test_channels_flag_when_any_passes_its_own_threshold():
    tasks = {
        "smoking": {
            "threshold": 0.0,
            "channels": [
                {
                    "name": "scene and person views",
                    "threshold": 0.5,
                    "members": [text_member()],
                },
                {
                    "name": "smoking detector",
                    "threshold": 0.3,
                    "members": [{"kind": "detector", "mean": 0.1, "std": 0.5}],
                },
            ],
        }
    }
    vectors = {("full", "walker"): [0.0, 1.0], ("full", "smoker"): [1.0, 0.0]}
    # Only the detector sees it: (0.6 - 0.1) / 0.5 = 1.0, margin 0.7.
    strategy = bare_strategy(tasks, vectors, smoking=0.6)
    score, channel = strategy.score_detail("smoking", ["walker"], None)
    assert channel == "smoking detector" and score == pytest.approx(0.7)
    # Only the views see it: margin 1.0 - 0.5.
    strategy = bare_strategy(tasks, vectors, smoking=0.0)
    score, channel = strategy.score_detail("smoking", ["smoker"], None)
    assert channel == "scene and person views" and score == pytest.approx(0.5)
    # Neither: the score is negative, so the window is not flagged at 0.
    assert strategy.score("smoking", ["walker"], None) < 0


def test_operating_point_selects_thresholds_and_calibration(monkeypatch):
    def tasks():
        return {
            "smoking": {
                "threshold": 0.0,
                "thresholds": {"balanced": 0.0, "low_false_alarm": 0.0},
                "calibrations": {
                    "balanced": {"slope": 1.0, "intercept": 0.0},
                    "low_false_alarm": {"slope": 2.0, "intercept": 0.5},
                },
                "channels": [
                    {
                        "threshold": 9,
                        "thresholds": {"balanced": 0.2, "low_false_alarm": 0.6},
                    }
                ],
            },
            "littering": {
                "threshold": 9,
                "thresholds": {"balanced": -0.1, "low_false_alarm": 0.4},
            },
        }

    bundle = {
        "operating_points": ["balanced", "low_false_alarm"],
        "default_operating_point": "low_false_alarm",
    }
    monkeypatch.delenv("VISION_OPERATING_POINT", raising=False)
    chosen = tasks()
    assert operating_point(bundle, chosen) == "low_false_alarm"
    assert chosen["littering"]["threshold"] == 0.4
    assert chosen["smoking"]["channels"][0]["threshold"] == 0.6
    assert chosen["smoking"]["calibration"]["slope"] == 2.0
    monkeypatch.setenv("VISION_OPERATING_POINT", "balanced")
    chosen = tasks()
    assert operating_point(bundle, chosen) == "balanced"
    assert chosen["littering"]["threshold"] == -0.1
    assert chosen["smoking"]["channels"][0]["threshold"] == 0.2
    monkeypatch.setenv("VISION_OPERATING_POINT", "strict")
    with pytest.raises(ValueError, match="VISION_OPERATING_POINT"):
        operating_point(bundle, tasks())
    # Bundles before 20261114 have one threshold and ignore the setting.
    old = {"littering": {"threshold": 0.3}}
    assert operating_point({}, old) is None and old["littering"]["threshold"] == 0.3


installed = [
    name for name in ENSEMBLES if (models_root() / name / "ensemble.json").is_file()
]


def frames(folder, count, seed):
    rng = np.random.default_rng(seed)
    paths = []
    for k in range(count):
        path = Path(folder) / f"f{seed}_{k}.jpg"
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    return paths


@pytest.mark.skipif(not installed, reason="ensemble bundle is not installed")
@pytest.mark.parametrize("name", installed)
def test_installed_ensemble_scores_every_supported_task(tmp_path, name):
    strategy = load_classifier(name)
    assert strategy.metadata["model"] == name
    long_paths = frames(tmp_path, 21, 1)
    short_paths = frames(tmp_path, 8, 2)
    candidate = {
        "event_id": "fixture",
        "timestamp_s": 20.0,
        "tasks": [
            "smoking",
            "littering",
            "loitering",
            "rough_sleeping",
            "unattended_items",
        ],
        "views": {
            "long": {"paths": long_paths, "times_s": [float(t) for t in range(21)]},
            "short": {
                "paths": short_paths,
                "times_s": [16 + 0.5 * k for k in range(8)],
            },
        },
        "reference_frame_path": frames(tmp_path, 1, 3)[0],
    }
    result = strategy.classify(candidate)["predictions"]
    for task, value in result.items():
        assert np.isfinite(value["score"]), task
        assert value["decision"] in {"normal", "anomaly"}
        if "calibration" in strategy.tasks[task]:
            assert 0 <= value["probability"] <= 1
            # The calibration is monotone in the score.
            assert (
                strategy._probability(task, value["score"] + 1)["probability"]
                > (value["probability"])
            )
    short = dict(
        candidate,
        views={
            **candidate["views"],
            "long": {"paths": long_paths[:6], "times_s": [float(t) for t in range(6)]},
        },
    )
    later = strategy.classify(short)["predictions"]
    assert later["loitering"]["decision"] == "unknown"
    assert later["unattended_items"]["decision"] == "unknown"
