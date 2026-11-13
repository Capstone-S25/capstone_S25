"""Ensemble classifier: feature contracts and the installed bundle."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_runtime.config import models_root
from vision_runtime.ensemble import LRU, EnsembleStrategy, crop_box, dwell, pool


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


installed = (models_root() / "ensemble-20261111" / "ensemble.json").is_file()


def frames(folder, count, seed):
    rng = np.random.default_rng(seed)
    paths = []
    for k in range(count):
        path = Path(folder) / f"f{seed}_{k}.jpg"
        Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)).save(path)
        paths.append(str(path))
    return paths


@pytest.mark.skipif(not installed, reason="ensemble bundle is not installed")
def test_installed_ensemble_scores_every_supported_task(tmp_path):
    strategy = EnsembleStrategy(device="cpu")
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
