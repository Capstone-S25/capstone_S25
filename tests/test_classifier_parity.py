"""The classifier computes only the features its selected heads read.

Scores must match the original dense algorithm, which encoded every frame at
every interpolation weight and detected objects in every frame. That algorithm
is frozen below as the reference.
"""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from vision_runtime.classifier import ClipStrategy
from vision_runtime.config import model_paths
from vision_runtime.encoder import load_image, pool_views
from vision_runtime.evidence import (
    TASKS,
    VIEW_FOR_TASK,
    history_problem,
    unknown,
    validated_views,
)
from vision_runtime.features import (
    Views,
    matrix,
    predict,
    temporal_indices,
    temporal_support,
)

DEPLOYED = {
    "smoking": "full_crop_pool_a0",
    "littering": "full_pool_a0.5",
    "loitering": "full_geometry_a0",
    "rough_sleeping": "full_pool_a0.25",
    "unattended_items": "full_ordered_a1",
}
# Exercises endpoint views, ordered crops/geometry and crop at a nonzero weight.
VARIED = {
    "smoking": "endpoint_crop_a0.5",
    "littering": "full_geometry_ordered_a0",
    "loitering": "full_crop_geometry_pool_a1",
    "rough_sleeping": "endpoint_a0",
    "unattended_items": "full_pool_a0.25",
}


def reference_classify(self, candidate):
    """Frozen copy of the original dense ClipStrategy.classify scoring."""
    views = validated_views(candidate)
    out, inputs = {}, {}
    for task in candidate["tasks"]:
        view = views[VIEW_FOR_TASK[task]]
        if history_problem(task, view, clip=True) or not temporal_support(
            task, view["times_s"], len(view["paths"])
        ):
            out[task] = unknown("history")
            continue
        n = len(view["paths"])
        indices = (
            [n - 1]
            if task in {"smoking", "rough_sleeping"}
            else np.rint(np.linspace(0, n - 1, 8)).astype(int).tolist()
            if task == "littering"
            else temporal_indices(view["times_s"], n)
        )
        inputs[task] = [view["paths"][i] for i in indices]
    unique = list(dict.fromkeys(p for paths in inputs.values() for p in paths))
    if not unique:
        return out
    policies = {t: self.policy[t]["selected"] for t in inputs}
    strategies = [p["selection"]["strategy"] for p in policies.values()]
    with torch.inference_mode():
        detections = (
            self._detections(unique)
            if any("crop" in s or "geometry" in s for s in strategies)
            else {p: [] for p in unique}
        )
        full = torch.stack([Views(unique, self.encoder)[i] for i in range(len(unique))])
        need_crop = any("crop" in s for s in strategies)
        crop = (
            torch.stack(
                [Views(unique, self.encoder, detections)[i] for i in range(len(unique))]
            )
            if need_crop
            else None
        )
        alphas = sorted(
            {float(s.rsplit("a", 1)[1]) for s in strategies}
            | ({0.0} if need_crop else set())
        )
        features = {}
        for alpha in alphas:
            for name, base in self.original.items():
                self.parameters[name].copy_(
                    torch.lerp(base, self.previous["parameters"][name], alpha)
                )
            batch = torch.cat([full, crop]) if alpha == 0 and crop is not None else full
            pooled = (
                self.encoder.encode(batch, [1] * len(batch))
                .cpu()
                .numpy()[:, : self.encoder.dimension]
            )
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            features["full", alpha] = dict(zip(unique, pooled[: len(unique)]))
            if alpha == 0 and crop is not None:
                features["crop", 0] = dict(zip(unique, pooled[len(unique) :]))
        for task, paths in inputs.items():
            chosen = policies[task]
            score = float(
                predict(
                    self.bundle["heads"][task],
                    matrix(
                        [{"views": paths}],
                        chosen["selection"]["strategy"],
                        features,
                        detections,
                    ),
                )[0]
            )
            threshold = float(chosen["thresholds"][self.point])
            out[task] = {
                "decision": "anomaly" if score >= threshold else "normal",
                "score": score,
                "threshold": threshold,
            }
    return out


def fake_detections(paths):
    """Deterministic in the image bytes, like a real detector: some frames have
    no boxes, some only a person, some a person and a bag."""
    found = {}
    for path in paths:
        seed = int(hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        kind = seed % 3
        boxes = []
        if kind >= 1:
            x, y = rng.uniform(0.05, 0.5, 2)
            boxes.append(
                {"box": [x, y, x + 0.3, y + 0.4], "label": 1, "confidence": 0.9}
            )
        if kind == 2:
            x, y = rng.uniform(0.1, 0.6, 2)
            label = int(rng.choice([27, 31, 33]))
            boxes.append(
                {"box": [x, y, x + 0.2, y + 0.2], "label": label, "confidence": 0.7}
            )
        found[path] = boxes
    return found


def candidate(folder, long_span=20.0, name="event"):
    """Rolling-buffer style evidence: the trigger frame is saved once per view."""
    rng = np.random.default_rng(len(name) + int(long_span))

    def frame(tag, k):
        path = folder / f"{name}_{tag}_{k}.png"
        Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(path)
        return str(path)

    trigger = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    views = {}
    for tag, span in (("short", 4.0), ("long", long_span)):
        times = [round(float(t), 2) for t in np.linspace(0, span, 8)]
        paths = [frame(tag, k) for k in range(7)]
        last = folder / f"{name}_{tag}_trigger.png"
        Image.fromarray(trigger).save(last)
        views[tag] = {"paths": [*paths, str(last)], "times_s": times}
    end = max(v["times_s"][-1] for v in views.values())
    # Both views must end at the same observed instant.
    views["short"]["times_s"] = [t + end - 4.0 for t in views["short"]["times_s"]]
    return {
        "event_id": name,
        "timestamp_s": end,
        "tasks": list(TASKS),
        "views": views,
    }


def input_dimension(strategy, d):
    per = 6 * d if "ordered" in strategy else 4 * d
    size = d if strategy.startswith("endpoint") else per
    if "crop" in strategy:
        size += per
    if "geometry" in strategy:
        size += 24
    return size


class FakeEncoder(nn.Module):
    """Small stand-in with the encoder interface and interpolated parameters."""

    name = "mobileclip2-s2"
    dimension = 16

    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.visual = nn.Linear(3 * 8 * 8, self.dimension)

    def processor(self, image):
        small = np.asarray(image.convert("RGB").resize((8, 8)), np.float32) / 255
        return torch.from_numpy(small.transpose(2, 0, 1).copy())

    def image_tensor(self, path):
        return self.processor(load_image(path, 768, "letterbox"))

    def encode(self, images, lengths):
        features = F.normalize(self.backbone.visual(images.flatten(1)), dim=-1)
        return pool_views(features, lengths)


def fake_strategy(recipes):
    torch.manual_seed(0)
    encoder = FakeEncoder().eval().requires_grad_(False)
    parameters = dict(encoder.named_parameters())
    original = {n: p.detach().clone() for n, p in parameters.items()}
    previous = {
        "parameters": {n: p + 0.5 * torch.randn_like(p) for n, p in original.items()}
    }
    rng = np.random.default_rng(1)
    heads, policy = {}, {}
    for task, recipe in recipes.items():
        n = input_dimension(recipe, encoder.dimension)
        heads[task] = {
            "kind": "linear",
            "weight": torch.from_numpy(rng.normal(size=(1, n)).astype(np.float32)),
            "bias": torch.zeros(1),
        }
        policy[task] = {
            "selected": {
                "selection": {"strategy": recipe},
                "thresholds": {"dev_max_f1": 0.0},
            }
        }
    strategy = object.__new__(ClipStrategy)
    strategy.__dict__.update(
        torch=torch,
        device="cpu",
        model="fixture",
        point="dev_max_f1",
        policy=policy,
        bundle={"heads": heads},
        encoder=encoder,
        previous=previous,
        parameters=parameters,
        original=original,
        detector=None,
        metadata={},
    )
    return strategy


def instrument(strategy):
    """Count detector frames and encoded rows on the instance."""
    work = SimpleNamespace(detected=0, encoded=0)
    detect, encode = fake_detections, strategy.encoder.encode

    def detections(paths):
        work.detected += len(paths)
        return detect(paths)

    def counted(images, lengths):
        work.encoded += len(images)
        return encode(images, lengths)

    strategy._detections = detections
    strategy.encoder.encode = counted
    return work


def assert_same(actual, expected, tolerance):
    assert set(actual) == set(expected)
    for task, value in expected.items():
        got = actual[task]
        assert got["decision"] == value["decision"], task
        if "score" in value:
            assert got["score"] == pytest.approx(value["score"], abs=tolerance), task
            assert got["threshold"] == value["threshold"]


@pytest.mark.parametrize("recipes", [DEPLOYED, VARIED], ids=["deployed", "varied"])
@pytest.mark.parametrize(
    "long_span", [20.0, 10.0], ids=["full-history", "short-history"]
)
def test_scores_match_dense_reference(tmp_path, recipes, long_span):
    strategy = fake_strategy(recipes)
    event = candidate(tmp_path, long_span)
    reference_work = instrument(strategy)
    expected = reference_classify(strategy, event)
    lean = fake_strategy(recipes)
    work = instrument(lean)
    result = lean.classify(event)
    assert_same(result["predictions"], expected, 1e-5)
    assert result["seconds"] >= 0
    assert work.encoded < reference_work.encoded
    assert work.detected <= reference_work.detected


def test_deployed_policy_skips_unread_frames(tmp_path):
    strategy = fake_strategy(DEPLOYED)
    work = instrument(strategy)
    strategy.classify(candidate(tmp_path))
    # Smoking reads one crop and loitering four geometry frames, all in the long
    # view; the eight littering frames are never detected or cropped.
    assert work.detected == 4
    # a0: 4 loitering frames + 1 smoking crop; a0.25: 1 rough-sleeping frame;
    # a0.5: 8 littering frames (one shares bytes with a long frame but is
    # needed at a different weight); a1: 4 unattended-item frames.
    assert work.encoded == 5 + 1 + 8 + 4

    short_history = fake_strategy(DEPLOYED)
    work = instrument(short_history)
    short_history.classify(candidate(tmp_path, 10.0, "early"))
    # Loitering and unattended items abstain; only the smoking frame is detected.
    assert work.detected == 1


def test_identical_frame_bytes_are_encoded_once(tmp_path):
    strategy = fake_strategy({"smoking": "full_pool_a0", "littering": "full_pool_a0"})
    work = instrument(strategy)
    event = candidate(tmp_path)
    event["tasks"] = ["smoking", "littering"]
    expected = reference_classify(fake_strategy(strategy_recipes(strategy)), event)
    assert_same(strategy.classify(event)["predictions"], expected, 1e-5)
    # Smoking reads the long trigger frame; littering reads the eight short frames,
    # whose last one has the same bytes.
    assert work.encoded == 8


def strategy_recipes(strategy):
    return {
        t: p["selected"]["selection"]["strategy"] for t, p in strategy.policy.items()
    }


installed = (model_paths().directory / "state.json").is_file()


@pytest.mark.skipif(not installed, reason="trained model bundle is not installed")
def test_installed_model_matches_dense_reference(tmp_path):
    strategy = ClipStrategy()
    strategy._detections = fake_detections
    for long_span in (20.0, 10.0):
        event = candidate(tmp_path, long_span, f"real{long_span:.0f}")
        expected = reference_classify(strategy, event)
        assert_same(strategy.classify(event)["predictions"], expected, 1e-4)
