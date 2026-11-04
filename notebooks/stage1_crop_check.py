"""Crop check: a second look at reach_down and pose-cue alerts using the pixels, not just the body points.

Body points cannot tell a hand carrying a bag at knee height from a hand letting go of rubbish, or a phone at the ear
from a cigarette. The crop check embeds the person's crop (person box, + the hand square for reach_down, padded 15 %)
with a frozen MobileCLIP2-S2 image encoder and scores it with a logistic regression per rule, trained on ICVL odd-half
alerts: 1 = looks like a real littering / smoking alert, 0 = a person with no incident. Alerts below the threshold are dropped before the person dedup. It runs once per alert.

Weights, one file per checked rule (weights_for: weights/crop_check.npz for reach_down,
weights/crop_check_pose_action.npz for the pose cue):
  version 2 (default, exp/icvl/crop_v2.py export): crops at t - dt, t and t + dt (a drop is a change: something in
            the hand, then not). The gate holds each alert dt seconds for the "after" crop.
  version 1 (weights/crop_check_v1.npz, exp/icvl/hand_probe.py export): the crop at t only, no hold.

Replaying cached runs (no frames): scores come from SCORES, {(sequence_id, event_id): score}, which evaluation
scripts fill from precomputed embeddings; an alert with no score is kept.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "crop_check.npz"   # reach_down
DEFAULT_ENCODER = "hf-hub:timm/MobileCLIP2-S2-OpenCLIP"
CROP_PAD = 0.15
SCORES: dict[tuple[str, str], float] = {}
_ENCODERS: dict[str, tuple] = {}   # one encoder per name, shared by the per-rule checks


def weights_for(rule: str) -> Path:
    """reach_down: weights/crop_check.npz; other rules: weights/crop_check_<rule>.npz."""
    return DEFAULT_WEIGHTS if rule == "reach_down" else DEFAULT_WEIGHTS.with_name(f"crop_check_{rule}.npz")


def crop(img: np.ndarray, box, pad: float = CROP_PAD) -> np.ndarray:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1, Y1 = int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h))
    X2, Y2 = int(min(img.shape[1], x2 + pad * w)), int(min(img.shape[0], y2 + pad * h))
    return img[Y1:Y2, X1:X2]


def unit(v: np.ndarray) -> np.ndarray:
    return (v / max(float(np.linalg.norm(v)), 1e-8)).astype(np.float32)


class CropCheck:
    def __init__(self, weights: str | Path | None = None, encoder: str | None = None):
        d = np.load(weights or DEFAULT_WEIGHTS)
        self.coef, self.intercept = d["coef"].astype(np.float32), float(d["intercept"])
        self.threshold = float(d["threshold"])
        self.version = int(d["version"]) if "version" in d else 1
        self.dt = float(d["dt"]) if "dt" in d else 0.0
        self.encoder_name = encoder or DEFAULT_ENCODER   # open_clip model id, or "local-dir:<snapshot folder>"
        self.model = self.prep = None

    def _load(self):
        import open_clip

        if self.encoder_name not in _ENCODERS:
            model, _, prep = open_clip.create_model_and_transforms(self.encoder_name)
            _ENCODERS[self.encoder_name] = (model.eval(), prep)
        self.model, self.prep = _ENCODERS[self.encoder_name]

    def embed(self, img: np.ndarray, box) -> np.ndarray | None:
        """Unit-length embedding of the padded crop (None if the crop is empty)."""
        return self.embed_crop(crop(img, box))

    def embed_crop(self, c: np.ndarray | None) -> np.ndarray | None:
        """Unit-length embedding of an already cut BGR crop (None for a missing or empty crop)."""
        import torch
        from PIL import Image

        if c is None or c.size == 0:
            return None
        if self.model is None:
            self._load()
        with torch.no_grad():
            f = self.model.encode_image(self.prep(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))[None])
        return unit(f.numpy()[0])

    def _prob(self, x: np.ndarray) -> float:
        return 1.0 / (1.0 + float(np.exp(-(x @ self.coef + self.intercept))))

    def score_embedding(self, embedding: np.ndarray, rule: str) -> float:
        """Version 1: one embedding plus a reach_down flag."""
        return self._prob(np.concatenate([unit(embedding), [float(rule == "reach_down")]]).astype(np.float32))

    def score_embeddings(self, embs) -> float:
        """Version 2: [before, now, after] embeddings; a missing before / after is replaced by now."""
        before, now, after = embs
        before = now if before is None else before
        after = now if after is None else after
        return self._prob(np.concatenate([unit(before), unit(now), unit(after)]))

    def score(self, img: np.ndarray, box, rule: str) -> float | None:
        """Version 1 only: score the crop at the alert time."""
        e = self.embed(img, box)
        return None if e is None else self.score_embedding(e, rule)
