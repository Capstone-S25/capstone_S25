"""Data pipeline for the smoking-video corpus (frame-level VadCLIP supervision).

The smoking dataset ships as COCO annotations over extracted frames. Frames are grouped
into *sessions* by ``(camera, date)`` parsed from the file name; within a session they are
ordered by the integer frame index in the name. Every frame carries a dense per-frame label:
category ``Smoking`` -> 1 (positive), ``Not Smoking`` -> 0 (negative).

Why frame-level supervision (the one deliberate deviation from the paper):
    All 16 sessions contain at least one positive frame, so at *clip* level every video is
    "anomaly" and the paper's video-MIL weak label degenerates to all-ones. The meaningful
    unit here is the frame -- which is also exactly what VadCLIP's headline metric
    (frame-level average precision) measures. We therefore supervise each representative frame with a
    soft target equal to the *fraction* of positive frames in its extraction block, which for
    short sessions (<=256 frames) reduces to exact per-frame 0/1 labels.

No image content is displayed; only file paths and integer labels are read here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from anomaly_eval.data import resolve_original

# --- binary task prompts (spliced into the learnable template by encode_textprompt) ----
NORMAL_PROMPT = "a normal scene with no one doing anything unusual"
SMOKING_PROMPT = "a person smoking a cigarette in a public place"
PROMPTS = [NORMAL_PROMPT, SMOKING_PROMPT]  # index 0 = normal, index 1 = anomaly

FRAME_RE = re.compile(r"frame_(c\d+)_(\d+)_")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass
class Session:
    key: str  # "camera:date", e.g. "c4:2025-10-26"
    frames: list[str] = field(default_factory=list)  # ordered image paths (str)
    frame_indices: list[int] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)  # per-frame 0/1, same order


def _load_coco(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    cats = {c["id"]: c.get("name", str(c["id"])) for c in d.get("categories", [])}
    imgs = {i["id"]: i.get("file_name", "") for i in d.get("images", [])}
    anns = list(d.get("annotations", []))
    return cats, imgs, anns


def load_smoking_sessions(root: str | Path) -> dict[str, Session]:
    """Group the smoking-video frames into ordered sessions with dense per-frame labels."""
    root = Path(root)
    base = root / "smoking" / "smoking-video.v1i.coco"
    sessions: dict[str, list[tuple[int, str, bool]]] = {}

    for split in ("train", "valid"):
        cp = resolve_original(base / split / "_annotations.coco.json")
        if cp is None:
            continue
        cats, imgs, anns = _load_coco(cp)
        smoke_ids = {
            a["image_id"] for a in anns if cats.get(a.get("category_id")) == "Smoking"
        }
        nonsmoke_ids = {
            a["image_id"]
            for a in anns
            if cats.get(a.get("category_id")) == "Not Smoking"
        }
        for iid, fname in imgs.items():
            path = resolve_original(cp.parent / fname)
            if path is None:
                continue
            m = FRAME_RE.search(fname)
            dm = DATE_RE.search(fname)
            order = int(m.group(2)) if m else -1
            key = f"{m.group(1)}:{dm.group(1)}" if (m and dm) else fname
            if iid not in smoke_ids and iid not in nonsmoke_ids:
                continue  # missing/unknown annotation is NOT an explicit negative
            is_smoke = iid in smoke_ids
            sessions.setdefault(key, []).append((order, str(path), bool(is_smoke)))

    out: dict[str, Session] = {}
    for key, rows in sessions.items():
        rows.sort(key=lambda t: (t[0], t[1]))
        s = Session(key=key)
        seen = {}
        for order, path, lab in rows:
            # Do not construct a fake temporal sequence by stacking augmented copies
            # of the same source moment. Unknown filenames retain their own identity.
            identity = order if order >= 0 else path
            if identity in seen:
                if seen[identity] != lab:
                    raise ValueError(
                        f"Conflicting annotation at source moment {key}:{order}"
                    )
                continue
            seen[identity] = lab
            s.frames.append(path)
            s.frame_indices.append(order)
            s.labels.append(int(lab))
        out[key] = s
    return out


# --- clip assembly (faithful to tools.uniform_extract / pad, with label alignment) -----


def build_clip(feat: np.ndarray, labels: np.ndarray, visual_length: int = 256):
    """Reduce a session's per-frame features/labels to exactly ``visual_length`` reps.

    Mirrors the repo's ``process_feat`` (uniform_extract when T>length else pad) but also
    carries the label through each extraction block as its positive fraction, so supervision
    stays aligned with the averaged feature. Returns (feat [L,W], soft_label [L], length).
    """
    T = feat.shape[0]
    if (
        T < 1
        or visual_length < 1
        or labels.shape != (T,)
        or not np.isin(labels, [0, 1]).all()
    ):
        raise ValueError(
            "Clip assembly requires nonempty features, positive length and aligned binary raw labels"
        )
    if T > visual_length:
        r = np.linspace(0, T, visual_length + 1, dtype=np.int32)
        new_feat = np.zeros((visual_length, feat.shape[1]), dtype=np.float32)
        new_lab = np.zeros(visual_length, dtype=np.float32)
        for i in range(visual_length):
            blk_f = feat[r[i] : r[i + 1]]
            blk_l = labels[r[i] : r[i + 1]]
            if r[i] != r[i + 1]:
                new_feat[i] = blk_f.mean(axis=0)
                new_lab[i] = blk_l.mean()
            else:
                new_feat[i] = feat[r[i]]
                new_lab[i] = labels[r[i]]
        return new_feat, new_lab, visual_length
    # T <= visual_length: keep real frames, zero-pad the tail (masked out via length)
    pad_n = visual_length - T
    new_feat = np.pad(
        feat, ((0, pad_n), (0, 0)), mode="constant", constant_values=0
    ).astype(np.float32)
    new_lab = np.pad(
        labels.astype(np.float32), (0, pad_n), mode="constant", constant_values=0
    )
    return new_feat, new_lab, T


class VadClipDataset(Dataset):
    """Yields precomputed session clips: (features [L,W], soft_label [L], length)."""

    def __init__(self, items: list[dict]):
        # each item: {"feat": np[L,W] float32, "label": np[L] float32, "length": int}
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return (
            torch.from_numpy(it["feat"]),
            torch.from_numpy(it["label"]),
            int(it["length"]),
        )


def collate(batch):
    """Pad to a common length within the batch (already fixed at visual_length here)."""
    feats = torch.stack([b[0] for b in batch])  # [B, L, W]
    labels = torch.stack([b[1] for b in batch])  # [B, L]
    lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)  # [B]
    return feats, labels, lengths


def expand_scores(scores: np.ndarray, raw_length: int) -> np.ndarray:
    """Repeat representative scores over original extraction blocks; never reduce evaluation labels."""
    scores = np.asarray(scores)
    if not len(scores) or raw_length < len(scores):
        raise ValueError("Invalid representative/raw frame lengths")
    bounds = np.linspace(0, raw_length, len(scores) + 1, dtype=int)
    return np.repeat(scores, np.diff(bounds))


def make_three_way_split(keys: list[str], valid_frac=0.2, test_frac=0.2, seed=234):
    if (
        len(keys) < 3
        or not 0 < valid_frac < 1
        or not 0 < test_frac < 1
        or valid_frac + test_frac >= 1
    ):
        raise ValueError(
            "Need at least three independent groups and valid train/dev/test fractions"
        )
    ordered = sorted(keys)
    np.random.default_rng(seed).shuffle(ordered)
    n_test = max(1, round(len(keys) * test_frac))
    n_dev = max(1, round(len(keys) * valid_frac))
    if n_test + n_dev >= len(keys):
        raise ValueError("Split leaves no training groups")
    return {
        "test": sorted(ordered[:n_test]),
        "dev": sorted(ordered[n_test : n_test + n_dev]),
        "train": sorted(ordered[n_test + n_dev :]),
    }
