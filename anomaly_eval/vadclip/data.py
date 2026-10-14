"""Data pipeline for the smoking-video corpus (frame-level VadCLIP supervision).

The smoking dataset ships as COCO annotations over extracted frames. Frames are grouped
into *sessions* by ``(camera, date)`` parsed from the file name; within a session they are
ordered by the full capture timestamp when available. Frame indices can restart within
the same camera/day. Timestamp-free legacy sessions use integer frame order; mixing
timestamped and timestamp-free frames in one session is rejected as ambiguous.
Every frame carries a dense per-frame label:
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
from datetime import datetime
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
TIME_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{3}_[AP]M)(?=_|\.|$)",
    re.IGNORECASE,
)
SOURCE_IDENTITY_POLICY = "camera_date_frame_capture_timestamp_v2"


@dataclass
class Session:
    key: str  # "camera:date", e.g. "c4:2025-10-26"
    frames: list[str] = field(default_factory=list)  # ordered image paths (str)
    frame_indices: list[int] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)  # per-frame 0/1, same order
    timestamps: list[str | None] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    ordering: str = "frame_index"


class SmokingSourceError(ValueError):
    """Source validation failed; the report retains all offending records."""

    def __init__(self, report: dict):
        self.report = report
        details = []
        for issue in report["issues"][:5]:
            details.append(issue["message"])
            for record in issue.get("records", []):
                details.append(
                    f"  {record['label']}: {record['path']} "
                    f"(COCO image {record['image_id']} in {record['annotations']})"
                )
        super().__init__(
            f"Smoking source validation found {len(report['issues'])} issue(s):\n"
            + "\n".join(details)
            + "\nReview the source annotations; labels were not changed. "
            "Use --validation-report to save every issue."
        )


@dataclass(frozen=True)
class _SourceFrame:
    index: int
    path: str
    label: int
    timestamp: str | None
    source_id: str
    annotations: str
    image_id: int

    def record(self):
        return {
            "path": self.path,
            "label": "Smoking" if self.label else "Not Smoking",
            "frame_index": self.index,
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "annotations": self.annotations,
            "image_id": self.image_id,
        }


def _frame_identity(fname: str):
    m, dm = FRAME_RE.search(fname), DATE_RE.search(fname)
    index = int(m.group(2)) if m else -1
    key = f"{m.group(1)}:{dm.group(1)}" if (m and dm) else fname
    clock = TIME_RE.search(fname)
    timestamp = None
    if clock:
        # The filenames carry a camera-local clock, with no timezone annotation.
        timestamp = datetime.strptime(
            clock.group(1).upper(), "%Y-%m-%d_%I-%M-%S-%f_%p"
        ).isoformat(timespec="milliseconds")
    elif dm and re.match(r"_\d{2}-", fname[dm.end() :]):
        raise ValueError(f"Incomplete capture timestamp in {fname}")
    # Export hashes do not identify new moments. Keep the index as well as the
    # timestamp: different source frames can share a capture clock value.
    source_id = f"{key}:frame={index}" if index >= 0 else f"path={fname}"
    if timestamp is not None:
        source_id += f":timestamp={timestamp}"
    return key, index, timestamp, source_id


def _load_coco(p: Path):
    d = json.loads(p.read_text(encoding="utf-8"))
    cats = {c["id"]: c.get("name", str(c["id"])) for c in d.get("categories", [])}
    imgs = {i["id"]: i.get("file_name", "") for i in d.get("images", [])}
    anns = list(d.get("annotations", []))
    return cats, imgs, anns


def load_smoking_sessions(
    root: str | Path, *, report: dict | None = None
) -> dict[str, Session]:
    """Group the smoking-video frames into ordered sessions with dense per-frame labels."""
    root = Path(root)
    base = root / "smoking" / "smoking-video.v1i.coco"
    sessions: dict[str, list[_SourceFrame]] = {}
    report = {} if report is None else report
    report.update(
        identity_policy=SOURCE_IDENTITY_POLICY,
        annotation_files=0,
        images_listed=0,
        unlabeled_images=0,
        missing_images=0,
        duplicate_exports=0,
        issues=[],
        sessions={},
    )

    for split in ("train", "valid"):
        cp = resolve_original(base / split / "_annotations.coco.json")
        if cp is None:
            continue
        report["annotation_files"] += 1
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
            report["images_listed"] += 1
            if iid not in smoke_ids and iid not in nonsmoke_ids:
                report["unlabeled_images"] += 1
                continue  # missing/unknown annotation is NOT an explicit negative
            path = resolve_original(cp.parent / fname)
            if path is None:
                report["missing_images"] += 1
                report["issues"].append(
                    {
                        "kind": "missing_image",
                        "message": f"Missing labeled image: {cp.parent / fname}",
                        "annotations": str(cp),
                        "image_id": iid,
                    }
                )
                continue
            try:
                key, index, timestamp, source_id = _frame_identity(fname)
            except ValueError as exc:
                report["issues"].append(
                    {
                        "kind": "invalid_timestamp",
                        "message": f"Invalid capture timestamp in {path}: {exc}",
                        "annotations": str(cp),
                        "image_id": iid,
                    }
                )
                continue
            sessions.setdefault(key, []).append(
                _SourceFrame(
                    index,
                    str(path),
                    int(iid in smoke_ids),
                    timestamp,
                    source_id,
                    str(cp),
                    iid,
                )
            )

    out: dict[str, Session] = {}
    for key, rows in sessions.items():
        clock_count = sum(row.timestamp is not None for row in rows)
        if clock_count not in (0, len(rows)):
            report["issues"].append(
                {
                    "kind": "ambiguous_order",
                    "message": f"Mixed capture timestamps and legacy frame indices in {key}",
                    "records": [row.record() for row in rows],
                }
            )
        ordering = "capture_timestamp" if clock_count else "frame_index"
        rows.sort(key=lambda row: (row.timestamp or "", row.index, row.path))
        s = Session(key=key, ordering=ordering)
        seen = {}
        for row in rows:
            # Do not construct a fake temporal sequence by stacking augmented copies
            # of the same source moment. Unknown filenames retain their own identity.
            if row.source_id in seen:
                first = seen[row.source_id]
                if first.label != row.label:
                    report["issues"].append(
                        {
                            "kind": "conflicting_annotation",
                            "message": f"Conflicting annotation at source moment {row.source_id}",
                            "records": [first.record(), row.record()],
                        }
                    )
                else:
                    report["duplicate_exports"] += 1
                continue
            seen[row.source_id] = row
            s.frames.append(row.path)
            s.frame_indices.append(row.index)
            s.labels.append(row.label)
            s.timestamps.append(row.timestamp)
            s.source_ids.append(row.source_id)
        out[key] = s
        report["sessions"][key] = {
            "n_frames": len(s.frames),
            "n_positive": sum(s.labels),
            "n_negative": len(s.labels) - sum(s.labels),
            "ordering": s.ordering,
            "source_ids": s.source_ids,
        }
    report["session_count"] = len(out)
    report["retained_frames"] = sum(len(s.frames) for s in out.values())
    report["status"] = "error" if report["issues"] else "ok"
    if report["issues"]:
        raise SmokingSourceError(report)
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
