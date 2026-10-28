"""Rebuild the 30-frame littering clips from the Roboflow export.

The export names every image `video_<V>_<F>_jpg.rf.<hash>.jpg`, but each class's recordings
were numbered video_1..video_12 independently, so one stem such as `video_10_7` belongs to
several different clips (one per class, and 5-6 recordings per class). Roboflow also scattered
the frames of each clip across train/valid/test at random.

Clips are recovered by chaining frames: for every (V, class) group, the copies of frame F are
matched to the copies of frame F+1 with the Hungarian algorithm on a thumbnail difference
(consecutive frames of one recording are nearly identical, other recordings are not) plus the
label box position. Frames with an empty label file are matched without the class constraint.

Output (raw export untouched; files are hard-linked when possible). Each clip stays intact in one
split folder, so no frame of a clip leaks into another split:
    <dst>/{train,valid,test}/<clip_id>/frame_01.jpg + frame_01.txt ... frame_30
    <dst>/data.yaml      class names (same YOLO class ids as the export)
    <dst>/manifest.csv   one row per frame, incl. original split and filename
    <dst>/clips.csv      one row per clip with its split and chaining quality scores

Four frames in the export carry a different class from the other 29 frames of their clip; they are
label errors (visually continuous) and are listed in clips.csv (class_consistent=False).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

NAME = re.compile(r"video_(\d+)_(\d+)_jpg\.rf\.([0-9a-f]+)\.jpg$")
CLASS_MISMATCH_COST = 1e3


def read_class_names(src: Path) -> list[str]:
    text = (src / "data.yaml").read_text()
    match = re.search(r"names:\s*\[(.*?)\]", text)
    if not match:
        raise ValueError("No names: [...] list in data.yaml")
    return [n.strip().strip("'\"") for n in match.group(1).split(",")]


def read_label(path: Path) -> tuple[int | None, list[list[float]]]:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    classes = {int(float(r[0])) for r in rows}
    if len(classes) > 1:
        raise ValueError(f"{path} has several classes {classes}")
    return (classes.pop() if classes else None), [list(map(float, r[1:5])) for r in rows]


def thumbnail(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if img is None:
        raise ValueError(f"Unreadable image {path}")
    return cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def load_frames(src: Path) -> list[dict]:
    frames = []
    for split in ("train", "valid", "test"):
        for img in sorted((src / split / "images").glob("*.jpg")):
            m = NAME.search(img.name)
            if not m:
                raise ValueError(f"Unexpected filename {img.name}")
            label = src / split / "labels" / (img.stem + ".txt")
            cls, boxes = read_label(label)
            frames.append({"video": int(m.group(1)), "frame": int(m.group(2)), "split": split, "image": img,
                           "label": label, "cls": cls, "boxes": boxes, "thumb": thumbnail(img)})
    return frames


def box_center(f):
    if not f["boxes"]:
        return None
    b = np.array(f["boxes"])
    return b[:, 0].mean(), b[:, 1].mean()


def pair_cost(a: dict, b: dict) -> float:
    cost = float(np.mean(np.abs(a["thumb"] - b["thumb"])))
    if a["cls"] is not None and b["cls"] is not None and a["cls"] != b["cls"]:
        cost += CLASS_MISMATCH_COST
    ca, cb = box_center(a), box_center(b)
    if ca is not None and cb is not None:
        cost += 20.0 * float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
    return cost


def chain_video(frames: list[dict]) -> list[list[dict]]:
    """Chain all copies of one video id into clips, frame by frame."""
    by_frame = defaultdict(list)
    for f in frames:
        by_frame[f["frame"]].append(f)
    order = sorted(by_frame)
    sizes = {len(by_frame[k]) for k in order}
    if len(sizes) != 1:
        raise ValueError(f"video_{frames[0]['video']}: uneven copies per frame {sizes}")
    chains = [[f] for f in by_frame[order[0]]]
    for k in order[1:]:
        nxt = by_frame[k]
        cost = np.array([[pair_cost(_tail_ref(c), f) for f in nxt] for c in chains])
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            chains[r].append(nxt[c])
    return chains


def _tail_ref(chain: list[dict]) -> dict:
    """Last frame of the chain, but with the chain's class so unlabeled tails keep the constraint."""
    tail = chain[-1]
    if tail["cls"] is None:
        known = [f["cls"] for f in chain if f["cls"] is not None]
        if known:
            return {**tail, "cls": max(set(known), key=known.count)}
    return tail


def chain_quality(chain: list[dict]) -> dict:
    steps = [float(np.mean(np.abs(a["thumb"] - b["thumb"]))) for a, b in zip(chain, chain[1:])]
    classes = [f["cls"] for f in chain if f["cls"] is not None]
    return {
        "max_step_diff": round(max(steps), 2),
        "mean_step_diff": round(float(np.mean(steps)), 2),
        "class_consistent": len(set(classes)) <= 1,
        "unlabeled_frames": sum(f["cls"] is None for f in chain),
    }


def assign_splits(clips: list[dict], mode: str = "stratified", fractions=(0.7, 0.15, 0.15), seed: int = 0) -> dict[str, str]:
    """Clip-level split so no clip has frames in two splits.
    original:   each clip goes to the split most of its frames were in (keeps the Roboflow split
                comparable; only the few clips Roboflow spread over two splits move).
    stratified: fresh random split per class with the given fractions."""
    if mode == "original":
        out = {}
        for c in clips:
            votes = [f["split"] for f in c["frames"]]
            out[c["clip_id"]] = max(("train", "valid", "test"), key=votes.count)
        return out
    rng = np.random.default_rng(seed)
    out = {}
    by_class = defaultdict(list)
    for c in clips:
        by_class[c["class"]].append(c["clip_id"])
    for ids in by_class.values():
        ids = sorted(ids)
        rng.shuffle(ids)
        n_train = round(fractions[0] * len(ids))
        n_valid = round(fractions[1] * len(ids))
        for i, cid in enumerate(ids):
            out[cid] = "train" if i < n_train else "valid" if i < n_train + n_valid else "test"
    return out


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, default=Path("Littering"), help="Roboflow export (with data.yaml)")
    parser.add_argument("--dst", type=Path, default=Path("data/littering"))
    parser.add_argument("--max-step-diff", type=float, default=20.0,
                        help="Warn about clips whose largest frame-to-frame thumbnail change exceeds this")
    parser.add_argument("--split", choices=("stratified", "original"), default="stratified",
                        help="stratified: new per-class 70/15/15 clip split (default; the Roboflow split puts only dropping+dumping in valid/test); original: majority Roboflow split per clip")
    parser.add_argument("--force", action="store_true", help="Replace an existing --dst")
    args = parser.parse_args(argv)

    if args.dst.exists():
        if not args.force:
            print(f"{args.dst} exists; pass --force to rebuild it", file=sys.stderr)
            return 1
        shutil.rmtree(args.dst)

    names = read_class_names(args.src)
    frames = load_frames(args.src)
    by_video = defaultdict(list)
    for f in frames:
        by_video[f["video"]].append(f)

    clips = []
    for video in sorted(by_video):
        chains = chain_video(by_video[video])
        chains.sort(key=lambda c: (min(f["cls"] for f in c if f["cls"] is not None), str(c[0]["image"].name)))
        per_class = defaultdict(int)
        for chain in chains:
            known = [f["cls"] for f in chain if f["cls"] is not None]
            cls = max(set(known), key=known.count)
            per_class[cls] += 1
            clip_id = f"{names[cls]}_v{video:02d}_{per_class[cls]}"
            clips.append({"clip_id": clip_id, "class": names[cls], "video": video, "frames": chain,
                          **chain_quality(chain)})

    splits = assign_splits(clips, args.split)
    manifest = []
    for clip in clips:
        out_dir = args.dst / splits[clip["clip_id"]] / clip["clip_id"]
        out_dir.mkdir(parents=True)
        for f in clip["frames"]:
            stem = f"frame_{f['frame']:02d}"
            link_or_copy(f["image"], out_dir / f"{stem}.jpg")
            link_or_copy(f["label"], out_dir / f"{stem}.txt")
            manifest.append({"clip_id": clip["clip_id"], "class": clip["class"], "frame": f["frame"],
                             "split": splits[clip["clip_id"]], "orig_split": f["split"],
                             "orig_file": f"{f['split']}/images/{f['image'].name}",
                             "label_class": "" if f["cls"] is None else names[f["cls"]]})

    with open(args.dst / "manifest.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    clip_rows = [{k: v for k, v in c.items() if k != "frames"} | {"split": splits[c["clip_id"]]} for c in clips]
    with open(args.dst / "clips.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(clip_rows[0]))
        writer.writeheader()
        writer.writerows(clip_rows)
    (args.dst / "data.yaml").write_text(
        "train: train\nval: valid\ntest: test\n\n"
        f"nc: {len(names)}\nnames: {json.dumps(names)}\n\n"
        "# Each split folder holds one sub-folder per 30-frame clip (frame_NN.jpg + YOLO frame_NN.txt).\n")

    suspicious = [c for c in clip_rows if c["max_step_diff"] > args.max_step_diff or not c["class_consistent"]]
    print(f"{len(frames)} frames -> {len(clips)} clips in {args.dst}")
    print("clips per class:", {n: sum(c["class"] == n for c in clips) for n in names})
    print("clips per split:", {s: sum(v == s for v in splits.values()) for s in ("train", "valid", "test")})
    print(f"{len(suspicious)} clips to inspect (step diff > {args.max_step_diff} or mixed classes)")
    for c in suspicious:
        print("  ", c["clip_id"], "max_step_diff", c["max_step_diff"], "class_consistent", c["class_consistent"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
