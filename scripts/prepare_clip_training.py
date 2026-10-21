"""Prepare masked, source-grouped supervision for the five requested targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from anomaly_eval.data import image_pixel_sha256
from anomaly_eval.datasets.labels import collect_smoking
from anomaly_eval.paths import resolve_path
from anomaly_eval.training.data import TASKS, validate_records

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/training_sources_20261015"
OUT = ROOT / "data/clip_training_20261015"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def grouped_split(rows, *, official_test=False):
    """Choose a reproducible, label-balanced group split without model scores."""
    import random

    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group_id"]].append(row)
    groups = sorted(by_group)
    if len(groups) < 3:
        raise ValueError("Insufficient independent groups")
    # Only annotation support/count balance is used to choose a feasible split.
    best = None
    for seed in range(42, 542):
        shuffled = groups.copy()
        random.Random(seed).shuffle(shuffled)
        if official_test:
            assignment = {
                g: ("dev" if i < max(1, round(len(groups) * 0.2)) else "train")
                for i, g in enumerate(shuffled)
            }
            splits = ("train", "dev")
        else:
            count = max(1, round(len(groups) * 0.2))
            assignment = {
                g: ("test" if i < count else "dev" if i < 2 * count else "train")
                for i, g in enumerate(shuffled)
            }
            splits = ("train", "dev", "test")
        counts = Counter((assignment[r["group_id"]], r["target"]) for r in rows)
        if any(not counts[s, y] for s in splits for y in [0, 1]):
            continue
        fraction = sum(r["target"] for r in rows) / len(rows)
        cost = sum(
            abs(counts[s, 1] / (counts[s, 0] + counts[s, 1]) - fraction) for s in splits
        )
        if best is None or cost < best[0]:
            best = (cost, seed, assignment)
    if best is None:
        raise ValueError("Cannot preserve both classes in group holdouts")
    for row in rows:
        row["split"] = best[2][row["group_id"]]
    return best[1]


def smoking():
    positives, negatives = collect_smoking(ROOT / "data/raw/datasets")
    rows = []
    for r in positives + negatives:
        camera = r.group_id.split(":")[1]
        rows.append(
            {
                "task": "smoking",
                "target": int(r.label == "smoking"),
                "source": "smoking_cctv",
                "group_id": "smoking_camera:" + camera,
                "source_id": r.group_id + ":" + r.path.name,
                "views": [str(r.path.relative_to(ROOT))],
                "split": {"c4": "train", "c2": "dev", "c1": "test"}[camera],
                "evidence": "source_COCO_Smoking_or_Not_Smoking",
                "qualification": "Retrospective camera holdout; images previously evaluated as development",
            }
        )
    # Add photo-domain smoking training; preserve the official photo test set.
    base = SOURCES / "stanford40"
    images = list(base.rglob("JPEGImages/smoking_*.jpg"))
    if images:
        image_dir = images[0].parent
        split_dir = base / "ImageSplits"
        official_train = {
            name.strip()
            for p in split_dir.glob("*_train.txt")
            for name in p.read_text().splitlines()
        }
        photo_rows = []
        hard_negatives = {
            "drinking",
            "blowing_bubbles",
            "brushing_teeth",
            "phoning",
            "texting_message",
        }
        for p in sorted(image_dir.glob("*.jpg")):
            action = re.sub(r"_\d+$", "", p.stem)
            if action != "smoking" and action not in hard_negatives:
                continue
            photo_rows.append(
                {
                    "task": "smoking",
                    "target": int(action == "smoking"),
                    "source": "stanford40",
                    "group_id": "stanford40:" + p.stem,
                    "source_id": p.stem,
                    "views": [str(p.relative_to(ROOT))],
                    "split": "train" if p.name in official_train else "external",
                    "official_split": "train" if p.name in official_train else "test",
                    "evidence": "source_primary_action_class",
                    "qualification": "Official Stanford40 photo holdout plus camera-held-out CCTV; photo domain contributes train data, so this is not unseen-dataset transfer; primary action classification, source identities beyond image unknown",
                }
            )
        pairs = scene_groups(photo_rows)
        reserved = {r["group_id"] for r in photo_rows if r["split"] == "external"}
        # Remove any tiny external smoke-check images from later held-out claims.
        debug_path = OUT / "smokecheck.jsonl"
        seen = (
            {
                r["source_id"]
                for r in map(json.loads, debug_path.read_text().splitlines())
                if r["split"] == "external"
            }
            if debug_path.exists()
            else set()
        )
        excluded = [
            r
            for r in photo_rows
            if (r["split"] == "train" and r["group_id"] in reserved)
            or (r["split"] == "external" and r["source_id"] in seen)
        ]
        photo_rows = [r for r in photo_rows if r not in excluded]
        grouped_split(
            [r for r in photo_rows if r["split"] == "train"], official_test=True
        )
        dump(
            OUT / "stanford_source_audit.json",
            {
                "near_duplicate_pairs": pairs,
                "excluded": [r["source_id"] for r in excluded],
                "policy": "Official photo test retained, 20% grouped training subset for development; related train photos and prior debug-test images excluded",
            },
        )
        rows += photo_rows
    return rows


def unattended():
    manifest = ROOT / "data/expanded_20261013/cohorts/unattended_expanded_frame.csv"
    videos = {}
    for p in (ROOT / "data/expanded_20261013/video_frames").glob("*/frames.json"):
        videos[p.parent.name] = json.loads(p.read_text())["samples"]
    rows = []
    for r in csv.DictReader(manifest.open()):
        tag = r["source"]
        frames = videos[tag]
        timestamp = float(r["origin"].rsplit("@", 1)[1])
        selected = [
            min(
                frames,
                key=lambda f: abs(f["timestamp"] - max(0, timestamp - 20 + 20 * j / 3)),
            )
            for j in range(4)
        ]
        group = r["group_id"]
        split = (
            "dev"
            if group == "aboda:entrance"
            else "test"
            if group in {"aboda:classroom", "aboda:walkway"}
            else "train"
        )
        rows.append(
            {
                "task": "unattended_items",
                "target": int(r["label"] == "unattended_items"),
                "source": "unattended_reviewed",
                "group_id": group,
                "source_id": r["origin"],
                "views": [f["path"] for f in selected],
                "split": split,
                "evidence": "assistant_timeline_review_XML_supported_CAVIAR",
                "qualification": "Provisional labels; old test already inspected; source-family holdout, not a fresh untouched deployment test",
            }
        )
    return rows


def scene_groups(rows):
    """Conservatively connect source groups with near-identical first/last views."""
    import cv2
    import numpy as np

    parent = {r["group_id"]: r["group_id"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    def phash(path):
        with Image.open(ROOT / path) as image:
            gray = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
        low = cv2.dct(gray)[:8, :8].flatten()[1:]
        bits = low > np.median(low)
        return sum(int(value) << i for i, value in enumerate(bits))

    signatures = []
    for row in rows:
        signatures.append([phash(row["views"][i]) for i in {0, -1}])
    pairs = []
    for i, row in enumerate(rows):
        for j in range(i):
            other = rows[j]
            if row["group_id"] == other["group_id"]:
                continue
            distance = min(
                (a ^ b).bit_count() for a in signatures[i] for b in signatures[j]
            )
            if distance <= 6:
                union(row["group_id"], other["group_id"])
                pairs.append(
                    {
                        "a": row["source_id"],
                        "b": other["source_id"],
                        "phash_distance": distance,
                    }
                )
    for row in rows:
        row["group_id"] = find(row["group_id"])
    return pairs


def decode_iwdd(allow_incomplete=False):
    import cv2

    base = SOURCES / "mivia_iwdd"
    records = []
    missing = []
    for split in ["train", "test"]:
        for p in sorted((base / split / "labels").glob("vid*.json")):
            if ".receipt." in p.name:
                continue
            label = json.loads(p.read_text())
            video = base / split / "videos" / label["metadata"]["video_id"]
            if not video.exists():
                missing.append(str(video.relative_to(ROOT)))
                if not allow_incomplete:
                    raise ValueError(f"IWDD transfer incomplete: {video}")
                continue
            records.append((split, p, label, video))

    def one(item):
        split, p, label, video = item
        directory = OUT / "iwdd_frames" / video.stem
        metadata = directory / "frames.json"
        if metadata.exists():
            return json.loads(metadata.read_text())
        cv2.setNumThreads(1)
        capture = cv2.VideoCapture(str(video))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if count < 8 or fps <= 0:
            raise ValueError(f"Invalid video {video}")
        indices = [round((count - 1) * i / 7) for i in range(8)]
        decoded = {}
        decode_mode = "indexed_seek"
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            decoded[index] = frame
        capture.release()
        if len(decoded) != len(indices):
            # Some VFR clips advertise an inaccessible final frame or cannot seek
            # precisely. Count actual decoded frames and retry sequentially.
            decode_mode = "sequential_actual_frame_count"
            capture = cv2.VideoCapture(str(video))
            actual_count = 0
            while capture.grab():
                actual_count += 1
            capture.release()
            if actual_count < 8:
                raise ValueError(f"Too few decodable frames {video}")
            count = actual_count
            indices = [round((count - 1) * i / 7) for i in range(8)]
            capture = cv2.VideoCapture(str(video))
            decoded = {}
            index = 0
            while capture.grab():
                if index in indices:
                    ok, frame = capture.retrieve()
                    if not ok:
                        raise ValueError(f"Sequential decode failed {video}/{index}")
                    decoded[index] = frame
                index += 1
            capture.release()
            if len(decoded) != 8:
                raise ValueError(f"Cannot construct complete uniform views: {video}")
        views = []
        directory.mkdir(parents=True, exist_ok=True)
        for index in indices:
            frame = decoded[index]
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            path = directory / f"{index:07d}.png"
            image.save(path, compress_level=1)
            views.append(str(path.relative_to(ROOT)))
        row = {
            "task": "littering",
            "target": int(str(label["label"]).lower() == "true"),
            "source": "mivia_iwdd",
            "group_id": "iwdd:" + video.stem,
            "source_id": video.stem,
            "views": views,
            "split": split,
            "official_split": split,
            "evidence": "author_video_label_and_onset_annotation",
            "qualification": "Offline video-level waste dumping classification; uniformly sampled 8 frames, not dense event localization; camera identity unavailable",
            "duration_seconds": count / fps,
            "onset_seconds": label["timestamp"],
            "annotation_path": str(p.relative_to(ROOT)),
            "sampled_frame_indices": indices,
            "decode_mode": decode_mode,
        }
        dump(metadata, row)
        return row

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, records))
    pairs = scene_groups(rows)
    test = [r for r in rows if r["official_split"] == "test"]
    excluded = []
    if len({r["target"] for r in test}) == 2:
        reserved = {r["group_id"] for r in test}
        excluded = [
            r
            for r in rows
            if r["official_split"] == "train" and r["group_id"] in reserved
        ]
        rows = [r for r in rows if r not in excluded]
        grouped_split(
            [r for r in rows if r["official_split"] == "train"], official_test=True
        )
        policy = "Official test retained; related training groups excluded; train grouped into train/dev"
    else:
        grouped_split(rows)
        policy = "Research train/dev/test re-split of available videos; official test unavailable/incomplete"
    dump(
        OUT / "iwdd_source_audit.json",
        {
            "available_videos": len(rows),
            "missing": missing,
            "related_training_excluded": [r["source_id"] for r in excluded],
            "near_duplicate_pairs": pairs,
            "split_policy": policy,
        },
    )
    return rows


def commons():
    path = SOURCES / "commons_review/review_labels.json"
    if not path.exists():
        return []
    review = json.loads(path.read_text())
    if (
        review["candidate_manifest_sha256"]
        != hashlib.sha256((path.parent / "candidates.json").read_bytes()).hexdigest()
    ):
        raise ValueError("Commons candidates changed after review")
    labels = review["labels_by_page_id"]
    candidates = json.loads((path.parent / "candidates.json").read_text())
    rows = []
    for r in candidates:
        label = labels.get(r["id"])
        if label not in [0, 1]:
            continue
        artist = r["metadata"].get("Artist", {}).get("value", "")
        artist = re.sub("<[^>]+>", "", artist).strip()
        if not artist:
            artist = r["metadata"].get("Credit", {}).get("value", r["id"])
        rows.append(
            {
                "task": "rough_sleeping",
                "target": label,
                "source": "commons_visible_sleeping",
                "group_id": "commons_author:" + artist,
                "source_id": r["id"],
                "views": [r["path"]],
                "evidence": "assistant_visual_review_visible_public_bedding_or_control",
                "qualification": "Provisional visual labels; visible posture/bedding/public context only, no housing-status inference; small author-held-out photo study",
                "source_page": r["source_page"],
            }
        )
    pairs = scene_groups(rows)
    dump(
        OUT / "commons_source_audit.json",
        {"review": review, "near_duplicate_pairs": pairs},
    )
    grouped_split(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-external", action="store_true")
    parser.add_argument("--allow-incomplete-downloads", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = smoking() + unattended()
    if not args.without_external:
        rows += decode_iwdd(args.allow_incomplete_downloads)
        street = json.loads((SOURCES / "street_scene/samples.json").read_text())
        grouped_split(street)
        rows += street + commons()
    for row in rows:
        row["id"] = hashlib.sha256(
            (row["task"] + ":" + row["source_id"]).encode()
        ).hexdigest()[:24]
        row["views"] = [str(resolve_path(p).relative_to(ROOT)) for p in row["views"]]
    paths = sorted({p for r in rows for p in r["views"]})
    with ThreadPoolExecutor(12) as pool:
        hashes = dict(
            zip(paths, pool.map(lambda p: image_pixel_sha256(ROOT / p), paths))
        )
    # A duplicate across datasets/splits is excluded from later folds, never leaked.
    known = {}
    kept = []
    removed = []
    for row in sorted(
        rows, key=lambda r: {"test": 0, "external": 1, "dev": 2, "train": 3}[r["split"]]
    ):
        row["pixel_hashes"] = [hashes[p] for p in row["views"]]
        conflicts = [
            h for h in row["pixel_hashes"] if h in known and known[h] != row["split"]
        ]
        if conflicts:
            removed.append(
                {
                    "id": row["id"],
                    "reason": "decoded pixels also occur in a reserved fold",
                }
            )
            continue
        for h in row["pixel_hashes"]:
            known[h] = row["split"]
        kept.append(row)
    kept.sort(key=lambda r: r["id"])
    counts = validate_records(kept, ROOT)
    (OUT / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in kept) + "\n")
    dump(
        OUT / "dataset.json",
        {
            "tasks": list(TASKS),
            "counts": counts,
            "samples": len(kept),
            "unique_view_images": len(paths),
            "removed": removed,
            "mask_policy": "Only the sample's explicitly annotated task contributes a binary loss; other targets are unknown, not normal",
            "test_policy": "No test inputs used by optimization or early stopping; historical local test results are retrospective; newly acquired sources identified separately",
            "sources": dict(Counter(r["source"] for r in kept)),
        },
    )
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    main()
