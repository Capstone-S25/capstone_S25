"""Source-disjoint training records; one known binary target per sample."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

TASKS = ("smoking", "littering", "loitering", "rough_sleeping", "unattended_items")


def validate_records(rows, root, *, check_files=True):
    """Reject leakage and unknown targets instead of treating them as negatives."""
    groups, pixels, identifiers = {}, {}, set()
    counts = defaultdict(Counter)
    for row in rows:
        if row["task"] not in TASKS or row["target"] not in (0, 1):
            raise ValueError("Each record needs a known task and a binary target")
        if row["split"] not in {"train", "dev", "test", "external"}:
            raise ValueError("Unknown split")
        if row["id"] in identifiers:
            raise ValueError("Duplicate training sample ID")
        identifiers.add(row["id"])
        if not row["views"] or not row.get("evidence") or not row.get("group_id"):
            raise ValueError("Missing views, evidence or source group")
        group = row["group_id"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError(f"Source group crosses splits: {group}")
        groups[group] = row["split"]
        for view in row["views"]:
            if check_files and not (root / view).is_file():
                raise ValueError(f"Missing training image: {view}")
        for pixel in row.get("pixel_hashes", []):
            if pixel in pixels and pixels[pixel] != row["split"]:
                raise ValueError("Decoded image pixels cross splits")
            pixels[pixel] = row["split"]
        counts[row["task"]][(row["split"], row["target"])] += 1
    for task in TASKS:
        if counts[task] and any(
            not counts[task][split, target]
            for split in ("train", "dev", "test")
            for target in (0, 1)
        ):
            raise ValueError(f"{task} needs both classes in train/dev/test")
    return {
        task: {f"{split}/{target}": n for (split, target), n in count.items()}
        for task, count in counts.items()
    }


def read_records(path, root, check_files=True):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    validate_records(rows, root, check_files=check_files)
    if check_files:
        from concurrent.futures import ThreadPoolExecutor

        from anomaly_eval.data import image_pixel_sha256

        expected = {}
        for row in rows:
            if len(row["views"]) != len(row.get("pixel_hashes", [])):
                raise ValueError(
                    "Training manifests require a decoded pixel hash for every view"
                )
            for view, pixel in zip(row["views"], row["pixel_hashes"]):
                if view in expected and expected[view] != pixel:
                    raise ValueError("Conflicting expected pixel hash")
                expected[view] = pixel
        views = list(expected)
        with ThreadPoolExecutor(12) as pool:
            for view, actual in zip(
                views, pool.map(lambda view: image_pixel_sha256(root / view), views)
            ):
                if actual != expected[view]:
                    raise ValueError(
                        f"Training image changed since preparation: {view}"
                    )
    return rows


def group_class_weights(rows):
    """Balance task/class, then dataset source, then recording/author group."""
    sizes = Counter(
        (r["task"], r["target"], r.get("source", "unknown"), r["group_id"])
        for r in rows
    )
    groups = defaultdict(set)
    sources = defaultdict(set)
    for task, target, source, group in sizes:
        groups[task, target, source].add(group)
        sources[task, target].add(source)
    return [
        1
        / (
            len(sources[r["task"], r["target"]])
            * len(groups[r["task"], r["target"], r.get("source", "unknown")])
            * sizes[r["task"], r["target"], r.get("source", "unknown"), r["group_id"]]
        )
        for r in rows
    ]


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
