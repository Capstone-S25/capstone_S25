"""Prepare new source holdouts without changing the previous training manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

from anomaly_eval.data import image_pixel_sha256

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/strategy_sources_20261016"
OUT = ROOT / "data/strategy_study_20261016"


def signature(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((32, 32))
        low = dctn(np.asarray(im, dtype=np.float32), norm="ortho")[:8, :8]
    values = low.ravel()[1:]
    return sum(int(b) << i for i, b in enumerate(values > np.median(values)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "samples.jsonl"
    if destination.exists():
        raise ValueError("Prepared study already exists; use the frozen manifest")
    previous = ROOT / "data/clip_training_20261015/samples.jsonl"
    old = [json.loads(line) for line in previous.read_text().splitlines()]
    paths = sorted({ROOT / p for row in old for p in row["views"]})
    with ThreadPoolExecutor(12) as pool:
        old_signatures = list(pool.map(signature, paths))
    candidates = sorted((SOURCE / "smoking_kaggle").rglob("*.jpg"))
    with ThreadPoolExecutor(12) as pool:
        new_signatures = list(pool.map(signature, candidates))
    parent = list(range(len(candidates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(new_signatures):
        for j in range(i):
            if (a ^ new_signatures[j]).bit_count() <= 6:
                parent[find(i)] = find(j)
    labels = [int(not p.name.lower().startswith("notsmoking")) for p in candidates]
    original_splits = [
        "test"
        if "Testing" in p.parts
        else "dev"
        if "Validation" in p.parts
        else "train"
        for p in candidates
    ]
    groups = {}
    for i in range(len(candidates)):
        groups.setdefault(find(i), []).append(i)
    rejected, accepted = [], []
    for members in groups.values():
        reason = None
        if len({labels[i] for i in members}) != 1:
            reason = "near_duplicate_label_conflict"
        elif any(
            (new_signatures[i] ^ s).bit_count() <= 6
            for i in members
            for s in old_signatures
        ):
            reason = "related_to_previous_study_image"
        if reason:
            rejected.extend(
                {"file": str(candidates[i]), "reason": reason} for i in members
            )
            continue
        priority = {"train": 0, "dev": 1, "test": 2}
        split = max((original_splits[i] for i in members), key=priority.get)
        # Related lower-priority examples are excluded, never moved into the test.
        for i in members:
            if original_splits[i] != split:
                rejected.append(
                    {
                        "file": str(candidates[i]),
                        "reason": "related_to_higher_priority_fold",
                    }
                )
                continue
            path = candidates[i]
            accepted.append(
                {
                    "id": "kaggle_smoking:" + path.stem,
                    "task": "smoking",
                    "target": labels[i],
                    "source": "kaggle_sujaykapadnis_smoking",
                    "group_id": "kaggle_smoking_group:" + str(min(members)),
                    "views": [str(path.relative_to(ROOT))],
                    "split": split,
                    "pixel_hashes": [image_pixel_sha256(path)],
                    "evidence": "publisher_image_class_and_official_split",
                    "qualification": "New photo source; original identity/camera genealogy unavailable; perceptual duplicate screening cannot prove subject independence",
                }
            )
    for split in ["train", "dev", "test"]:
        assert {r["target"] for r in accepted if r["split"] == split} == {0, 1}
    # Old holdouts do not become training or selection data in this study.
    rows = [r for r in old if r["split"] in {"train", "dev"}] + accepted
    destination.write_text("".join(json.dumps(r) + "\n" for r in rows))
    metadata = {
        "previous_manifest_sha256": hashlib.sha256(previous.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "previous_test_used_for_selection": False,
        "source": "https://www.kaggle.com/datasets/sujaykapadnis/smoking",
        "new_smoking_counts": dict(
            Counter(f"{r['split']}/{r['target']}" for r in accepted)
        ),
        "source_candidates": len(candidates),
        "new_smoking_accepted": len(accepted),
        "rejected": rejected,
        "samples": len(rows),
        "unique_images": len({p for r in rows for p in r["views"]}),
    }
    (OUT / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print({k: v for k, v in metadata.items() if k != "rejected"}, flush=True)


if __name__ == "__main__":
    main()
