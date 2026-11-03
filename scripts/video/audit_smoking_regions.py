"""Measure source-annotated smoking-person size and automatic crop coverage."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.video.extract_video_pilot import regions

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    old = ROOT / "data/strategy_study_20261016"
    rows = {
        r["views"][0]: r
        for n in ["samples", "historical"]
        for r in map(json.loads, (old / (n + ".jsonl")).read_text().splitlines())
        if r["source"] == "smoking_cctv" and r["target"] == 1
    }
    detections = {
        r["path"]: r["detections"]
        for r in map(json.loads, (old / "detections.jsonl").read_text().splitlines())
    }
    tracks = {
        r["path"]: r["tracks"]
        for r in map(json.loads, (data / "tracks.jsonl").read_text().splitlines())
    }
    summaries = {}
    details = []
    for path in (ROOT / "data/raw/datasets/smoking/smoking-video.v1i.coco").glob(
        "*/_annotations.coco.json"
    ):
        coco = json.loads(path.read_text())
        images = {r["id"]: r for r in coco["images"]}
        ann = defaultdict(list)
        for a in coco["annotations"]:
            if a["category_id"] == 2:
                ann[a["image_id"]].append(a["bbox"])
        for image_id, boxes in ann.items():
            image = images[image_id]
            name = str((path.parent / image["file_name"]).relative_to(ROOT))
            if name not in rows:
                continue
            w, h = image["width"], image["height"]
            targets = [
                [x / w, y / h, (x + bw) / w, (y + bh) / h] for x, y, bw, bh in boxes
            ]
            chosen = []
            for labels in [{1}, {27, 31, 33}]:
                options = [r for r in detections[name] if r["label"] in labels]
                if options:
                    b = max(options, key=lambda x: x["confidence"])["box"]
                    dx = (b[2] - b[0]) * 0.2
                    dy = (b[3] - b[1]) * 0.2
                    chosen.append(
                        [
                            max(0, b[0] - dx),
                            max(0, b[1] - dy),
                            min(1, b[2] + dx),
                            min(1, b[3] + dy),
                        ]
                    )
            prior = (
                [
                    [
                        min(b[0] for b in chosen),
                        min(b[1] for b in chosen),
                        max(b[2] for b in chosen),
                        max(b[3] for b in chosen),
                    ]
                ]
                if chosen
                else [[0, 0, 1, 1]]
            )
            current = regions(tracks[name]) or [[0, 0, 1, 1]]

            def coverage(crops, targets=targets):
                return max(
                    max(0, min(a[2], b[2]) - max(a[0], b[0]))
                    * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                    / ((a[2] - a[0]) * (a[3] - a[1]))
                    for a in targets
                    for b in crops
                )

            details.append(
                {
                    "id": rows[name]["id"],
                    "split": rows[name]["split"],
                    "path": name,
                    "largest_smoker_height_at_224": max(
                        bh * 224 / max(w, h) for x, y, bw, bh in boxes
                    ),
                    "prior_target_coverage": coverage(prior),
                    "pilot_target_coverage": coverage(current),
                    "pilot_region_count": len(current),
                    "pilot_scene_fallback": not bool(regions(tracks[name])),
                }
            )
    for split in ["train", "dev", "test"]:
        values = [r for r in details if r["split"] == split]
        summaries[split] = {
            "positive_frames": len(values),
            "median_largest_smoker_height_at_224": float(
                np.median([r["largest_smoker_height_at_224"] for r in values])
            ),
            "all_smokers_under_12_pixels_fraction": float(
                np.mean([r["largest_smoker_height_at_224"] < 12 for r in values])
            ),
            "prior_under_half_target_coverage_fraction": float(
                np.mean([r["prior_target_coverage"] < 0.5 for r in values])
            ),
            "pilot_under_half_target_coverage_fraction": float(
                np.mean([r["pilot_target_coverage"] < 0.5 for r in values])
            ),
            "pilot_scene_fallback_count": sum(
                r["pilot_scene_fallback"] for r in values
            ),
        }
    args.output.write_text(
        json.dumps(
            {
                "summary": summaries,
                "frames": details,
                "qualification": "Retrospective geometric coverage, not smoking accuracy. Different fixed detector resolutions and multiple crops are both changed. Source boxes never select inference crops.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
