"""Conservative transfer-photo/video duplicate audit against all strategy inputs."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.strategy.prepare_strategy_data import ROOT, signature

DATA = ROOT / "data/strategy_study_20261016"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Screened cohort already frozen")
    training = [
        json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines()
    ]
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    reference = sorted({p for r in training for p in r["views"]})
    views = sorted({p for r in rows for p in r["views"]})
    cache = DATA / "reference_phashes.json"
    if cache.exists():
        hashes = json.loads(cache.read_text())
        assert sorted(hashes) == reference
    else:
        with ThreadPoolExecutor(12) as pool:
            hashes = dict(
                zip(reference, pool.map(lambda p: signature(ROOT / p), reference))
            )
        cache.write_text(json.dumps(hashes) + "\n")
    known = list(hashes.values())
    with ThreadPoolExecutor(12) as pool:
        signatures = dict(zip(views, pool.map(lambda p: signature(ROOT / p), views)))
    matched = {
        p: min((s ^ other).bit_count() for other in known)
        for p, s in signatures.items()
    }
    rejected, retained = [], []
    for row in rows:
        distance = min(matched[p] for p in row["views"])
        if distance <= 6:
            rejected.append({"id": row["id"], "nearest_phash_distance": distance})
        else:
            retained.append(row)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in retained))
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(
            {
                "input": str(args.manifest),
                "retained": len(retained),
                "positive": sum(r["target"] for r in retained),
                "excluded_related_to_any_strategy_image": rejected,
                "policy": "Any of the fixed input views within 6/63 phash bits of any strategy train/dev/new-photo-test image excludes the entire transfer example; screening is independent of model predictions",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "SCREENED",
        args.output.name,
        len(retained),
        "excluded",
        len(rejected),
        flush=True,
    )


if __name__ == "__main__":
    main()
