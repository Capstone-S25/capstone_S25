"""Audit video input identities, fold isolation and causal window boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anomaly_eval.data import image_pixel_sha256

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    data = args.data.resolve()
    rows = [
        r
        for n in ["windows.jsonl", "fresh_avss.jsonl"]
        for r in map(json.loads, (data / n).read_text().splitlines())
    ]
    group_folds = defaultdict(set)
    record_folds = defaultdict(set)
    path_folds = defaultdict(set)
    ids = set()
    for row in rows:
        assert row["id"] not in ids, row["id"]
        ids.add(row["id"])
        assert row["target"] in {-1, 0, 1}
        assert len(row["timestamps"]) == len(row["views"]) > 0
        assert all(b > a for a, b in zip(row["timestamps"], row["timestamps"][1:]))
        if row["task"] == "unattended_items":
            assert row["timestamps"][-1] - row["timestamps"][0] <= 20.00001
        group_folds[row["group_id"]].add(row["split"])
        record_folds[row["recording_id"]].add(row["split"])
        for p in row["views"]:
            path_folds[p].add(row["split"])
    for name in ["recordings.jsonl", "fresh_recordings.jsonl"]:
        for record in map(json.loads, (data / name).read_text().splitlines()):
            folds = record_folds[record["id"]]
            for frame in record["frames"]:
                path_folds[frame["path"]].update(folds)
    assert all(len(v) == 1 for v in group_folds.values()), "Group leakage"
    assert all(len(v) == 1 for v in record_folds.values()), "Recording leakage"
    assert all(len(v) == 1 for v in path_folds.values()), "Path leakage"
    paths = sorted(path_folds)

    def one(p):
        path = ROOT / p
        return {
            "path": p,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pixel_sha256": image_pixel_sha256(path),
            "split": next(iter(path_folds[p])),
        }

    with ThreadPoolExecutor(12) as pool:
        identities = list(pool.map(one, paths))
    pixels = defaultdict(set)
    pixel_paths = defaultdict(list)
    for r in identities:
        pixels[r["pixel_sha256"]].add(r["split"])
        pixel_paths[r["pixel_sha256"]].append(r["path"])
    conflicts = [
        {"pixel_sha256": h, "splits": sorted(folds), "paths": pixel_paths[h]}
        for h, folds in pixels.items()
        if len(folds) > 1
    ]
    (data / "input_identities.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in identities)
    )
    report = {
        "status": "pass" if not conflicts else "fail",
        "windows": len(rows),
        "groups": len(group_folds),
        "recordings": len(record_folds),
        "unique_window_images": len(paths),
        "decoded_pixel_conflicts": conflicts,
        "strictly_causal_windows": True,
        "source_groups": "supplied camera/date/title prefix groups; movie/actor identities not independently verified",
        "manifest_sha256": {
            n: hashlib.sha256((data / n).read_bytes()).hexdigest()
            for n in ["windows.jsonl", "fresh_avss.jsonl"]
        },
    }
    by_path = {r["path"]: r["pixel_sha256"] for r in identities}
    report["canonical_manifests"] = {}
    # Complete/repair inherited sparse-view hashes in separate audited copies.
    # Prepared files stay fixed; paths, labels, splits and times are identical.
    for name in ["windows", "fresh_avss"]:
        prepared = data / (name + ".jsonl")
        verified = data / ("verified_" + name + ".jsonl")
        corrected = []
        amended = 0
        for row in map(json.loads, prepared.read_text().splitlines()):
            hashes = [by_path[p] for p in row["views"]]
            amended += row.get("pixel_hashes") != hashes
            corrected.append({**row, "pixel_hashes": hashes})
        content = "".join(json.dumps(r) + "\n" for r in corrected)
        if verified.exists() and verified.read_text() != content:
            raise ValueError("Canonical inputs changed after verification")
        if not verified.exists():
            verified.write_text(content)
        report["canonical_manifests"][name] = {
            "path": str(verified),
            "sha256": hashlib.sha256(verified.read_bytes()).hexdigest(),
            "prepared_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(),
            "pixel_hash_fields_completed_or_repaired": amended,
            "other_fields_unchanged": True,
        }
    (data / "data_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "VIDEO DATA AUDIT",
        report["status"],
        len(paths),
        "images",
        len(conflicts),
        "cross-fold pixel conflicts",
        flush=True,
    )
    if conflicts:
        raise ValueError("Resolve source identity conflicts before fitting")


if __name__ == "__main__":
    main()
