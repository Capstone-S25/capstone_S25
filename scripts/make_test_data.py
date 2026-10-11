"""Copy the vetted head-to-head test frames into a standalone bundle.

Reads the sample entries from the benchmark run.json (each with path, label,
split, group_id) and copies every source image into <out-dir>/<label>/ while
preserving its original basename, then writes MANIFEST.csv mapping each file to
its label, group, split, original source path, and SHA-256 digest so the bundle
can be re-verified against the original datasets.

Usage:
    python -I scripts/make_test_data.py \
        --run-json results/anomaly_headtohead/run.json \
        --out-dir test_data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def find_samples(run: dict) -> list[dict]:
    for value in run.values():
        if (
            isinstance(value, list)
            and len(value) > 0
            and all(isinstance(e, dict) for e in value)
            and "path" in value[0]
            and "label" in value[0]
        ):
            return value
    raise SystemExit("no sample list with path/label entries found in run.json")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.run_json.read_text())
    samples = find_samples(run)
    rows: list[dict] = []
    for sample in samples:
        src = Path(sample["path"])
        if not src.is_file():
            raise SystemExit(f"missing source frame: {src}")
        dest_dir = args.out_dir / str(sample["label"])
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        digest_src = sha256_of(src)
        if dest.exists():
            # Never silently overwrite a different file.
            if sha256_of(dest) != digest_src:
                raise SystemExit(f"refusing to overwrite differing file: {dest}")
        else:
            shutil.copy2(src, dest)
        rows.append(
            {
                "file": f"{sample['label']}/{src.name}",
                "label": sample["label"],
                "group_id": sample.get("group_id", ""),
                "split": sample.get("split", ""),
                "source_path": str(src),
                "sha256": digest_src,
            }
        )

    manifest = args.out_dir / "MANIFEST.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_label: dict[str, int] = {}
    for row in rows:
        by_label[row["label"]] = by_label.get(row["label"], 0) + 1
    total = sum(
        p.stat().st_size
        for p in args.out_dir.rglob("*")
        if p.is_file() and p.name != "MANIFEST.csv"
    )
    print(f"copied {len(rows)} frames -> {args.out_dir}")
    for label in sorted(by_label):
        print(f"  {label}: {by_label[label]}")
    print(f"total image bytes: {total} ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
