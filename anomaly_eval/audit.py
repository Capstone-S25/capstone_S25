"""Check dataset integrity, duplicates and source diversity without model weights.

Run ``anomaly-eval audit`` with the configured bundle and results roots. Numerical
evaluation regressions are covered separately by the test suite.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from anomaly_eval.data import code_hashes, read_definitions, read_manifest
from anomaly_eval.metrics import detection_metrics
from anomaly_eval.paths import (
    Workspace,
    portable_path,
    resolve_arguments,
    resolve_path,
)


def bundle_checks(root: Path) -> tuple[dict, list[dict]]:
    from PIL import Image, ImageOps

    with (root / "MANIFEST.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    hashes = defaultdict(list)
    dimensions = defaultdict(Counter)
    errors = []
    diagnostic_rows = []
    perceptual_hashes = []
    for row in rows:
        path = root / row["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[digest].append(row["file"])
        if digest != row["sha256"]:
            errors.append({"file": row["file"], "error": "SHA-256 mismatch"})
        with Image.open(path) as image:
            image.load()
            size = image.size
            dimensions[row["label"]][str(size)] += 1
            pixels = (
                ImageOps.exif_transpose(image).convert("L").resize((9, 8)).tobytes()
            )
            value = sum(
                int(pixels[y * 9 + x] > pixels[y * 9 + x + 1]) << (y * 8 + x)
                for y in range(8)
                for x in range(8)
            )
            perceptual_hashes.append((row["file"], row["label"], value))
        # Post-hoc dataset-confounding diagnostic, not a fitted/held-out detector.
        guess = {
            (640, 640): "loitering",
            (1920, 1080): "littering",
            (1024, 1024): "rough_sleeping",
        }.get(size, "normal")
        diagnostic_rows.append(
            {
                "repeat": 0,
                "label": row["label"],
                "prediction": guess,
                "status": "ok",
                "anomaly_score": float(guess != "normal"),
            }
        )
    labels = list(read_definitions(Workspace.current().definitions))
    similar = []
    for index, first in enumerate(perceptual_hashes):
        for second in perceptual_hashes[index + 1 :]:
            distance = (first[2] ^ second[2]).bit_count()
            if distance <= 5:
                similar.append(
                    {
                        "first": first[0],
                        "second": second[0],
                        "first_label": first[1],
                        "second_label": second[1],
                        "dhash_distance": distance,
                    }
                )
    return {
        "n_images": len(rows),
        "class_counts": dict(Counter(row["label"] for row in rows)),
        "unique_groups": len({row["group_id"] for row in rows}),
        "groups_per_class": {
            label: len({row["group_id"] for row in rows if row["label"] == label})
            for label in labels
        },
        "group_counts": dict(Counter(row["group_id"] for row in rows)),
        "dimensions": {label: dict(counts) for label, counts in dimensions.items()},
        "hash_errors": errors,
        "exact_duplicate_groups": [
            files for files in hashes.values() if len(files) > 1
        ],
        "similarity_candidates": similar,
        "manifest_columns": list(rows[0]),
        "original_source_files_available": sum(
            resolve_path(row["source_path"]).is_file()
            if "://" not in row["source_path"]
            else False
            for row in rows
        ),
        "dimension_only_diagnostic": detection_metrics(diagnostic_rows, labels),
        "diagnostic_caveat": "The dimension rule was designed after inspecting this dataset; it diagnoses source confounding and is not a model benchmark.",
    }, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-data", type=Path, default=Workspace.current().bundle_dir
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Workspace.current().results_dir / "audits" / "checks.json",
    )
    parser.add_argument(
        "--portable-manifest",
        type=Path,
        default=Workspace.current().results_dir / "audits" / "historical_manifest.csv",
    )
    args = resolve_arguments(parser.parse_args(argv))
    bundle, rows = bundle_checks(args.test_data.resolve())
    args.portable_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.portable_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "label", "split", "group_id"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "path": portable_path(
                    args.test_data.resolve() / row["file"],
                    args.portable_manifest.resolve().parent,
                ),
                "label": row["label"],
                "split": "dev",
                "group_id": row["group_id"],
            }
            for row in rows
        )
    loaded = read_manifest(
        args.portable_manifest.resolve(),
        read_definitions(Workspace.current().definitions),
    )
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "Bundle integrity and source diversity; numerical regressions are covered by the test suite",
        "python": platform.python_version(),
        "bundle": bundle,
        "portable_manifest_samples": len(loaded),
        "source_sha256": code_hashes(),
        "historical_raw_results_present": (
            Workspace.current().results_dir / "anomaly_headtohead"
        ).is_dir(),
    }

    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    confirmed = args.test_data / "EVALUATION.csv"
    if confirmed.is_file():
        verified = read_manifest(
            confirmed, read_definitions(Workspace.current().definitions)
        )
        result["confirmed_manifest"] = {
            "samples": len(verified),
            "groups": len({s.group_id for s in verified}),
        }
    result = json_safe(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Audit evidence: {args.output}")
    print(
        f"Portable manifest: {args.portable_manifest} ({len(loaded)} samples; inherited proxy labels, not newly verified ground truth)"
    )
    print(
        f"Verified {len(loaded)} historical image hashes; {len(bundle['similarity_candidates'])} near-duplicate candidates require review"
    )
    return 0
