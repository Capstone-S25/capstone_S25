"""Replay metrics and check real checkpoint-backed strategy inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.data import image_pixel_sha256
from anomaly_eval.training.clip import binary_metrics, select_points
from anomaly_eval.training.data import TASKS
from scripts.predict_strategy import main as predict_live

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/strategy_study_20261016"
RUN = ROOT / "results/strategy_study_20261016"


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    main_rows = [
        json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines()
    ]
    manifests = {"photo": [r for r in main_rows if r["split"] == "test"]}
    for tag, name in [
        ("historical", "historical"),
        ("hmdb", "hmdb_screened"),
        ("meva", "meva_screened"),
        ("rough", "rough_screened"),
    ]:
        manifests[tag] = [
            json.loads(line)
            for line in (DATA / (name + ".jsonl")).read_text().splitlines()
        ]
    expected = {
        p: h
        for r in main_rows + [row for rr in manifests.values() for row in rr]
        for p, h in zip(r["views"], r["pixel_hashes"])
    }
    with ThreadPoolExecutor(12) as pool:
        for (name, saved), actual in zip(
            expected.items(),
            pool.map(lambda name: image_pixel_sha256(ROOT / name), expected),
        ):
            assert saved == actual, name
    audit = {
        "status": "complete",
        "date": "2026-10-16",
        "unique_input_images_verified": len(expected),
        "model_target_candidate_fits": 0,
        "predictions_recomputed": 0,
        "routes_selected_without_old_test": True,
        "source_separation": True,
        "inference_checks": [],
    }
    group_fold, pixel_fold = {}, {}
    for row in main_rows:
        for storage, keys in [
            (group_fold, [row["group_id"]]),
            (pixel_fold, row["pixel_hashes"]),
        ]:
            for key in keys:
                if key in storage:
                    assert storage[key] == row["split"]
                storage[key] = row["split"]
    for tag in ["hmdb", "meva", "rough"]:
        assert not {p for r in manifests[tag] for p in r["pixel_hashes"]}.intersection(
            pixel_fold
        )
    for model in ["mobileclip2-s2", "siglip2-b16-224"]:
        directory = RUN / model
        selection = read(directory / "selection.json")
        assert (
            selection["manifest_sha256"]
            == hashlib.sha256((DATA / "samples.jsonl").read_bytes()).hexdigest()
        )
        results = read(directory / "transfer_evaluation/results.json")
        policies = read(directory / "transfer_evaluation/frozen_policies.json")
        development = read(
            directory / "transfer_evaluation/development_predictions.json"
        )
        bundle = torch.load(
            directory / "selected_heads.pt", weights_only=True, map_location="cpu"
        )
        assert bundle["model"] == model
        for task in TASKS:
            candidates = torch.load(
                directory / f"{task}_candidates.pt",
                weights_only=True,
                map_location="cpu",
            )["candidates"]
            assert len(candidates) == 33
            audit["model_target_candidate_fits"] += len(candidates)
            best = max(candidates, key=lambda c: c["selection"]["mean_source_ap"])
            for key in ["strategy", "classifier", "mean_source_ap"]:
                assert (
                    best["selection"][key]
                    == selection["tasks"][task][key]
                    == policies[task]["selected"]["selection"][key]
                )
            for name, points in development[task].items():
                if name == "previous_bundle":
                    continue
                labels = {
                    r["id"]: r["target"]
                    for r in main_rows
                    if r["task"] == task and r["split"] == "dev"
                }
                expected_points = select_points(
                    [labels[key] for key in points["ids"]], points["scores"]
                )
                assert expected_points == policies[task][name]["thresholds"]
        for tag, rows in manifests.items():
            index = {r["id"]: r for r in rows}
            records = [
                json.loads(line)
                for line in (
                    directory / "transfer_evaluation" / (tag + "_predictions.jsonl")
                )
                .read_text()
                .splitlines()
            ]
            groups = defaultdict(list)
            for record in records:
                row = index[record["id"]]
                assert all(
                    record[k] == row[k]
                    for k in ["task", "target", "group_id", "source"]
                )
                assert np.isfinite(record["score"])
                groups[record["task"], record["family"]].append(record)
            for (task, family), values in groups.items():
                assert len(values) == sum(r["task"] == task for r in rows)
                points = policies[task][family]["thresholds"]
                metrics = {
                    op: binary_metrics(
                        [r["target"] for r in values],
                        [r["score"] for r in values],
                        threshold,
                    )
                    for op, threshold in points.items()
                }
                expected_metrics = results[tag][task][family]["operating_points"]
                for op, metric in metrics.items():
                    assert all(
                        np.isclose(
                            metric[k], expected_metrics[op][k], atol=1e-12, rtol=0
                        )
                        for k in metric
                    )
            assert (directory / "selection.json").stat().st_mtime_ns < (
                directory / "transfer_evaluation" / (tag + "_predictions.jsonl")
            ).stat().st_mtime_ns
            audit["predictions_recomputed"] += len(records)
        print("METRICS AND ROUTES VERIFIED", model, flush=True)
    if args.live:
        cases = [
            ("photo", "smoking"),
            ("historical", "littering"),
            ("historical", "loitering"),
            ("rough", "rough_sleeping"),
            ("meva", "unattended_items"),
        ]
        for model in ["mobileclip2-s2", "siglip2-b16-224"]:
            for tag, task in cases:
                row = next(
                    r for r in manifests[tag] if r["task"] == task and r["target"] == 1
                )
                output = RUN / f"inference_check_{model}_{task}.json"
                arguments = [
                    "--model",
                    model,
                    "--images",
                    *row["views"],
                    "--device",
                    "cuda",
                    "--operating-point",
                    "raw",
                    "--output",
                    str(output),
                ]
                if task in {"loitering", "unattended_items"}:
                    arguments.extend(
                        ["--timestamps", "0", "6.666667", "13.333333", "20"]
                    )
                predict_live(arguments)
                actual = read(output)["predictions"]
                predicted = next(
                    r
                    for r in map(
                        json.loads,
                        (
                            RUN
                            / model
                            / "transfer_evaluation"
                            / (tag + "_predictions.jsonl")
                        )
                        .read_text()
                        .splitlines(),
                    )
                    if r["family"] == "selected" and r["id"] == row["id"]
                )
                delta = abs(actual[task]["score"] - predicted["score"])
                assert delta < 0.005, (model, task, delta)
                assert actual[task]["decision"] != "unknown"
                if len(row["views"]) == 1:
                    assert all(
                        actual[t]["decision"] == "unknown"
                        for t in ["littering", "loitering", "unattended_items"]
                    )
                audit["inference_checks"].append(
                    {
                        "model": model,
                        "task": task,
                        "absolute_score_difference": delta,
                        "tolerance": 0.005,
                    }
                )
                print("LIVE INFERENCE VERIFIED", model, task, delta, flush=True)
    audit["live_inference_completed"] = args.live
    (
        RUN / ("verification.json" if args.live else "metric_verification.json")
    ).write_text(json.dumps(audit, indent=2) + "\n")
    print(
        "STRATEGY STUDY VERIFIED",
        audit["predictions_recomputed"],
        "predictions",
        flush=True,
    )


if __name__ == "__main__":
    main()
