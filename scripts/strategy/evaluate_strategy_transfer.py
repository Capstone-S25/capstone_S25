"""Compare fixed development-selected strategy families on qualified holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.training.clip import binary_metrics, select_points
from anomaly_eval.training.data import TASKS
from anomaly_eval.training.strategies import geometry, normalize, represent
from scripts.strategy.run_strategy_screen import predict

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/strategy_study_20261016"
RUN = ROOT / "results/strategy_study_20261016"


def read(path):
    return json.loads(path.read_text())


def load_features(model, tag):
    output = {}
    for stage in ["full", "crop"]:
        for alpha in [0, 0.25, 0.5, 1]:
            path = DATA / "features" / model / f"{tag}_{stage}_alpha_{alpha:g}.npz"
            if path.exists():
                with np.load(path) as value:
                    output[stage, alpha] = dict(
                        zip(value["paths"].tolist(), value["features"])
                    )
    return output


def matrix(rows, recipe, features, detections):
    alpha = float(recipe.rsplit("a", 1)[1])
    output = []
    for row in rows:
        if recipe.startswith("endpoint"):
            full = features["full", alpha][row["views"][-1]]
        else:
            full = represent(
                [features["full", alpha][p] for p in row["views"]],
                ordered="ordered" in recipe,
            )
        if "crop" in recipe:
            local = represent(
                [features["crop", 0][p] for p in row["views"]],
                ordered="ordered" in recipe,
            )
            full = normalize(np.concatenate([full, local]))
        if "geometry" in recipe:
            full = normalize(
                np.concatenate([full, geometry([detections[p] for p in row["views"]])])
            )
        output.append(full)
    return np.stack(output).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=["mobileclip2-s2", "siglip2-b16-224"]
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    directory = RUN / args.model
    output = directory / "transfer_evaluation"
    output.mkdir(exist_ok=False)
    selected = read(directory / "selection.json")["tasks"]
    bundle = torch.load(
        directory / "selected_heads.pt", weights_only=True, map_location="cpu"
    )
    main_rows = [
        json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines()
    ]
    detections = {
        r["path"]: r["detections"]
        for r in map(json.loads, (DATA / "detections.jsonl").read_text().splitlines())
    }
    rough_cache = DATA / "rough_detections.jsonl"
    if rough_cache.exists():
        detections.update(
            {
                r["path"]: r["detections"]
                for r in map(json.loads, rough_cache.read_text().splitlines())
            }
        )
    main_features = load_features(args.model, "main")
    old = ROOT / "results/clip_finetuning_20261015/seed_42" / args.model
    old_selection = read(old / "float32_evaluation/selection.json")["task_routes"]
    old_metrics = read(old / "float32_evaluation/results.json")
    prior = {
        m: torch.load(old / name, weights_only=True, map_location="cpu")
        for m, name in [
            ("linear_probe", "linear_probe.pt"),
            ("finetuned", "finetuned_best.pt"),
        ]
    }
    policies, thresholds, dev_scores = {}, {}, {}
    for task in TASKS:
        ti = TASKS.index(task)
        candidates = torch.load(
            directory / f"{task}_candidates.pt", weights_only=True, map_location="cpu"
        )["candidates"]
        families = {
            c["selection"]["strategy"] + "/" + c["selection"]["classifier"]
            for c in candidates
        }
        entries = {
            "selected": {"selection": selected[task], "head": bundle["heads"][task]}
        }
        for family in sorted(families):
            values = [
                c
                for c in candidates
                if c["selection"]["strategy"] + "/" + c["selection"]["classifier"]
                == family
            ]
            entries[family] = max(
                values, key=lambda c: c["selection"]["mean_source_ap"]
            )
        entries["worst_source_selection"] = max(
            candidates, key=lambda c: c["selection"]["worst_source_ap"]
        )
        method = old_selection[task]
        if method == "zero_shot":
            recipe = "endpoint_a0"
            head = next(
                c["head"]
                for c in candidates
                if c["selection"]["classifier"] == "zero_shot"
            )
        else:
            recipe = "full_pool_a0" if method == "linear_probe" else "full_pool_a1"
            params = prior[method]["parameters"]
            head = {
                "kind": "linear",
                "weight": params["head.weight"][ti : ti + 1],
                "bias": params["head.bias"][ti : ti + 1],
            }
        entries["previous_bundle"] = {
            "selection": {"strategy": recipe, "classifier": method},
            "head": head,
        }
        dev = [r for r in main_rows if r["task"] == task and r["split"] == "dev"]
        policies[task], thresholds[task], dev_scores[task] = entries, {}, {}
        for name, entry in entries.items():
            values = predict(
                entry["head"],
                matrix(dev, entry["selection"]["strategy"], main_features, detections),
            )
            dev_scores[task][name] = {
                "ids": [r["id"] for r in dev],
                "scores": values.tolist(),
            }
            thresholds[task][name] = (
                old_metrics[method]["dev"][task]["thresholds"]
                if name == "previous_bundle"
                else select_points([r["target"] for r in dev], values)
            )
    # Preserve original route choices; serialized-head calibration uses dev only.
    frozen = {
        task: {
            name: {
                "selection": entry["selection"],
                "thresholds": thresholds[task][name],
            }
            for name, entry in entries.items()
        }
        for task, entries in policies.items()
    }
    (output / "frozen_policies.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (output / "development_predictions.json").write_text(json.dumps(dev_scores) + "\n")
    summary = {}
    for tag, manifest in [
        ("photo", "samples.jsonl"),
        ("historical", "historical.jsonl"),
        ("hmdb", "hmdb_screened.jsonl"),
        ("meva", "meva_screened.jsonl"),
        ("rough", "rough_screened.jsonl"),
    ]:
        rows = [json.loads(line) for line in (DATA / manifest).read_text().splitlines()]
        if tag == "photo":
            rows = [r for r in rows if r["split"] == "test"]
        features = main_features if tag == "photo" else load_features(args.model, tag)
        summary[tag] = {}
        with (output / (tag + "_predictions.jsonl")).open("w") as stream:
            for task in sorted({r["task"] for r in rows}):
                subset = [r for r in rows if r["task"] == task]
                y = [r["target"] for r in subset]
                summary[tag][task] = {}
                for name, entry in policies[task].items():
                    recipe = entry["selection"]["strategy"]
                    alpha = float(recipe.rsplit("a", 1)[1])
                    if ("full", alpha) not in features:
                        continue  # External single-task cohorts need only relevant weights.
                    values = predict(
                        entry["head"], matrix(subset, recipe, features, detections)
                    )
                    metrics = {
                        op: binary_metrics(y, values, threshold)
                        for op, threshold in thresholds[task][name].items()
                    }
                    summary[tag][task][name] = {
                        "operating_points": metrics,
                        "groups": len({r["group_id"] for r in subset}),
                        "strategy": recipe,
                        "sources": sorted({r["source"] for r in subset}),
                    }
                    for row, score in zip(subset, values):
                        stream.write(
                            json.dumps(
                                {
                                    "id": row["id"],
                                    "task": task,
                                    "target": row["target"],
                                    "group_id": row["group_id"],
                                    "source": row["source"],
                                    "split": row["split"],
                                    "family": name,
                                    "score": float(score),
                                }
                            )
                            + "\n"
                        )
        print("evaluated", args.model, tag, len(rows), flush=True)
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [directory / "selection.json", directory / "selected_heads.pt"]
    }
    (output / "state.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "selection_checkpoint_sha256": hashes,
                "routes_changed": False,
                "serialized_head_thresholds": "development only; previous bundle thresholds retained",
                "qualifications": {
                    "photo": "New official image holdout in a photo source that contributes training data; subject identity unavailable",
                    "historical": "Previously inspected retrospective holdouts, not fresh confirmation",
                    "hmdb": "Entirely new movie/web action source; offline eight-view classification, primary-action negatives not exhaustive",
                    "meva": "New annotated abandonment-action windows, not independently verified persistent unattended-object status",
                },
            },
            indent=2,
        )
        + "\n"
    )
    print("TRANSFER EVALUATION COMPLETE", args.model, flush=True)


if __name__ == "__main__":
    main()
