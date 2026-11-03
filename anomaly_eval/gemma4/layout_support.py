"""Explicit retrospective correction for a layout never exercised in development."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from anomaly_eval.datasets.transfer import atomic_json, digest
from anomaly_eval.gemma4.protocol import BASELINE, POLICIES, ROOT, TASKS, read_rows
from anomaly_eval.training.clip import binary_metrics


def branch(row, policy, development_multiview):
    """Never select a branch using source identity, labels or test outcomes."""
    if policy.layout == "sheet" and not development_multiview and len(row["views"]) > 1:
        return "baseline"
    return "optimized"


def source_metrics(rows, scores, branches, calibration):
    result = {}
    for source in sorted({r["source"] for r in rows}):
        members = [r for r in rows if r["source"] == source]
        labels = [r["target"] for r in members]
        margins = [scores[branches[r["id"]]][r["id"]] for r in members]
        ranking = binary_metrics(labels, margins, 0)
        points = {}
        for point in calibration["baseline"]:
            cutoffs = {
                method: calibration[method][point]
                for method in ["baseline", "optimized"]
            }
            predictions = [
                int(scores[branches[r["id"]]][r["id"]] >= cutoffs[branches[r["id"]]])
                for r in members
            ]
            value = binary_metrics(labels, predictions, 0.5)
            value.update(
                ap=ranking["ap"],
                auroc=ranking["auroc"],
                threshold=cutoffs[branches[members[0]["id"]]]
                if len({branches[r["id"]] for r in members}) == 1
                else None,
                route_cutoffs=cutoffs,
                route_counts=dict(Counter(branches[r["id"]] for r in members)),
                groups=len({r["group_id"] for r in members}),
            )
            value["accuracy"] = (value["tp"] + value["tn"]) / value["n"]
            value["balanced_accuracy"] = (
                (value["recall"] + 1 - value["fpr"]) / 2
                if value["positive"] and value["negative"]
                else None
            )
            value["prevalence_ap_reference"] = value["positive"] / value["n"]
            points[point] = value
        result[source] = points
    return result


def derive(output, name, original):
    output = Path(output)
    protocol = output / "layout_support_protocol.json"
    expected = ROOT / "anomaly_eval/config/gemma4_layout_support.json"
    if json.loads(protocol.read_text()) != json.loads(expected.read_text()):
        raise ValueError("Recorded layout-support rule differs")
    directory = output / name
    rows = read_rows(output / "samples.jsonl")
    dev = [r for r in rows if r["cohort"] == "main" and r["split"] == "dev"]
    heldout = [r for r in rows if r["split"] in {"test", "external"}]
    selection = json.loads((directory / "selection.json").read_text())["routes"]
    calibration = json.loads((directory / "calibration.json").read_text())
    policies = {p.name: p for p in POLICIES}
    multiview = {
        task: sum(len(r["views"]) > 1 for r in dev if r["task"] == task)
        for task in TASKS
    }
    branches = {
        r["id"]: branch(r, policies[selection[r["task"]]], multiview[r["task"]])
        for r in heldout
    }
    scores = {}
    for method in ["baseline", "optimized"]:
        predictions = read_rows(directory / (method + "_heldout_predictions.jsonl"))
        if {r["id"] for r in predictions} != {r["id"] for r in heldout}:
            raise ValueError("Incomplete original layout comparison")
        scores[method] = {r["id"]: r["score"] for r in predictions}
    tasks = {}
    for task in TASKS:
        members = [r for r in heldout if r["task"] == task]
        points = {
            method: calibration[method][task] for method in ["baseline", "optimized"]
        }
        computed = source_metrics(members, scores, branches, points)
        # All current source cohorts take one branch. Recheck numerical equality
        # to the corresponding independently audited original result.
        for source, metrics in computed.items():
            modes = {branches[r["id"]] for r in members if r["source"] == source}
            if len(modes) == 1:
                method = next(iter(modes))
                previous = original["routes"][method]["tasks"][task]["source_metrics"][
                    source
                ]
                for point, values in previous.items():
                    if any(values[key] != metrics[point][key] for key in values):
                        raise ValueError("Layout-derived metric differs from original")
        label = selection[task]
        if policies[label].layout == "sheet" and not multiview[task]:
            label += "; unsupported multi-image sheet -> " + BASELINE.name
        tasks[task] = {
            "policy": label,
            "development_multiview_examples": multiview[task],
            "source_metrics": computed,
        }
        if task == "smoking":
            secondary = {
                method: calibration[method]["xrf55_source_dev"]
                for method in ["baseline", "optimized"]
            }
            tasks[task]["xrf55_source_recalibrated"] = source_metrics(
                [r for r in members if r["cohort"] == "xrf55"],
                scores,
                branches,
                secondary,
            )
    prediction_path = directory / "layout_supported_heldout_predictions.jsonl"
    prediction_path.write_text(
        "".join(
            json.dumps(
                {
                    **{
                        key: r[key]
                        for key in [
                            "id",
                            "task",
                            "source",
                            "group_id",
                            "split",
                            "target",
                            "cohort",
                        ]
                    },
                    "route": branches[r["id"]],
                    "score": scores[branches[r["id"]]][r["id"]],
                }
            )
            + "\n"
            for r in heldout
        )
    )
    result = {
        "tasks": tasks,
        "heldout_examples": len(heldout),
        "baseline_fallback_cases": sum(v == "baseline" for v in branches.values()),
        "predictions_sha256": digest(prediction_path),
        "identity": {
            "code_sha256": digest(Path(__file__)),
            "protocol_sha256": digest(protocol),
            "original_results_sha256": digest(directory / "results.json"),
        },
        "qualification": "Post-result E2B diagnostic; branch uses input layout support, not test source or labels. Original results remain unchanged.",
    }
    atomic_json(directory / "layout_supported_results.json", result)
    return result
