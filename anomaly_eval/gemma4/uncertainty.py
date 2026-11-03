"""Conditional group-bootstrap uncertainty for paired prompt comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from anomaly_eval.datasets.transfer import atomic_json, digest
from anomaly_eval.gemma4.protocol import read_rows


def interval(rows, baseline, optimized, repetitions=500):
    groups = sorted({r["group_id"] for r in rows})
    result = {"cases": len(rows), "groups": len(groups), "repetitions": repetitions}
    if len(groups) < 3:
        return {
            **result,
            "status": "too_few_groups",
            "baseline_ci95": None,
            "optimized_ci95": None,
            "paired_delta_ci95": None,
        }
    index = {group: i for i, group in enumerate(groups)}
    aliases = np.array([index[r["group_id"]] for r in rows])
    labels = np.array([r["target"] for r in rows])
    first = np.array([baseline[r["id"]] for r in rows])
    second = np.array([optimized[r["id"]] for r in rows])
    rng = np.random.default_rng(20261027)
    values = []
    for _ in range(repetitions):
        weights = np.bincount(
            rng.integers(0, len(groups), len(groups)), minlength=len(groups)
        )[aliases]
        if not weights[labels == 0].sum() or not weights[labels == 1].sum():
            continue
        ap1 = float(average_precision_score(labels, first, sample_weight=weights))
        ap2 = float(average_precision_score(labels, second, sample_weight=weights))
        values.append([ap1, ap2, ap2 - ap1])
    result["valid_repetitions"] = len(values)
    if len(values) < min(100, repetitions):
        return {
            **result,
            "status": "insufficient_two_class_resamples",
            "baseline_ci95": None,
            "optimized_ci95": None,
            "paired_delta_ci95": None,
        }
    bounds = np.quantile(np.asarray(values), [0.025, 0.975], axis=0)
    result.update(
        status="conditional_on_observed_groups",
        baseline_ci95=bounds[:, 0].tolist(),
        optimized_ci95=bounds[:, 1].tolist(),
        paired_delta_ci95=bounds[:, 2].tolist(),
    )
    return result


def compute(directory):
    identity = {
        "results_sha256": digest(directory / "results.json"),
        "code_sha256": digest(Path(__file__)),
        "repetitions": 500,
    }
    destination = directory / "ap_uncertainty.json"
    if destination.exists():
        result = json.loads(destination.read_text())
        if result["identity"] == identity:
            return result
    rows = read_rows(directory / "baseline_heldout_predictions.jsonl")
    baseline = {r["id"]: r["score"] for r in rows}
    optimized = {
        r["id"]: r["score"]
        for r in read_rows(directory / "optimized_heldout_predictions.jsonl")
    }
    result = {
        "identity": identity,
        "sources": {},
        "qualification": "Paired resampling of observed source groups; does not estimate generalization to new cameras, correct weak labels or undo prior holdout reuse. Fewer than three groups yields no interval.",
    }
    for task, source in sorted({(r["task"], r["source"]) for r in rows}):
        result["sources"][task + "/" + source] = interval(
            [r for r in rows if r["task"] == task and r["source"] == source],
            baseline,
            optimized,
        )
    atomic_json(destination, result)
    return result


def main(argv=None):
    """Export paired source-group intervals for completed second-stage routes."""
    from anomaly_eval.gemma4.optimization import OUTPUT
    from anomaly_eval.gemma4.protocol import DEFAULT_OUTPUT, MODELS, ROOT
    from anomaly_eval.gemma4.report import write_csv

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--previous", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--repetitions", type=int, default=500)
    args = parser.parse_args(argv)
    if args.repetitions < 100:
        parser.error("Use at least 100 paired group resamples")
    rows = []
    identity = {
        "code_sha256": digest(Path(__file__)),
        "repetitions": args.repetitions,
        "inputs": {},
    }
    for name in MODELS:
        result = args.output / name / "results.json"
        if not result.exists():
            continue
        baseline_path = args.previous / name / "baseline_heldout_predictions.jsonl"
        examples = read_rows(baseline_path)
        baseline = {r["id"]: r["score"] for r in examples}
        identity["inputs"][str(baseline_path)] = digest(baseline_path)
        for method in json.loads(result.read_text())["methods"]:
            path = args.output / name / (method + "_predictions.jsonl")
            predictions = read_rows(path)
            if {r["id"] for r in predictions} != set(baseline) or len(
                predictions
            ) != len(examples):
                raise ValueError(
                    "Bootstrap inputs do not contain the identical held-out cohort"
                )
            indexed = {r["id"]: r for r in examples}
            for r in predictions:
                if any(
                    r[k] != indexed[r["id"]][k]
                    for k in ["task", "source", "group_id", "target"]
                ):
                    raise ValueError("Bootstrap pair identity/label mismatch")
            scores = {r["id"]: r["score"] for r in predictions}
            identity["inputs"][str(path)] = digest(path)
            for task, source in sorted({(r["task"], r["source"]) for r in examples}):
                members = [
                    r for r in examples if r["task"] == task and r["source"] == source
                ]
                rows.append(
                    {
                        "model": name,
                        "method": method,
                        "task": task,
                        "source": source,
                        **interval(members, baseline, scores, args.repetitions),
                    }
                )
        print(name, "group-bootstrap complete", flush=True)
    if not rows:
        print("No completed second-stage model is available yet.")
        return 0
    write_csv(args.output / "ap_uncertainty.csv", rows)
    atomic_json(
        args.output / "ap_uncertainty.json",
        {
            "identity": identity,
            "qualification": "Paired group resampling conditional on observed source groups and labels, not new-camera or prospective validation. Sources with fewer than three groups have no interval.",
            "sources": rows,
        },
    )
    tracked = ROOT / "reports/gemma4_optimization_20261028"
    tracked.mkdir(exist_ok=True)
    for file in ["ap_uncertainty.csv", "ap_uncertainty.json"]:
        (tracked / file).write_bytes((args.output / file).read_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
