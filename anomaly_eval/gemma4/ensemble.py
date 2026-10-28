"""Predeclared development-only Gemma family routing and equal-weight ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anomaly_eval.datasets.transfer import atomic_json, digest
from anomaly_eval.gemma4.protocol import (
    BASELINE,
    DEFAULT_OUTPUT,
    MODELS,
    ROOT,
    TASKS,
    read_rows,
    source_ap,
)
from anomaly_eval.gemma4.report import write_csv
from anomaly_eval.gemma4.study import calibrate, metrics


def fit_development(dev, predictions):
    if any(r["split"] != "dev" or r["cohort"] != "main" for r in dev):
        raise ValueError("Ensemble fitting requires primary development only")
    models = list(predictions)
    fitted = {"routes": {}, "normalization": {}, "development_ap": {}}
    for task in TASKS:
        members = [r for r in dev if r["task"] == task]
        performance = {name: source_ap(dev, predictions[name], task) for name in models}
        fitted["development_ap"][task] = performance
        fitted["routes"][task] = max(
            models, key=lambda name: (performance[name], -models.index(name))
        )
        fitted["normalization"][task] = {}
        for name in models:
            values = [predictions[name][r["id"]] for r in members]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite development ensemble margin")
            scale = float(np.std(values))
            fitted["normalization"][task][name] = {
                "mean": float(np.mean(values)),
                "std": scale if scale > 1e-8 else 1.0,
            }
    return fitted


def transform(rows, predictions, fitted, method):
    result = {}
    for row in rows:
        task = row["task"]
        if method == "development_best_model_per_task":
            value = predictions[fitted["routes"][task]][row["id"]]
        elif method == "equal_weight_mean_of_development_z_scores":
            value = float(
                np.mean(
                    [
                        (predictions[name][row["id"]] - stats["mean"]) / stats["std"]
                        for name, stats in fitted["normalization"][task].items()
                    ]
                )
            )
        else:
            raise ValueError("Unknown predeclared ensemble")
        if not np.isfinite(value):
            raise ValueError("Nonfinite ensemble score")
        result[row["id"]] = value
    return result


def evaluate(output):
    output = Path(output)
    protocol_path = output / "ensemble_protocol.json"
    if not protocol_path.exists():
        raise ValueError("Declare the ensemble protocol before held-out model scoring")
    expected = ROOT / "anomaly_eval/config/gemma4_ensembles.json"
    if json.loads(protocol_path.read_text()) != json.loads(expected.read_text()):
        raise ValueError("The predeclared ensemble protocol differs")
    incomplete = [
        name for name in MODELS if not (output / name / "results.json").exists()
    ]
    if incomplete:
        print("Ensembles wait for completed sizes:", ", ".join(incomplete))
        return None
    rows = read_rows(output / "samples.jsonl")
    dev = [r for r in rows if r["cohort"] == "main" and r["split"] == "dev"]
    source_dev = [r for r in rows if r["cohort"] == "xrf55" and r["split"] == "dev"]
    heldout = [r for r in rows if r["split"] in {"test", "external"}]
    model_scores = {}
    route_scores = {}
    for name in MODELS:
        directory = output / name
        routes = json.loads((directory / "selection.json").read_text())["routes"]
        cached = {
            policy: {
                r["id"]: r
                for r in read_rows(directory / "scores" / (policy + ".jsonl"))
            }
            for policy in set(routes.values()) | {BASELINE.name}
        }
        for method in ["baseline", "optimized"]:
            scores = {}
            for row in rows:
                policy = BASELINE.name if method == "baseline" else routes[row["task"]]
                value = cached[policy][row["id"]]
                if any(
                    value[k] != row[k]
                    for k in ["task", "target", "source", "split", "cohort"]
                ):
                    raise ValueError("Ensemble source prediction identity mismatch")
                scores[row["id"]] = value["score"]
            route_scores[name + "/" + method] = scores
        model_scores[name] = route_scores[name + "/optimized"]
    fitted = fit_development(dev, model_scores)
    safeguard_path = output / "safeguard_protocol.json"
    safeguard_method = "development_best_baseline_or_optimized_model_per_task"
    safeguard = None
    if safeguard_path.exists():
        if json.loads(safeguard_path.read_text()) != json.loads(
            (ROOT / "anomaly_eval/config/gemma4_safeguard.json").read_text()
        ):
            raise ValueError("The recorded safeguard protocol differs")
        safeguard = fit_development(dev, route_scores)

    def combined_scores(members, method):
        if method == safeguard_method:
            return transform(
                members, route_scores, safeguard, "development_best_model_per_task"
            )
        return transform(members, model_scores, fitted, method)

    identity = {
        "protocol_sha256": digest(protocol_path),
        "code_sha256": digest(Path(__file__)),
        "study_protocol_sha256": digest(output / "protocol.json"),
        "base_results_sha256": {
            name: digest(output / name / "results.json") for name in MODELS
        },
        "uses_test_for_fit": False,
        "safeguard_protocol_sha256": digest(safeguard_path)
        if safeguard is not None
        else None,
    }
    directory = output / "ensembles"
    directory.mkdir(exist_ok=True)
    identity_path = directory / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Frozen ensemble identity changed")
    atomic_json(identity_path, identity)
    atomic_json(
        directory / "selection.json",
        {"primary": fitted, "baseline_safeguard": safeguard},
    )
    methods = json.loads(protocol_path.read_text())["candidates"]
    if safeguard is not None:
        methods.append(safeguard_method)
    cutoffs = {}
    for method in methods:
        scores = combined_scores(dev + source_dev, method)
        cutoffs[method] = {
            task: calibrate([r for r in dev if r["task"] == task], scores)
            for task in TASKS
        }
        cutoffs[method]["xrf55_source_dev"] = calibrate(source_dev, scores)
    atomic_json(directory / "calibration.json", cutoffs)
    # No ensemble held-out score is constructed until all calibrations exist.
    result = {
        "identity": identity,
        "methods": {},
        "heldout_examples": len(heldout),
        "status": "complete",
    }
    flat = []
    for method in methods:
        scores = combined_scores(heldout, method)
        tasks = {}
        for task in TASKS:
            members = [r for r in heldout if r["task"] == task]
            task_metrics = metrics(members, scores, cutoffs[method][task])
            tasks[task] = {"source_metrics": task_metrics}
            for source, points in task_metrics.items():
                for point, values in points.items():
                    flat.append(
                        {
                            "model": "gemma4_ensemble",
                            "method": method,
                            "task": task,
                            "source": source,
                            "operating_point": point,
                            "calibration": "primary_development",
                            **values,
                        }
                    )
        secondary = metrics(
            [r for r in heldout if r["cohort"] == "xrf55"],
            scores,
            cutoffs[method]["xrf55_source_dev"],
        )
        tasks["smoking"]["xrf55_source_recalibrated"] = secondary
        for source, points in secondary.items():
            for point, values in points.items():
                flat.append(
                    {
                        "model": "gemma4_ensemble",
                        "method": method,
                        "task": "smoking",
                        "source": source,
                        "operating_point": point,
                        "calibration": "xrf55_source_development",
                        **values,
                    }
                )
        (directory / (method + "_predictions.jsonl")).write_text(
            "".join(
                json.dumps(
                    {
                        **{
                            k: r[k]
                            for k in [
                                "id",
                                "task",
                                "source",
                                "split",
                                "cohort",
                                "target",
                                "group_id",
                            ]
                        },
                        "score": scores[r["id"]],
                    }
                )
                + "\n"
                for r in heldout
            )
        )
        result["methods"][method] = tasks
    atomic_json(directory / "results.json", result)
    write_csv(output / "ensemble_metrics.csv", flat)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    evaluate(args.output)


if __name__ == "__main__":
    main()
