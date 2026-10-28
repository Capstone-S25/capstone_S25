"""Replay Gemma development decisions and every published held-out metric."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from anomaly_eval.datasets.transfer import atomic_json, digest
from anomaly_eval.gemma4.protocol import (
    BASELINE,
    DEFAULT_OUTPUT,
    MODELS,
    POLICIES,
    ROOT,
    TASKS,
    choose_routes,
    read_rows,
)
from anomaly_eval.gemma4.study import calibrate, metrics


def equal(actual, expected, context):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("Audit key mismatch: " + context)
        for key in expected:
            equal(actual[key], expected[key], context + "/" + key)
    elif isinstance(expected, float):
        if not math.isfinite(actual) or not math.isclose(
            actual, expected, abs_tol=1e-9, rel_tol=1e-9
        ):
            raise ValueError("Audit numeric mismatch: " + context)
    elif actual != expected:
        raise ValueError("Audit value mismatch: " + context)


def verify_model(output, name, rows, protocol):
    directory = output / name
    results = json.loads((directory / "results.json").read_text())
    settings = json.loads((directory / "runtime.json").read_text())
    identity = json.loads((directory / "scores/identity.json").read_text())
    equal(
        identity,
        {"manifest_sha256": protocol["manifest_sha256"], "settings": settings},
        name + " identity",
    )
    equal(results["runtime"], settings, name + " runtime")
    for filename, key in [
        ("study.py", "study_sha256"),
        ("backends.py", "image_helpers_sha256"),
    ]:
        equal(
            digest(directory / "implementation" / filename),
            settings[key],
            name + " implementation/" + filename,
        )
    class_file = (
        "llama_backend.py"
        if settings["backend"]["backend"] == "llama.cpp"
        else "backends.py"
    )
    equal(
        digest(directory / "implementation" / class_file),
        settings["backends_sha256"],
        name + " backend",
    )
    equal(
        digest(directory / "checkpoint.json"),
        settings["checkpoint_verification_sha256"],
        name + " checkpoint receipt",
    )
    equal(
        digest(output / "protocol.json"),
        settings["protocol_sha256"],
        name + " protocol",
    )
    all_rows = {r["id"]: r for r in rows}
    caches = {}
    for policy in POLICIES:
        values = read_rows(directory / "scores" / (policy.name + ".jsonl"))
        if len(values) != len({r["id"] for r in values}):
            raise ValueError("Duplicate score ID")
        for prediction in values:
            sample = all_rows[prediction["id"]]
            for key in [
                "id",
                "task",
                "source",
                "group_id",
                "split",
                "cohort",
                "target",
            ]:
                equal(prediction[key], sample[key], name + " score " + key)
            if not math.isfinite(prediction["score"]):
                raise ValueError("Nonfinite margin")
            mass = prediction.get("answer_token_mass")
            if mass is not None and (
                not math.isfinite(mass) or not -1e-6 <= mass <= 1.0001
            ):
                raise ValueError("Invalid native answer mass")
            if sample["split"] not in {"dev", "test", "external"}:
                raise ValueError("Training input entered inference benchmark")
            equal(prediction["policy"], policy.name, name + " policy")
        caches[policy.name] = {r["id"]: r["score"] for r in values}
    dev = [r for r in rows if r["cohort"] == "main" and r["split"] == "dev"]
    screen = [r for r in dev if r["id"] in set(protocol["development_screen_ids"])]
    heldout = [r for r in rows if r["split"] in {"test", "external"}]
    source_dev = [r for r in rows if r["cohort"] == "xrf55" and r["split"] == "dev"]
    selected = choose_routes(screen, caches)
    saved_selection = json.loads((directory / "selection.json").read_text())
    for key, value in selected.items():
        equal(saved_selection[key], value, name + " selection/" + key)
    saved_calibration = json.loads((directory / "calibration.json").read_text())
    for method in ["baseline", "optimized"]:
        task_policies = {
            task: BASELINE.name if method == "baseline" else selected["routes"][task]
            for task in TASKS
        }
        score = {r["id"]: caches[task_policies[r["task"]]][r["id"]] for r in rows}
        calibration = {
            task: calibrate([r for r in dev if r["task"] == task], score)
            for task in TASKS
        }
        calibration["xrf55_source_dev"] = calibrate(source_dev, score)
        equal(
            saved_calibration[method], calibration, name + " " + method + " calibration"
        )
        for task in TASKS:
            expected = {
                "policy": task_policies[task],
                "source_metrics": metrics(
                    [r for r in heldout if r["task"] == task], score, calibration[task]
                ),
            }
            if task == "smoking":
                expected["xrf55_source_recalibrated"] = metrics(
                    [r for r in heldout if r["cohort"] == "xrf55"],
                    score,
                    calibration["xrf55_source_dev"],
                )
            equal(
                results["routes"][method]["tasks"][task],
                expected,
                name + " " + method + " " + task,
            )
        prediction_file = directory / (method + "_heldout_predictions.jsonl")
        equal(
            digest(prediction_file),
            results["routes"][method]["predictions_sha256"],
            name + " predictions hash",
        )
        predictions = read_rows(prediction_file)
        equal(len(predictions), len(heldout), name + " coverage")
        for actual, row in zip(predictions, heldout):
            expected = {
                key: row[key]
                for key in [
                    "id",
                    "task",
                    "source",
                    "group_id",
                    "split",
                    "target",
                    "cohort",
                ]
            }
            expected["score"] = score[row["id"]]
            equal(actual, expected, name + " heldout " + row["id"])
    return {
        "status": "passed",
        "heldout_cases": len(heldout),
        "development_cases": len(dev),
        "source_development_cases": len(source_dev),
        "cached_forwards": sum(len(c) for c in caches.values()),
        "result_sha256": digest(directory / "results.json"),
        "replayed": [
            "prompt routes",
            "both method calibrations",
            "all heldout scores",
            "every source/operating metric",
            "secondary source calibration",
            "implementation and checkpoint receipts",
        ],
    }


def verify(output):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    equal(digest(output / "samples.jsonl"), protocol["manifest_sha256"], "manifest")
    equal(
        digest(ROOT / "anomaly_eval/config/anomalies.json"),
        protocol["definitions_sha256"],
        "definitions",
    )
    for path, sha in protocol["sources"].items():
        equal(digest(ROOT / path), sha, "source manifest " + path)
    rows = read_rows(output / "samples.jsonl")
    results = {
        name: verify_model(output, name, rows, protocol)
        for name in MODELS
        if (output / name / "results.json").exists()
    }
    receipt = {
        "models": results,
        "status": "passed" if results else "pending",
        "models_complete": len(results),
        "planned_models": len(MODELS),
        "does_not_validate": "Weak labels, independent cameras, continuous event performance or deployment suitability",
    }
    atomic_json(output / "verification.json", receipt)
    return receipt


def historical_cohorts(output):
    """Validate ID/label/source/group equality, beyond equal row denominators."""
    output = Path(output)
    expected = {
        r["id"]: r
        for r in read_rows(output / "samples.jsonl")
        if r["split"] in {"test", "external"} and r["cohort"] != "xrf55"
    }
    families = {}
    for name in ["mobileclip2-s2", "siglip2-b16-224"]:
        directory = (
            ROOT / "results/strategy_study_20261016" / name / "transfer_evaluation"
        )
        files = sorted(directory.glob("*_predictions.jsonl"))
        families[name] = (files, "selected")
    for variant in ["original", "seed_42__lr_0.0001", "seed_42__lr_3e-05"]:
        files = sorted(
            (ROOT / "results/smolvlm_finetuning_20261019/evaluation" / variant).glob(
                "*_predictions.jsonl"
            )
        )
        files += sorted(
            (
                ROOT
                / "results/smolvlm_finetuning_20261019/additional_transfer/evaluation"
                / variant
            ).glob("*_predictions.jsonl")
        )
        families["smolvlm-500m/" + variant] = (files, None)
    verified = {}
    for family, (files, selected) in families.items():
        prior = {}
        for path in files:
            for row in read_rows(path):
                if selected is not None and row.get("family") != selected:
                    continue
                if row["id"] in prior:
                    raise ValueError("Duplicate prior prediction identity")
                prior[row["id"]] = row
        equal(set(prior), set(expected), family + " matched IDs")
        role_aliases = Counter()
        for identifier, row in prior.items():
            if row["split"] not in {"test", "external"}:
                raise ValueError("Prior comparison is not held out")
            if row["split"] != expected[identifier]["split"]:
                role_aliases[row["split"] + "->" + expected[identifier]["split"]] += 1
            for key in ["task", "source", "group_id", "target"]:
                equal(row[key], expected[identifier][key], family + " matched " + key)
        verified[family] = {
            "status": "passed",
            "matched_heldout_cases": len(prior),
            "heldout_partition_name_aliases": dict(role_aliases),
            "files_sha256": {
                str(path.relative_to(ROOT)): digest(path) for path in files
            },
        }
    receipt = {
        "status": "passed",
        "matched_cases": len(expected),
        "families": verified,
        "qualification": "Matching IDs and annotations does not make crops, precision, token budgets, supervision or view counts identical. XRF55 comparisons are separate.",
    }
    atomic_json(output / "historical_cohort_verification.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.output), indent=2))


if __name__ == "__main__":
    main()
