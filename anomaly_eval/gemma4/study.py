"""Development-only optimization and complete matched Gemma 4 evaluation."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from anomaly_eval.datasets.transfer import atomic_json, digest
from anomaly_eval.gemma4.protocol import (
    BASELINE,
    DEFAULT_OUTPUT,
    MODELS,
    POLICIES,
    ROOT,
    TASKS,
    choose_routes,
    prepare,
    read_rows,
)
from anomaly_eval.training.clip import binary_metrics, select_points


def every_source_cutoff(rows, scores):
    """One global threshold satisfying the empirical budget in every dev source."""
    if any(r["split"] != "dev" for r in rows):
        raise ValueError("Threshold calibration requires development data")
    values = np.asarray([scores[r["id"]] for r in rows], dtype=float)
    best = None
    for cutoff in np.r_[np.nextafter(values.max(), np.inf), np.unique(values)]:
        recalls = []
        fprs = []
        for source in sorted({r["source"] for r in rows}):
            subset = [i for i, r in enumerate(rows) if r["source"] == source]
            y = np.asarray([rows[i]["target"] for i in subset])
            pred = values[subset] >= cutoff
            if not (y == 0).any() or not (y == 1).any():
                continue
            recalls.append(float(pred[y == 1].mean()))
            fprs.append(float(pred[y == 0].mean()))
        if not recalls or max(fprs) > 0.05:
            continue
        key = (min(recalls), float(np.mean(recalls)), -max(fprs), float(cutoff))
        if best is None or key > best[0]:
            best = (key, float(cutoff))
    if best is None:
        raise ValueError("Unable to construct source-budget threshold")
    return best[1]


def calibrate(rows, scores):
    if any(r["split"] != "dev" for r in rows):
        raise ValueError("Threshold calibration rejects held-out labels")
    points = select_points([r["target"] for r in rows], [scores[r["id"]] for r in rows])
    points["dev_every_source_fpr_budget"] = every_source_cutoff(rows, scores)
    return points


def metrics(rows, scores, points):
    result = {}
    for source in sorted({r["source"] for r in rows}):
        subset = [r for r in rows if r["source"] == source]
        result[source] = {}
        for point, cutoff in points.items():
            value = binary_metrics(
                [r["target"] for r in subset], [scores[r["id"]] for r in subset], cutoff
            )
            value.update(
                threshold=cutoff,
                groups=len({r["group_id"] for r in subset}),
                accuracy=(value["tp"] + value["tn"]) / value["n"],
            )
            value["balanced_accuracy"] = (
                (value["recall"] + 1 - value["fpr"]) / 2
                if value["positive"] and value["negative"]
                else None
            )
            value["prevalence_ap_reference"] = value["positive"] / value["n"]
            result[source][point] = value
    return result


class Scores:
    """Append-only scores with strict model/manifest identity and safe resume."""

    def __init__(self, directory, manifest_sha, settings):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        identity = {"manifest_sha256": manifest_sha, "settings": settings}
        path = self.directory / "identity.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError(
                "Refuse to reuse scores after model, code or input changes"
            )
        atomic_json(path, identity)
        self.memory = {}

    def load(self, policy):
        if policy.name not in self.memory:
            values = {}
            path = self.directory / (policy.name + ".jsonl")
            if path.exists():
                for line in path.read_text().splitlines():
                    record = json.loads(line)
                    if record["id"] in values or not math.isfinite(record["score"]):
                        raise ValueError("Duplicate/nonfinite cached prediction")
                    values[record["id"]] = record
            self.memory[policy.name] = values
        return self.memory[policy.name]

    def score(self, rows, policy, backend, progress):
        values = self.load(policy)
        for row in rows:
            if row["id"] in values and any(
                values[row["id"]][key] != row[key]
                for key in ["task", "source", "split", "target", "cohort"]
            ):
                raise ValueError(
                    "Cached prediction identity differs from the current record"
                )
        pending = [r for r in rows if r["id"] not in values]
        started = time.monotonic()
        with (self.directory / (policy.name + ".jsonl")).open("a") as stream:
            for start in range(0, len(pending), backend.batch_size):
                batch = pending[start : start + backend.batch_size]
                predictions = backend.score(batch, policy)
                if len(predictions) != len(batch):
                    raise ValueError("Model dropped an input")
                for row, prediction in zip(batch, predictions):
                    if not math.isfinite(prediction["score"]):
                        raise ValueError("Nonfinite prediction")
                    record = {
                        key: row[key]
                        for key in [
                            "id",
                            "task",
                            "source",
                            "split",
                            "target",
                            "cohort",
                            "group_id",
                        ]
                    }
                    record.update(prediction, policy=policy.name)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    values[row["id"]] = record
                stream.flush()
                completed = min(start + len(batch), len(pending))
                if completed == len(pending) or start % (backend.batch_size * 10) == 0:
                    atomic_json(
                        self.directory.parent / "progress.json",
                        {
                            "stage": progress,
                            "policy": policy.name,
                            "completed_new": completed,
                            "planned_new": len(pending),
                            "seconds": round(time.monotonic() - started, 2),
                        },
                    )
                    print(
                        f"{progress} {policy.name}: {completed}/{len(pending)} new cases",
                        flush=True,
                    )
        return {row["id"]: values[row["id"]]["score"] for row in rows}


def run(name, output, backend, settings):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    if digest(output / "samples.jsonl") != protocol["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    if (
        digest(ROOT / "anomaly_eval/config/anomalies.json")
        != protocol["definitions_sha256"]
    ):
        raise ValueError("Frozen definitions changed")
    if digest(ROOT / "anomaly_eval/gemma4/protocol.py") != protocol["code_sha256"]:
        raise ValueError("Frozen prompt/selection code changed")
    directory = output / name
    directory.mkdir(exist_ok=True)
    if (directory / "results.json").exists():
        print(name, "already complete", flush=True)
        return json.loads((directory / "results.json").read_text())
    rows = read_rows(output / "samples.jsonl")
    dev = [r for r in rows if r["cohort"] == "main" and r["split"] == "dev"]
    screen = [r for r in dev if r["id"] in set(protocol["development_screen_ids"])]
    heldout = [r for r in rows if r["split"] in {"test", "external"}]
    xrf_dev = [r for r in rows if r["cohort"] == "xrf55" and r["split"] == "dev"]
    settings = {
        **settings,
        "backend": backend.metadata,
        "backends_sha256": digest(Path(inspect.getfile(type(backend)))),
        "image_helpers_sha256": digest(ROOT / "anomaly_eval/gemma4/backends.py"),
        "study_sha256": digest(Path(__file__)),
        "protocol_sha256": digest(output / "protocol.json"),
    }
    cache = Scores(directory / "scores", protocol["manifest_sha256"], settings)
    snapshot = directory / "implementation"
    snapshot.mkdir(exist_ok=True)
    for source in [
        Path(__file__),
        ROOT / "anomaly_eval/gemma4/protocol.py",
        ROOT / "anomaly_eval/gemma4/backends.py",
        Path(inspect.getfile(type(backend))),
    ]:
        target = snapshot / source.name
        if target.exists() and digest(target) != digest(source):
            raise ValueError("Frozen implementation snapshot differs")
        if not target.exists():
            shutil.copyfile(source, target)
    atomic_json(directory / "runtime.json", settings)
    if not (directory / "batch_verification.json").exists():
        chosen = []
        for task in TASKS:
            chosen.append(next(r for r in dev if r["task"] == task))
        verification = backend.verify_batch(chosen[:3], BASELINE)
        atomic_json(directory / "batch_verification.json", verification)
    predictions = {}
    for policy in POLICIES:
        predictions[policy.name] = cache.score(
            screen, policy, backend, "development screen"
        )
    selection = choose_routes(screen, predictions)
    selection["screen_ids_sha256"] = digest(output / "protocol.json")
    selection["uses_test"] = False
    selection_path = directory / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Frozen development selection changed")
    atomic_json(selection_path, selection)
    policies = {p.name: p for p in POLICIES}
    results = {
        "model": name,
        "checkpoint": MODELS[name],
        "runtime": settings,
        "development_selection": selection,
        "calibration_uses_test": False,
        "routes": {},
    }
    score_sets = {}
    calibration = {}
    for method in ["baseline", "optimized"]:
        score_sets[method] = {}
        calibration[method] = {}
        for task in TASKS:
            members = [r for r in dev if r["task"] == task]
            policy = (
                BASELINE
                if method == "baseline"
                else policies[selection["routes"][task]]
            )
            scores = cache.score(members, policy, backend, method + " full development")
            score_sets[method].update(scores)
            calibration[method][task] = calibrate(members, scores)
        xrf_policy = (
            BASELINE
            if method == "baseline"
            else policies[selection["routes"]["smoking"]]
        )
        xrf_scores = cache.score(
            xrf_dev, xrf_policy, backend, method + " source development"
        )
        calibration[method]["xrf55_source_dev"] = calibrate(xrf_dev, xrf_scores)
    calibration_path = directory / "calibration.json"
    if (
        calibration_path.exists()
        and json.loads(calibration_path.read_text()) != calibration
    ):
        raise ValueError("Frozen calibration changed")
    atomic_json(calibration_path, calibration)
    # No held-out forward pass occurs until BOTH methods' routes and cutoffs exist.
    for method in ["baseline", "optimized"]:
        tasks = {}
        all_scores = {}
        for task in TASKS:
            members = [r for r in heldout if r["task"] == task]
            policy = (
                BASELINE
                if method == "baseline"
                else policies[selection["routes"][task]]
            )
            scores = cache.score(members, policy, backend, method + " held out")
            all_scores.update(scores)
            tasks[task] = {
                "policy": policy.name,
                "source_metrics": metrics(members, scores, calibration[method][task]),
            }
        source_test = [r for r in heldout if r["cohort"] == "xrf55"]
        tasks["smoking"]["xrf55_source_recalibrated"] = metrics(
            source_test, all_scores, calibration[method]["xrf55_source_dev"]
        )
        prediction_file = directory / (method + "_heldout_predictions.jsonl")
        prediction_file.write_text(
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
                        "score": all_scores[r["id"]],
                    }
                )
                + "\n"
                for r in heldout
            )
        )
        results["routes"][method] = {
            "tasks": tasks,
            "heldout_examples": len(heldout),
            "predictions_sha256": digest(prediction_file),
        }
    results.update(
        status="complete",
        expected_unique_cases=len(rows),
        heldout_examples=len(heldout),
        source_development_cases=len(xrf_dev),
        primary_development_cases=len(dev),
        caveat="Retrospective source-held-out classifier/clip tests; AP and frame FPR are not event recall or false alerts per hour.",
    )
    atomic_json(directory / "results.json", results)
    atomic_json(
        directory / "progress.json",
        {"stage": "complete", "unique_cases": len(rows), "heldout_cases": len(heldout)},
    )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--server-url")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.prepare:
        print(json.dumps(prepare(args.output), indent=2))
        return 0
    if not args.model:
        parser.error("--model is required")
    from anomaly_eval.gemma4.backends import TransformersBackend
    from anomaly_eval.gemma4.llama_backend import (
        LlamaServerBackend,
        ManagedLlamaBackend,
    )

    checkpoint_json = args.output / args.model / "checkpoint.json"
    if not checkpoint_json.exists():
        parser.error(
            "Fetch and verify the pinned model first with anomaly_eval/gemma4/fetch.py"
        )
    verified = json.loads(checkpoint_json.read_text())
    if (
        args.checkpoint
        and Path(verified["path"]).resolve() != args.checkpoint.resolve()
    ):
        parser.error("Checkpoint path differs from the verified model identity")

    if args.server_url:
        backend = LlamaServerBackend(args.server_url, batch_size=args.batch_size)
    elif args.checkpoint:
        backend = (
            ManagedLlamaBackend(
                args.checkpoint,
                args.model,
                args.output / args.model,
                batch_size=args.batch_size,
            )
            if verified["backend"] == "llama.cpp"
            else TransformersBackend(args.checkpoint, batch_size=args.batch_size)
        )
    else:
        parser.error("Supply --checkpoint or --server-url")
    settings = {
        "model": args.model,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "server_url": args.server_url,
        "checkpoint_verification_sha256": digest(checkpoint_json),
        "checkpoint_identity": {
            k: verified[k] for k in ["repo", "revision", "precision", "verified_files"]
        },
    }
    try:
        run(args.model, args.output, backend, settings)
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
