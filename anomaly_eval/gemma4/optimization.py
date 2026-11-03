"""Second-stage Gemma input/prompt experiments with development-only decisions.

The previous study remains immutable. Its margins are imported only after a
development numerical parity check. New routes and thresholds are frozen before
this stage constructs held-out predictions. Existing holdouts are retrospective.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from anomaly_eval.datasets.transfer import atomic_json, digest
from anomaly_eval.gemma4.backends import _image, images_for
from anomaly_eval.gemma4.protocol import (
    BASELINE,
    DEFAULT_OUTPUT,
    MODELS,
    POLICIES,
    ROOT,
    TASKS,
    Policy,
    question,
    read_rows,
    source_ap,
    view_indices,
)
from anomaly_eval.gemma4.report import macro_ap, table, write_csv
from anomaly_eval.gemma4.study import Scores, calibrate, metrics

OUTPUT = ROOT / "results/gemma4_optimization_20261028"
NEW_POLICIES = (
    Policy("detail_native_1120", "evidence", 1120),
    Policy("contrastive_native_560", "evidence", 560),
    Policy("contextual_crops_560", "evidence", 560, "native", 4),
)
ALL_POLICIES = {p.name: p for p in (*POLICIES, *NEW_POLICIES)}
DETECTIONS = [
    ROOT / "data/strategy_study_20261016/detections.jsonl",
    ROOT / "data/strategy_study_20261016/rough_detections.jsonl",
    ROOT / "data/generalization_pilot_20261020/xrf55/detections.jsonl",
]
EXPLANATIONS = {
    "smoking": "Positive: a cigarette/cigar/pipe used near the mouth or visible smoking plume. Alternative: eating, drinking, phoning or an ambiguous hand near the face.",
    "littering": "Positive: the ordered observations show a person discarding rubbish onto the ground. Alternative: carrying, retrieving, using a bin, benign placement, or waste already present.",
    "loitering": "Positive: the same person stays in the area across observations without apparent activity. Alternative: purposeful movement, a brief pause, or an unestablished identity/duration.",
    "rough_sleeping": "Positive: apparent sleep or a lying person with bedding in a public area. Alternative: awake sitting/resting, indoor sleep, or unoccupied bedding. Housing status is never inferred.",
    "unattended_items": "Positive: an item persists while its associated person moves away or is absent across observations. Alternative: attended placement, carrying, retrieval, or an isolated item with unknown owner history.",
}


class Inputs:
    def __init__(self, detections):
        self.detections = detections

    def images(self, row, policy):
        if policy.name != "contextual_crops_560":
            return images_for(row, policy)
        result = []
        for i in view_indices(row, policy.max_views):
            path = row["views"][i]
            image = _image(path).copy()
            result.append(image)
            found = self.detections[path]
            persons = sorted(
                (d for d in found if d["label"] == 1 and d["confidence"] >= 0.5),
                key=lambda d: -d["confidence"],
            )
            objects = sorted(
                (
                    d
                    for d in found
                    if d["label"] in {27, 31, 33} and d["confidence"] >= 0.5
                ),
                key=lambda d: -d["confidence"],
            )
            boxes = [d["box"][:] for d in persons[:1]]
            if row["task"] == "smoking" and boxes:
                boxes[0][3] = boxes[0][1] + 0.65 * (boxes[0][3] - boxes[0][1])
            elif objects:
                boxes.append(objects[0]["box"][:])
            if not boxes:
                continue
            x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            dx, dy = (x2 - x1) * 0.25, (y2 - y1) * 0.25
            box = (max(0, x1 - dx), max(0, y1 - dy), min(1, x2 + dx), min(1, y2 + dy))
            result.append(
                image.crop(
                    tuple(v * s for v, s in zip(box, [image.width, image.height] * 2))
                )
            )
        return result

    def prompt(self, row, policy):
        text = question(row, policy)
        if policy.name == "contrastive_native_560":
            text = text.replace(
                "Does this evidence match the target? Answer Yes or No.",
                EXPLANATIONS[row["task"]]
                + " Compare both explanations using the actual visible evidence. Does the positive explanation satisfy the definition? Answer exactly Yes or No.",
            )
        if policy.name == "contextual_crops_560":
            text = (
                "Each sampled moment is shown as its full scene followed, when detected, by one enlarged person/object detail from that SAME moment. Detail crops are not later frames or independent events. Use full scenes for context and time; crops can reveal small objects but cannot establish an absent owner. "
                + text
            )
        return text

    def messages(self, row, policy):
        images = self.images(row, policy)
        return [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images]
                + [{"type": "text", "text": self.prompt(row, policy)}],
            }
        ], images


def load_detections(rows):
    needed = {p for r in rows for p in r["views"]}
    found = {}
    for path in DETECTIONS:
        for row in read_rows(path):
            if row["path"] not in needed:
                continue
            for d in row["detections"]:
                box = d["box"]
                if len(box) != 4 or not all(
                    math.isfinite(v) and -1e-6 <= v <= 1 + 1e-6 for v in box
                ):
                    raise ValueError("Invalid normalized automatic detection")
                d["box"] = box = [min(1.0, max(0.0, v)) for v in box]
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError("Degenerate automatic detection")
            if row["path"] in found and found[row["path"]] != row["detections"]:
                raise ValueError("Conflicting detection caches")
            found[row["path"]] = row["detections"]
    if needed - found.keys():
        raise ValueError("Automatic detection cache is incomplete")
    return found


def prepare(previous, output):
    output.mkdir(parents=True, exist_ok=True)
    source = previous / "samples.jsonl"
    original = json.loads((previous / "protocol.json").read_text())
    if digest(source) != original["manifest_sha256"]:
        raise ValueError("Previous frozen manifest changed")
    protocol = {
        "previous": str(previous),
        "manifest_sha256": digest(source),
        "definitions_sha256": digest(ROOT / "anomaly_eval/config/anomalies.json"),
        "code_sha256": digest(Path(__file__)),
        "previous_results_sha256": {
            n: digest(previous / n / "results.json") for n in MODELS
        },
        "previous_routes_sha256": {
            n: digest(previous / n / "selection.json") for n in MODELS
        },
        "checkpoint_receipt_sha256": {
            n: digest(previous / n / "checkpoint.json") for n in MODELS
        },
        "scorer_sha256": {
            n: digest(ROOT / "anomaly_eval/gemma4" / n)
            for n in ["backends.py", "llama_backend.py", "protocol.py", "study.py"]
        },
        "new_policies": [asdict(p) for p in NEW_POLICIES],
        "detection_sha256": {str(p.relative_to(ROOT)): digest(p) for p in DETECTIONS},
        "detector_bounds": "Accept <=1e-6 floating-point boundary roundoff and clamp it to [0,1]; reject larger excursions, nonfinite or degenerate boxes.",
        "screen_ids": original["development_screen_ids"],
        "selection": "Screen three new policies on primary development only. Per task, fully score the new screen winner, baseline and supported prior optimized route on ALL primary development. Select full-dev mean-source AP; also test a source safeguard and an equal baseline/candidate development-z-score blend.",
        "safeguard": "Candidate mean-source AP must improve baseline by >=0.005 and every dev source AP must remain within 0.01 of baseline. Otherwise baseline. Ties prefer baseline, prior route, then new candidate.",
        "layout_guard": "Exclude sheet candidates for tasks with no multi-image primary development; native high-detail remains eligible.",
        "calibration": "Freeze all routes and primary/secondary development cutoffs before any held-out scoring in this stage.",
        "retrospective": True,
        "uses_test_for_fit": False,
        "precision": "Unchanged pinned model/precision and canonical one-example GPU forwards; no pacing or latency study",
    }
    frozen = output / "protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise ValueError(
                "Optimization protocol changed; use a new output directory"
            )
    else:
        atomic_json(frozen, protocol)
        atomic_json(
            output / "declaration.json",
            {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "prior_holdout_results_already_reviewed": True,
            },
        )
        shutil.copyfile(source, output / "samples.jsonl")
    if digest(output / "samples.jsonl") != protocol["manifest_sha256"]:
        raise ValueError("Optimization manifest changed")
    return protocol


def eligible(policy, task, dev):
    return policy.layout != "sheet" or any(
        r["task"] == task and len(r["views"]) > 1 for r in dev
    )


def validate_pixels(rows, output):
    from anomaly_eval.data import image_pixel_sha256

    expected = {}
    for row in rows:
        if len(row["views"]) != len(row["pixel_hashes"]):
            raise ValueError("Unaligned input pixel identities")
        for path, sha in zip(row["views"], row["pixel_hashes"]):
            if path in expected and expected[path] != sha:
                raise ValueError("Conflicting frozen pixel identities")
            expected[path] = sha

    def check(path):
        if image_pixel_sha256(ROOT / path) != expected[path]:
            raise ValueError("Frozen input image changed: " + path)

    with ThreadPoolExecutor(12) as pool:
        for _ in pool.map(check, expected):
            pass
    atomic_json(
        output / "input_verification.json",
        {
            "status": "passed",
            "distinct_images": len(expected),
            "cases": len(rows),
            "check": "All referenced decoded RGB pixel hashes match the frozen manifest",
        },
    )


def fit(dev, predictions, finalists):
    if any(r["split"] != "dev" or r["cohort"] != "main" for r in dev):
        raise ValueError("Selection requires primary development only")
    fitted = {"full_dev_selected": {}, "source_safeguarded": {}, "baseline_blend": {}}
    evidence = {}
    for task in TASKS:
        members = [r for r in dev if r["task"] == task]
        names = finalists[task]
        performance = {n: source_ap(members, predictions[n], task) for n in names}
        best = max(names, key=lambda n: (performance[n], -names.index(n)))
        fitted["full_dev_selected"][task] = best
        by_source = {
            n: {
                s: source_ap(
                    [r for r in members if r["source"] == s], predictions[n], task
                )
                for s in sorted({r["source"] for r in members})
            }
            for n in names
        }
        accepted = [BASELINE.name] + [
            n
            for n in names
            if n != BASELINE.name
            and performance[n] >= performance[BASELINE.name] + 0.005
            and all(
                by_source[n][s] >= value - 0.01
                for s, value in by_source[BASELINE.name].items()
            )
        ]
        fitted["source_safeguarded"][task] = max(
            accepted, key=lambda n: (performance[n], -names.index(n))
        )
        normalization = {}
        for n in dict.fromkeys([BASELINE.name, best]):
            values = [predictions[n][r["id"]] for r in members]
            normalization[n] = {
                "mean": float(np.mean(values)),
                "std": max(float(np.std(values)), 1e-8),
            }
        fitted["baseline_blend"][task] = normalization
        evidence[task] = {"mean_source_ap": performance, "source_ap": by_source}
    return {"routes": fitted, "development": evidence, "uses_test_for_fit": False}


def transform(rows, predictions, fitted, method):
    result = {}
    for r in rows:
        route = fitted["routes"][method][r["task"]]
        result[r["id"]] = (
            float(
                np.mean(
                    [
                        (predictions[n][r["id"]] - v["mean"]) / v["std"]
                        for n, v in route.items()
                    ]
                )
            )
            if method == "baseline_blend"
            else predictions[route][r["id"]]
        )
    return result


def make_backend(spec, name, directory, inputs):
    from anomaly_eval.gemma4.backends import TransformersBackend
    from anomaly_eval.gemma4.llama_backend import ManagedLlamaBackend

    backend = (
        ManagedLlamaBackend(spec["path"], name, directory, batch_size=8)
        if spec["backend"] == "llama.cpp"
        else TransformersBackend(spec["path"], batch_size=8)
    )
    backend.messages = inputs.messages
    backend.images_for = inputs.images
    backend.question = inputs.prompt
    return backend


def run_model(previous, output, name, protocol, rows, inputs):
    directory = output / name
    directory.mkdir(exist_ok=True)
    if (directory / "results.json").exists():
        return json.loads((directory / "results.json").read_text())
    dev = [r for r in rows if r["split"] == "dev" and r["cohort"] == "main"]
    secondary = [r for r in rows if r["split"] == "dev" and r["cohort"] == "xrf55"]
    heldout = [r for r in rows if r["split"] in {"test", "external"}]
    old = previous / name
    spec = json.loads((old / "checkpoint.json").read_text())
    if not Path(spec["path"]).is_dir():
        raise FileNotFoundError("Checkpoint unavailable: " + spec["path"])
    for expected in spec.get("verified_files", []):
        path = Path(spec["path"]) / expected["path"]
        if (
            path.stat().st_size != expected["bytes"]
            or digest(path) != expected["sha256"]
        ):
            raise ValueError("Pinned checkpoint bytes changed: " + name)
    backend = make_backend(spec, name, directory, inputs)
    try:
        settings = {
            "model": name,
            "backend": backend.metadata,
            "protocol_sha256": digest(output / "protocol.json"),
            "checkpoint_sha256": digest(old / "checkpoint.json"),
            "inherited_scores_sha256": {
                p.name: digest(old / "scores" / (p.name + ".jsonl")) for p in POLICIES
            },
        }
        cache = Scores(directory / "scores", protocol["manifest_sha256"], settings)
        atomic_json(directory / "settings.json", settings)
        snapshot = directory / "implementation"
        snapshot.mkdir(exist_ok=True)
        for filename in [
            "optimization.py",
            "backends.py",
            "llama_backend.py",
            "protocol.py",
            "study.py",
        ]:
            source = ROOT / "anomaly_eval/gemma4" / filename
            target = snapshot / filename
            if target.exists() and digest(target) != digest(source):
                raise ValueError("Frozen optimization implementation changed")
            if not target.exists():
                shutil.copyfile(source, target)
        # All reused margins must match the current numerical path on development.
        if not (directory / "inheritance.json").exists():
            parity = []
            for policy in POLICIES:
                records = read_rows(old / "scores" / (policy.name + ".jsonl"))
                available = {r["id"]: r for r in records}
                probe = [
                    max(
                        (r for r in dev if r["task"] == task and r["id"] in available),
                        key=lambda r: (len(r["views"]), r["id"]),
                    )
                    for task in TASKS
                ]
                scores = backend.score(probe, policy)
                error = max(
                    abs(x["score"] - available[r["id"]]["score"])
                    for x, r in zip(scores, probe)
                )
                if error > 0.02:
                    raise ValueError(
                        f"Inherited {policy.name} margin drift {error:g}; no reuse permitted"
                    )
                parity.append(
                    {
                        "policy": policy.name,
                        "max_error": error,
                        "ids": [r["id"] for r in probe],
                    }
                )
                target = directory / "scores" / (policy.name + ".jsonl")
                if not target.exists():
                    shutil.copyfile(old / "scores" / (policy.name + ".jsonl"), target)
            atomic_json(
                directory / "inheritance.json",
                {"status": "passed", "development_only": True, "parity": parity},
            )
        screen = [r for r in dev if r["id"] in set(protocol["screen_ids"])]
        candidates = {}
        preflight = {}
        for policy in NEW_POLICIES:
            probes = [
                max(
                    (r for r in dev if r["task"] == task),
                    key=lambda r: (min(len(r["views"]), policy.max_views), r["id"]),
                )
                for task in TASKS
            ]
            preflight[policy.name] = backend.verify_batch(probes, policy)
            candidates[policy.name] = cache.score(
                screen, policy, backend, "new-policy development screen"
            )
        atomic_json(
            directory / "preflight.json", {"status": "passed", "policies": preflight}
        )
        prior = json.loads((old / "selection.json").read_text())["routes"]
        finalists = {}
        for task in TASKS:
            new = max(
                NEW_POLICIES,
                key=lambda p: (
                    source_ap(screen, candidates[p.name], task),
                    -NEW_POLICIES.index(p),
                ),
            )
            names = [BASELINE.name, prior[task], new.name]
            finalists[task] = [
                n for n in dict.fromkeys(names) if eligible(ALL_POLICIES[n], task, dev)
            ]
        atomic_json(
            directory / "screen.json",
            {
                "finalists": finalists,
                "uses_test": False,
                "new_policy_source_ap": {
                    t: {
                        p.name: source_ap(screen, candidates[p.name], t)
                        for p in NEW_POLICIES
                    }
                    for t in TASKS
                },
            },
        )
        predictions = {}
        for policy_name in dict.fromkeys(
            n for names in finalists.values() for n in names
        ):
            members = [r for r in dev if policy_name in finalists[r["task"]]]
            predictions[policy_name] = cache.score(
                members,
                ALL_POLICIES[policy_name],
                backend,
                "full development finalists",
            )
        fitted = fit(dev, predictions, finalists)
        if (directory / "selection.json").exists() and json.loads(
            (directory / "selection.json").read_text()
        ) != fitted:
            raise ValueError("Frozen full-development routes changed")
        atomic_json(directory / "selection.json", fitted)
        required = {t: set() for t in TASKS}
        for task in TASKS:
            required[task].update(
                fitted["routes"][m][task]
                for m in ["full_dev_selected", "source_safeguarded"]
            )
            required[task].update(fitted["routes"]["baseline_blend"][task])
        for policy_name in dict.fromkeys(
            n for names in required.values() for n in sorted(names)
        ):
            members = [r for r in secondary if policy_name in required[r["task"]]]
            predictions.setdefault(policy_name, {}).update(
                cache.score(
                    members, ALL_POLICIES[policy_name], backend, "secondary development"
                )
            )
        cutoffs = {}
        for method in fitted["routes"]:
            scores = transform(dev + secondary, predictions, fitted, method)
            cutoffs[method] = {
                t: calibrate([r for r in dev if r["task"] == t], scores) for t in TASKS
            }
            cutoffs[method]["xrf55_source_dev"] = calibrate(secondary, scores)
        if (directory / "calibration.json").exists() and json.loads(
            (directory / "calibration.json").read_text()
        ) != cutoffs:
            raise ValueError("Frozen optimization thresholds changed")
        atomic_json(directory / "calibration.json", cutoffs)
        # No new held-out forward until every method's route and cutoff exists.
        for policy_name in dict.fromkeys(
            n for names in required.values() for n in sorted(names)
        ):
            members = [r for r in heldout if policy_name in required[r["task"]]]
            predictions.setdefault(policy_name, {}).update(
                cache.score(
                    members, ALL_POLICIES[policy_name], backend, "frozen-route held out"
                )
            )
        result = {
            "status": "complete",
            "model": name,
            "methods": {},
            "heldout_examples": len(heldout),
            "protocol_sha256": digest(output / "protocol.json"),
            "retrospective": True,
        }
        for method in fitted["routes"]:
            scores = transform(heldout, predictions, fitted, method)
            tasks = {
                t: metrics(
                    [r for r in heldout if r["task"] == t], scores, cutoffs[method][t]
                )
                for t in TASKS
            }
            secondary_metrics = metrics(
                [r for r in heldout if r["cohort"] == "xrf55"],
                scores,
                cutoffs[method]["xrf55_source_dev"],
            )
            filename = directory / (method + "_predictions.jsonl")
            filename.write_text(
                "".join(
                    json.dumps(
                        {
                            **{
                                k: r[k]
                                for k in [
                                    "id",
                                    "task",
                                    "target",
                                    "source",
                                    "split",
                                    "cohort",
                                    "group_id",
                                ]
                            },
                            "score": scores[r["id"]],
                        },
                        allow_nan=False,
                    )
                    + "\n"
                    for r in heldout
                )
            )
            result["methods"][method] = {
                "tasks": tasks,
                "xrf55_source_recalibrated": secondary_metrics,
                "predictions_sha256": digest(filename),
            }
        atomic_json(directory / "results.json", result)
        atomic_json(
            directory / "progress.json",
            {"stage": "complete", "heldout_cases": len(heldout)},
        )
        return result
    finally:
        backend.close()


def verify_model(previous, output, name):
    """Replay selection, cutoffs and every metric from immutable score margins."""
    from anomaly_eval.gemma4.verify import equal

    protocol = json.loads((output / "protocol.json").read_text())
    rows = read_rows(output / "samples.jsonl")
    if digest(output / "samples.jsonl") != protocol["manifest_sha256"]:
        raise ValueError("Audit manifest mismatch")
    directory = output / name
    for filename in ["results.json", "selection.json", "checkpoint.json"]:
        key = {
            "results.json": "previous_results_sha256",
            "selection.json": "previous_routes_sha256",
            "checkpoint.json": "checkpoint_receipt_sha256",
        }[filename]
        equal(
            digest(previous / name / filename),
            protocol[key][name],
            "previous " + filename,
        )
    settings = json.loads((directory / "settings.json").read_text())
    equal(
        json.loads((directory / "scores/identity.json").read_text()),
        {"manifest_sha256": protocol["manifest_sha256"], "settings": settings},
        "score identity",
    )
    for filename, expected in {
        "optimization.py": protocol["code_sha256"],
        **protocol["scorer_sha256"],
    }.items():
        equal(
            digest(directory / "implementation" / filename),
            expected,
            "implementation " + filename,
        )
    indexed = {r["id"]: r for r in rows}
    predictions = {}
    for p in (directory / "scores").glob("*.jsonl"):
        values = read_rows(p)
        if len(values) != len({v["id"] for v in values}):
            raise ValueError("Duplicate optimization cache ID")
        for record in values:
            row = indexed[record["id"]]
            for k in ["id", "task", "target", "source", "split", "cohort", "group_id"]:
                equal(record[k], row[k], "cached " + k)
            if not math.isfinite(record["score"]):
                raise ValueError("Nonfinite optimization score")
            if record["policy"] != p.stem:
                raise ValueError("Cached optimization policy mismatch")
        predictions[p.stem] = {r["id"]: r["score"] for r in values}
    dev = [r for r in rows if r["cohort"] == "main" and r["split"] == "dev"]
    secondary = [r for r in rows if r["cohort"] == "xrf55" and r["split"] == "dev"]
    heldout = [r for r in rows if r["split"] in {"test", "external"}]
    screen = [r for r in dev if r["id"] in set(protocol["screen_ids"])]
    prior = json.loads((previous / name / "selection.json").read_text())["routes"]
    finalists = {}
    for task in TASKS:
        best = max(
            NEW_POLICIES,
            key=lambda p: (
                source_ap(screen, predictions[p.name], task),
                -NEW_POLICIES.index(p),
            ),
        )
        finalists[task] = [
            n
            for n in dict.fromkeys([BASELINE.name, prior[task], best.name])
            if eligible(ALL_POLICIES[n], task, dev)
        ]
    equal(
        json.loads((directory / "screen.json").read_text())["finalists"],
        finalists,
        "finalists",
    )
    fitted = fit(dev, predictions, finalists)
    equal(
        json.loads((directory / "selection.json").read_text()),
        fitted,
        "full development fit",
    )
    cutoffs = json.loads((directory / "calibration.json").read_text())
    result = json.loads((directory / "results.json").read_text())
    for method in fitted["routes"]:
        scores = transform(dev + secondary, predictions, fitted, method)
        expected = {
            t: calibrate([r for r in dev if r["task"] == t], scores) for t in TASKS
        }
        expected["xrf55_source_dev"] = calibrate(secondary, scores)
        equal(cutoffs[method], expected, method + " cutoffs")
        scores = transform(heldout, predictions, fitted, method)
        filename = directory / (method + "_predictions.jsonl")
        saved = read_rows(filename)
        if len(saved) != len(heldout) or {r["id"] for r in saved} != {
            r["id"] for r in heldout
        }:
            raise ValueError("Missing or duplicate held-out predictions")
        for r in saved:
            equal(
                r,
                {
                    **{
                        k: indexed[r["id"]][k]
                        for k in [
                            "id",
                            "task",
                            "target",
                            "source",
                            "split",
                            "cohort",
                            "group_id",
                        ]
                    },
                    "score": scores[r["id"]],
                },
                "held-out prediction",
            )
        expected = {
            t: metrics(
                [r for r in heldout if r["task"] == t], scores, cutoffs[method][t]
            )
            for t in TASKS
        }
        equal(result["methods"][method]["tasks"], expected, method + " source metrics")
        equal(
            result["methods"][method]["xrf55_source_recalibrated"],
            metrics(
                [r for r in heldout if r["cohort"] == "xrf55"],
                scores,
                cutoffs[method]["xrf55_source_dev"],
            ),
            method + " secondary metrics",
        )
        equal(
            result["methods"][method]["predictions_sha256"],
            digest(filename),
            method + " predictions hash",
        )
    receipt = {
        "status": "passed",
        "heldout_cases": len(heldout),
        "primary_development_cases": len(dev),
        "source_development_cases": len(secondary),
        "result_sha256": digest(directory / "results.json"),
        "replayed": [
            "new-policy finalists",
            "full-development routes and blend constants",
            "all cutoffs",
            "all held-out margins",
            "all source metrics",
        ],
    }
    atomic_json(directory / "verification.json", receipt)
    return receipt


def report(previous, output):
    flat = []
    audit = {}
    for name in MODELS:
        path = output / name / "results.json"
        if not path.exists():
            continue
        audit[name] = verify_model(previous, output, name)
        result = json.loads(path.read_text())
        for method, values in result["methods"].items():
            for task, sources in values["tasks"].items():
                for source, points in sources.items():
                    for point, value in points.items():
                        flat.append(
                            {
                                "model": name,
                                "method": method,
                                "task": task,
                                "source": source,
                                "operating_point": point,
                                "calibration": "primary_development",
                                **value,
                            }
                        )
            for source, points in values["xrf55_source_recalibrated"].items():
                for point, value in points.items():
                    flat.append(
                        {
                            "model": name,
                            "method": method,
                            "task": "smoking",
                            "source": source,
                            "operating_point": point,
                            "calibration": "xrf55_source_development",
                            **value,
                        }
                    )
    if not flat:
        return
    write_csv(output / "metrics.csv", flat)
    oldrows = list(csv.DictReader((previous / "metrics.csv").open()))
    combined = [
        {**r, **{k: float(r[k]) for k in ["ap", "auroc", "fpr", "recall"]}}
        for r in oldrows
        if r["method"] in {"baseline", "optimized"}
    ] + flat
    summary = []
    for name, method in dict.fromkeys((r["model"], r["method"]) for r in combined):
        members = [
            r
            for r in combined
            if r["model"] == name
            and r["method"] == method
            and r["operating_point"] == "raw"
            and r["calibration"] == "primary_development"
        ]
        per_task = macro_ap(members)
        common = macro_ap(
            members, {r["source"] for r in members if r["source"] != "xrf55_rgb_public"}
        )
        summary.append(
            {
                "model": name,
                "method": method,
                "all_source_macro_ap": float(np.mean(list(per_task.values()))),
                "common_source_macro_ap": float(np.mean(list(common.values()))),
                **{"ap_" + t: per_task[t] for t in TASKS},
            }
        )
    write_csv(output / "summary.csv", summary)
    atomic_json(
        output / "verification.json",
        {
            "status": "passed",
            "completed_models": len(audit),
            "planned_models": len(MODELS),
            "models": audit,
        },
    )
    text = (
        "\n\n".join(
            [
                "# Gemma 4 second-stage optimization — 28 October 2026",
                f"Completed variants: {len({r['model'] for r in flat})}/5. Existing baseline and first-stage optimized results are preserved. This is a retrospective study using previously inspected holdouts.",
                "Three new input policies test native 1,120-token detail, explicit competing explanations at 560 tokens, and full-scene plus automatic person/object crops at 560 tokens. Crops use existing label-free Faster R-CNN detections, not annotation boxes. Smoking crops emphasize the upper body. Cropped sequences use four chronological time points, each followed by an optional detail crop; other native routes retain up to eight time points.",
                "The original primary-development screen chooses a new-policy finalist per task. That finalist, baseline and the supported prior route are compared on the entire primary development set. Unsupported sheet layouts are excluded. Three methods compare full-development source-AP selection, a per-source development safeguard, and equal baseline/selected standardized-score blending. All routes, blend constants and four operating-point thresholds are frozen before new held-out inference. XRF development supplies a separate secondary cutoff only.",
                table(
                    [
                        "Model",
                        "Method",
                        "All-source macro AP",
                        "Common-source macro AP",
                        *TASKS,
                    ],
                    [
                        [
                            r["model"],
                            r["method"],
                            f"{r['all_source_macro_ap']:.4f}",
                            f"{r['common_source_macro_ap']:.4f}",
                            *[f"{r['ap_' + t]:.4f}" for t in TASKS],
                        ]
                        for r in summary
                    ],
                ),
                "AP averages sources within each of the five equally weighted objectives. Common-source macro excludes the weak XRF smoking proxy, matching the previous 3,792-case comparison; all-source macro includes all 4,032 held-out cases. Complete AUROC, AP, precision, recall, F1, FPR, accuracy, balanced accuracy and confusion counts are in metrics.csv. A 5% development FPR constraint is not a test guarantee.",
                "Scores are imported from the previous stage only after an exact-path numerical development parity check (maximum allowed margin drift 0.02). Raw caches bind model receipts, prior score hashes, detector hashes, prompts, implementation and the frozen manifest. No Gemma weights are trained. These are classification/clip experiments, not continuous detector/tracker event validation. Reused holdouts, weak labels and limited camera/owner evidence constrain generalization claims.",
            ]
        )
        + "\n"
    )
    (output / "report.md").write_text(text)
    tracked = ROOT / "reports/gemma4_optimization_20261028"
    tracked.mkdir(exist_ok=True)
    for name in [
        "report.md",
        "summary.csv",
        "metrics.csv",
        "protocol.json",
        "declaration.json",
        "verification.json",
    ]:
        shutil.copyfile(output / name, tracked / name)
    for name in MODELS:
        directory = tracked / name
        for file in [
            "selection.json",
            "screen.json",
            "calibration.json",
            "inheritance.json",
            "preflight.json",
            "verification.json",
        ]:
            p = output / name / file
            if p.exists():
                directory.mkdir(exist_ok=True)
                shutil.copyfile(p, directory / file)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        default=[
            "gemma4-12b",
            "gemma4-e2b",
            "gemma4-e4b",
            "gemma4-26b-a4b",
            "gemma4-31b",
        ],
    )
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = prepare(args.previous, args.output)
        if args.report_only:
            report(args.previous, args.output)
            return 0
        rows = read_rows(args.output / "samples.jsonl")
        validate_pixels(rows, args.output)
        inputs = Inputs(load_detections(rows))
        state = {}
        for name in args.models:
            state[name] = {"status": "running"}
            atomic_json(args.output / "suite_status.json", state)
            try:
                run_model(args.previous, args.output, name, protocol, rows, inputs)
                state[name] = {"status": "complete"}
                report(args.previous, args.output)
            except (OSError, RuntimeError, ValueError, KeyError) as e:
                state[name] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
                import traceback

                traceback.print_exc()
            atomic_json(args.output / "suite_status.json", state)
        return int(any(v["status"] != "complete" for v in state.values()))


if __name__ == "__main__":
    raise SystemExit(main())
