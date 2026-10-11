"""Matched-cohort statistics, calibration, figures and reports."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from anomaly_eval.metrics import (
    detection_metrics,
    frame_metrics,
    percentile,
    select_threshold,
)
from anomaly_eval.paths import resolve_arguments, resolve_path


def _ci(values):
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return (
        [percentile(finite, 0.025), percentile(finite, 0.975)]
        if len(finite) > 1
        else None
    )


def _extract(m):
    return {
        "accuracy_including_failures": m["accuracy_including_failures"],
        "macro_f1": m["macro_f1"],
        "binary_f1": m["binary"]["f1"],
        "roc_auc": m["binary"]["roc_auc"],
    }


def analyze(directory, bootstrap=2000, seed=12345):
    run = json.loads((directory / "run.json").read_text())
    labels = list(run["definitions"])
    if run.get("metric_version") != 2 or not run.get("experiment_fingerprint"):
        raise ValueError(
            "Legacy run lacks corrected metric version/provenance; rerun inference"
        )
    expected = {
        (r["path"], r["label"], r["group_id"])
        for r in run["samples"]
        if r["split"] == "test"
    }
    allrows = {}
    summaries = {}
    for model in run["models"]:
        summary = json.loads((directory / model / "summary.json").read_text())
        summaries[model] = summary
        raw = [
            json.loads(line)
            for line in (directory / model / "predictions.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        rows = [r for r in raw if r["repeat"] == 0 and r["label"] is not None]
        actual = [(r["path"], r["label"], r["group_id"]) for r in rows]
        if (
            len(set(actual)) != len(actual)
            or set(actual) != expected
            or len(raw) != len(expected) * run["options"]["repeats"]
        ):
            raise ValueError(
                f"Incomplete or mismatched planned prediction cohort: {model}"
            )
        point = detection_metrics(rows, labels)
        if point != summary.get("metrics"):
            raise ValueError(
                f"Saved metrics disagree with corrected predictions: {model}"
            )
        allrows[model] = {r["path"]: r for r in rows}
    first = next(iter(allrows.values()))
    supported = [l for l in labels if any(r["label"] == l for r in first.values())]
    groups = defaultdict(list)
    for path, r in first.items():
        groups[r["group_id"]].append(path)
    keys = sorted(groups)
    rng = random.Random(seed)
    metrics = {
        m: _extract(detection_metrics(list(rows.values()), labels, supported))
        for m, rows in allrows.items()
    }
    draws = {m: {k: [] for k in metrics[m]} for m in metrics}
    for _ in range(bootstrap):
        paths = [p for group in rng.choices(keys, k=len(keys)) for p in groups[group]]
        for m, rows in allrows.items():
            values = _extract(
                detection_metrics([rows[p] for p in paths], labels, supported)
            )
            for k, v in values.items():
                draws[m][k].append(v)
    result = {
        "unit": "source_group",
        "n_groups": len(keys),
        "uncertainty_warning": "Few independent source groups; intervals are exploratory"
        if len(keys) < 10
        else None,
        "seed": seed,
        "bootstrap": bootstrap,
        "experiment_fingerprint": run["experiment_fingerprint"],
        "labels": labels,
        "scored_labels": supported,
        "models": {},
        "paired_differences": {},
    }
    for m, point in metrics.items():
        full = detection_metrics(list(allrows[m].values()), labels, supported)
        result["models"][m] = {
            "point": point,
            "ci95": {k: _ci(v) for k, v in draws[m].items()},
            "per_class": full["per_class"],
            "coverage": full["prediction_coverage"],
            "abstentions": full["abstentions"],
            "errors": full["errors"],
        }
    for a, b in combinations(metrics, 2):
        result["paired_differences"][a + " minus " + b] = {
            k: {
                "point": metrics[a][k] - metrics[b][k]
                if metrics[a][k] is not None and metrics[b][k] is not None
                else None,
                "ci95": _ci(
                    [
                        x - y if x is not None and y is not None else None
                        for x, y in zip(draws[a][k], draws[b][k])
                    ]
                ),
            }
            for k in metrics[a]
        }
    (directory / "improvements.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def load_rows(directory, model):
    return [
        json.loads(l)
        for l in (directory / model / "predictions.jsonl").read_text().splitlines()
        if l.strip() and json.loads(l)["repeat"] == 0
    ]


def fpr_threshold(y, score, target=0.05):
    if not ((y == 0).any() and (y == 1).any()):
        raise ValueError("Both classes required in development")
    candidates = np.r_[np.nextafter(score.max(), np.inf), np.unique(score)]
    feasible = [t for t in candidates if np.mean(score[y == 0] >= t) <= target]
    return float(min(feasible))


def calibrate(dev_dir: Path, test_dir: Path, target_fpr=0.05):
    dev = json.loads((dev_dir / "run.json").read_text())
    test = json.loads((test_dir / "run.json").read_text())
    if (
        dev["evaluation_split"] != "dev"
        or test["evaluation_split"] != "test"
        or dev["definitions"] != test["definitions"]
        or dev["models"] != test["models"]
    ):
        raise ValueError("Development/test protocols or model lists disagree")
    settings = (
        "models",
        "seed",
        "threads",
        "duty_cycle",
        "max_side",
        "image_policy",
        "max_new_tokens",
        "cpu_cores",
        "repeats",
    )
    if any(
        dev.get("options", {}).get(k) != test.get("options", {}).get(k)
        for k in settings
    ) or dev.get("code_sha256") != test.get("code_sha256"):
        raise ValueError("Development/test inference settings or code differ")
    dev_groups = {r["group_id"] for r in dev["samples"]}
    test_groups = {r["group_id"] for r in test["samples"]}
    if dev_groups & test_groups:
        raise ValueError("Development/test groups overlap")
    task = next(l for l in test["definitions"] if l != "normal")
    result = {"task": task, "target_fpr": target_fpr, "models": {}}
    for model in test["models"]:
        dr, tr = load_rows(dev_dir, model), load_rows(test_dir, model)
        expected_dev = sum(r["split"] == "test" for r in dev["samples"])
        expected_test = sum(r["split"] == "test" for r in test["samples"])
        if len(dr) != expected_dev or len(tr) != expected_test:
            raise ValueError("Calibration needs complete planned cohorts")
        if any(r.get("anomaly_score") is None for r in dr + tr):
            result["models"][model] = {
                "status": "no_continuous_score",
                "categorical": detection_metrics(tr, list(test["definitions"])),
                "note": "No invented hard-label AUC or threshold for VLMs",
            }
            continue
        yd = np.array([r["label"] != "normal" for r in dr], dtype=int)
        sd = np.array([r["anomaly_score"] for r in dr])
        yt = np.array([r["label"] != "normal" for r in tr], dtype=int)
        st = np.array([r["anomaly_score"] for r in tr])
        thresholds = {
            "dev_max_f1": select_threshold(yd, sd),
            "dev_fpr_budget": fpr_threshold(yd, sd, target_fpr),
        }
        out = {
            "thresholds": thresholds,
            "development_negatives": int((yd == 0).sum()),
            "development_fpr_resolution": 1 / int((yd == 0).sum()),
            "operating_points": {},
        }
        for name, t in thresholds.items():
            scored = [
                {**r, "prediction": task if r["anomaly_score"] >= t else "normal"}
                for r in tr
            ]
            out["operating_points"][name] = {
                "all_samples": detection_metrics(scored, list(test["definitions"])),
                "score_metrics": frame_metrics(yt, st, t),
            }
        result["models"][model] = out
    (test_dir / "calibration.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def generate(results_dir: Path, out_dir: Path):
    import os
    import tempfile

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "capstone-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run = json.loads((results_dir / "run.json").read_text())
    if run.get("metric_version") != 2:
        raise ValueError("Figures require corrected metric version 2")
    results = json.loads((results_dir / "results.json").read_text())["models"]
    improvements = (
        json.loads((results_dir / "improvements.json").read_text())
        if (results_dir / "improvements.json").is_file()
        else {}
    )
    models = [r["model"] for r in results]
    xs = np.arange(len(models))
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Accuracy, all samples", "accuracy_including_failures"),
        ("Binary anomaly F1", "binary_f1"),
        ("Continuous score AUROC", "roc_auc"),
        ("Prediction coverage", "prediction_coverage"),
        ("End-to-end p95 seconds", "latency_p95_seconds"),
        ("Peak worker RSS MiB", "peak_rss_mb"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (title, key) in zip(axes.flat, panels):
        for i, r in enumerate(results):
            m = r.get("metrics", {})
            b = m.get("binary") or {}
            v = (
                b.get("f1")
                if key == "binary_f1"
                else b.get("roc_auc")
                if key == "roc_auc"
                else m.get(key, r.get(key))
            )
            if v is None:
                ax.text(i, 0, "NA", ha="center")
                continue
            ax.bar(i, v, color="#4C72B0")
            ci = (
                improvements.get("models", {})
                .get(r["model"], {})
                .get("ci95", {})
                .get(key)
            )
            if ci:
                ax.vlines(i, ci[0], ci[1], color="black")
                ax.hlines(ci, [i - 0.08] * 2, [i + 0.08] * 2, color="black")
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    fig.suptitle(
        f"{run['evaluation_split']} · identical cohort · cluster intervals where available"
    )
    fig.tight_layout()
    path = out_dir / "comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_report(output, state, args):
    lines = [
        "# Corrected evaluation run",
        "",
        f"Run: `{output.name}`. Execution status: **{state['status']}**.",
        "",
        "Measurements describe the recorded host. CPU throttling does not emulate a Ryzen AI Max+ 395.",
        "",
        "All models in each task/protocol see the same images and source groups. Quality scores include failures and abstentions. Continuous-score operating thresholds are chosen on development data; VLM hard decisions have no invented AUC.",
        "",
        "| Task / input | Coverage |",
        "|---|---|",
    ]
    for task, protocols in state["dataset"]["coverage"].items():
        if not protocols:
            lines.append(f"| {task} | Needs verified event and matched normal labels |")
        for protocol, c in protocols.items():
            lines.append(
                f"| {task} / {protocol} | {'Dev/test available' if c['eligible_for_calibrated_test'] else 'Development only'}; counts {c['counts']} |"
            )
    lines += [
        "",
        "## Measured model comparisons",
        "",
        "| Task / input / split | Model | Coverage | Binary F1 | FPR | p95 seconds | RSS MiB | Host budget |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    def fmt(v):
        return "NA" if v is None else f"{v:.3f}"

    for record in state["benchmarks"]:
        path = resolve_path(record["output"], output) / "results.json"
        if not path.is_file():
            continue
        for model in json.loads(path.read_text())["models"]:
            m = model.get("metrics", {})
            b = m.get("binary") or {}
            lat = model.get("latency_p95_seconds")
            rss = model.get("peak_rss_mb")
            baselines = model.get("naive_baselines", {})
            if baselines:
                state.setdefault("naive_baselines", {})[
                    record["cohort"] + "/" + record["split"]
                ] = {
                    k: {
                        "accuracy": v["accuracy_including_failures"],
                        "binary_f1": v["binary"]["f1"],
                    }
                    for k, v in baselines.items()
                }
            feasible = (
                lat is not None
                and rss is not None
                and lat <= 1 / args.camera_fps
                and rss <= args.max_rss_mb
                and model["status"] == "ok"
            )
            values = [
                record["cohort"] + " / " + record["split"],
                model["model"],
                fmt(m.get("prediction_coverage")),
                fmt(b.get("f1")),
                fmt(b.get("false_positive_rate")),
                fmt(lat),
                fmt(rss),
                "passes sampled frame budget" if feasible else "fails/unknown",
            ]
            lines.append("| " + " | ".join(values) + " |")
    if state.get("naive_baselines"):
        lines += [
            "",
            "Constant-prediction baselines (quality only):",
            "",
            "```json",
            json.dumps(state["naive_baselines"], indent=2),
            "```",
        ]
    lines += [
        "",
        f"Resource screen: one stream sampled at {args.camera_fps:g} frames/s, p95 inference ≤ {1 / args.camera_fps:g}s, worker/process-tree RSS cap {args.max_rss_mb:g} MiB. This excludes queueing and camera decoding; verify on the deployment box.",
        "",
        "Raw model argmax and development-calibrated operating points are separate in `calibration.json`. Group-bootstrap intervals and paired differences are in each run's `improvements.json`. Unsupported tasks are not zero-scored or pooled into a model ranking.",
        "",
        "A final choice requires adequate recall at an acceptable false-alert rate on verified events from the deployment domain. A small staged set cannot establish a five-task deployment winner.",
        "",
        "## Limitations",
        "",
    ]
    lines += ["- " + s for s in state["dataset"]["limitations"]]
    lines += [
        "- Data preparation issue: " + issue
        for issue in state["dataset"].get("issues", [])
    ]
    if state.get("vadclip"):
        lines += [
            "",
            "VadCLIP results are a supervised offline smoking adaptation with frozen CLIP-B/16 features and a separate train/dev/test protocol. Adapter timing excludes the image encoder and is not comparable to end-to-end VLM FPS. It uses future context within each sparse session and is not a streaming detector.",
        ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--dev-dir",
        type=Path,
        help="Optional development run for frozen-threshold calibration",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--no-figures", action="store_true")
    args = resolve_arguments(parser.parse_args(argv))
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    directory = resolve_path(args.results_dir)
    result = analyze(directory, args.bootstrap, args.seed)
    if args.dev_dir:
        calibrate(resolve_path(args.dev_dir), directory, args.target_fpr)
    if not args.no_figures:
        generate(directory, directory / "figures")
    print(
        f"Analyzed {len(result['models'])} models in {result['n_groups']} independent source groups"
    )
    return 0
