"""Audit immutable run artifacts and generate the paper's numerical evidence.

Run with the repository requirements installed. No inference is performed and no
original result is edited. Recomputed bootstrap/calibration live in a temporary
copy. Tables are generated from JSONL, not transcribed from report.md.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import math
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = next(p for p in PAPER.parents if (p / "anomaly_eval").is_dir())
sys.path.insert(0, str(REPO))

from anomaly_eval.analysis import analyze, calibrate
from anomaly_eval.data import (
    experiment_options,
    fingerprint,
    image_pixel_sha256,
)
from anomaly_eval.metrics import detection_metrics, percentile

NAMES = {
    "mobileclip2-s2": "MobileCLIP2-S2",
    "siglip2-b16-224": "SigLIP2-B/16",
    "smolvlm-500m": "SmolVLM-500M",
    "minicpm-v4": "MiniCPM-V 4.0",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex(value):
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(c, c) for c in str(value))


def fmt(value, digits=3):
    return "--" if value is None else f"{value:.{digits}f}"


def interval(value):
    return "--" if value is None else f"[{value[0]:.3f}, {value[1]:.3f}]"


def table(filename, columns, headers, rows):
    text = [
        r"\begin{longtable}{" + columns + "}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule\endhead",
    ]
    text.extend(" & ".join(str(x) for x in row) + r" \\" for row in rows)
    text.extend([r"\bottomrule", r"\end{longtable}"])
    (PAPER / "generated" / filename).write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=PAPER.parent)
    args = parser.parse_args()
    root = args.run.resolve()
    for folder in ("generated", "figures"):
        (PAPER / folder).mkdir(exist_ok=True)
    e = read(root / "evaluation.json")
    inventory = {
        p.relative_to(root).as_posix(): sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and PAPER not in p.parents and p.name != "inspection.md"
    }
    checks = []

    def check(name, passed, details=None):
        checks.append({"check": name, "passed": bool(passed), "details": details})

    mismatches = [
        p
        for p, h in e["code_sha256"].items()
        if not (REPO / p).is_file() or sha(REPO / p) != h
    ]
    check("recorded_source_hashes_match_checkout", not mismatches, mismatches)
    expected_fp = fingerprint(
        {
            "options": experiment_options(e["options"]),
            "code": e["code_sha256"],
            "dataset": e["dataset_fingerprint"],
        }
    )
    check(
        "top_level_experiment_fingerprint", expected_fp == e["experiment_fingerprint"]
    )
    (
        cohorts,
        metrics_rows,
        timing_rows,
        responses,
        groups,
        calibration_rows,
        ci_rows,
        pair_rows,
    ) = [], [], [], [], [], [], [], []
    cases, all_identities = [], []
    total_rows = 0
    for b in e["benchmarks"]:
        directory = root / b["output"]
        run = read(directory / "run.json")
        key = b["output"]
        label = (
            "Smoking dev"
            if b["cohort"] == "smoking_frame"
            else ("Frame " if b["cohort"].endswith("frame") else "Window ") + b["split"]
        )
        ids = run["sample_identity"]
        all_identities.extend((key, i) for i in ids)
        fp = fingerprint(
            {
                "options": experiment_options(run["options"]),
                **{
                    k: run[k]
                    for k in ("definitions", "sample_identity", "code_sha256", "models")
                },
            }
        )
        check(key + ":fingerprint", fp == run["experiment_fingerprint"])
        check(key + ":code_hashes", run["code_sha256"] == e["code_sha256"])
        check(
            key + ":no_pixel_duplicates",
            len({i["pixel_sha256"] for i in ids}) == len(ids),
        )
        moments = [tuple(i["source_moment"]) for i in ids if i.get("source_moment")]
        check(key + ":no_source_moment_duplicates", len(set(moments)) == len(moments))
        expected = [
            (s["path"], s["label"], s["group_id"])
            for s in run["samples"]
            if s["split"] == "test"
        ]
        local_verified, unavailable, image_mismatches = [], [], []
        for i in ids:
            path = (
                i["path"]
                .replace("${BUNDLE_ROOT}", str(REPO / "test_data"))
                .replace("${DATA_ROOT}", str(REPO / "data"))
            )
            file = Path(path)
            if file.is_file():
                good = (
                    sha(file) == i["sha256"]
                    and image_pixel_sha256(file) == i["pixel_sha256"]
                )
                (local_verified if good else image_mismatches).append(i["path"])
            else:
                unavailable.append(i["path"])
        check(
            key + ":available_local_image_byte_and_pixel_hashes",
            not image_mismatches,
            {
                "verified": len(local_verified),
                "unavailable": len(unavailable),
                "mismatches": image_mismatches,
            },
        )
        bygroup = collections.defaultdict(collections.Counter)
        for s in run["samples"]:
            bygroup[s["group_id"]][s["label"]] += 1
        for group, counts in sorted(bygroup.items()):
            groups.append(
                [
                    tex(label),
                    tex(group),
                    counts.get("normal", 0),
                    sum(v for k, v in counts.items() if k != "normal"),
                ]
            )
        cohort = {
            "cohort": key,
            "label": label,
            "planned_samples": len(expected),
            "host": {
                k: run[k]
                for k in (
                    "python",
                    "platform",
                    "cpu_model",
                    "physical_cpus",
                    "logical_cpus",
                    "total_ram_mb",
                )
            },
            "local_images_verified": len(local_verified),
            "local_images_unavailable": len(unavailable),
            "models": {},
        }
        for model in run["models"]:
            directory_model = directory / model
            summary = read(directory_model / "summary.json")
            rows = [
                json.loads(line)
                for line in (directory_model / "predictions.jsonl")
                .read_text()
                .splitlines()
                if line.strip()
            ]
            total_rows += len(rows)
            first = [r for r in rows if r["repeat"] == 0]
            counts = dict(collections.Counter(r["repeat"] for r in rows))
            complete = len(rows) == len(expected) * run["options"]["repeats"] and all(
                [
                    (r["path"], r["label"], r["group_id"])
                    for r in rows
                    if r["repeat"] == rep
                ]
                == expected
                for rep in range(run["options"]["repeats"])
            )
            check(
                key + ":" + model + ":planned_cohort_complete",
                complete,
                {
                    "expected_rows": len(expected) * run["options"]["repeats"],
                    "observed_rows": len(rows),
                    "repeat_counts": counts,
                },
            )
            lineage = {i["path"]: i for i in ids}
            check(
                key + ":" + model + ":prediction_lineage",
                all(
                    all(
                        r.get(k) == lineage[r["path"]].get(k)
                        for k in (
                            "label",
                            "group_id",
                            "origin",
                            "pixel_sha256",
                            "protocol",
                        )
                    )
                    for r in rows
                ),
            )
            calculated = detection_metrics(rows, list(run["definitions"]))
            if complete:
                check(
                    key + ":" + model + ":metrics_reproduce",
                    calculated == summary.get("metrics"),
                )
                latencies = [
                    r["latency_seconds"] for r in rows if r["status"] != "error"
                ]
                check(
                    key + ":" + model + ":p95_reproduces",
                    math.isclose(
                        percentile(latencies, 0.95),
                        summary["latency_p95_seconds"],
                        rel_tol=1e-12,
                    ),
                )
                check(
                    key + ":" + model + ":no_pacing",
                    summary["throttle_sleep_seconds"] == 0,
                )
            decision_sets = collections.defaultdict(set)
            for r in rows:
                decision_sets[r["path"]].add((r["status"], r.get("prediction")))
            formats = collections.Counter(
                r.get("response_format")
                for r in rows
                if r.get("raw_response") is not None
            )
            texts = collections.Counter(
                r.get("raw_response") for r in rows if r.get("raw_response") is not None
            )
            entry = {
                "complete": complete,
                "status": summary["status"],
                "expected_rows": len(expected) * run["options"]["repeats"],
                "observed_rows": len(rows),
                "repeat_counts": counts,
                "first_pass_complete": len(first) == len(expected),
                "calculated_first_pass_metrics": calculated,
                "unstable_decisions": sum(len(v) > 1 for v in decision_sets.values()),
                "response_formats": dict(formats),
                "raw_response_counts": dict(texts),
                "summary": summary,
            }
            cohort["models"][model] = entry
            for r in first:
                cases.append({"cohort": key, "model": model, **r})
            if complete:
                m, bi = calculated, calculated["binary"]
                metrics_rows.append(
                    [
                        tex(label),
                        NAMES[model],
                        bi["tp"],
                        bi["fp"],
                        bi["fn"],
                        bi["tn"],
                        fmt(m["accuracy_including_failures"]),
                        fmt(bi["precision"]),
                        fmt(bi["recall"]),
                        fmt(bi["f1"]),
                        fmt(bi["false_positive_rate"]),
                        fmt(m["prediction_coverage"]),
                        fmt(m["macro_f1"]),
                    ]
                )
            else:
                metrics_rows.append(
                    [tex(label), NAMES[model] + r"$^\dagger$", *(["--"] * 11)]
                )
            timing_rows.append(
                [
                    tex(label),
                    NAMES[model],
                    fmt(summary.get("load_seconds")),
                    fmt(summary.get("warmup_seconds")),
                    fmt(summary.get("latency_p50_seconds")),
                    fmt(summary.get("latency_p95_seconds")),
                    fmt(summary.get("active_samples_per_second")),
                    fmt(summary.get("average_cpu_cores"), 1),
                    fmt(summary["peak_rss_mb"] / 1024, 2),
                ]
            )
            if formats:
                predictions = collections.Counter(
                    r.get("prediction") or "unanswered" for r in first
                )
                fraction = formats.get("json_label", 0) / sum(formats.values())
                responses.append(
                    [
                        tex(label),
                        NAMES[model],
                        len(rows),
                        fmt(fraction),
                        tex(
                            ", ".join(
                                f"{k}:{v}" for k, v in sorted(predictions.items())
                            )
                        ),
                    ]
                )
        if (directory / "improvements.json").is_file():
            im = read(directory / "improvements.json")
            positive_groups = {
                s["group_id"] for s in run["samples"] if s["label"] != "normal"
            }
            rng = random.Random(im["seed"])
            keys = sorted(bygroup)
            no_positive = sum(
                not (set(rng.choices(keys, k=len(keys))) & positive_groups)
                for _ in range(im["bootstrap"])
            )
            cohort["bootstrap_draws_without_positive_group"] = no_positive
            for model, result in im["models"].items():
                ci_rows.append(
                    [
                        tex(label),
                        NAMES[model],
                        fmt(result["point"]["binary_f1"]),
                        interval(result["ci95"]["binary_f1"]),
                        fmt(result["point"]["roc_auc"]),
                        interval(result["ci95"]["roc_auc"]),
                    ]
                )
            for pair, results in im["paired_differences"].items():
                a, bmodel = pair.split(" minus ")
                pair_rows.append(
                    [
                        tex(label),
                        NAMES[a] + " - " + NAMES[bmodel],
                        fmt(results["binary_f1"]["point"]),
                        interval(results["binary_f1"]["ci95"]),
                    ]
                )
        if (directory / "calibration.json").is_file():
            c = read(directory / "calibration.json")
            for model, value in c["models"].items():
                for name, operating in value.get("operating_points", {}).items():
                    m = operating["all_samples"]["binary"]
                    calibration_rows.append(
                        [
                            tex(label),
                            NAMES[model],
                            "Max dev F1" if name == "dev_max_f1" else "Dev FPR budget",
                            fmt(value["thresholds"][name], 6),
                            m["tp"],
                            m["fp"],
                            m["fn"],
                            m["tn"],
                            fmt(m["recall"]),
                            fmt(m["f1"]),
                            fmt(m["false_positive_rate"]),
                            fmt(operating["score_metrics"]["aupr"]),
                        ]
                    )
        cohorts.append(cohort)

    for protocol in ("frame", "window"):
        dev = read(root / f"unattended_items_{protocol}/dev/run.json")[
            "sample_identity"
        ]
        test = read(root / f"unattended_items_{protocol}/test/run.json")[
            "sample_identity"
        ]
        for attribute in ("group_id", "sha256", "pixel_sha256", "source_moment"):
            dv = {json.dumps(i[attribute], sort_keys=True) for i in dev}
            tv = {json.dumps(i[attribute], sort_keys=True) for i in test}
            check(protocol + ":dev_test_disjoint_" + attribute, not (dv & tv))

    with tempfile.TemporaryDirectory(prefix="capstone-paper-audit-") as temporary:
        copy = Path(temporary) / "run"
        shutil.copytree(
            root,
            copy,
            ignore=lambda directory, names: [
                n for n in names if n == "paper" or n == "inspection.md"
            ],
        )
        for b in e["benchmarks"]:
            d = copy / b["output"]
            if (d / "improvements.json").is_file():
                saved = read(d / "improvements.json")
                result = analyze(d, saved["bootstrap"], saved["seed"])
                check(
                    b["output"] + ":bootstrap_and_paired_results_reproduce",
                    result == saved,
                )
        for protocol in ("frame", "window"):
            d = copy / f"unattended_items_{protocol}/test"
            saved = read(d / "calibration.json")
            result = calibrate(
                copy / f"unattended_items_{protocol}/dev", d, saved["target_fpr"]
            )
            check(protocol + ":calibration_reproduces", result == saved)
    check(
        "original_artifacts_unchanged_by_audit",
        all(sha(root / p) == h for p, h in inventory.items()),
    )
    result = {
        "run": root.name,
        "audit_date": "2026-10-12",
        "status": e["status"],
        "checks": checks,
        "recorded_code_matches": not mismatches,
        "source_artifact_sha256": inventory,
        "total_prediction_rows": total_rows,
        "planned_prediction_rows": e["dataset"]["confirmed_rows"]
        * 4
        * e["options"]["repeats"],
        "total_stage_seconds": sum(s["seconds"] for s in e["steps"]),
        "cohorts": cohorts,
        "unsupported_tasks": [
            k
            for k, v in e["requested_task_coverage"].items()
            if not v["available_protocols"]
        ],
        "vadclip_executed": e["options"]["vadclip"],
        "checkout_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "verification_environment": {
            "python": sys.version,
            "packages": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "scipy", "scikit-learn", "matplotlib", "Pillow")
            },
        },
    }
    (PAPER / "audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (PAPER / "first_pass_cases.jsonl").write_text(
        "".join(json.dumps(c, allow_nan=False) + "\n" for c in cases)
    )
    table(
        "quality.tex",
        "llrrrrrrrrrrr",
        [
            "Cohort",
            "Model",
            "TP",
            "FP",
            "FN",
            "TN",
            "Acc.",
            "Prec.",
            "Recall",
            "F1",
            "FPR",
            "Cov.",
            "Macro F1",
        ],
        metrics_rows,
    )
    table(
        "timing.tex",
        "llrrrrrrr",
        [
            "Cohort",
            "Model",
            "Load s",
            "Warm s",
            "p50 s",
            "p95 s",
            "Active/s",
            "CPU eq.",
            "GiB",
        ],
        timing_rows,
    )
    table(
        "timing_details.tex",
        "llrrrrr",
        [
            "Cohort",
            "Model",
            "Mean s",
            "Wall/s",
            "Loop wall s",
            "Loop CPU s",
            "Worker wall s",
        ],
        [
            [
                tex(c["label"]),
                NAMES[m],
                fmt(v["summary"].get("latency_mean_seconds")),
                fmt(v["summary"].get("wall_samples_per_second")),
                fmt(v["summary"].get("measured_wall_seconds")),
                fmt(v["summary"].get("measured_cpu_seconds")),
                fmt(v["summary"].get("total_wall_seconds")),
            ]
            for c in cohorts
            for m, v in c["models"].items()
        ],
    )
    table(
        "responses.tex",
        "llrrP{5.0cm}",
        ["Cohort", "Model", "Rows", "JSON frac.", "First-pass decisions"],
        responses,
    )
    table("groups.tex", "llrr", ["Cohort", "Source group", "Normal", "Event"], groups)
    table(
        "calibration.tex",
        "lllrrrrrrrrr",
        [
            "Protocol",
            "Model",
            "Rule",
            "Threshold",
            "TP",
            "FP",
            "FN",
            "TN",
            "Recall",
            "F1",
            "FPR",
            "AP",
        ],
        calibration_rows,
    )
    table(
        "intervals.tex",
        "llrlrl",
        ["Cohort", "Model", "F1", "95\\% range", "AUROC", "95\\% range"],
        ci_rows,
    )
    table(
        "paired.tex",
        "llrl",
        ["Cohort", "Model difference", "$\\Delta$F1", "95\\% range"],
        pair_rows,
    )
    table(
        "heldout.tex",
        "llrrrrrr",
        ["Input", "Model", "TP/4", "FP/12", "Recall", "F1", "FPR", "Acc."],
        [
            [r[0].replace(" test", ""), r[1], r[2], r[3], r[8], r[9], r[10], r[6]]
            for r in metrics_rows
            if r[0].endswith("test")
        ],
    )
    review = read(REPO / "anomaly_eval/config/caviar_review_intervals.json")
    table(
        "review_intervals.tex",
        "lrrlp{7cm}",
        ["Sequence", "Start s", "End s", "Label", "Recorded rationale"],
        [
            [
                tex(r["sequence"]),
                fmt(r["start"], 1),
                fmt(r["end"], 1),
                "Event" if r["label"] != "normal" else "Normal",
                tex(r["note"]),
            ]
            for r in review["intervals"]
        ],
    )
    (PAPER / "generated" / "command.tex").write_text(
        "\\begin{verbatim}\nanomaly-eval run --mode evaluation --skip-prepare \\\n  --threads 32 --repeats 3 --protocols frame window \\\n  --limit 0 --bootstrap 2000 --seed 42 \\\n  --timeout-seconds 3600 --max-rss-mb 24576 \\\n  --image-policy letterbox --max-side 768 --max-new-tokens 64\n\\end{verbatim}\n"
    )
    make_figures(root, cohorts, cases)
    print(
        json.dumps(
            {
                "checks": len(checks),
                "failed_checks": [c["check"] for c in checks if not c["passed"]],
                "rows": total_rows,
                "planned_rows": result["planned_prediction_rows"],
                "bootstrap_without_positive": {
                    c["cohort"]: c.get("bootstrap_draws_without_positive_group")
                    for c in cohorts
                },
                "local_verified": sum(c["local_images_verified"] for c in cohorts),
                "local_unavailable": sum(
                    c["local_images_unavailable"] for c in cohorts
                ),
            },
            indent=2,
        )
    )
    expected_failure = "smoking_frame/dev:minicpm-v4:planned_cohort_complete"
    unexpected = [
        c["check"] for c in checks if not c["passed"] and c["check"] != expected_failure
    ]
    if unexpected:
        raise SystemExit(
            "Unexpected evidence validation failures: " + ", ".join(unexpected)
        )


def make_figures(root, cohorts, cases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    models = list(NAMES)
    colors = ["#2573a6", "#32936f", "#d99131", "#9a578f"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, label in zip(axs, ("Frame test", "Window test")):
        c = next(c for c in cohorts if c["label"] == label)
        x = np.arange(4)
        for offset, metric, text in (
            (-0.19, "recall", "Event recall"),
            (0.19, "false_positive_rate", "False-positive rate"),
        ):
            vals = [
                c["models"][m]["calculated_first_pass_metrics"]["binary"][metric]
                for m in models
            ]
            ax.bar(
                x + offset,
                vals,
                width=0.36,
                label=text,
                color="#2573a6" if offset < 0 else "#c46241",
            )
        ax.set(
            xticks=x,
            xticklabels=["MobileCLIP2", "SigLIP2", "SmolVLM", "MiniCPM"],
            ylim=(0, 1.12),
            title=label,
            ylabel="Fraction",
        )
        ax.tick_params(axis="x", labelrotation=20)
    axs[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "recall_fpr.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for i, m in enumerate(models):
        cs = [c for c in cohorts if c["label"].endswith("test")]
        axs[0].plot(
            [0, 1],
            [c["models"][m]["summary"]["latency_p95_seconds"] for c in cs],
            marker="o",
            label=NAMES[m],
            color=colors[i],
        )
        axs[1].plot(
            [0, 1],
            [c["models"][m]["summary"]["peak_rss_mb"] / 1024 for c in cs],
            marker="o",
            label=NAMES[m],
            color=colors[i],
        )
    axs[0].set(
        yscale="log",
        ylabel="p95 active seconds (log scale)",
        xticks=[0, 1],
        xticklabels=["Frame test", "Window test"],
    )
    axs[0].axhline(1, color="gray", ls="--", label="1 input/s budget")
    axs[1].set(
        ylabel="Sampled peak RSS (GiB)",
        xticks=[0, 1],
        xticklabels=["Frame test", "Window test"],
    )
    axs[1].axhline(24, color="gray", ls="--", label="24 GiB cap")
    axs[1].set_ylim(0, 26)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "resources.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.8))
    for ax, (protocol, model) in zip(
        axs.flat, [(p, m) for p in ("frame", "window") for m in models[:2]]
    ):
        for label, col in [("normal", "#2573a6"), ("unattended_items", "#c46241")]:
            rr = [
                r
                for r in cases
                if r["cohort"] == f"unattended_items_{protocol}/test"
                and r["model"] == model
                and r["label"] == label
            ]
            ax.scatter(
                [r["anomaly_score"] for r in rr],
                np.arange(len(rr)) * 0.08,
                marker="o" if label == "normal" else "^",
                color=col,
                label=label.replace("_", " "),
                alpha=0.85,
            )
        cal = read(root / f"unattended_items_{protocol}/test/calibration.json")[
            "models"
        ][model]["thresholds"]
        for rule, style in [("dev_max_f1", "--"), ("dev_fpr_budget", ":")]:
            ax.axvline(cal[rule], color="black", ls=style, label=rule.replace("_", " "))
        ax.set(
            title=f"{NAMES[model]} / {protocol}",
            xlabel="Continuous anomaly score",
            yticks=[],
        )
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "score_distributions.pdf")
    plt.close(fig)
    # Actual unchanged CAVIAR inputs: selected held-out positive and matched control.
    examples = [
        ("frame", "caviar_LeftBag_AtChair_2_0_3.png", "Event endpoint: 28.0 s"),
        ("frame", "caviar_LeftBag_AtChair_3_0_3.png", "Attended placement: 19.6 s"),
        ("window", "caviar_LeftBag_AtChair_2_0.png", "Event: past 6 s sheet"),
        ("window", "caviar_LeftBag_AtChair_3_0.png", "Attended: past 6 s sheet"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (protocol, name, title) in zip(axs.flat, examples):
        path = REPO / "test_data" / "caviar" / protocol / name
        with Image.open(path) as image:
            ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(PAPER / "figures" / "input_examples.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
