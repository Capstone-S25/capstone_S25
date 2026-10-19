"""Generate source-derived LaTeX tables/figures and compile the technical report.

This reads completed study artifacts only. It performs no model inference.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/technical_runthrough_20261018"
STEM = "model_evaluation_runthrough_20261018"
MODELS = ["mobileclip2-s2", "siglip2-b16-224"]
ALL_MODELS = [*MODELS, "smolvlm-500m", "minicpm-v4"]
SHORT = dict(zip(ALL_MODELS, ["MobileCLIP", "SigLIP", "SmolVLM", "MiniCPM"]))
TASKS = ["smoking", "littering", "loitering", "rough_sleeping", "unattended_items"]
TASK_NAMES = dict(
    zip(
        TASKS,
        ["Smoking", "Dumping/littering", "Loitering", "Public sleep", "Unattended"],
    )
)
SOURCES: dict[str, str] = {}
TABLES: dict[str, int] = {}


def register(path):
    p = ROOT / path
    SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def read(path):
    return json.loads(register(path).read_text())


def rows(path):
    return list(csv.DictReader(register(path).open()))


def esc(value):
    if value is None:
        return "--"
    if value == "MobileCLIP":
        return r"Mobile\allowbreak{}CLIP"
    value = str(value).replace("→", " to ").replace("±", "+/-").replace("–", "--")
    value = value.replace("Dumping/littering", "Dumping / littering")
    return "".join(
        {
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
        }.get(c, c)
        for c in value
    )


def num(v, digits=3):
    return "--" if v is None or v == "" else f"{float(v):.{digits}f}"


def pct(v):
    return "--" if v is None or v == "" else f"{float(v) * 100:.1f}%"


def compact(v):
    return str(v).replace("_", " ")


def table(name, caption, headings, data, weights=None, raw_columns=()):
    data = list(data)
    n = len(headings)
    weights = weights or [1] * n
    width = [0.84 * w / sum(weights) for w in weights]
    columns = (
        "@{}"
        + "".join(
            r">{\raggedright\arraybackslash}p{" + f"{w:.4f}" + r"\linewidth}"
            for w in width
        )
        + "@{}"
    )
    aliases = {
        "Threshold": "Cutoff",
        "FP/repeat/unknown": "FP / dup / unk",
        "Events": "Hit / N",
        "Setting": "C/ep",
        "Epoch": "Ep.",
        "Pick": "*",
        "Heads": "n",
    }
    heading = (
        " & ".join(
            r"\textbf{"
            + esc("M/S" if h == "Model" and width[i] < 0.08 else aliases.get(h, h))
            + "}"
            for i, h in enumerate(headings)
        )
        + r" \\"
    )
    parts = [
        r"\begingroup\footnotesize",
        r"\setlength{\LTpre}{5pt}\setlength{\LTpost}{8pt}",
        r"\begin{longtable}{" + columns + "}",
        r"\caption["
        + esc(name.replace("_", " ").capitalize())
        + "]{"
        + esc(caption)
        + r"}\label{tab:"
        + name
        + r"}\\",
        r"\toprule",
        heading,
        r"\midrule\endfirsthead",
        r"\multicolumn{" + str(n) + r"}{l}{\textit{Table \thetable\ continued}}\\",
        r"\toprule",
        heading,
        r"\midrule\endhead",
        r"\midrule\multicolumn{"
        + str(n)
        + r"}{r}{\textit{Continued on next page}}\\\endfoot",
        r"\bottomrule\endlastfoot",
    ]
    for row in data:
        assert len(row) == n, (name, row)
        parts.append(
            " & ".join(
                str(v) if i in raw_columns else esc(v) for i, v in enumerate(row)
            )
            + r" \\"
        )
    parts += [r"\end{longtable}\endgroup", ""]
    (OUT / "generated" / (name + ".tex")).write_text("\n".join(parts))
    TABLES[name] = len(data)


def baseline_tables():
    audit = register("reports/archive/2026-10-11/evaluation_audit.md").read_text()
    findings = re.findall(r"^### (F\d+) — ([^:]+): (.+)$", audit, re.MULTILINE)
    assert len(findings) == 15
    table(
        "historical_audit",
        "Fifteen archived audit findings; original reproductions are in the registered audit source.",
        ["ID", "Severity", "Historical issue"],
        findings,
        [0.5, 0.7, 4.5],
    )
    archived = []
    for name in ["full", "no_graph", "win_1", "win_16", "vl_128", "vl_512"]:
        j = read(f"results/vadclip/runs/{name}/eval.json")
        c, v = j["config"], j["metrics"]["dual_mean"]
        archived.append(
            [
                compact(name),
                c["visual_length"],
                c["attn_window"],
                "off" if c["no_graph"] else "on",
                v["n"],
                v["n_pos"],
                num(v["aupr"], 4),
                num(v["f1_opt"], 4),
            ]
        )
    table(
        "historical_vadclip_ablation",
        "Superseded validation-only adapter records; target/compression and implementation defects preclude a clean architecture claim.",
        ["Run", "Length", "Window", "Graph", "Reps", "+ reps", "Old AP", "Old opt F1"],
        archived,
        [1.3, 0.7, 0.8, 0.7, 0.7, 0.7, 0.9, 1.1],
    )
    definitions = read("anomaly_eval/config/anomalies.json")
    table(
        "definitions",
        "Recorded visible-evidence target definitions.",
        ["Label", "Definition"],
        [(k.replace("_", " "), v) for k, v in definitions.items()],
        [1, 4],
    )
    source = register("anomaly_eval/models.py").read_text()
    tree = ast.parse(source)
    node = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(k, ast.Name) and k.id == "MODELS" for k in n.targets)
    )
    specs = {
        ast.literal_eval(k): [ast.literal_eval(a) for a in v.args]
        for k, v in zip(node.keys, node.values)
    }
    data = [
        [
            SHORT[m],
            specs[m][0],
            r"\path{" + specs[m][1] + "}",
            "Frozen image comparison; CLIP later adapted"
            if m in MODELS
            else "Prompted image comparison only",
        ]
        for m in ALL_MODELS
    ]
    data.append(
        [
            "VadCLIP",
            "Supervised video adapter",
            "Frozen CLIP-B/16 features",
            "Separate offline smoking protocol",
        ]
    )
    table(
        "models",
        "Models and comparison scope; pinned revisions appear in the appendix.",
        ["Model", "Family", "Checkpoint / feature source", "Role"],
        data,
        [1.1, 1, 2, 2],
        raw_columns=(2,),
    )
    table(
        "model_revisions",
        "Pinned image-model identities from the adapter registry.",
        ["Model", "Repository", "Revision"],
        [[SHORT[m], specs[m][1], r"\path{" + specs[m][2] + "}"] for m in ALL_MODELS],
        [1, 2.6, 3],
        raw_columns=(2,),
    )
    table(
        "chronology",
        "Distinct stages of the recorded project.",
        ["Dates", "Stage", "Completed evidence / important qualification"],
        [
            [
                "11 Oct",
                "Early benchmark and audit",
                "80 images; proxy labels and no unattended class. Historical, superseded.",
            ],
            [
                "12 Oct",
                "Corrected CPU-era run",
                "236 smoking frames; run status error; MiniCPM smoking missing.",
            ],
            [
                "13 Oct",
                "Full GPU comparison",
                "13,400 predictions across four models; separate supervised VadCLIP and CPU pilot.",
            ],
            [
                "14 Oct",
                "Expanded accuracy",
                "55,156 predictions; 1,770 unattended endpoints and qualified weak diagnostics.",
            ],
            [
                "15--16 Oct",
                "Visual fine-tuning",
                "Two backbones x three seeds; six actual changed-visual-weight runs.",
            ],
            [
                "16 Oct",
                "Strategy screen",
                "330 configurations, new-source transfer and provided thermal trajectories.",
            ],
            [
                "16 Oct",
                "Causal video pilot",
                "50 heads; real tracks/regions; two clean new AVSS events, one censored.",
            ],
            [
                "17 Oct",
                "Focused improvements",
                "38 policies; state/memory ablations; 830 tiled-detector GPU views.",
            ],
            [
                "18 Oct",
                "This documentation",
                "Source reconstruction and LaTeX/PDF compilation; no new model experiment.",
            ],
        ],
        [1, 1.6, 4],
    )
    smoking = []
    unattended = []
    latency = []
    for m in ALL_MODELS:
        j = read(f"results/gpu_all_models_20261013/smoking_frame/dev/{m}/summary.json")
        b = j["metrics"]["binary"]
        smoking.append(
            [
                SHORT[m],
                pct(b["precision"]),
                pct(b["recall"]),
                num(b["f1"]),
                pct(b["false_positive_rate"]),
                num(b.get("average_precision")),
                num(b.get("roc_auc")),
                f"{b['tp']}/{b['fp']}",
            ]
        )
        for protocol in ["frame", "window"]:
            q = read(
                f"results/gpu_all_models_20261013/unattended_items_{protocol}/test/{m}/summary.json"
            )["metrics"]["binary"]
            unattended.append(
                [
                    SHORT[m],
                    protocol,
                    pct(q["precision"]),
                    pct(q["recall"]),
                    num(q["f1"]),
                    pct(q["false_positive_rate"]),
                    num(q.get("average_precision")),
                ]
            )
        q = read(f"results/cpu_latency_20261013/{m}/summary.json")
        latency.append(
            [
                SHORT[m],
                num(q["latency_p50_seconds"]),
                num(q["latency_p95_seconds"]),
                num(q["peak_rss_mb"] / 1024, 2),
                num(q["load_seconds"], 2),
            ]
        )
    table(
        "baseline_smoking",
        "Initial complete GPU smoking development results: 3,278 frames, previously inspected sources.",
        ["Model", "PPV", "Recall", "F1", "FPR", "AP", "AUROC", "TP/FP"],
        smoking,
        [1.6, 1, 1, 0.8, 1, 0.8, 0.8, 0.9],
    )
    table(
        "baseline_unattended",
        "Initial unattended raw test results: each protocol has four positives and twelve normals.",
        ["Model", "Input", "PPV", "Recall", "F1", "FPR", "AP"],
        unattended,
        [1.5, 1, 1, 1, 0.8, 1, 0.8],
    )
    table(
        "historical_latency",
        "Historical CPU pilot only: eight images, three measured repeats, four logical CPUs; not rerun.",
        ["Model", "p50 (s)", "p95 (s)", "Peak RSS (GiB)", "Load (s)"],
        latency,
        [1.6, 1, 1, 1.4, 1],
    )
    vad = read("results/gpu_all_models_20261013/vadclip/adapter/eval.json")
    data = [
        [
            k,
            pct(v["precision"]),
            pct(v["recall"]),
            num(v["f1"]),
            pct(v["false_positive_rate"]),
            num(v["aupr"]),
            num(v["auroc"]),
        ]
        for k, v in vad["metrics"].items()
    ]
    table(
        "vadclip",
        "Separate supervised VadCLIP test: 1,056 frames; dual_mean is the development-selected branch.",
        ["Branch", "PPV", "Recall", "F1", "FPR", "AP", "AUROC"],
        data,
        [1.6, 1, 1, 0.8, 1, 0.8, 0.8],
    )


def expanded_tables():
    q = read("results/expanded_accuracy_20261013/quality_summary.json")
    assert q["completed_predictions"] == 55156 and q["completed_summaries"] == 40
    data = []
    for c in q["cohorts"]:
        counts = c["counts"]
        data.append(
            [
                compact(c["name"]),
                str(sum(counts.get("dev", {}).values())),
                str(sum(counts.get("test", {}).values())),
                "Provisional intervals"
                if "unattended" in c["name"]
                else "Oracle posture crops"
                if "oracle" in c["name"]
                else "Weak / component diagnostic",
            ]
        )
    table(
        "expanded_cohorts",
        "Expanded study cohort sizes; frame/window unattended inputs share endpoints.",
        ["Cohort", "Dev", "Test", "Evidence"],
        data,
        [2.4, 0.6, 0.6, 1.7],
    )
    data, weak, oracle = [], [], []
    for r in q["rows"]:
        if r["split"] == "test" and r["cohort"] in {
            "unattended_expanded_frame",
            "unattended_expanded_window",
        }:
            data.append(
                [
                    SHORT[r["model"]],
                    r["cohort"].rsplit("_", 1)[1],
                    pct(r["precision"]),
                    pct(r["recall"]),
                    num(r["f1"]),
                    pct(r["fpr"]),
                    num(r["ap"]),
                    num(r["auroc"]),
                ]
            )
        if r["cohort"] in {
            "littering_weak",
            "loitering_weak",
            "rough_sleeping_weak",
            "person_presence",
        }:
            weak.append(
                [
                    compact(r["cohort"]),
                    SHORT[r["model"]],
                    r["inputs"],
                    pct(r["target_prediction_rate"]),
                    r["abstentions"],
                    r["errors"],
                ]
            )
        if r["cohort"] == "lying_pose_oracle" and r["split"] == "test":
            oracle.append(
                [
                    SHORT[r["model"]],
                    pct(r["precision"]),
                    pct(r["recall"]),
                    num(r["f1"]),
                    pct(r["fpr"]),
                    num(r["ap"]),
                    num(r["auroc"]),
                ]
            )
    table(
        "expanded_unattended",
        "Expanded unattended raw test decisions: 962 endpoints per input protocol, provisional labels.",
        ["Model", "Input", "PPV", "Recall", "F1", "FPR", "AP", "AUROC"],
        data,
        [1.5, 0.9, 0.9, 0.9, 0.7, 0.9, 0.7, 0.9],
    )
    table(
        "weak_diagnostics",
        "Weak/coarse labels: target-prediction rates, not verified event recall or precision.",
        ["Diagnostic", "Model", "Inputs", "Target rate", "Abstain", "Errors"],
        weak,
        [2, 1.4, 0.8, 1.1, 0.8, 0.7],
    )
    table(
        "oracle_pose",
        "Oracle posture test: 44 lying and four standing source-box crops; not rough-sleeping events.",
        ["Model", "PPV", "Recall", "F1", "FPR", "AP", "AUROC"],
        oracle,
        [1.6, 1, 1, 0.8, 1, 0.8, 1],
    )


def fine_tables():
    dataset = read("data/clip_training_20261015/dataset.json")
    assert dataset["samples"] == 7498 and dataset["unique_view_images"] == 12478
    table(
        "training_counts",
        "First supervised study: normal/positive counts, with a separate smoking photo holdout.",
        ["Target", "Train N/+", "Dev N/+", "Test N/+", "External N/+"],
        [
            [TASK_NAMES[t]]
            + [
                f"{dataset['counts'][t].get(s + '/0', 0)}/{dataset['counts'][t].get(s + '/1', 0)}"
                if any(k.startswith(s + "/") for k in dataset["counts"][t])
                else "--"
                for s in ["train", "dev", "test", "external"]
            ]
            for t in TASKS
        ],
        [1.7, 1, 1, 1, 1.2],
    )
    main, operating, allop, seeds = [], [], [], []
    fine_values = {}
    for m in MODELS:
        for seed in [42, 43, 44]:
            base = f"results/clip_finetuning_20261015/seed_{seed}/{m}"
            sel = read(base + "/selection.json")
            result = read(base + "/float32_evaluation/results.json")
            for task in TASKS:
                route = sel["task_routes"][task]
                r = result[route]["test"][task]
                v = r["operating_points"]["dev_max_f1"]
                seeds.append(
                    [
                        SHORT[m],
                        seed,
                        TASK_NAMES[task],
                        compact(route),
                        num(v["ap"]),
                        num(v["f1"]),
                        pct(v["recall"]),
                        pct(v["fpr"]),
                    ]
                )
                if seed != 42:
                    continue
                z = result["zero_shot"]["test"][task]["operating_points"]["raw"]["ap"]
                p = result["linear_probe"]["test"][task]["operating_points"]["raw"][
                    "ap"
                ]
                f = result["finetuned"]["test"][task]["operating_points"]["raw"]["ap"]
                fine_values[m, task] = (z, v["ap"])
                main.append(
                    [
                        SHORT[m],
                        TASK_NAMES[task],
                        v["n"],
                        num(z),
                        num(p),
                        num(f),
                        compact(route),
                        num(v["ap"]),
                    ]
                )
                for point in ["raw", "dev_max_f1", "dev_fpr_budget"]:
                    a = r["operating_points"][point]
                    allop.append(
                        [
                            SHORT[m],
                            TASK_NAMES[task],
                            compact(point),
                            num(r["thresholds"][point]),
                            pct(a["precision"]),
                            pct(a["recall"]),
                            num(a["f1"]),
                            pct(a["fpr"]),
                        ]
                    )
                    if task in {"smoking", "unattended_items"}:
                        operating.append(allop[-1])
    table(
        "finetuning_main",
        "Primary seed 42 AP and development routing; pretrained baseline is last-frame text scoring.",
        ["Model", "Target", "N", "Zero", "Probe", "Tuned", "Dev route", "Route AP"],
        main,
        [1.1, 1.45, 0.55, 0.65, 0.65, 0.65, 1.2, 0.75],
    )
    head = ["Model", "Target", "Cutoff rule", "Threshold", "PPV", "Recall", "F1", "FPR"]
    weights = [1.1, 1.4, 1.3, 0.9, 0.7, 0.8, 0.7, 0.8]
    table(
        "finetuning_operating",
        "Primary smoking/unattended threshold transfer: fixed weights, development-only cutoffs.",
        head,
        operating,
        weights,
    )
    table(
        "finetuning_all_operating",
        "All primary routed operating points; historical/provisional cohort qualifications apply.",
        head,
        allop,
        weights,
    )
    table(
        "finetuning_seeds",
        "All seed-specific routed results at maximum-dev-F1 cutoffs; seed 42 remains primary.",
        ["Model", "Seed", "Target", "Dev route", "AP", "F1", "Recall", "FPR"],
        seeds,
        [1.1, 0.5, 1.5, 1.3, 0.65, 0.65, 0.8, 0.8],
    )
    return fine_values


def strategy_tables():
    q = read("results/strategy_study_20261016/report_data.json")
    hist, transfer = [], []
    transfer_values = []
    for r in q["primary_rows"]:
        row = [
            r[0].replace("2-S2", "").replace("2-B/16-224", ""),
            r[1],
            r[2],
            r[3],
            r[4],
            r[5],
            r[6],
            r[7],
            r[8],
        ]
        (hist if "historical" in r[1] else transfer).append(row)
        if "historical" not in r[1]:
            transfer_values.append(r)
    head = [
        "Model",
        "Source/cohort",
        "N",
        "+",
        "Old AP",
        "New AP",
        "F1",
        "Recall",
        "FPR",
    ]
    weights = [1, 2.5, 0.5, 0.5, 0.65, 0.65, 0.65, 0.8, 0.8]
    table(
        "strategy_historical",
        "Previously inspected source holdouts: previous bundle versus primary selected strategy; max-dev-F1 decisions.",
        head,
        hist,
        weights,
    )
    table(
        "strategy_transfer",
        "New-source/image-holdout results at first strategy evaluation; not later HMDB-adapted cohorts.",
        head,
        transfer,
        weights,
    )
    route_rows, candidate_rows = [], []
    for m in MODELS:
        sel = read(f"results/strategy_study_20261016/{m}/selection.json")
        dev = read(f"results/strategy_study_20261016/{m}/development_results.json")
        for task, choice in sel["tasks"].items():
            route_rows.append(
                [
                    SHORT[m],
                    TASK_NAMES[task],
                    compact(choice["strategy"]),
                    choice["classifier"],
                    str(choice.get("C", "--")),
                    num(choice["mean_source_ap"]),
                    num(choice["worst_source_ap"]),
                ]
            )
        for task, candidates in dev.items():
            assert len(candidates) == 33
            for c in candidates:
                candidate_rows.append(
                    [
                        "M" if m == MODELS[0] else "S",
                        TASK_NAMES[task],
                        compact(c["strategy"]),
                        c["classifier"],
                        str(c.get("C", c.get("epoch", "--"))),
                        num(c["mean_source_ap"], 4),
                        num(c["worst_source_ap"], 4),
                    ]
                )
    assert len(candidate_rows) == 330
    table(
        "strategy_routes",
        "Primary strategy routes and development scores; alpha suffixes denote visual interpolation.",
        ["Model", "Target", "Representation", "Head", "C", "Mean AP", "Worst AP"],
        route_rows,
        [1.1, 1.4, 2.2, 0.8, 0.5, 0.9, 0.9],
    )
    table(
        "strategy_all_candidates",
        "All 330 strategy candidates: development scores only; M=MobileCLIP, S=SigLIP.",
        ["Model", "Target", "Representation", "Head", "Setting", "Mean AP", "Worst AP"],
        candidate_rows,
        [0.5, 1.5, 2.2, 0.8, 0.6, 0.85, 0.85],
    )
    thermal = read("results/strategy_study_20261016/thermal_trajectories/results.json")
    t = [
        [
            compact(k),
            num(v["dev"]["mean_source_ap"]),
            num(v["test"]["dev_max_f1"]["ap"]),
            pct(v["test"]["dev_max_f1"]["recall"]),
            num(v["test"]["dev_max_f1"]["f1"]),
            pct(v["test"]["dev_max_f1"]["fpr"]),
        ]
        for k, v in thermal.items()
    ]
    table(
        "thermal",
        "Provided thermal tracks: mean development-date AP and held-out-date decisions.",
        ["Candidate", "Dev AP", "Test AP", "Recall", "F1", "FPR"],
        t,
        [2.2, 1, 1, 1, 1, 1],
    )
    return transfer_values


def pilot_tables():
    pcsv = rows("results/video_pilot_20261016/comparison.csv")
    ecsv = rows("results/video_pilot_20261016/events.csv")
    candidates, smoke, primary_events, event_all = [], [], [], []
    for m in MODELS:
        joint = read(f"results/video_pilot_20261016/{m}/joint_selection.json")
        for task, details in joint["tasks"].items():
            for c in details["candidates"]:
                candidates.append(
                    [
                        "M" if m == MODELS[0] else "S",
                        TASK_NAMES[task],
                        compact(c["name"]),
                        c.get("seed", "--"),
                        c.get("epoch", "--"),
                        num(c["selection"]["worst_source_ap"], 4),
                        num(c["selection"]["mean_source_ap"], 4),
                        "*" if c["name"] == details["primary"]["name"] else "",
                    ]
                )
        name = joint["tasks"]["smoking"]["primary"]["name"]
        for source in [
            "smoking_cctv",
            "hmdb51_action_transfer",
            "kaggle_sujaykapadnis_smoking",
            "stanford40",
        ]:
            a = next(
                r
                for r in pcsv
                if r["model"] == m
                and r["task"] == "smoking"
                and r["candidate"] == name
                and r["source"] == source
                and r["operating_point"] == "dev_fpr_budget"
            )
            b = next(
                r
                for r in pcsv
                if r["model"] == m
                and r["task"] == "smoking"
                and r["candidate"] == "scene_linear"
                and r["source"] == source
                and r["operating_point"] == "raw"
            )
            smoke.append(
                [
                    SHORT[m],
                    compact(source),
                    a["n"],
                    num(b["ap"]),
                    num(a["ap"]),
                    pct(a["recall"]),
                    pct(a["fpr"]),
                ]
            )
    assert len(candidates) == 50
    table(
        "pilot_all_heads",
        "All final source-guarded pilot heads; asterisk marks the primary development route.",
        ["Model", "Target", "Head", "Seed", "Epoch", "Worst AP", "Mean AP", "Pick"],
        candidates,
        [0.5, 1.2, 3.5, 0.5, 0.6, 0.8, 0.8, 0.4],
    )
    table(
        "pilot_smoking",
        "Pilot smoking: same-pilot scene AP versus selected route, with dev-FPR-budget recall/FPR.",
        ["Model", "Source", "N", "Scene AP", "Pick AP", "Recall", "FPR"],
        smoke,
        [1.1, 2.5, 0.6, 0.9, 0.9, 0.9, 0.9],
    )
    for r in ecsv:
        row = [
            SHORT[r["model"]],
            compact(r["cohort"]),
            compact(r["candidate"]),
            f"{r['detected_events']}/{r['events']}",
            f"{r['normal_false_alerts']}/{r['duplicate_alerts']}/{r['unknown_time_alerts']}",
            num(r["normal_seconds"], 1),
            num(r["median_delay_seconds"], 1),
        ]
        event_all.append(row)
        if r["primary"] == "True" and r["cohort"].startswith("fresh_avss_clean_"):
            primary_events.append(row)
    # Exact source cohort names are retained in the appendix; main AVSS rows
    # are selected from the primary flag and clean-primary prefix.
    if not primary_events:
        raise ValueError("Clean AVSS primary rows not found")
    heads = [
        "Model",
        "Protocol/cohort",
        "Head",
        "Events",
        "FP/repeat/unknown",
        "Normal s",
        "Delay s",
    ]
    weights = [1, 2.2, 2.5, 0.65, 1.1, 0.8, 0.8]
    table(
        "pilot_all_events",
        "Pilot event outcomes for all scored family routes; delay is median onset-to-emission time.",
        heads,
        event_all,
        weights,
    )
    table(
        "pilot_avss_primary",
        "Clean AVSS primary source subset; global and known-source criteria remain distinct.",
        heads,
        primary_events,
        weights,
    )
    fresh = [
        json.loads(s)
        for s in register("data/video_pilot_20261016/fresh_recordings.jsonl")
        .read_text()
        .splitlines()
    ]
    table(
        "avss_sources",
        "AVSS source timing and censoring before pilot scoring.",
        [
            "Source",
            "Encoded FPS",
            "Decoded frames",
            "Original event frames",
            "Censored?",
        ],
        [
            [
                r["id"].split(":")[-1],
                num(r["fps"], 5),
                r["decoded_frames"],
                str(r["original_events_frames"]),
                "yes" if r["right_censored_annotation"] else "no",
            ]
            for r in fresh
        ],
        [1.6, 1, 1.1, 1.8, 0.8],
    )
    crop = read("results/video_pilot_20261016/error_audit/pilot_region_coverage.json")[
        "summary"
    ]
    table(
        "crop_audit",
        "Smoking geometric coverage, independent of recognition accuracy; positive frames with target boxes.",
        [
            "Split",
            "+ frames",
            "Median height at 224",
            "All targets <12 px",
            "Prior <half coverage",
            "New <half coverage",
        ],
        [
            [
                k,
                v["positive_frames"],
                num(v["median_largest_smoker_height_at_224"], 1),
                pct(v["all_smokers_under_12_pixels_fraction"]),
                pct(v["prior_under_half_target_coverage_fraction"]),
                pct(v["pilot_under_half_target_coverage_fraction"]),
            ]
            for k, v in crop.items()
        ],
        [0.7, 0.8, 1.3, 1.3, 1.4, 1.4],
    )


def latest_tables():
    selection = read("results/event_improvements_20261017/smoking_selection.json")
    result = read("results/event_improvements_20261017/smoking_results.json")
    smoke, policies = [], []
    for m in MODELS:
        for label, name in [
            ("Previous", "previous_primary_original_policy"),
            ("17 Oct pick", selection[m]["selected"]),
        ]:
            c = result[m][name]["sources"]["smoking_cctv"]
            h = result[m][name]["sources"]["hmdb51_action_transfer"]
            smoke.append(
                [
                    SHORT[m],
                    label,
                    num(c["ap"], 4),
                    f"{pct(c['recall'])}/{pct(c['fpr'])}",
                    num(h["ap"], 4),
                    f"{pct(h['recall'])}/{pct(h['fpr'])}",
                ]
            )
        for name, c in selection[m]["candidates"].items():
            assert all(v["fpr"] <= 0.05 for v in c["development"].values())
            policies.append(
                [
                    "M" if m == MODELS[0] else "S",
                    compact(name),
                    len(c["members"]),
                    num(c["threshold"], 5),
                    pct(c["worst_recall"]),
                    pct(c["mean_recall"]),
                    num(c["worst_ap"], 4),
                    "*" if name == selection[m]["selected"] else "",
                ]
            )
    table(
        "latest_smoking",
        "Latest source-budget/ensemble policies versus previous primary policies, on the same retrospective cases.",
        ["Model", "Policy", "CCTV AP", "CCTV recall/FPR", "HMDB AP", "HMDB recall/FPR"],
        smoke,
        [1.1, 1.1, 1, 1.8, 1, 1.8],
    )
    table(
        "latest_all_policies",
        "All 38 smoking-policy development summaries; equal sigmoid averages are not calibrated probabilities.",
        [
            "Model",
            "Policy",
            "Heads",
            "Cutoff",
            "Worst recall",
            "Mean recall",
            "Worst AP",
            "Pick",
        ],
        policies,
        [0.5, 3.5, 0.6, 0.9, 1, 1, 0.9, 0.4],
    )
    ev = read("results/event_improvements_20261017/event_results.json")["aggregate"]
    allrows, mainrows = [], []
    for group, variants in ev.items():
        for name, v in variants.items():
            a = [
                compact(group),
                compact(name),
                f"{v['detected_events']}/{v['events']}",
                f"{v['normal_false_alerts']}/{v['duplicate_alerts']}/{v['unknown_time_alerts']}",
                num(v["normal_seconds"], 2),
                num(v["median_delay_seconds"], 2),
                num(v["normal_far_upper_95_per_hour"], 1),
            ]
            allrows.append(a)
            if group == "avss_primary_retrospective":
                mainrows.append(a[1:])
    table(
        "latest_events",
        "Latest AVSS primary state ablations; all results retrospective; upper bound assumes a Poisson process.",
        [
            "Policy",
            "Events",
            "FP/repeat/unknown",
            "Normal s",
            "Delay s",
            "95% upper/hr",
        ],
        mainrows,
        [3.3, 0.6, 1.1, 0.8, 0.8, 1],
    )
    table(
        "latest_all_event_groups",
        "Latest state ablations by source/split; historical intervals have different source criteria.",
        [
            "Source",
            "Policy",
            "Events",
            "FP/repeat/unknown",
            "Normal s",
            "Delay s",
            "95% upper/hr",
        ],
        allrows,
        [1.5, 2.6, 0.6, 1.1, 0.8, 0.8, 1],
    )
    component = read("results/event_improvements_20261017/component_results.json")
    comp = []
    for r in component["aggregate"]:
        comp.append(
            [
                "CAVIAR train" if "caviar" in r["source"] else "Highway debris",
                compact(r["mode"]),
                r["iou"],
                f"{r['matched']}/{r['reference_boxes']}",
                pct(r["recall"]),
                r["predicted_items"],
            ]
        )
    table(
        "latest_components",
        "Full-image versus fixed full-plus-tiles item reference recall; partial references do not support precision.",
        ["Source", "Input", "IoU", "Matches", "Recall", "Predicted items"],
        comp,
        [1.3, 1.5, 0.5, 0.8, 0.8, 1],
    )
    return component


def evidence_registry():
    # Register narrative evidence and implementation sources without loading
    # checkpoint tensors or changing historical reports.
    for p in sorted((OUT / "chapters").glob("*.tex")):
        for relative in re.findall(r"\\evidence\{([^}]+)\}", p.read_text()):
            register(relative)
    for path in [
        "reports/workflow_review_20261013.md",
        "reports/clip_finetuning_findings_20261016.md",
        "reports/strategy_findings_20261016.md",
        "reports/video_pilot_findings_20261016.md",
        "reports/event_improvements_findings_20261017.md",
        "reports/strategy_protocol_20261016.md",
        "reports/video_pilot_protocol_20261016.md",
        "reports/data_readiness_20261013.json",
        "reports/vadclip_source_validation.json",
        "reports/hardware_20261013.json",
        "data/clip_training_20261015/stanford_source_audit.json",
        "data/clip_training_20261015/iwdd_source_audit.json",
        "data/clip_training_20261015/commons_source_audit.json",
        "data/strategy_study_20261016/dataset.json",
        "data/strategy_sources_20261016/small_sources_receipts.json",
        "data/strategy_sources_20261016/meva/index.json",
        "data/video_pilot_20261016/source_guard.json",
        "data/video_pilot_20261016/data_verification.json",
        "data/video_pilot_20261016/fresh_cohort.json",
        "data/video_pilot_20261016/detector.json",
        "data/video_pilot_20261016/grounding_all.json",
        "data/event_improvements_20261017/receipt.json",
        "data/event_improvements_20261017/component_protocol.json",
        "results/event_improvements_20261017/protocol.json",
        "results/event_improvements_20261017/validation.json",
        "results/expanded_accuracy_20261013/verification.json",
        "results/clip_finetuning_20261015/verification.json",
        "results/strategy_study_20261016/verification.json",
        "results/video_pilot_20261016/verification.json",
        "results/event_improvements_20261017/verification.json",
        "results/strategy_study_20261016/run.json",
        "results/video_pilot_20261016/run.json",
        "results/clip_finetuning_20261015/run.json",
        "reports/slides/progress_validation_20261017.json",
        "anomaly_eval/training/clip.py",
        "anomaly_eval/training/data.py",
        "anomaly_eval/training/strategies.py",
        "anomaly_eval/training/video.py",
        "anomaly_eval/training/temporal.py",
        "anomaly_eval/training/event_memory.py",
        "scripts/run_strategy_screen.py",
        "scripts/train_video_pilot.py",
        "scripts/train_person_pilot.py",
        "scripts/train_grounded_pilot.py",
        "scripts/evaluate_video_pilot.py",
        "scripts/benchmark_loitering_trajectories.py",
        "scripts/run_event_improvements.py",
        "scripts/benchmark_item_tiles.py",
        "scripts/report_event_improvements.py",
        "scripts/prepare_item_component.py",
        "scripts/fetch_item_component.py",
    ]:
        register(path)
    inventory = []
    for directory in sorted((ROOT / "results").iterdir()):
        if not directory.is_dir():
            continue
        choices = [
            directory / n
            for n in ["run.json", "state.json", "evaluation.json"]
            if (directory / n).exists()
        ]
        if not choices:
            continue
        p = choices[0]
        j = read(str(p.relative_to(ROOT)))
        inventory.append(
            [
                r"\path{" + directory.name + "}",
                p.name,
                compact(j.get("status", "not recorded")),
            ]
        )
    table(
        "run_inventory",
        "Preserved root run metadata status; not a combined performance leaderboard.",
        ["Result directory", "Metadata", "Recorded status"],
        inventory,
        [3.7, 1.2, 2],
        raw_columns=(0,),
    )
    data = [
        [f"S{i:03d}", r"\path{" + p + "}", h[:16]]
        for i, (p, h) in enumerate(sorted(SOURCES.items()), 1)
    ]
    table(
        "source_registry",
        "Documentation source registry; complete SHA-256 identities are in source_manifest.json.",
        ["ID", "Repository-relative source", "SHA-256 prefix"],
        data,
        [0.55, 5, 1.6],
        raw_columns=(1,),
    )
    table(
        "verification",
        "Historical completion audits; counts are replayed predictions, not independent events.",
        ["Stage", "Test snapshot", "Prediction/identity evidence"],
        [
            [
                "Corrected GPU baseline",
                "144 tests",
                "13,400 planned predictions checked; 20 summaries recomputed.",
            ],
            [
                "Expanded accuracy",
                "Existing tests retained",
                "40/40 successful CUDA summaries; 55,156 predictions replayed.",
            ],
            [
                "Actual fine-tuning",
                "153 tests",
                "46,926 held-out + 29,394 dev predictions; six changed visual checkpoints.",
            ],
            [
                "Strategy screen",
                "159 tests",
                "117,612 predictions; 330 selections; 20,980 input image identities; ten live cases.",
            ],
            [
                "Causal video pilot",
                "171 tests",
                "99,528 predictions; 50 selections/heads; 25,452 input images and live encoder checks.",
            ],
            [
                "Latest improvements",
                "182 tests",
                "5,643 smoking scores; 38 dev policies; 6,390 trace decisions; eight component cells.",
            ],
        ],
        [1.5, 1.2, 4],
    )


def figures(fine, transfer, component):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/tools/report_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )
    colors = ["#007E80", "#C86535"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    x = np.arange(5)
    for ax, m in zip(axes, MODELS):
        ax.bar(
            x - 0.18,
            [fine[m, t][0] for t in TASKS],
            0.36,
            label="Pretrained text",
            color="#b6c4cb",
        )
        ax.bar(
            x + 0.18,
            [fine[m, t][1] for t in TASKS],
            0.36,
            label="Dev-routed",
            color=colors[MODELS.index(m)],
        )
        ax.set_xticks(
            x, ["Smoking\nCCTV", "Dumping", "Loitering", "Public\nsleep", "Unattended"]
        )
        ax.set_ylim(0, 1.05)
        ax.set_title(SHORT[m])
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Average precision")
    fig.tight_layout()
    fig.savefig(OUT / "figures/finetuning_ap.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, model in zip(axes, ["MobileCLIP2-S2", "SigLIP2-B/16-224"]):
        v = [r for r in transfer if r[0] == model]
        xx = np.arange(len(v))
        ax.bar(
            xx - 0.18,
            [float(r[4]) for r in v],
            0.36,
            label="Previous bundle",
            color="#b6c4cb",
        )
        ax.bar(
            xx + 0.18,
            [float(r[5]) for r in v],
            0.36,
            label="Selected strategy",
            color=colors[["MobileCLIP2-S2", "SigLIP2-B/16-224"].index(model)],
        )
        ax.set_xticks(xx, ["Smoking\nphotos", "HMDB", "Public\nsleep", "MEVA"])
        ax.set_ylim(0, 1.05)
        ax.set_title(model)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Average precision")
    fig.tight_layout()
    fig.savefig(OUT / "figures/strategy_transfer.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    hours = np.geomspace(0.02, 10, 300)
    ax.loglog(
        hours,
        -np.log(0.05) / hours,
        color=colors[0],
        label="Zero-alert Poisson upper bound",
    )
    ax.axhline(1, color=colors[1], ls="--", label="1 false alert/hour")
    h = 165.73886666666664 / 3600
    ax.scatter([h], [-np.log(0.05) / h], color=colors[1], zorder=3)
    ax.annotate(
        "AVSS: 2.76 min; upper 65.1/hour",
        (h, -np.log(0.05) / h),
        xytext=(0.11, 85),
        arrowprops={"arrowstyle": "->"},
        fontsize=9,
    )
    ax.set_xlabel("Reviewed normal exposure (hours)")
    ax.set_ylabel("95% one-sided upper rate (per hour)")
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures/normal_exposure_bound.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(2)
    for i, mode in enumerate(["full", "full_plus_tiles"]):
        values = [
            next(
                r["recall"]
                for r in component["aggregate"]
                if r["source"] == s and r["mode"] == mode and r["iou"] == 0.5
            )
            for s in ["caviar_training_component", "author_highway_debris_component"]
        ]
        bars = ax.bar(
            x + (i - 0.5) * 0.35,
            values,
            0.35,
            label="Full image" if i == 0 else "Full + four fixed tiles",
            color="#b6c4cb" if i == 0 else colors[0],
        )
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=4)
    ax.set_xticks(x, ["CAVIAR training-source items", "Highway-debris stress cases"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Reference-box recall at IoU 0.5")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "figures/tiled_recall.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile using the workspace-local Tectonic binary",
    )
    args = parser.parse_args()
    for name in ["generated", "figures", "numeric_exports"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    baseline_tables()
    expanded_tables()
    fine = fine_tables()
    transfer = strategy_tables()
    pilot_tables()
    component = latest_tables()
    figures(fine, transfer, component)
    for stage in [
        "clip_finetuning_20261015",
        "strategy_study_20261016",
        "video_pilot_20261016",
    ]:
        shutil.copyfile(
            register(f"results/{stage}/comparison.csv"),
            OUT / f"numeric_exports/{stage}_comparison.csv",
        )
    shutil.copyfile(
        register("results/video_pilot_20261016/events.csv"),
        OUT / "numeric_exports/video_pilot_20261016_events.csv",
    )
    (OUT / "tools").mkdir(exist_ok=True)
    shutil.copyfile(
        register("scripts/bootstrap_report_latex.py"),
        OUT / "tools/bootstrap_report_latex.py",
    )
    evidence_registry()
    inputs = sorted(SOURCES.items())
    manifest = {
        "report_date": "2026-10-18",
        "timezone": "Asia/Jakarta",
        "experiment_cutoff": "2026-10-17",
        "no_new_model_inference": True,
        "source_files_sha256": dict(inputs),
        "table_rows": TABLES,
        "latex_files_sha256": {
            str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(OUT.rglob("*.tex"))
        },
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figures_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((OUT / "figures").glob("*.pdf"))
        },
        "numeric_exports_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((OUT / "numeric_exports").glob("*.csv"))
        },
        "figure_build_packages": {
            name: importlib.metadata.version(name) for name in ["matplotlib", "numpy"]
        },
        "numeric_exports": "Preserved CSV exports supplement generated tables; cohort/field semantics remain as in source reports.",
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.compile:
        compiler = ROOT / "data/tools/tectonic/tectonic"
        if not compiler.exists():
            raise FileNotFoundError("Run scripts/bootstrap_report_latex.py first")
        env = {**os.environ, "XDG_CACHE_HOME": str(ROOT / "data/tools/tectonic/cache")}
        with (OUT / "compile_console.log").open("w") as log:
            subprocess.run(
                [
                    str(compiler),
                    "--untrusted",
                    "--keep-logs",
                    "--keep-intermediates",
                    STEM + ".tex",
                ],
                cwd=OUT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    print(
        "Generated",
        len(TABLES),
        "tables;",
        sum(TABLES.values()),
        "table rows;",
        len(inputs),
        "source identities; four scientific figures.",
    )


if __name__ == "__main__":
    main()
