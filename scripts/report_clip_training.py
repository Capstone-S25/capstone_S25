"""Report trained CLIP quality without choosing checkpoints from test scores."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/clip_finetuning_20261015"
NAMES = ["mobileclip2-s2", "siglip2-b16-224"]
LABELS = ["MobileCLIP2-S2", "SigLIP2-B/16-224"]
TASKS = ["smoking", "littering", "loitering", "rough_sleeping", "unattended_items"]
KINDS = {
    "smoking": "source action labels; retrospective CCTV camera holdout",
    "littering": "author-labelled video-level waste dumping; broader than pedestrian littering",
    "loitering": "author track/type labels; supervised video holdout on one camera",
    "rough_sleeping": "provisional assistant-reviewed public-sleep photographs",
    "unattended_items": "provisional assistant-reviewed source-video events; retrospective camera holdout",
}


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def f(value):
    return "—" if value is None else f"{value:.3f}"


def p(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def prediction_rows(directory, method, split):
    path = directory / f"{method}_{split}_predictions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_model_bundle(selections, results):
    """Index the predesignated primary checkpoints without selecting on test."""
    provenance = read(RUN / "provenance.json")
    audit = read(RUN / "verification.json")
    models = {}
    card_rows = []
    for model, label in zip(NAMES, LABELS):
        directory = RUN / "seed_42" / model
        checkpoints = {}
        for name in ["linear_probe.pt", "finetuned_best.pt"]:
            path = directory / name
            checkpoints[name] = {
                "path_relative_to_study": str(path.relative_to(RUN)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        revision = next(
            row["base_revision"]
            for row in audit["checkpoint_checks"]
            if row["model"] == model and row["seed"] == 42
        )
        choice = selections[model, 42]
        models[model] = {
            "seed": 42,
            "base_revision": revision,
            "checkpoints": checkpoints,
            "evaluation_relative_to_study": str(
                (directory / "float32_evaluation").relative_to(RUN)
            ),
            "routes_selected_on_development": choice["task_routes"],
            "operating_thresholds": {
                task: results[model, 42][method]["dev"][task]["thresholds"]
                for task, method in choice["task_routes"].items()
            },
        }
        for task, method in choice["task_routes"].items():
            card_rows.append([label, task, method])
    bundle = {
        "schema_version": 1,
        "status": "research_candidates; generalization limitations documented",
        "study": RUN.name,
        "finalized_date": audit["finalized_date"],
        "manifest_sha256": audit["manifest_sha256"],
        "definitions_sha256": provenance["definitions_sha256"],
        "task_order": TASKS,
        "precision": "float32; TF32 disabled",
        "default_operating_point": "dev_max_f1",
        "seed_selection": "42 predesignated; 43/44 are replications, not test-based selection",
        "models": models,
    }
    (RUN / "model_bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    (RUN / "model_card.md").write_text(
        "# Trained CLIP research candidates\n\n"
        "Completed 16 October 2026. MobileCLIP2-S2 and SigLIP2-B/16-224 have "
        "five binary target heads and genuinely updated final visual layers. "
        "The [bundle index](model_bundle.json) records exact base revisions, "
        "checkpoint paths/SHA-256 hashes, frozen development routes and thresholds. "
        "These checkpoints store the trained parameters; loading also requires "
        "the pinned pretrained base model. They are not standalone full models.\n\n"
        "Seed 42 was designated before test evaluation; seeds 43 and 44 "
        "measure optimization variation. No checkpoint or route is selected "
        "using held-out performance. Final inference is float32 with TF32 disabled.\n\n"
        + table(["Backbone", "Objective", "Development-selected method"], card_rows)
        + "\n\n"
        "Use [the inference guide](../../docs/training.md) and the "
        "`anomaly-eval predict-clip` command. Static targets use the endpoint "
        "image. Loitering/unattended-item targets require four distinct views "
        "around the last 20 seconds, with timestamps. Waste dumping uses eight "
        "uniform views over a complete clip, so that result is an offline "
        "video-level decision. The interface returns unknown when the necessary "
        "input history is missing. Scores are logits, not calibrated probabilities.\n\n"
        "The default threshold maximizes development F1. It is retained for "
        "reproducibility, but transfers poorly to some test cameras: unattended "
        "false-positive rates can approach 100%. The predefined zero-logit and "
        "development-FPR-budget points are also reported, without test refitting. "
        "Choose and validate operating rules on separately labelled deployment "
        "development data before using alerts.\n\n"
        "[Full evaluation](report.md) documents gains on waste dumping and "
        "unattended-item ranking, mixed CCTV-smoking transfer, and unresolved "
        "loitering. Public-sleep evidence is only 18 test photos (six positives); "
        "those and unattended labels are provisional assistant review. Camera "
        "coverage is limited, and old CCTV sources were historically inspected. "
        "These classifiers provide scene/clip labels; they do not localize boxes, "
        "prove item ownership or infer housing status.\n\n"
        "[Protocol](../../reports/clip_training_protocol_20261015.md), "
        "[data metadata](../../data/clip_training_20261015/dataset.json), "
        "[provenance](provenance.json) and [verification](verification.json) "
        "describe source qualifications and reproducibility.\n"
    )


def group_bootstrap(rows, baseline, trained, seed=20261015, draws=2000):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["group_id"]].append(i)
    if len(groups) < 3:
        return {
            "groups": len(groups),
            "draws": 0,
            "warning": "Too few independent source groups for a useful interval",
            "ap_ci95": None,
            "delta_ap_ci95": None,
        }
    keys = list(groups)
    rng = np.random.default_rng(seed)
    y = np.array([r["target"] for r in rows])
    base = np.array(baseline)
    score = np.array(trained)
    ap = []
    difference = []
    for _ in range(draws):
        selected = rng.integers(0, len(keys), len(keys))
        idx = np.array([i for k in selected for i in groups[keys[k]]])
        if len(set(y[idx])) < 2:
            continue
        value = average_precision_score(y[idx], score[idx])
        original = average_precision_score(y[idx], base[idx])
        ap.append(float(value))
        difference.append(float(value - original))
    return {
        "groups": len(groups),
        "draws": len(ap),
        "seed": seed,
        "ap_ci95": np.quantile(ap, [0.025, 0.975]).tolist() if ap else None,
        "delta_ap_ci95": np.quantile(difference, [0.025, 0.975]).tolist()
        if difference
        else None,
        "warning": "Exploratory source-group interval; authors/recordings do not create new cameras",
    }


def main():
    state = read(RUN / "run.json")
    dataset = read(ROOT / "data/clip_training_20261015/dataset.json")
    final_fp32 = all(
        (RUN / f"seed_{seed}" / model / "float32_evaluation/state.json").exists()
        and read(RUN / f"seed_{seed}" / model / "float32_evaluation/state.json")[
            "status"
        ]
        == "complete"
        for model in NAMES
        for seed in [42, 43, 44]
    )
    precision = "float32" if final_fp32 else "bfloat16 autocast (initial evaluation)"
    complete = []
    numeric = []
    main_rows = []
    raw_rows = []
    points = []
    routes = []
    source_rows = []
    intervals = {}
    seeds_rows = []
    selections = {}
    results = {}
    for model, label in zip(NAMES, LABELS):
        for seed in [42, 43, 44]:
            directory = RUN / f"seed_{seed}" / model
            if final_fp32:
                directory = directory / "float32_evaluation"
            path = directory / "results.json"
            if not path.exists():
                continue
            value = read(path)
            choice = read(directory / "selection.json")
            complete.append((model, seed))
            results[model, seed] = value
            selections[model, seed] = choice
            for method, splits in value.items():
                for split, metrics in splits.items():
                    for task, record in metrics.items():
                        for op, metric in record["operating_points"].items():
                            numeric.append(
                                {
                                    "model": model,
                                    "precision": precision,
                                    "seed": seed,
                                    "method": method,
                                    "split": split,
                                    "task": task,
                                    "qualification": KINDS[task],
                                    "operating_point": op,
                                    "threshold": record["thresholds"][op],
                                    "groups": record["groups"],
                                    **metric,
                                }
                            )
            for task in TASKS:
                method = choice["task_routes"][task]
                primary = value[method]["test"][task]["operating_points"]["dev_max_f1"]
                seeds_rows.append(
                    [
                        label,
                        seed,
                        task,
                        method,
                        f(primary["ap"]),
                        f(primary["f1"]),
                        p(primary["recall"]),
                        p(primary["fpr"]),
                    ]
                )
            if seed != 42:
                continue
            for task in TASKS:
                method = choice["task_routes"][task]
                metrics = [
                    value[m]["test"][task]["operating_points"]["dev_max_f1"]
                    for m in ["zero_shot", "linear_probe", "finetuned"]
                ]
                chosen = value[method]["test"][task]
                c = chosen["operating_points"]["dev_max_f1"]
                main_rows.append(
                    [
                        label,
                        task,
                        c["n"],
                        f(metrics[0]["ap"]),
                        f(metrics[1]["ap"]),
                        f(metrics[2]["ap"]),
                        method,
                        f(c["ap"]),
                    ]
                )
                routes.append(
                    [
                        label,
                        task,
                        method,
                        p(c["precision"]),
                        p(c["recall"]),
                        f(c["f1"]),
                        p(c["fpr"]),
                    ]
                )
                raw = chosen["operating_points"]["raw"]
                baseline = value["zero_shot"]["test"][task]["operating_points"]["raw"]
                raw_rows.append(
                    [
                        label,
                        task,
                        method,
                        p(raw["precision"]),
                        p(raw["recall"]),
                        f(raw["f1"]),
                        p(raw["fpr"]),
                        f(baseline["f1"]),
                    ]
                )
                for op in ["raw", "dev_max_f1", "dev_fpr_budget"]:
                    m = chosen["operating_points"][op]
                    points.append(
                        [
                            label,
                            task,
                            op,
                            p(m["precision"]),
                            p(m["recall"]),
                            f(m["f1"]),
                            p(m["fpr"]),
                            f(chosen["thresholds"][op]),
                        ]
                    )
                for source, sources in chosen["sources"].items():
                    m = sources["dev_max_f1"]
                    source_rows.append(
                        [
                            label,
                            task,
                            source,
                            m["n"],
                            f(m["ap"]),
                            f(m["f1"]),
                            p(m["recall"]),
                            p(m["fpr"]),
                        ]
                    )
                base_rows = [
                    r
                    for r in prediction_rows(directory, "zero_shot", "test")
                    if r["task"] == task
                ]
                tuned = {r["id"]: r for r in prediction_rows(directory, method, "test")}
                assert all(r["id"] in tuned for r in base_rows)
                base = [r["scores"][task] for r in base_rows]
                adapted = [tuned[r["id"]]["scores"][task] for r in base_rows]
                intervals[model + "/" + task] = group_bootstrap(
                    base_rows, base, adapted
                )
            if "external" in value["zero_shot"]:
                for method in ["zero_shot", "linear_probe", "finetuned"]:
                    m = value[method]["external"]["smoking"]["operating_points"][
                        "dev_max_f1"
                    ]
                    source_rows.append(
                        [
                            label,
                            "smoking",
                            f"stanford40 official test / {method}",
                            m["n"],
                            f(m["ap"]),
                            f(m["f1"]),
                            p(m["recall"]),
                            p(m["fpr"]),
                        ]
                    )
    seed_summary = []
    for model, label in zip(NAMES, LABELS):
        for task in TASKS:
            values = []
            for seed in [42, 43, 44]:
                if (model, seed) not in results:
                    continue
                method = selections[model, seed]["task_routes"][task]
                values.append(
                    results[model, seed][method]["test"][task]["operating_points"][
                        "raw"
                    ]
                )
            if not values:
                continue

            def spread(key, values=values):
                v = [x[key] for x in values]
                if key == "fpr":
                    return (
                        f"{100 * np.mean(v):.1f}% ± {100 * np.std(v, ddof=1):.1f}%"
                        if len(v) > 1
                        else p(v[0])
                    )
                return (
                    f"{np.mean(v):.3f} ± {np.std(v, ddof=1):.3f}"
                    if len(v) > 1
                    else f(v[0])
                )

            seed_summary.append(
                [label, task, len(values), spread("ap"), spread("f1"), spread("fpr")]
            )
    findings = []
    for model, label in zip(NAMES, LABELS):
        if (model, 42) not in results:
            continue
        value = results[model, 42]
        choice = selections[model, 42]
        u = value[choice["task_routes"]["unattended_items"]]["test"][
            "unattended_items"
        ]["operating_points"]
        baseline = value["zero_shot"]["test"]["unattended_items"]["operating_points"][
            "raw"
        ]
        litter = value[choice["task_routes"]["littering"]]["test"]["littering"][
            "operating_points"
        ]["raw"]
        findings.append(
            f"{label}: unattended-item AP changes from {f(baseline['ap'])} to {f(u['raw']['ap'])}; the fixed zero-logit decision gives F1 {f(u['raw']['f1'])}, recall {p(u['raw']['recall'])} and FPR {p(u['raw']['fpr'])}. The same model's littering/waste-dumping route gives AP {f(litter['ap'])}, raw F1 {f(litter['f1'])} and FPR {p(litter['fpr'])}. These are the tested source conditions, with provisional unattended labels."
        )
    if ("siglip2-b16-224", 42) in results:
        value = results["siglip2-b16-224", 42]
        choice = selections["siglip2-b16-224", 42]
        smoke = value[choice["task_routes"]["smoking"]]
        a = smoke["test"]["smoking"]["operating_points"]["raw"]
        b = smoke["external"]["smoking"]["operating_points"]["raw"]
        findings.append(
            f"Smoking transfer is mixed: SigLIP's selected route has AP {f(a['ap'])} on held-out CCTV camera c1 and {f(b['ap'])} on the official photo holdout. Photo-domain gains do not establish the same improvement on distant CCTV views. Inspect source-specific results rather than a pooled accuracy."
        )
    for model, label in zip(NAMES, LABELS):
        if (model, 42) not in results:
            continue
        value = results[model, 42]
        choice = selections[model, 42]
        comparison = []
        for task in ["smoking", "loitering"]:
            baseline = value["zero_shot"]["test"][task]["operating_points"]["raw"]
            routed = value[choice["task_routes"][task]]["test"][task][
                "operating_points"
            ]["raw"]
            comparison.append(f"{task} AP {f(baseline['ap'])} → {f(routed['ap'])}")
        findings.append(
            f"{label} does not improve every objective: "
            + "; ".join(comparison)
            + ". Development selection cannot prevent these source-transfer failures. In particular, the selected loitering routes are below the held-out positive prevalence (11/75 = 0.147); loitering remains unresolved."
        )
    if ("siglip2-b16-224", 42) in results:
        value = results["siglip2-b16-224", 42]
        choice = selections["siglip2-b16-224", 42]
        points_u = value[choice["task_routes"]["unattended_items"]]["test"][
            "unattended_items"
        ]["operating_points"]
        findings.append(
            f"SigLIP unattended-item threshold transfer fails: the maximum-development-F1 point yields test FPR {p(points_u['dev_max_f1']['fpr'])}, and the development 5% FPR-budget point yields {p(points_u['dev_fpr_budget']['fpr'])}. The improved ranking and the predefined zero-logit result do not justify deploying the default development threshold on new cameras. New-camera calibration requires separate labelled development data."
        )
    findings.extend(
        [
            "Score ranking and threshold transfer are different findings. Development-fitted thresholds can produce almost universal alarms on a new camera even when AP improves. The report retains the predefined raw zero-logit point and both development operating points; no threshold is refitted using test labels.",
            "Public-sleep photo scores concern only 18 held-out photos, including six positives, with provisional visual labels. Loitering uses a single camera, sparse global-view context and few positive held-out windows. These studies do not establish deployment readiness; learned global CLIP features do not provide a person trajectory or prove ownership/duration.",
        ]
    )
    lines = [
        "# Trained CLIP model evaluation",
        "",
        f"Study status: **{state['status']}**; completed model/seed evaluations: **{len(complete)}/6**. Study identifier: 20261015. No latency experiment is included.",
        "",
        f"Final quality precision: **{precision}**. Training used bfloat16 autocast with float32 weights. Float32 re-evaluation disables TF32, keeps the trained weights and development-selected methods fixed, and refits operating thresholds on development scores only. Original AMP evaluation artifacts remain alongside the float32 artifacts.",
        "",
        "This study trains MobileCLIP2-S2 and SigLIP2-B/16-224 for the five existing target definitions. It compares pretrained last-frame text scores, learned frozen-feature temporal probes, and actual last-stage visual fine-tuning. The candidate routing choice is frozen **per target on development data**, never chosen from test performance. All methods are reported, including unsuccessful changes.",
        "",
        "## Main findings",
        "",
        "\n\n".join(findings),
        "",
        "## Data and generalization design",
        "",
        table(
            [
                "Task",
                "Train 0/1",
                "Dev 0/1",
                "Test 0/1",
                "Additional holdout 0/1",
                "Interpretation",
            ],
            [
                [
                    t,
                    " / ".join(
                        str(dataset["counts"][t].get(f"train/{y}", 0)) for y in [0, 1]
                    ),
                    " / ".join(
                        str(dataset["counts"][t].get(f"dev/{y}", 0)) for y in [0, 1]
                    ),
                    " / ".join(
                        str(dataset["counts"][t].get(f"test/{y}", 0)) for y in [0, 1]
                    ),
                    " / ".join(
                        str(dataset["counts"][t].get(f"external/{y}", 0))
                        for y in [0, 1]
                    ),
                    KINDS[t],
                ]
                for t in TASKS
            ],
        ),
        "",
        "`0` means target-specific normal; `1` means the named target. Unknown objectives are masked, not assigned normal labels. Every held-out example uses exactly the same input views across learned methods. The pretrained baseline uses the endpoint frame; probe/fine-tuning additionally use ordered temporal pooling. Comparisons with earlier reports also change splits, training supervision and input protocol; they are not isolated backbone-only gains.",
        "",
        "CCTV smoking holds camera c1 out from training (c4) and development (c2). Stanford40 contributes official-training images and hard negatives; its official test images remain separate, after perceptual-neighbour exclusions and removal of prior debug-test images. It is an additional photo-domain image holdout, not unseen-dataset transfer. IWDD retains its official test; related training video groups are excluded by first/last-frame perceptual hashes. Whole Street Scene recordings and whole photo authors/perceptual groups stay together. Street Scene has one camera; the unattended holdout has two camera families. No universal deployment-generalization claim follows from these limited sources.",
        "",
        "Unattended and public-sleep photograph labels remain **provisional assistant review**, not independently human verified. The synthetic lying dataset is not used as rough-sleeping event supervision. Waste dumping is broader than small pedestrian littering. These qualifications apply to every corresponding score below.",
        "",
        "## Primary seed: all candidate test AP scores",
        "",
        table(
            [
                "Model",
                "Target",
                "N",
                "Pretrained AP",
                "Probe AP",
                "Fine-tuned AP",
                "Dev-selected route",
                "Route AP",
            ],
            main_rows,
        )
        if main_rows
        else "Training is in progress.",
        "",
        "AP is threshold-free; compare it with each cohort's positive prevalence. All routes are selected before computing held-out metrics. A higher test value in another column does not retrospectively change the selected route. Seed 42 is the primary candidate; extra seeds test optimization variability.",
        "",
        "## Development-selected route: test operating point",
        "",
        table(
            ["Model", "Target", "Method", "Precision", "Recall", "F1", "FPR"], routes
        ),
        "",
        "## Fixed zero-logit decisions",
        "",
        "The zero-logit point was evaluated for every candidate before test inspection. These are raw binary decisions, alongside the development-fitted thresholds above; this table does not refit a threshold on test or change the development-selected route.",
        "",
        table(
            [
                "Model",
                "Target",
                "Dev-selected method",
                "Precision",
                "Recall",
                "F1",
                "FPR",
                "Pretrained raw F1",
            ],
            raw_rows,
        ),
        "",
        "## Threshold tradeoffs",
        "",
        table(
            [
                "Model",
                "Target",
                "Operating point",
                "Test precision",
                "Test recall",
                "Test F1",
                "Test FPR",
                "Threshold",
            ],
            points,
        ),
        "",
        "The FPR-budget point constrains empirical development FPR to at most 5%. It is not a guarantee for new sources, nor an alerts/hour estimate. Scores are logits, not calibrated event probabilities. Few positive sources and small control counts can make apparently excellent results unstable.",
        "",
        "## Source-specific and photo-holdout results",
        "",
        table(
            ["Model", "Target", "Source", "N", "AP", "F1", "Recall", "FPR"], source_rows
        ),
        "",
        "## Replication across training seeds",
        "",
        table(
            [
                "Model",
                "Target",
                "Seeds",
                "Routed AP mean ± SD",
                "Raw F1 mean ± SD",
                "Raw FPR mean ± SD",
            ],
            seed_summary,
        ),
        "",
        "The summary uses each seed's development-selected route with the same predefined zero-logit decision. It measures optimization/routing variation on the same sources, not population uncertainty. The detailed rows below use maximum-development-F1 thresholds.",
        "",
        table(
            [
                "Model",
                "Seed",
                "Target",
                "Dev-selected method",
                "Test AP",
                "Test F1",
                "Recall",
                "FPR",
            ],
            seeds_rows,
        ),
        "",
        "Seeds 43 and 44 use seed 42's development-selected visual learning rate. Repeating seeds does not add independent cameras, people, or events. No seed is selected for deployment using its test results.",
        "",
        "## Source-group uncertainty for the primary routed AP",
        "",
        table(
            [
                "Model / target",
                "Groups",
                "AP 95% interval",
                "Paired AP gain 95% interval",
            ],
            [
                [
                    key,
                    v["groups"],
                    "—"
                    if v["ap_ci95"] is None
                    else " to ".join(f(x) for x in v["ap_ci95"]),
                    "—"
                    if v["delta_ap_ci95"] is None
                    else " to ".join(f(x) for x in v["delta_ap_ci95"]),
                ]
                for key, v in intervals.items()
            ],
        ),
        "",
        "Bootstrap uses 2,000 paired source-group draws, seed 20261015. Intervals are omitted with fewer than three groups. Photographer/recording groups do not establish unseen-camera independence; single-camera loitering and small photo studies remain preliminary.",
        "",
        "## Training, artifacts and reproduction",
        "",
        "Training uses masked binary loss; task/class/source/group-balanced sampling; mild brightness/contrast/colour jitter and flip; frozen text/early vision and batch-normalization statistics; feature distillation and anchoring to the pretrained visual weights. Fine-tuning updates 18.1M MobileCLIP visual parameters or 14.2M SigLIP visual parameters plus classifier heads. Checkpoints retain the pinned base revision and trained tensors. Source hashes, ordered view hashes, exact commands, development histories and raw held-out scores are retained. Training is unpaced CUDA with float32 weights and bfloat16 autocast.",
        "",
        "- [Pre-registered protocol](../../reports/clip_training_protocol_20261015.md) and [training guide](../../docs/training.md).",
        "- [Study state and commands](run.json), [frozen dataset metadata](../../data/clip_training_20261015/dataset.json), [IWDD acquisition/split audit](../../data/clip_training_20261015/iwdd_source_audit.json), [photo source audit](../../data/clip_training_20261015/commons_source_audit.json).",
        "- Per-model/seed `linear_probe.pt`, `finetuned_best.pt`, `selection.json`, `results.json`, `history.json` and `*_predictions.jsonl` live under `seed_42/`, `seed_43/`, `seed_44/`.",
        "- [Full numeric export](comparison.csv), [group bootstrap results](group_uncertainty.json), [AP figure](figures/test_ap.png) and [training curves](figures/training_curves.png).",
        "- [Checkpoint/metric verification](verification.json), [study provenance](provenance.json), and the per-model `float32_evaluation/` folders contain final precision-consistent scores and operating points.",
        "- [Research model card](model_card.md) and [checkpoint bundle index](model_bundle.json) record the primary checkpoints, hashes, development routes, input requirements and operating limitations.",
        "",
        "These are scene/clip classifiers, not box-localization detectors. Independent label review, more cameras and matched real deployment events are still required, particularly for rough sleeping and loitering. Duration or ownership cannot be established from an isolated image; the prediction interface returns unknown for duration-dependent targets without timestamped past context.",
        "",
    ]
    (RUN / "report.md").write_text("\n".join(lines))
    if numeric:
        with (RUN / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(numeric[0]))
            writer.writeheader()
            writer.writerows(numeric)
    (RUN / "group_uncertainty.json").write_text(json.dumps(intervals, indent=2) + "\n")
    (RUN / "report_data.json").write_text(
        json.dumps(
            {
                "completed": len(complete),
                "precision": precision,
                "main": main_rows,
                "routes": routes,
                "raw_routes": raw_rows,
                "seeds": seeds_rows,
                "sources": source_rows,
            },
            indent=2,
        )
        + "\n"
    )
    if final_fp32 and (RUN / "verification.json").exists():
        write_model_bundle(selections, results)
    if len([x for x in complete if x[1] == 42]) == 2:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        directory = RUN / "figures"
        directory.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for axis, model, label in zip(axes, NAMES, LABELS):
            values = results[model, 42]
            x = np.arange(len(TASKS))
            for offset, method, color in [
                (-0.24, "zero_shot", "#8799a5"),
                (0, "linear_probe", "#007e80"),
                (0.24, "finetuned", "#c86535"),
            ]:
                ap = [
                    values[method]["test"][t]["operating_points"]["raw"]["ap"]
                    for t in TASKS
                ]
                axis.bar(
                    x + offset, ap, 0.23, label=method.replace("_", " "), color=color
                )
            n = [
                values["zero_shot"]["test"][t]["operating_points"]["raw"]["n"]
                for t in TASKS
            ]
            prevalence = [
                values["zero_shot"]["test"][t]["operating_points"]["raw"]["positive"]
                / count
                for t, count in zip(TASKS, n)
            ]
            axis.scatter(
                x,
                prevalence,
                marker="_",
                s=360,
                color="#142c40",
                zorder=5,
                label="Positive prevalence",
            )
            axis.set_xticks(
                x,
                [t.replace("_", "\n") + f"\n(n={count})" for t, count in zip(TASKS, n)],
                fontsize=9,
            )
            axis.set_ylim(0, 1.05)
            axis.set_title(label)
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        axes[0].set_ylabel("Held-out average precision")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=4,
            fontsize=9,
        )
        fig.suptitle(
            "Primary seed: all fixed candidate methods; routes selected on development only"
        )
        fig.text(
            0.5,
            0.012,
            "Unattended/public-sleep labels are provisional; loitering uses one camera; rough sleeping has only six positive test photos.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=[0, 0.055, 1, 0.88])
        fig.savefig(directory / "test_ap.png", dpi=180)
        fig.savefig(directory / "test_ap.pdf")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for axis, model, label in zip(axes, NAMES, LABELS):
            history = read(RUN / "seed_42" / model / "history.json")
            for lr in sorted({x["learning_rate"] for x in history}):
                selected = [x for x in history if x["learning_rate"] == lr]
                axis.plot(
                    [x["epoch"] for x in selected],
                    [x["dev_macro_ap"] for x in selected],
                    marker="o",
                    label=f"lr={lr:g}",
                )
            axis.set_title(label)
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.2)
            axis.legend()
        axes[0].set_ylabel("Development macro task/source AP")
        fig.tight_layout()
        fig.savefig(directory / "training_curves.png", dpi=180)
        plt.close(fig)
    print("Report updated", len(complete), "/6 model/seed evaluations", flush=True)


if __name__ == "__main__":
    main()
