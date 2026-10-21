"""Export matched original/LoRA/routed metrics, findings and an inference bundle."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/capstone_smolvlm_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from anomaly_eval.training.clip import save_json
from anomaly_eval.training.data import TASKS
from anomaly_eval.training.smolvlm import ROOT, SPEC, digest


def read(path):
    return json.loads(path.read_text())


def pct(value):
    return f"{100 * value:.1f}%"


def metric_cells(study, selection):
    routes = {"original": "original", "joint": selection["joint_adapter"]}
    cells = []
    for kind in ("test", "external", "additional_transfer"):
        base = (
            study / "evaluation"
            if kind != "additional_transfer"
            else study / "additional_transfer/evaluation"
        )
        required = set(selection["task_routes"].values()) | set(routes.values())
        all_results = {}
        for route in required:
            path = base / route.replace("/", "__") / "results.json"
            if path.exists():
                results = read(path)
                all_results[route] = (
                    results[kind] if kind != "additional_transfer" else results
                )
        if "original" not in all_results:
            continue
        for task in TASKS:
            if task not in all_results["original"]:
                continue
            for source in all_results["original"][task]["sources"]:
                for method, route in {
                    **routes,
                    "routed": selection["task_routes"][task],
                }.items():
                    values = all_results[route][task]["sources"][source]
                    cells.append(
                        {
                            "seed": selection["seed"],
                            "cohort": kind,
                            "task": task,
                            "source": source,
                            "method": method,
                            "route": route,
                            "ap": values["raw"]["ap"],
                            "auroc": values["raw"]["auroc"],
                            "n": values["raw"]["n"],
                            "positive": values["raw"]["positive"],
                            "constant_score_ap": values["raw"]["positive"]
                            / values["raw"]["n"],
                            "f1_at_raw": values["raw"]["f1"],
                            "recall_at_raw": values["raw"]["recall"],
                            "fpr_at_raw": values["raw"]["fpr"],
                            "f1_at_dev_f1": values["dev_max_f1"]["f1"],
                            "recall_at_dev_f1": values["dev_max_f1"]["recall"],
                            "fpr_at_dev_f1": values["dev_max_f1"]["fpr"],
                            "recall_at_dev_5pct": values["dev_fpr_budget"]["recall"],
                            "fpr_at_dev_5pct": values["dev_fpr_budget"]["fpr"],
                            "precision_at_dev_5pct": values["dev_fpr_budget"][
                                "precision"
                            ],
                            "tp_at_dev_5pct": values["dev_fpr_budget"]["tp"],
                            "fp_at_dev_5pct": values["dev_fpr_budget"]["fp"],
                        }
                    )
    return cells


def compare_existing_clip(study, primary_cells):
    """Use existing primary CLIP routes only, after checking exact held-out IDs."""
    records = [
        json.loads(s) for s in (study / "samples.jsonl").read_text().splitlines()
    ]
    transfer = study / "additional_transfer/samples.jsonl"
    if transfer.exists():
        records += [json.loads(s) for s in transfer.read_text().splitlines()]
    labels = {
        r["id"]: (r["task"], r["target"], r["source"])
        for r in records
        if r["split"] in {"test", "external"}
    }
    values, inputs = {}, {}
    source_cases = {
        "hmdb51_action_transfer": "hmdb",
        "commons_new_author_transfer": "rough",
        "meva_abandon_action_transfer": "meva",
        "kaggle_sujaykapadnis_smoking": "photo",
    }
    for model in ("mobileclip2-s2", "siglip2-b16-224"):
        directory = (
            ROOT / "results/strategy_study_20261016" / model / "transfer_evaluation"
        )
        metric_path = directory / "per_source_results.json"
        sources = read(metric_path)
        inputs[str(metric_path.relative_to(ROOT))] = digest(metric_path)
        predicted = {}
        for case in ("historical", "photo", "hmdb", "rough", "meva"):
            path = directory / f"{case}_predictions.jsonl"
            for line in path.read_text().splitlines():
                record = json.loads(line)
                if record["family"] == "selected":
                    predicted[record["id"]] = (
                        record["task"],
                        record["target"],
                        record["source"],
                    )
        assert predicted == labels, (
            "Context comparison must use identical examples and labels"
        )
        for cell in primary_cells:
            if cell["method"] != "routed":
                continue
            case = source_cases.get(cell["source"], "historical")
            key = f"{case}/{cell['task']}/{cell['source']}"
            metric = sources[key]["selected"]["raw"]
            assert (metric["n"], metric["positive"]) == (cell["n"], cell["positive"])
            values.setdefault((cell["cohort"], cell["task"], cell["source"]), {})[
                model
            ] = metric["ap"]
    save_json(
        study / "clip_comparison_provenance.json",
        {
            "inputs": inputs,
            "exact_heldout_labels_checked": len(labels),
            "qualification": "Existing 16 October primary CLIP strategy routes on identical training/development/held-out splits; crop, representation, resolution and optimization differ from SmolVLM. Later smoking video heads use additional HMDB training and are a separate protocol.",
        },
    )
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=ROOT / "results/smolvlm_finetuning_20261019"
    )
    args = parser.parse_args()
    study = args.study
    selections = [read(p) for p in sorted(study.glob("seed_*/selection.json"))]
    primary = next(s for s in selections if s["seed"] == 42)
    mass_statistics = {}
    for method, directory in {
        "original": study / "baseline",
        "joint": study / primary["joint_adapter"],
    }.items():
        values = [
            json.loads(s)["yes_no_token_mass"]
            for s in (directory / "dev_predictions.jsonl").read_text().splitlines()
        ]
        mass_statistics[method] = {
            "development_min": min(values),
            "development_median": float(np.median(values)),
        }
    save_json(study / "response_mass.json", mass_statistics)
    cells = [
        cell for selection in selections for cell in metric_cells(study, selection)
    ]
    if not cells:
        raise ValueError("Complete held-out inference before reporting results")
    verification = read(study / "verification.json")
    assert verification["status"] == "passed"
    protocol = read(study / "protocol.json")
    save_json(
        study / "report_data.json",
        {
            "protocol": protocol,
            "selections": selections,
            "cells": cells,
            "verification": verification,
        },
    )
    with (study / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    # Package only routes selected by development data, with explicit pinned base.
    bundle = {
        "status": "research_inference_configuration",
        "model": SPEC.checkpoint,
        "revision": SPEC.revision,
        "base_sha256": protocol["checkpoint_sha256"],
        "primary_seed": 42,
        "joint_adapter": primary["joint_adapter"],
        "task_routes": primary["task_routes"],
        "thresholds": primary["thresholds"],
        "adapter_weights": {
            route: {
                "path": route + "/adapter",
                "sha256": sha,
                "config_sha256": digest(study / route / "adapter/adapter_config.json"),
            }
            for route, sha in primary["adapters_sha256"].items()
        },
        "protocol_sha256": digest(study / "protocol.json"),
        "inference": "bash scripts/smolvlm_gpu.sh predict --task TASK --images ...",
        "qualification": protocol["qualification"],
    }
    save_json(study / "model_bundle.json", bundle)
    repeated_path = study / "repeated_gain_routes.json"
    if repeated_path.exists():
        repeated = read(repeated_path)
        bundle["optional_repeated_gain_routes"] = repeated["task_routes"]
        bundle["optional_repeated_gain_qualification"] = repeated["qualification"]
        save_json(study / "model_bundle.json", bundle)
    primary_cells = [c for c in cells if c["seed"] == 42]
    context = compare_existing_clip(study, primary_cells)
    keys = list(
        dict.fromkeys((c["cohort"], c["task"], c["source"]) for c in primary_cells)
    )
    lookup = {
        (c["cohort"], c["task"], c["source"], c["method"]): c for c in primary_cells
    }
    figures = study / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, max(5, 0.55 * len(keys))))
    y = np.arange(len(keys))
    for offset, method, color in [
        (-0.24, "original", "#778899"),
        (0, "joint", "#e59c42"),
        (0.24, "routed", "#2789a4"),
    ]:
        ax.barh(
            y + offset,
            [lookup[*key, method]["ap"] for key in keys],
            height=0.22,
            label=method,
            color=color,
        )
    ax.set_yticks(y, [f"{task}: {source}" for _, task, source in keys], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Average precision; identical inputs and binary scoring")
    ax.set_title("SmolVLM-500M: original vs development-selected adaptation (seed 42)")
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"matched_ap.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for selection in selections:
        for route in selection["adapters_sha256"]:
            history = read(study / route / "history.json")
            epochs = [entry["epoch"] for entry in history]
            axes[0].plot(
                epochs, [entry["loss"] for entry in history], "o-", label=route
            )
            axes[1].plot(
                epochs, [entry["dev_macro_ap"] for entry in history], "o-", label=route
            )
    axes[0].set_ylabel("Training first-response cross entropy")
    axes[1].set_ylabel("Development macro source AP")
    axes[1].axhline(
        primary["development_macro_ap"]["original"],
        color="gray",
        linestyle="--",
        label="Original model",
    )
    for ax in axes:
        ax.set_xlabel("Completed epoch")
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"learning_curves.{suffix}", dpi=180)
    plt.close(fig)
    rows = [json.loads(s) for s in (study / "samples.jsonl").read_text().splitlines()]
    lines = [
        "# SmolVLM fine-tuning for all five anomaly targets — 19 October 2026",
        "",
        (
            "Actual CUDA LoRA training of the pinned SmolVLM-500M model is complete. "
            "All five objectives share adapted language weights; development-selected target routes are reported separately. "
            "Original and adapted scores use identical task queries, 512-pixel images, ordered views and frozen visual tokens. "
            "This is retrospective source-held-out evaluation, not a fresh deployment test."
        ),
        "",
        "## Data and supervision",
        "",
        "| Target | Train N / positive | Dev N / positive | Test N / positive | External N / positive |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        values = []
        for split in ("train", "dev", "test", "external"):
            selected = [r for r in rows if r["task"] == task and r["split"] == split]
            values.append(f"{len(selected)} / {sum(r['target'] for r in selected)}")
        lines.append("| " + " | ".join([task, *values]) + " |")
    lines.extend(
        [
            "",
            (
                f"There are {protocol['examples']:,} main examples and {protocol['unique_images']:,} distinct decoded images. "
                "Additional transfer uses 906 HMDB action clips, 28 new-author public-sleep photos and 27 MEVA action windows; "
                "none enter training or selection. Photo and camera groups remain disjoint, and decoded pixels are checked. "
                "A target-negative annotation never labels the other four objectives as absent."
            ),
            "",
            (
                "Unattended items have two training source groups, one development group and two historical test groups; "
                "loitering uses different recordings from a single camera. Rough-sleeping labels remain provisional visible-sleep annotations, "
                "with no inference about housing status. HMDB controls are other primary actions, not exhaustive smoking-negative verification. "
                "MEVA labels describe abandonment action windows, not persistent unattended-item state."
            ),
            "",
            "## Optimization, selection and reproducibility",
            "",
            (
                "The vision tower, multimodal connector, token embeddings and vocabulary output head stay frozen. "
                "LoRA trains 4,341,760 language-projection parameters (rank 8, alpha 16, dropout 0.05), using full-vocabulary "
                "first-answer-token cross entropy for ` Yes`/` No`. All input-token loss positions are excluded. "
                "Each epoch visits all 3,973 training examples, then an equally sized task/class/source/group-balanced resampled pass. "
                "AdamW uses weight decay 0.01, 5% warmup, linear learning-rate decay, gradient clipping 1.0, "
                "microbatch 8 and accumulation 4. Training is bfloat16; all development selection and final inference are float32 with TF32 off."
            ),
            "",
            (
                "Primary seed 42 compares learning rates 0.00003 and 0.0001, with at most six epochs and patience two. "
                "The selection metric averages AP by dataset source within each target, then equally over the five targets. "
                "The best joint adapter and per-task original/adapter routes are frozen before held-out predictions. "
                "A second seed, when present, repeats only the primary winning learning rate; it never replaces the primary based on test results."
            ),
            "",
            "| Seed | Candidate | Best epoch | Completed epochs | Development macro source AP |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for selection in selections:
        for route in selection["adapters_sha256"]:
            state = read(study / route / "state.json")
            lines.append(
                f"| {selection['seed']} | {route} | {state['best_epoch']} | {state['epochs_completed']} | {state['best_dev_macro_ap']:.4f} |"
            )
    lines.extend(
        [
            "",
            "![Training loss and development checkpoint selection](figures/learning_curves.png)",
            "",
            (
                f"Matched original development macro source AP: **{primary['development_macro_ap']['original']:.4f}**. "
                f"Primary joint adapter: `{primary['joint_adapter']}` "
                f"(**{primary['development_macro_ap'][primary['joint_adapter']]:.4f}** development macro source AP)."
            ),
            "",
            "| Target | Development-selected route | Development source AP |",
            "|---|---|---:|",
        ]
    )
    for task, route in primary["task_routes"].items():
        lines.append(
            f"| {task} | {route} | {primary['route_dev_source_ap'][task]:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                f"Response-format check on development data: original Yes+No token mass has minimum {pct(mass_statistics['original']['development_min'])} "
                f"and median {pct(mass_statistics['original']['development_median'])}; selected joint LoRA has minimum {pct(mass_statistics['joint']['development_min'])} "
                f"and median {pct(mass_statistics['joint']['development_median'])}. Both already concentrate probability on the required binary responses."
            ),
            "",
            "## Matched held-out results",
            "",
            (
                "AP measures ranking, not false-alert control. The original, joint adapter and per-target route receive the same inputs. "
                "Routing chooses development AP, so it can regress on held-out sources. All regressions below are retained."
            ),
            "",
            "| Target / source | N / positive | Original AP | Joint LoRA AP | Routed AP |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in keys:
        original = lookup[*key, "original"]
        joint = lookup[*key, "joint"]
        routed = lookup[*key, "routed"]
        lines.append(
            f"| {key[1]} / {key[2]} | {original['n']} / {original['positive']} | {original['ap']:.4f} | {joint['ap']:.4f} | {routed['ap']:.4f} |"
        )
    loiter = next(key for key in keys if key[1] == "loitering")
    unattended = next(
        key for key in keys if key[1] == "unattended_items" and key[0] == "test"
    )
    lines.extend(
        [
            "",
            (
                f"The primary routed configuration does not improve every objective: loitering AP changes from {lookup[*loiter, 'original']['ap']:.4f} "
                f"to {lookup[*loiter, 'routed']['ap']:.4f}. Unattended-item ranking improves, but the development-budget threshold gives "
                f"{pct(lookup[*unattended, 'routed']['fpr_at_dev_5pct'])} held-out FPR at {pct(lookup[*unattended, 'routed']['recall_at_dev_5pct'])} recall. "
                "These results do not justify replacing an operational alert pipeline with this research adapter."
            ),
        ]
    )
    cctv = next(key for key in keys if key[2] == "smoking_cctv")
    lines.extend(
        [
            "",
            (
                f"AP is ranking performance, not classification accuracy. A constant-score baseline has AP equal to the positive fraction. "
                f"For CCTV smoking that reference is {lookup[*cctv, 'original']['constant_score_ap']:.4f}, "
                f"versus {lookup[*cctv, 'routed']['ap']:.4f} for the routed adapter: the improvement is small and this remains a weak CCTV detector."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "![Matched original and adapted AP](figures/matched_ap.png)",
            "",
            "## Threshold transfer",
            "",
            (
                "Each method has its own development-only threshold. The following point allows at most 5% pooled development FPR "
                "for that target; the held-out FPR can be much higher. It is not a 5% deployment guarantee or an event false-alert rate."
            ),
            "",
            "| Target / source | Original recall | Original FPR | Routed recall | Routed FPR | Routed TP / FP |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in keys:
        a, b = lookup[*key, "original"], lookup[*key, "routed"]
        lines.append(
            f"| {key[1]} / {key[2]} | {pct(a['recall_at_dev_5pct'])} | {pct(a['fpr_at_dev_5pct'])} | {pct(b['recall_at_dev_5pct'])} | {pct(b['fpr_at_dev_5pct'])} | {b['tp_at_dev_5pct']} / {b['fp_at_dev_5pct']} |"
        )
    lines.extend(
        [
            "",
            "The predeclared raw threshold is margin zero: choose Yes when its logit is at least No's logit, without a fitted cutoff.",
            "",
            "| Target / source | Original raw recall | Original raw FPR | Routed raw recall | Routed raw FPR | Routed raw F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in keys:
        a, b = lookup[*key, "original"], lookup[*key, "routed"]
        lines.append(
            f"| {key[1]} / {key[2]} | {pct(a['recall_at_raw'])} | {pct(a['fpr_at_raw'])} | {pct(b['recall_at_raw'])} | {pct(b['fpr_at_raw'])} | {b['f1_at_raw']:.4f} |"
        )
    if len(selections) > 1:
        lines.extend(
            [
                "",
                "## Seed stability",
                "",
                "| Target / source | Seed 42 joint AP | Seed 43 joint AP |",
                "|---|---:|---:|",
            ]
        )
        other = {
            (c["cohort"], c["task"], c["source"]): c
            for c in cells
            if c["seed"] == 43 and c["method"] == "joint"
        }
        for key in keys:
            if key in other:
                lines.append(
                    f"| {key[1]} / {key[2]} | {lookup[*key, 'joint']['ap']:.4f} | {other[key]['ap']:.4f} |"
                )
    lines.extend(
        [
            "",
            "## Context against the previous five-target CLIP strategies",
            "",
            (
                "These are the existing 16 October primary development-selected CLIP routes, using identical training/development splits and exactly matched held-out IDs/labels. "
                "Crops, resolution, representation and training method differ, so this is a pipeline comparison; the original SmolVLM remains the controlled weight-adaptation baseline. "
                "Later smoking video heads use additional HMDB training and a different input protocol and are documented separately, rather than mixed into this same-cohort table."
            ),
            "",
            "| Target / source | MobileCLIP strategy AP | SigLIP strategy AP | SmolVLM routed AP |",
            "|---|---:|---:|---:|",
        ]
    )
    for key in keys:
        prior = context[key]
        lines.append(
            f"| {key[1]} / {key[2]} | {prior['mobileclip2-s2']:.4f} | {prior['siglip2-b16-224']:.4f} | {lookup[*key, 'routed']['ap']:.4f} |"
        )
    if repeated_path.exists():
        lines.extend(
            [
                "",
                "## Optional routing that requires development gains in both seeds",
                "",
                repeated["qualification"],
                "",
                repeated["rule"],
                "",
                "| Target | Primary development AP gain | Second-seed development AP gain | Optional route |",
                "|---|---:|---:|---|",
            ]
        )
        for task in TASKS:
            gains = repeated["development_gains"][task]
            lines.append(
                f"| {task} | {gains['primary_development_ap_gain']:+.4f} | {gains['secondary_development_ap_gain']:+.4f} | {repeated['task_routes'][task]} |"
            )
        lines.extend(
            [
                "",
                (
                    "This option leaves the original primary results above unchanged. On a rejected target it reuses the original model scores and original development thresholds; on other targets it retains the primary route. "
                    "It does not solve camera shift, temporal evidence or the unattended-item false-positive problem. Run with `predict --repeated-gains`; do not treat this retrospective rule as an independently validated deployment policy."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Verification and limits",
            "",
            (
                f"Verification passed: {verification['decoded_images_checked'] + verification.get('transfer_decoded_images_checked', 0):,} decoded image contents, "
                f"{verification['development_predictions_replayed']:,} development predictions and "
                f"{verification['heldout_predictions_replayed']:,} held-out predictions replayed; "
                f"{len(verification['adapter_checks'])} adapter checkpoints checked for finite weights and nonzero updated LoRA B matrices, "
                f"and {len(verification['live_checks'])} native-image GPU comparisons against cached/batched scores. "
                "Selections and frozen thresholds are reconstructed from development predictions. Full optimization coverage is checked for every completed epoch."
            ),
            "",
            (
                "The score is `logit( Yes) - logit( No)` at the first response position. The saved two-token probability mass "
                "helps reveal when the original vocabulary distribution prefers other answers. This forced binary interface "
                "has no trained abstention, explanations or bounding boxes. It differs from the older six-label JSON protocol, "
                "so the matched original here is the attribution baseline. Static and sparse-window AP does not prove event detection, "
                "item-owner association or continuous dwell time. No new latency testing was performed."
            ),
            "",
            (
                "The primary inference configuration is seed 42, chosen before seeing its held-out results. "
                "Keep the original fallback for targets selected that way; do not choose a seed or swap a route based on the tables above. "
                "A future deployment claim needs newly held-out cameras, independent review of temporal/ownership labels, "
                "and hours of normal video for event false-alert measurement."
            ),
            "",
            (
                "Artifacts: [protocol](protocol.json), [CSV metrics](metrics.csv), [model bundle](model_bundle.json), "
                "[verification](verification.json), [primary selection](seed_42/selection.json), and per-candidate adapter/history folders. "
                "See [the reproduction guide](../../docs/smolvlm_finetuning.md) and "
                "[the publisher's model card](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) / "
                "[PEFT LoRA reference](https://huggingface.co/docs/peft/v0.17.0/package_reference/lora) for implementation context."
            ),
        ]
    )
    software_path = study / "software_checks.json"
    if software_path.exists():
        software = read(software_path)
        lines.extend(
            [
                "",
                (
                    f"Software checks pass: {software['unit_tests_passed']} unit tests, Ruff lint and formatting, and whitespace validation. "
                    f"The actual inference CLI also passed {software['cli_cases_passed']} checkpoint/score or missing-context checks; these validate the interface, not accuracy on those individual examples."
                ),
            ]
        )
    report = "\n".join(lines) + "\n"
    (study / "report.md").write_text(report)
    findings = ROOT / "reports/smolvlm_finetuning_findings_20261019.md"
    findings.write_text(
        report.replace(
            "](figures/", "](../results/smolvlm_finetuning_20261019/figures/"
        )
        .replace(
            "](protocol.json)",
            "](../results/smolvlm_finetuning_20261019/protocol.json)",
        )
        .replace(
            "](metrics.csv)", "](../results/smolvlm_finetuning_20261019/metrics.csv)"
        )
        .replace(
            "](model_bundle.json)",
            "](../results/smolvlm_finetuning_20261019/model_bundle.json)",
        )
        .replace(
            "](verification.json)",
            "](../results/smolvlm_finetuning_20261019/verification.json)",
        )
        .replace(
            "](seed_42/selection.json)",
            "](../results/smolvlm_finetuning_20261019/seed_42/selection.json)",
        )
        .replace("](../../docs/", "](../docs/")
    )
    save_json(
        study / "run.json",
        {
            "status": "complete",
            "seeds": [s["seed"] for s in selections],
            "model": SPEC.checkpoint,
            "tasks": list(TASKS),
            "no_latency_testing": True,
            "metric_rows": len(cells),
            "verification": "passed",
        },
    )
    print("Report exported", study / "report.md", flush=True)


if __name__ == "__main__":
    main()
