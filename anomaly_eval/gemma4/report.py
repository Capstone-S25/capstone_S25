"""Reproducible Gemma result tables, matched historical context and figures."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from anomaly_eval.datasets.transfer import atomic_json, digest
from anomaly_eval.gemma4.protocol import (
    DEFAULT_OUTPUT,
    MODELS,
    ROOT,
    TASKS,
    read_rows,
    view_indices,
)


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(map(str, headers)) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def macro_ap(records, allowed_sources=None):
    by_task = defaultdict(list)
    for row in records:
        if allowed_sources is None or row["source"] in allowed_sources:
            by_task[row["task"]].append(row["ap"])
    return {task: float(np.mean(values)) for task, values in by_task.items()}


def answer_mass(output, results):
    sources = []
    summary = []
    for name, result in results.items():
        cached = {
            path.stem: {r["id"]: r.get("answer_token_mass") for r in read_rows(path)}
            for path in (output / name / "scores").glob("*.jsonl")
        }
        for method in result["routes"]:
            rows = read_rows(output / name / (method + "_heldout_predictions.jsonl"))
            grouped = defaultdict(list)
            for row in rows:
                actual_route = row.get("route", method)
                policy = result["routes"][actual_route]["tasks"][row["task"]]["policy"]
                grouped[row["task"], row["source"]].append(cached[policy][row["id"]])

            def describe(values):
                available = [value for value in values if value is not None]
                return {
                    "n": len(values),
                    "mass_available": len(available),
                    "minimum": min(available) if available else None,
                    "median": float(np.median(available)) if available else None,
                    "p01": float(np.quantile(available, 0.01)) if available else None,
                    "below_0_5": sum(value < 0.5 for value in available)
                    if available
                    else None,
                    "below_0_9": sum(value < 0.9 for value in available)
                    if available
                    else None,
                }

            for (task, source), values in grouped.items():
                sources.append(
                    {
                        "model": name,
                        "method": method,
                        "task": task,
                        "source": source,
                        **describe(values),
                    }
                )
            summary.append(
                {
                    "model": name,
                    "method": method,
                    **describe(
                        [value for values in grouped.values() for value in values]
                    ),
                }
            )
    return sources, summary


def generate(output):
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text())
    samples = read_rows(output / "samples.jsonl")
    selected_images = len(
        {row["views"][i] for row in samples for i in view_indices(row)}
    )
    from anomaly_eval.gemma4.verify import historical_cohorts

    historical_cohorts(output)
    catalog = json.loads((ROOT / "anomaly_eval/config/gemma4_models.json").read_text())[
        "models"
    ]
    source_cohorts = {r["source"]: r["cohort"] for r in samples}
    main_sources = {r["source"] for r in samples if r["cohort"] == "main"}
    results = {
        name: json.loads((output / name / "results.json").read_text())
        for name in MODELS
        if (output / name / "results.json").exists()
    }
    if results:
        from anomaly_eval.gemma4.verify import verify

        verify(output)
    from anomaly_eval.gemma4.uncertainty import compute

    uncertainty = []
    for name in results:
        intervals = compute(output / name)
        for source, values in intervals["sources"].items():
            task, source = source.split("/", 1)
            uncertainty.append(
                {"model": name, "task": task, "source": source, **values}
            )
    write_csv(output / "ap_uncertainty.csv", uncertainty)
    if (output / "layout_support_protocol.json").exists():
        from anomaly_eval.gemma4.layout_support import derive

        for name, result in results.items():
            result["routes"]["layout_supported"] = derive(output, name, result)
    mass_rows, mass_summary = answer_mass(output, results)
    write_csv(output / "answer_mass.csv", mass_rows)
    if len(results) == len(MODELS) and (output / "ensemble_protocol.json").exists():
        from anomaly_eval.gemma4.ensemble import evaluate

        evaluate(output)
    metrics = []
    routes = []
    for name, result in results.items():
        for method, method_result in result["routes"].items():
            for task, task_result in method_result["tasks"].items():
                routes.append(
                    {
                        "model": name,
                        "method": method,
                        "task": task,
                        "policy": task_result["policy"],
                    }
                )
                for source, points in task_result["source_metrics"].items():
                    for point, values in points.items():
                        metrics.append(
                            {
                                "model": name,
                                "method": method,
                                "task": task,
                                "source": source,
                                "cohort": source_cohorts[source],
                                "calibration": "primary_development",
                                "operating_point": point,
                                "policy": task_result["policy"],
                                **values,
                            }
                        )
                for source, points in task_result.get(
                    "xrf55_source_recalibrated", {}
                ).items():
                    for point, values in points.items():
                        metrics.append(
                            {
                                "model": name,
                                "method": method,
                                "task": task,
                                "source": source,
                                "cohort": "xrf55",
                                "calibration": "xrf55_source_development",
                                "operating_point": point,
                                "policy": task_result["policy"],
                                **values,
                            }
                        )
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "routes.csv", routes)
    counts = []
    for cohort, split, task, source in sorted(
        {(r["cohort"], r["split"], r["task"], r["source"]) for r in samples}
    ):
        members = [
            r
            for r in samples
            if (r["cohort"], r["split"], r["task"], r["source"])
            == (cohort, split, task, source)
        ]
        counts.append(
            {
                "cohort": cohort,
                "split": split,
                "task": task,
                "source": source,
                "n": len(members),
                "positive": sum(r["target"] for r in members),
                "negative": sum(1 - r["target"] for r in members),
                "groups": len({r["group_id"] for r in members}),
                "images": len({p for r in members for p in r["views"]}),
                "selected_unique_images": len(
                    {r["views"][i] for r in members for i in view_indices(r)}
                ),
                "selected_view_occurrences": sum(len(view_indices(r)) for r in members),
            }
        )
    write_csv(output / "dataset_counts.csv", counts)
    main = [
        r
        for r in metrics
        if r["operating_point"] == "raw" and r["calibration"] == "primary_development"
    ]
    summary = []
    for name in results:
        for method in results[name]["routes"]:
            members = [r for r in main if r["model"] == name and r["method"] == method]
            primary = macro_ap(members, main_sources)
            all_tasks = macro_ap(members)
            summary.append(
                {
                    "model": name,
                    "method": method,
                    "main_macro_task_source_ap": float(np.mean(list(primary.values()))),
                    "all_macro_task_source_ap": float(
                        np.mean(list(all_tasks.values()))
                    ),
                    **{"ap_" + task: all_tasks[task] for task in TASKS},
                }
            )
    write_csv(output / "summary.csv", summary)
    old_path = ROOT / "reports/slides/training_evaluation_20261020/metrics.csv"
    historical = []
    if old_path.exists():
        historical = list(csv.DictReader(old_path.open()))
        write_csv(output / "historical_metrics.csv", historical)
    comparisons = []
    for row in historical:
        if (
            row["study"] == "strategy_16oct"
            or (
                row["study"] == "smol_lora_19oct"
                and row["seed"] == "42"
                and row["stage"] in {"original", "routed"}
            )
        ) and row["operating_point"] == "raw":
            matching = next(
                (
                    r
                    for r in main
                    if r["task"] == row["task"] and r["source"] == row["source"]
                ),
                None,
            )
            if matching and (int(row["n"]), int(row["positive"])) != (
                matching["n"],
                matching["positive"],
            ):
                raise ValueError("Historical comparison denominators differ")
            comparisons.append(
                {
                    "model": row["model"],
                    "stage": row["stage"],
                    "task": row["task"],
                    "source": row["source"],
                    "n": row["n"],
                    "positive": row["positive"],
                    "ap": float(row["ap"]),
                    "auroc": float(row["auroc"]),
                }
            )
    comparisons += [
        {
            "model": r["model"],
            "stage": r["method"],
            "task": r["task"],
            "source": r["source"],
            "n": r["n"],
            "positive": r["positive"],
            "ap": r["ap"],
            "auroc": r["auroc"],
        }
        for r in main
    ]
    write_csv(output / "matched_ap_comparison.csv", comparisons)
    common_sources = {
        r["source"] for r in comparisons if not r["model"].startswith("gemma4")
    }
    common_summary = []
    for model, stage in dict.fromkeys((r["model"], r["stage"]) for r in comparisons):
        members = [
            r
            for r in comparisons
            if r["model"] == model
            and r["stage"] == stage
            and r["source"] in common_sources
        ]
        per_task = macro_ap(members)
        if set(per_task) == set(TASKS):
            common_summary.append(
                {
                    "model": model,
                    "stage": stage,
                    "common_macro_task_source_ap": float(
                        np.mean(list(per_task.values()))
                    ),
                    **{"ap_" + task: per_task[task] for task in TASKS},
                }
            )
    write_csv(output / "common_source_summary.csv", common_summary)
    sections = [
        "# Gemma 4 five-anomaly evaluation — 27 October 2026",
        f"Completed model sizes: **{len(results)}/5**. Only completed evaluations enter the result tables. Check individual `progress.json`, `run.log` and `suite_status.json` for active or failed work.",
        "## Scope and model identities",
        "This is an accuracy study of binary image/clip classification for smoking, littering, loitering, apparent public sleeping and unattended items. It does not benchmark runtime latency, train Gemma weights, or establish complete detector/tracker event performance.",
        table(
            ["Model", "Pinned repository", "Precision / runtime", "Weight GB"],
            [
                [
                    s["name"],
                    f"[{s['repo']}](https://huggingface.co/{s['repo']}/tree/{s['revision']})",
                    s["precision"] + " / " + s["backend"],
                    f"{sum(f['bytes'] for f in s['files']) / 1e9:.2f}",
                ]
                for s in catalog
            ],
        ),
        "E2B/E4B/12B use native BF16 Transformers. The 26B mixture-of-experts and dense 31B use official Google QAT Q4_0 weights on the 32 GB GPU. Precision, runtime, architecture and model size therefore change together. All downloaded weight files are checked against pinned publisher SHA-256 values; checkpoint receipts record exact revisions. Models are inference-only with thinking disabled.",
        "## Data and development-only optimization",
        f"The frozen manifest contains **{protocol['cases']:,} non-training cases**, referencing **{protocol['unique_images']:,} distinct image views**; the fixed eight-view selection actually consumes **{selected_images:,} distinct source views** before model resizing/contact-sheet construction. Main development has 1,812 cases; XRF source development has 160; the held-out comparison has 4,032. Prior training partitions are used only to check source-group and decoded-pixel separation. No Gemma fitting uses training or test labels.",
        table(
            ["Cohort / split", "Cases", "Positives"],
            [
                [
                    f"{cohort} / {split}",
                    sum(
                        r["n"]
                        for r in counts
                        if r["cohort"] == cohort and r["split"] == split
                    ),
                    sum(
                        r["positive"]
                        for r in counts
                        if r["cohort"] == cohort and r["split"] == split
                    ),
                ]
                for cohort, split in sorted({(r["cohort"], r["split"]) for r in counts})
            ],
        ),
        "`dataset_counts.csv` supplies every source/task/split denominator, positive count, negative count, group count and image count. Only the annotated objective is evaluated; unannotated objectives are not negative labels. Up to eight uniformly selected chronological views are retained, without inventing elapsed duration. XRF55’s original sixteen-view examples are uniformly subsampled to eight.",
        "Four predeclared policies compare the existing definition at 280 visual tokens/image, evidence/confuser guidance at 280 or 560 tokens/image, and a numbered chronological contact sheet at 1,120 tokens/sheet. A deterministic primary-development screen (274 cases, source/class/group balanced, at most 64/task) chooses each task’s mean-source AP winner; ties keep the simpler baseline. Full development then calibrates both baseline and optimized methods before any test forward pass. XRF source development calibrates only a secondary threshold; it cannot select the primary prompt route.",
        "The score is the real first-answer Yes-minus-No logit margin. Native BF16 mixed-length tensor batches showed answer-margin drift in preflight, so both backends use canonical single-example forwards without deliberate pacing. GPU inference remains unrestricted. Native Yes+No probability mass is saved. For GGUF, equal +100 label biases expose both labels in post-sampling probabilities with temperature 1 and no truncation/penalties; their log-probability difference cancels the common bias. Development preflight verifies +50/+100 bias invariance before accepting scores.",
        "Operating points are raw zero, maximum development F1, pooled development FPR ≤5%, and a single global threshold meeting ≤5% empirical FPR in every primary-development source. Test FPR can exceed that budget. Raw zero for a development-standardized ensemble is merely a centered-score control, not an uncalibrated Yes/No decision.",
        "## Results",
        table(
            ["Model", "Method", "Main macro AP", "All-source macro AP"] + list(TASKS),
            [
                [
                    r["model"],
                    r["method"],
                    f"{r['main_macro_task_source_ap']:.4f}",
                    f"{r['all_macro_task_source_ap']:.4f}",
                ]
                + [f"{r['ap_' + t]:.4f}" for t in TASKS]
                for r in summary
            ],
        )
        if summary
        else "Full model runs are in progress; no completed-model results are available yet.",
        "Macro AP first averages sources within each task, then weights the five tasks equally. Main macro excludes additional transfer and XRF; all-source macro includes them, including the weak XRF proxy. AP is ranking performance, not classification accuracy. Compare it with each source’s positive fraction (the constant-score AP reference), and inspect operating-point metrics.",
        "## Selected routes and source-level operating performance",
        table(
            ["Model", "Task", "Development-selected policy"],
            [
                [r["model"], r["task"], r["policy"]]
                for r in routes
                if r["method"] == "optimized"
            ],
        )
        if routes
        else "Route selection remains in progress.",
        "The following optimized-route table uses the single every-source primary-development FPR-budget threshold. Complete raw/F1/pooled-budget/source-budget results, confusion counts, accuracy, balanced accuracy, AUROC, AP and source-specific secondary calibration are in `metrics.csv`.",
        table(
            [
                "Model",
                "Task / source",
                "N / P / groups",
                "AP",
                "AUROC",
                "Precision",
                "Recall",
                "F1",
                "FPR",
            ],
            [
                [
                    r["model"],
                    r["task"] + " / " + r["source"],
                    f"{r['n']} / {r['positive']} / {r['groups']}",
                ]
                + [
                    f"{r[k]:.4f}" if r[k] is not None else "—"
                    for k in ["ap", "auroc", "precision", "recall", "f1", "fpr"]
                ]
                for r in metrics
                if r["method"] == "optimized"
                and r["operating_point"] == "dev_every_source_fpr_budget"
                and r["calibration"] == "primary_development"
            ],
        ),
        "## Historical context and limits",
        "`matched_ap_comparison.csv` includes the earlier development-selected MobileCLIP2/SigLIP2 pipelines and original/routed SmolVLM. The audit checks identical IDs, targets, sources and groups for all 3,792 non-XRF held-out cases, beyond matching denominators; older transfer rows called test are now called external, with the naming aliases recorded. Different crops, view processing, precision, training and selection budgets make this a pipeline comparison. XRF uses eight Gemma views versus the prior CLIP pilot's sixteen. `historical_metrics.csv` preserves all prior stages, including the VADCLIP study; its event/session cohorts are not substituted for this five-objective matched manifest.",
        "The small legacy `test_data/EVALUATION.csv` contains smoke/unattended partial labels, not an exhaustive six-class benchmark. Positive-only weak anomaly CSVs cannot establish specificity or AP against negatives. Oracle pose, person presence, detector box recall, thermal trajectories and full tracking/event tests address different outputs. These older artifacts are preserved separately rather than reported as new Gemma five-objective results.",
        "Historical holdouts have been inspected repeatedly: this is a retrospective comparison, not a new prospective acceptance test. Loitering uses recordings from a single camera; unattended evidence has few scene groups. Public-sleep labels are provisional visible-sleep proxies and do not imply housing status. HMDB primary actions are not exhaustive smoking-negative verification. MEVA labels describe abandonment action windows rather than persistent ownership state. XRF55 is a masked, scripted hand-to-mouth action proxy with no independently verified cigarette. Large row counts do not create independent cameras, subjects or events.",
        "Classifier FPR is not false alerts/hour, and clip recall is not complete event recall. YOLO/ByteTrack integration and continuous-stream onset/offset tests are outside this study. The existing Faster R-CNN/Grounding DINO and Hungarian/velocity tracking experiments remain separate.",
        "## Reproduction and audit artifacts",
        "See `docs/gemma4_evaluation.md` for environment setup and commands. Frozen manifests/protocols, pre-score amendment, checkpoint receipts, implementation snapshots, development selections, calibrations, all answer margins, token-mass diagnostics, preflights and progress logs live beside this report. The suite can resume without reusing scores after an input, model or scorer identity change.",
    ]
    ensemble_path = output / "ensemble_metrics.csv"
    if common_summary:
        sections += [
            "## Same-cohort comparison with the incumbent pipelines",
            "This table restricts every pipeline to the same ten sources and 3,792 held-out identities, excluding the newer XRF proxy for which the earlier five-task strategies lack the same protocol. AP is averaged over sources within each task and then across the five tasks. The historical CLIP and adapted SmolVLM pipelines use supervised training, crops and different image processing, so this compares complete classifier pipelines rather than isolating pretrained architecture. Local split checks do not establish absence from foundation-model pretraining.",
            table(
                ["Model", "Stage", "Common-source macro AP"] + list(TASKS),
                [
                    [r["model"], r["stage"], f"{r['common_macro_task_source_ap']:.4f}"]
                    + [f"{r['ap_' + task]:.4f}" for task in TASKS]
                    for r in common_summary
                ],
            ),
        ]
    if mass_summary:
        sections += [
            "## Binary-answer probability diagnostics",
            "The score compares the Yes and No alternatives; it does not establish that an unconstrained response would start with either word. Native runs retain the full-vocabulary Yes+No probability mass. All cases remain in the metrics, including low-mass cases. `answer_mass.csv` gives every held-out source's distribution and low-mass counts. GGUF's biased scoring path does not provide an unmodified mass for every case, so those entries are unavailable; its unbiased development preflight reports separate lower bounds and answer tokens.",
            table(
                [
                    "Model",
                    "Method",
                    "Available / N",
                    "Minimum mass",
                    "Median mass",
                    "Mass below 0.5",
                ],
                [
                    [
                        r["model"],
                        r["method"],
                        f"{r['mass_available']}/{r['n']}",
                        f"{r['minimum']:.3g}" if r["minimum"] is not None else "—",
                        f"{r['median']:.4f}" if r["median"] is not None else "—",
                        r["below_0_5"] if r["below_0_5"] is not None else "—",
                    ]
                    for r in mass_summary
                ],
            ),
        ]
    if (output / "layout_support_protocol.json").exists():
        affected = []
        for name, result in results.items():
            for task, task_result in result["routes"]["layout_supported"][
                "tasks"
            ].items():
                for source, points in task_result["source_metrics"].items():
                    if points["raw"]["route_counts"].get("baseline", 0):
                        affected.append(
                            [
                                name,
                                source,
                                points["raw"]["n"],
                                f"{result['routes']['baseline']['tasks'][task]['source_metrics'][source]['raw']['ap']:.4f}",
                                f"{result['routes']['optimized']['tasks'][task]['source_metrics'][source]['raw']['ap']:.4f}",
                                f"{points['raw']['ap']:.4f}",
                            ]
                        )
        sections += [
            "## Post-result layout-support correction",
            "E2B exposed a policy-selection problem: all primary smoking development cases are single images, so the selected sheet policy acted as a higher-resolution single image during selection and changed representation to an untested contact sheet on multi-frame transfer. The unchanged original run remains in every comparison. After reviewing E2B, a separate diagnostic rule was recorded in `layout_support_protocol.json`: when primary development contains no multi-image input for an objective, a multi-image request using a selected sheet policy falls back to that model's original baseline. Other requests keep the optimized route. The inference predicate uses input view count and development layout coverage, never source identity or a held-out label.",
            "Each branch uses its own original development-only cutoff, recorded in `route_cutoffs`/`route_counts`; mixed-mode sources do not have a single scalar cutoff. This corrects an unvalidated representation switch but is a retrospective E2B analysis, not a new untouched acceptance test. It cannot establish deployment generalization. The layout-supported row is separate from the original optimized row in all CSVs and macro-AP tables.",
            table(
                [
                    "Model",
                    "Affected source",
                    "N",
                    "Baseline AP",
                    "Screen-optimized AP",
                    "Layout-supported AP",
                ],
                affected,
            ),
        ]
    if uncertainty:
        sections += [
            "## Uncertainty from observed source groups",
            "`ap_uncertainty.csv` reports 500 paired group-bootstrap repetitions for baseline/optimized AP and their difference, using the same sampled groups for both methods. Rows within a recording/author/subject group are kept together. Sources with fewer than three groups receive no interval. These intervals condition on the observed source groups and inherit their labels; they do not measure generalization to unseen cameras, establish independent events or undo historical holdout reuse.",
        ]
    if ensemble_path.exists():
        ensemble_rows = list(csv.DictReader(ensemble_path.open()))
        ensemble_summary = []
        for method in sorted({r["method"] for r in ensemble_rows}):
            selected = [
                {**r, "ap": float(r["ap"])}
                for r in ensemble_rows
                if r["method"] == method
                and r["operating_point"] == "raw"
                and r["calibration"] == "primary_development"
            ]
            task_ap = macro_ap(selected)
            primary = macro_ap(selected, main_sources)
            ensemble_summary.append(
                [
                    method,
                    f"{np.mean(list(primary.values())):.4f}",
                    f"{np.mean(list(task_ap.values())):.4f}",
                ]
                + [f"{task_ap[t]:.4f}" for t in TASKS]
            )
        sections += [
            "## Predeclared model-family combinations",
            "Both combinations are fitted using primary development only. The per-task route selects the model with best full-development mean-source AP. The equal-weight combination averages margins normalized by each model's development mean and standard deviation. All five models contribute equally, and both combinations receive development-only thresholds. These are additional pipeline comparisons; neither is selected based on held-out performance.",
            "An optional third safeguard compares all five models' baseline and optimized routes using the complete primary development set, so a noisy screen winner can be rejected. Its separate `safeguard_protocol.json` was declared before the first completed Gemma result was reviewed, after initial held-out forwards had begun; it is an additional retrospective comparison, not part of the original pre-forward declaration. No held-out scores or labels select this route.",
            table(
                ["Combination", "Main macro AP", "All-source macro AP"] + list(TASKS),
                ensemble_summary,
            ),
            "Full operating-point results and confusion counts are in `ensemble_metrics.csv`; fitting constants, model routes and calibrations are in `ensembles/`. The centered ensemble raw-zero point has no uncalibrated Yes/No probability interpretation.",
        ]
        fitted = json.loads((output / "ensembles/selection.json").read_text())
        family_routes = []
        for family, values in fitted.items():
            if values is not None:
                for task, route in values["routes"].items():
                    family_routes.append(
                        [
                            family,
                            task,
                            route,
                            f"{values['development_ap'][task][route]:.4f}",
                        ]
                    )
        sections += [
            "The following family routes are chosen from complete primary development, before combining any held-out scores. They do not constitute a deployment promotion.",
            table(
                ["Family", "Task", "Selected model / route", "Development source AP"],
                family_routes,
            ),
        ]
    if len(results) == len(MODELS):
        original = [r for r in summary if r["method"] in {"baseline", "optimized"}]
        best = max(original, key=lambda r: r["all_macro_task_source_ap"])
        findings = [
            "## Main findings from the completed study",
            f"All five variants completed the same 6,004-case manifest, including 4,032 held-out cases each. The replay audit passed for every model. The highest observed standalone all-source macro AP is **{best['all_macro_task_source_ap']:.4f}**, from **{best['model']} / {best['method']}**. This describes the observed comparison; it is not a test-selected deployment rule.",
            "Prompt and visual-policy optimization improves some models and regresses others. The following deltas preserve the original baseline and development-screen-selected routes. FPR failures and zero recall refer to the optimized route at its frozen every-source development FPR-budget threshold, evaluated on the eleven held-out sources.",
        ]
        optimization = []
        for name in MODELS:
            by_method = {r["method"]: r for r in summary if r["model"] == name}
            baseline = by_method["baseline"]["all_macro_task_source_ap"]
            optimized = by_method["optimized"]["all_macro_task_source_ap"]
            points = [
                r
                for r in metrics
                if r["model"] == name
                and r["method"] == "optimized"
                and r["calibration"] == "primary_development"
                and r["operating_point"] == "dev_every_source_fpr_budget"
            ]
            optimization.append(
                [
                    name,
                    f"{baseline:.4f}",
                    f"{optimized:.4f}",
                    f"{optimized - baseline:+.4f}",
                    f"{sum(r['fpr'] > 0.05 + 1e-9 for r in points)}/{len(points)}",
                    f"{sum(r['recall'] == 0 for r in points)}/{len(points)}",
                ]
            )
        findings.append(
            table(
                [
                    "Model",
                    "Baseline macro AP",
                    "Optimized macro AP",
                    "Change",
                    "Sources with FPR >5%",
                    "Sources with zero recall",
                ],
                optimization,
            )
        )
        matched_gemma = [
            r
            for r in common_summary
            if r["model"] in MODELS and r["stage"] in {"baseline", "optimized"}
        ]
        matched_incumbents = [r for r in common_summary if r["model"] not in MODELS]
        if matched_gemma and matched_incumbents:
            candidate = max(
                matched_gemma, key=lambda r: r["common_macro_task_source_ap"]
            )
            incumbent = max(
                matched_incumbents, key=lambda r: r["common_macro_task_source_ap"]
            )
            findings.append(
                f"On the identical ten-source, 3,792-case cohort, the strongest observed standalone Gemma route is {candidate['model']} / {candidate['stage']} at **{candidate['common_macro_task_source_ap']:.4f}**, versus **{incumbent['common_macro_task_source_ap']:.4f}** for the existing {incumbent['model']} / {incumbent['stage']} pipeline. Gemma therefore does not displace the incumbent on this overall retrospective ranking metric. Individual objectives can favor different models; their source-level results and training/input differences remain essential."
            )
        findings += [
            "The remaining failures concern generalization and event evidence. All five optimized routes have zero recall on the CCTV smoking cohort at the frozen source-budget operating point. Loitering remains weak. Strong reviewed-unattended AP does not carry over consistently to MEVA abandonment-action windows, which label a different part of the event. High public-sleep AP comes from small provisional visible-sleep cohorts. A development FPR constraint does not guarantee its test counterpart.",
            "The next evaluation should use new camera/session holdouts, representative multi-frame development inputs, explicit duration and owner/item evidence, and continuous event recall plus false alerts/hour. The E2B layout correction is a separately labeled retrospective diagnostic. These results support further validation of task-specific confirmation models; they do not establish a general five-anomaly deployment model.",
        ]
        sections[2:2] = findings
    report = "\n\n".join(sections) + "\n"
    (output / "report.md").write_text(report)
    tracked = ROOT / "reports/gemma4_evaluation_20261027"
    tracked.mkdir(parents=True, exist_ok=True)
    (tracked / "report.md").write_text(report)
    for filename in [
        "metrics.csv",
        "routes.csv",
        "summary.csv",
        "dataset_counts.csv",
        "matched_ap_comparison.csv",
        "ensemble_metrics.csv",
        "ap_uncertainty.csv",
        "historical_metrics.csv",
        "historical_cohort_verification.json",
        "layout_support_protocol.json",
        "answer_mass.csv",
        "common_source_summary.csv",
        "verification.json",
        "protocol.json",
        "ensemble_protocol.json",
        "safeguard_protocol.json",
        "environment.json",
    ]:
        if (output / filename).exists():
            (tracked / filename).write_bytes((output / filename).read_bytes())
    atomic_json(
        output / "report_receipt.json",
        {
            "complete_models": list(results),
            "protocol_sha256": digest(output / "protocol.json"),
            "historical_metrics_sha256": digest(old_path)
            if old_path.exists()
            else None,
            "metrics_rows": len(metrics),
            "historical_cohort_verification_sha256": digest(
                output / "historical_cohort_verification.json"
            ),
            "result_sha256": {
                name: digest(output / name / "results.json") for name in results
            },
        },
    )
    (tracked / "report_receipt.json").write_bytes(
        (output / "report_receipt.json").read_bytes()
    )
    for filename in [
        "identity.json",
        "selection.json",
        "calibration.json",
        "results.json",
    ]:
        source = output / "ensembles" / filename
        if source.exists():
            (tracked / "ensembles").mkdir(exist_ok=True)
            (tracked / "ensembles" / filename).write_bytes(source.read_bytes())
    if main:
        figures(output, main, results)
        sections = "\n\n![Held-out source AP](figures/source_ap.png)\n\n![Precision–recall curves](figures/precision_recall.png)\n\n![ROC curves](figures/roc.png)\n"
        for path in [output / "report.md", tracked / "report.md"]:
            with path.open("a") as stream:
                stream.write(sections)
        import shutil

        shutil.copytree(output / "figures", tracked / "figures", dirs_exist_ok=True)
    return summary


def figures(output, metrics, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    destination = output / "figures"
    destination.mkdir(exist_ok=True)
    sources = sorted({(r["task"], r["source"]) for r in metrics})
    pairs = [(name, method) for name in results for method in results[name]["routes"]]
    matrix = np.array(
        [
            [
                next(
                    r["ap"]
                    for r in metrics
                    if r["model"] == name
                    and r["method"] == method
                    and (r["task"], r["source"]) == source
                )
                for source in sources
            ]
            for name, method in pairs
        ]
    )
    fig, ax = plt.subplots(figsize=(16, max(3, len(pairs) * 0.55)))
    chart = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(
        range(len(sources)),
        [source for task, source in sources],
        rotation=40,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(len(pairs)), [name + " / " + method for name, method in pairs])
    for i in range(len(pairs)):
        for j in range(len(sources)):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                color="black" if matrix[i, j] > 0.65 else "white",
                fontsize=8,
            )
    fig.colorbar(chart, ax=ax, label="Average precision")
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(destination / ("source_ap." + extension), dpi=160)
    plt.close(fig)
    for curve_name in ["precision_recall", "roc"]:
        fig, axes = plt.subplots(3, 4, figsize=(17, 12))
        for axis, (task, source) in zip(axes.flat, sources):
            for name in results:
                rows = [
                    r
                    for r in read_rows(
                        output / name / "optimized_heldout_predictions.jsonl"
                    )
                    if r["task"] == task and r["source"] == source
                ]
                labels = [r["target"] for r in rows]
                scores = [r["score"] for r in rows]
                if curve_name == "precision_recall":
                    precision, recall, _ = precision_recall_curve(labels, scores)
                    axis.plot(recall, precision, label=name)
                    axis.set(xlabel="Recall", ylabel="Precision")
                else:
                    fpr, recall, _ = roc_curve(labels, scores)
                    axis.plot(fpr, recall, label=name)
                    axis.set(xlabel="False-positive rate", ylabel="Recall")
            axis.set_title(source, fontsize=9)
            axis.set(xlim=(0, 1), ylim=(0, 1))
            axis.grid(alpha=0.2)
        for axis in list(axes.flat)[len(sources) :]:
            axis.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5)
        fig.suptitle(
            "Development-selected Gemma routes — held-out "
            + curve_name.replace("_", " ")
        )
        fig.tight_layout(rect=(0, 0.035, 1, 0.96))
        for extension in ["png", "pdf"]:
            fig.savefig(destination / (curve_name + "." + extension), dpi=160)
        plt.close(fig)


def optimization_overview(output):
    """Add operating-point comparisons without changing frozen model scorers."""
    from anomaly_eval.gemma4.optimization import report

    previous = Path(json.loads((output / "protocol.json").read_text())["previous"])
    report(previous, output)
    if not (output / "metrics.csv").exists():
        return []
    records = [
        r
        for r in csv.DictReader((previous / "metrics.csv").open())
        if r["method"] in {"baseline", "optimized"}
    ] + list(csv.DictReader((output / "metrics.csv").open()))
    columns = [
        "ap",
        "auroc",
        "precision",
        "recall",
        "f1",
        "fpr",
        "accuracy",
        "balanced_accuracy",
    ]
    summary = []
    keys = dict.fromkeys(
        (r["model"], r["method"], r["operating_point"])
        for r in records
        if r["calibration"] == "primary_development"
    )
    for model, method, point in keys:
        for cohort in ["all", "common"]:
            members = [
                r
                for r in records
                if (r["model"], r["method"], r["operating_point"])
                == (model, method, point)
                and r["calibration"] == "primary_development"
                and (cohort == "all" or r["source"] != "xrf55_rgb_public")
            ]
            if len(members) != len({(r["task"], r["source"]) for r in members}):
                raise ValueError("Duplicate operating-summary source")
            row = {
                "model": model,
                "method": method,
                "operating_point": point,
                "cohort": cohort,
                "n": sum(int(r["n"]) for r in members),
                "sources": len(members),
            }
            if row["n"] != (4032 if cohort == "all" else 3792):
                raise ValueError("Operating-summary cohort differs")
            for column in columns:
                row["macro_" + column] = float(
                    np.mean(
                        [
                            np.mean(
                                [float(r[column]) for r in members if r["task"] == task]
                            )
                            for task in TASKS
                        ]
                    )
                )
            row["sources_fpr_over_5pct"] = sum(
                float(r["fpr"]) > 0.05 + 1e-9 for r in members
            )
            row["zero_recall_sources"] = sum(float(r["recall"]) == 0 for r in members)
            row["max_source_fpr"] = max(float(r["fpr"]) for r in members)
            for count in ["tp", "fp", "fn", "tn"]:
                row[count] = sum(int(r[count]) for r in members)
            row["micro_f1"] = (
                2 * row["tp"] / max(1, 2 * row["tp"] + row["fp"] + row["fn"])
            )
            summary.append(row)
    write_csv(output / "operating_summary.csv", summary)
    selected = [
        r
        for r in summary
        if r["cohort"] == "all"
        and r["operating_point"] == "dev_every_source_fpr_budget"
    ]
    section = (
        "\n## Operating-point comparison\n\n"
        + (
            "Each metric below first averages sources within a task, then the five tasks equally. These are task/source macro values, not pooled case metrics. All rows use each method's frozen every-source primary-development FPR-budget threshold. AP measures ranking and can improve while F1 or recall falls. Test sources exceeding 5% FPR and sources with zero recall are reported explicitly. All four operating points, both the 4,032-case all-source and 3,792-case common-source cohorts, confusion totals and micro F1 are in `operating_summary.csv`. The blend's raw-zero point is a centered-score control.\n\n"
        )
        + table(
            [
                "Model",
                "Method",
                "AP",
                "AUROC",
                "F1",
                "Recall",
                "FPR",
                "Sources >5% FPR",
                "Zero-recall sources",
            ],
            [
                [
                    r["model"],
                    r["method"],
                    *[
                        f"{r['macro_' + c]:.4f}"
                        for c in ["ap", "auroc", "f1", "recall", "fpr"]
                    ],
                    f"{r['sources_fpr_over_5pct']}/{r['sources']}",
                    f"{r['zero_recall_sources']}/{r['sources']}",
                ]
                for r in selected
            ],
        )
        + "\n"
    )
    for name in MODELS:
        rows = {r["method"]: r for r in selected if r["model"] == name}
        if "full_dev_selected" in rows:
            old, new = rows["optimized"], rows["full_dev_selected"]
            section += f"\n{name}: compared with the previous optimized route, full-development selection changes all-source macro AP by {new['macro_ap'] - old['macro_ap']:+.4f} and macro F1 at this operating point by {new['macro_f1'] - old['macro_f1']:+.4f}. These are descriptive retrospective differences, not a deployment promotion.\n"
    historical = previous / "common_source_summary.csv"
    if historical.exists():
        incumbents = [
            r
            for r in csv.DictReader(historical.open())
            if not r["model"].startswith("gemma4")
        ]
        matched = [
            {
                "model": r["model"],
                "method": r["stage"],
                "common_source_macro_ap": float(r["common_macro_task_source_ap"]),
            }
            for r in incumbents
        ] + [
            {
                "model": r["model"],
                "method": r["method"],
                "common_source_macro_ap": r["macro_ap"],
            }
            for r in summary
            if r["cohort"] == "common"
            and r["operating_point"] == "dev_every_source_fpr_budget"
        ]
        write_csv(output / "common_source_comparison.csv", matched)
        section += (
            "\n## Matched incumbent comparison\n\n"
            "The common-source cohort contains the same 3,792 held-out IDs, labels, sources and groups as the earlier CLIP/SmolVLM comparison. Its original identity audit is retained in the previous report. Different training, crops, input processing and precision make these pipeline comparisons. The additional XRF gesture proxy is excluded from every row below and from `common_source_comparison.csv`.\n\n"
            + table(
                ["Incumbent", "Stage", "Common-source macro AP"],
                [
                    [r["model"], r["method"], f"{r['common_source_macro_ap']:.4f}"]
                    for r in matched
                    if not r["model"].startswith("gemma4")
                ],
            )
            + "\n"
        )
        completed = {r["model"] for r in selected if r["method"] == "full_dev_selected"}
        if len(completed) == len(MODELS) and incumbents:
            candidate = max(
                (r for r in matched if r["model"].startswith("gemma4")),
                key=lambda r: r["common_source_macro_ap"],
            )
            incumbent = max(
                (r for r in matched if not r["model"].startswith("gemma4")),
                key=lambda r: r["common_source_macro_ap"],
            )
            section += f"\nThe highest observed standalone Gemma common-source macro AP across the reported methods is {candidate['common_source_macro_ap']:.4f} ({candidate['model']}, {candidate['method']}); the incumbent reference is {incumbent['common_source_macro_ap']:.4f} ({incumbent['model']}, {incumbent['method']}). This describes the retrospective results; choosing an implementation from held-out scores does not create a development-selected deployment policy.\n"
    sources = [
        r
        for r in records
        if r["method"] == "full_dev_selected"
        and r["operating_point"] == "dev_every_source_fpr_budget"
        and r["calibration"] == "primary_development"
    ]
    source_rows = []
    for name in MODELS:
        by_source = {r["source"]: r for r in sources if r["model"] == name}
        if not by_source:
            continue
        smoke = by_source["smoking_cctv"]
        loiter = by_source["street_scene"]
        meva = by_source["meva_abandon_action_transfer"]
        reviewed = by_source["unattended_reviewed"]
        source_rows.append(
            [
                name,
                *[f"{float(smoke[c]):.4f}" for c in ["ap", "recall", "fpr"]],
                *[f"{float(loiter[c]):.4f}" for c in ["ap", "recall"]],
                *[f"{float(meva[c]):.4f}" for c in ["ap", "recall"]],
                f"{float(reviewed['fpr']):.4f}",
            ]
        )
    section += (
        "\n## Source-level failure checks\n\n"
        "All rows below use full-development selection and its frozen every-source development FPR-budget threshold. A ranking gain does not imply useful event recall. The CCTV smoking cohort has one camera, Street Scene loitering has six held-out recording groups from one camera, reviewed unattended items have two scene groups, and MEVA labels abandonment-action windows rather than persistent ownership state.\n\n"
        + table(
            [
                "Model",
                "CCTV AP",
                "CCTV recall",
                "CCTV FPR",
                "Loiter AP",
                "Loiter recall",
                "MEVA AP",
                "MEVA recall",
                "Reviewed-item FPR",
            ],
            source_rows,
        )
        + "\n\nThe frozen manifest has 6,004 cases: 1,812 primary development, 160 secondary source-development and 4,032 held out. The previous report's `dataset_counts.csv` gives the complete source/split denominators. Only the queried objective is annotated; other objectives are not automatically negative. Public-sleep labels describe provisional visible sleeping/bedding, without inferring housing status. Smoking primary development contains single images, so temporal crop/detail behavior remains weakly validated.\n\n"
        "E2B, E4B and 12B use native BF16 Transformers inference. The 26B-A4B and 31B variants use official Google QAT Q4_0 GGUF releases through the pinned CUDA llama.cpp runtime. This is a pipeline comparison, with size, architecture, precision and runtime differences. GPU inference has no deliberate pacing; no latency benchmark is included. The Yes-minus-No margin is a ranking score, not a calibrated probability.\n\n"
        "`ap_uncertainty.csv` and `ap_uncertainty.json`, generated with the separate uncertainty command, give paired observed-group bootstrap intervals against the original baseline. They do not estimate new-camera performance, remove weak-label uncertainty or undo repeated holdout inspection. Sources with fewer than three groups have no interval. No Gemma weights were updated; these experiments optimize input representation, prompts, development routing, score blending and thresholds. Continuous detector/tracker event recall and false alerts per hour still require a separate prospective evaluation.\n"
    )
    with (output / "report.md").open("a") as stream:
        stream.write(section)
    tracked = ROOT / "reports/gemma4_optimization_20261028"
    for filename in [
        "report.md",
        "operating_summary.csv",
        "common_source_comparison.csv",
    ]:
        if not (output / filename).exists():
            continue
        (tracked / filename).write_bytes((output / filename).read_bytes())
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--optimization",
        action="store_true",
        help="Report stage-two routes and operating-point trade-offs",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        if args.optimization:
            from anomaly_eval.gemma4.optimization import OUTPUT

            args.output = OUTPUT
        else:
            args.output = DEFAULT_OUTPUT
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(
            json.dumps(
                optimization_overview(args.output)
                if args.optimization
                else generate(args.output),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
