"""Source-separated, reproducible report for the expanded strategy experiments."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from anomaly_eval.training.clip import binary_metrics
from scripts.report_clip_training import group_bootstrap, table

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/strategy_study_20261016"
DATA = ROOT / "data/strategy_study_20261016"
MODELS = {"mobileclip2-s2": "MobileCLIP2-S2", "siglip2-b16-224": "SigLIP2-B/16-224"}
NAMES = {
    "kaggle_sujaykapadnis_smoking": "Smoking: new photo holdout",
    "hmdb51_action_transfer": "Smoking: new HMDB action source",
    "commons_new_author_transfer": "Public sleep: new author photos",
    "meva_abandon_action_transfer": "Unattended: new MEVA action proxy",
    "smoking_cctv": "Smoking: historical CCTV c1",
    "stanford40": "Smoking: historical Stanford photos",
    "mivia_iwdd": "Littering/waste dumping: historical IWDD",
    "street_scene": "Loitering: historical Street Scene",
    "commons_visible_sleeping": "Public sleep: historical photos",
    "unattended_reviewed": "Unattended: historical camera holdout",
}


def read(path):
    return json.loads(path.read_text())


def f(value):
    return "—" if value is None else f"{value:.3f}"


def pct(value):
    return f"{100 * value:.1f}%"


def main():
    all_metrics, all_rows, exported = {}, {}, []
    routes, rows, families, calibration, uncertainty = [], [], [], [], {}
    for model, label in MODELS.items():
        directory = RUN / model / "transfer_evaluation"
        assert read(directory / "state.json")["status"] == "complete"
        frozen = read(directory / "frozen_policies.json")
        grouped = defaultdict(list)
        for path in directory.glob("*_predictions.jsonl"):
            tag = path.name.replace("_predictions.jsonl", "")
            for row in map(json.loads, path.read_text().splitlines()):
                grouped[tag, row["task"], row["source"], row["family"]].append(row)
        per_source = {}
        for (tag, task, source, family), records in grouped.items():
            target = [r["target"] for r in records]
            score = [r["score"] for r in records]
            points = frozen[task][family]["thresholds"]
            metrics = {
                op: binary_metrics(target, score, threshold)
                for op, threshold in points.items()
            }
            key = tag + "/" + task + "/" + source
            per_source.setdefault(key, {})[family] = metrics
            all_metrics[model, source, family] = metrics
            all_rows[model, source, family] = records
            for op, value in metrics.items():
                exported.append(
                    {
                        "model": model,
                        "cohort": tag,
                        "task": task,
                        "source": source,
                        "family": family,
                        "operating_point": op,
                        "threshold": points[op],
                        "groups": len({r["group_id"] for r in records}),
                        **value,
                    }
                )
        (directory / "per_source_results.json").write_text(
            json.dumps(per_source, indent=2) + "\n"
        )
        for source, description in NAMES.items():
            current = all_metrics[model, source, "selected"]
            previous = all_metrics[model, source, "previous_bundle"]
            c = current["dev_max_f1"]
            rows.append(
                [
                    label,
                    description,
                    c["n"],
                    c["positive"],
                    f(previous["raw"]["ap"]),
                    f(c["ap"]),
                    f(c["f1"]),
                    pct(c["recall"]),
                    pct(c["fpr"]),
                ]
            )
            family_names = [
                "full_pool_a0/linear",
                "full_crop_pool_a0/linear",
                "full_ordered_a0/linear",
                "full_geometry_a0/linear",
            ]
            families.append(
                [
                    label,
                    description,
                    *[
                        f(all_metrics[model, source, name]["raw"]["ap"])
                        for name in family_names
                    ],
                    f(c["ap"]),
                ]
            )
            for op, metric in current.items():
                calibration.append(
                    [
                        label,
                        description,
                        op,
                        pct(metric["precision"]),
                        pct(metric["recall"]),
                        f(metric["f1"]),
                        pct(metric["fpr"]),
                    ]
                )
            record = all_rows[model, source, "selected"]
            old = {r["id"]: r for r in all_rows[model, source, "previous_bundle"]}
            uncertainty[model + "/" + source] = group_bootstrap(
                record,
                [old[r["id"]]["score"] for r in record],
                [r["score"] for r in record],
            )
        selection = read(RUN / model / "selection.json")["tasks"]
        for task, value in selection.items():
            routes.append(
                [
                    label,
                    task,
                    value["strategy"],
                    value["classifier"],
                    f(value["mean_source_ap"]),
                    f(value["worst_source_ap"]),
                ]
            )
    with (RUN / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    (RUN / "group_uncertainty.json").write_text(
        json.dumps(uncertainty, indent=2) + "\n"
    )
    thermal = read(RUN / "thermal_trajectories/results.json")
    thermal_selection = read(RUN / "thermal_trajectories/selection.json")
    selected_thermal = thermal[thermal_selection["selected"]]["test"]["dev_max_f1"]
    rough = read(DATA / "rough_screened.audit.json")
    lines = [
        "# Expanded CLIP strategy and new-source evaluation — 16 October 2026",
        "",
        "Both backbones completed 33 candidate fits/evaluations per objective (330 candidates total), using development-only strategy selection. Actual new training data include 715 additional smoking photos; 179 new photos join development and 224 are held out. Further sources are reserved for transfer evaluation. No latency experiment is included.",
        "",
        "## Findings",
        "",
        "Automatic contextual crops improve the historical CCTV smoking ranking for both models. Sparse detector geometry improves loitering ranking within Street Scene. Ordered-view probes on previously adapted vision features improve unattended-item ranking on historical cameras. These gains do not establish universal transfer: the primary routes regress slightly on the wholly new HMDB smoking source and MEVA abandonment-action proxy. The new public-sleep photo evidence remains small and provisionally labelled.",
        "",
        "Threshold transfer remains the main operational failure. Very high unattended-item AP on historical cameras coexists with nearly universal alarms at maximum-development-F1 thresholds. Fixed zero-logit recall can also collapse. The operating-point tables retain these failures; routes or thresholds are never selected using holdout labels.",
        "",
        "## What was acquired",
        "",
        table(
            ["Source", "Acquired/retained", "Use and qualification"],
            [
                [
                    "[Smoking photo collection](https://www.kaggle.com/datasets/sujaykapadnis/smoking)",
                    "1,120 acquired; 1,118 retained after duplicate screening",
                    "715 train / 179 dev / 224 official image holdout; identities/cameras not provided",
                ],
                [
                    "[Author-hosted HMDB51](https://huggingface.co/datasets/Serrelab/hmdb51)",
                    "Full archive acquired; 906 clips extracted from seven relevant classes",
                    "109 smoking / 797 other primary actions; entirely new action-domain transfer, not an official 51-class benchmark",
                ],
                [
                    "[Published thermal loitering annotations](https://github.com/johnnynunez/RS-WACV24_Loitering)",
                    "874,936 human observations; 19,737 tracks across 31 dates",
                    "17,878 tracks with at least eight observations; date-held-out trajectory component study on one thermal camera",
                ],
                [
                    "[Public MEVA release](https://mevadata.org/)",
                    "375 MB annotation archive; 13 relevant videos downloaded",
                    "27 action windows: 15 positive / 12 pre-event controls; one additional source video unavailable",
                ],
                [
                    "Commons / author-linked Flickr",
                    "148 new human-category candidates; 40 downloaded and reviewed",
                    f"29 usable provisional labels; {rough['retained']} retained new-author holdout photos ({rough['positive']} positive); ambiguous cases omitted",
                ],
                [
                    "Additional littering searches",
                    "No further matching public event corpus acquired",
                    "VADD requires an access form; several reported collections lack a public release. Garbage-object datasets are not substituted for disposal-event labels",
                ],
            ],
        ),
        "",
        "New photo/clip sources are checked against old inputs using decoded identities and conservative perceptual hashes. Whole related-photo groups remain in one split. The new public-sleep cohort excludes every previously used photographer and related images. The previous study's test set stays out of fitting/selection and is labelled historical below. Unseen-source evaluation and a same-source image holdout answer different questions.",
        "",
        "## Primary source-separated results",
        "",
        "AP measures ranking across thresholds. F1, recall and FPR use the primary route's maximum-development-F1 threshold. Previous AP uses the previous study's per-target development-selected bundle, not the best historical test column. Each source is reported separately; CCTV smoking is never pooled with easier photographs here.",
        "",
        table(
            [
                "Backbone",
                "Evidence",
                "N",
                "Positive",
                "Previous AP",
                "New route AP",
                "New F1",
                "Recall",
                "FPR",
            ],
            rows,
        ),
        "",
        "Photo-source gains do not establish CCTV or video-source transfer. HMDB labels identify a primary action, so other action categories are not exhaustive smoking-negative annotations. MEVA positives are windows ending at an annotated abandonment action; controls precede the first annotation. This event-anchored offline diagnostic does not certify persistent unattended-item status or continuous detection. Public-sleep and historical unattended labels remain provisional assistant review, not independent human ground truth.",
        "",
        "## Which changes helped",
        "",
        "Every family below chooses its regularization using development data. Full, crop, ordered and geometry rows use the same new training data and frozen pretrained backbone, which isolates representation changes more closely than comparison with the previous bundle. The primary route can instead select interpolated/adapted visual weights or an MLP. A higher test column does not change the frozen primary choice.",
        "",
        table(
            [
                "Backbone",
                "Evidence",
                "Full pooled AP",
                "Full + crop AP",
                "Ordered AP",
                "Full + geometry AP",
                "Primary route AP",
            ],
            families,
        ),
        "",
        "The interpolation grid (0, 0.25, 0.5, 1) is evaluated on development data; coefficients mix the pretrained and previously fine-tuned visual parameters. The classifier is fitted again on training labels. This screen does not assume interpolation or more fine-tuning is always beneficial: MobileCLIP's selected waste-dumping route regresses on the historical official video holdout. All candidate development results and saved heads are retained.",
        "",
        "## Development selections",
        "",
        table(
            [
                "Backbone",
                "Target",
                "Representation",
                "Head",
                "Mean source AP",
                "Worst source AP",
            ],
            routes,
        ),
        "",
        "The auxiliary worst-source selection is also retained in numeric outputs. It was specified before HMDB/MEVA scores and does not replace the primary route. For this run it selects the same effective primary candidates. Most objectives still have only one development dataset, so this policy cannot manufacture missing camera diversity.",
        "",
        "## Operating-point transfer",
        "",
        table(
            [
                "Backbone",
                "Evidence",
                "Dev operating point",
                "Precision",
                "Recall",
                "F1",
                "FPR",
            ],
            calibration,
        ),
        "",
        "`raw` means the predefined zero-logit decision, not a calibrated probability. The FPR-budget point constrains empirical development FPR, not new-camera FPR or alerts/hour. Thresholds are recomputed from serialized-head development scores only; primary weights/routes remain fixed and prior-bundle thresholds remain unchanged.",
        "",
        "## Separate loitering trajectory study",
        "",
        f"The development-selected {thermal_selection['selected']} has held-out-date AP {f(selected_thermal['ap'])}, F1 {f(selected_thermal['f1'])}, recall {pct(selected_thermal['recall'])} and FPR {pct(selected_thermal['fpr'])} on {selected_thermal['n']} complete trajectories. Its mean development-date AP is {f(thermal[thermal_selection['selected']]['dev']['mean_source_ap'])}; the drop on later dates remains visible.",
        "",
        table(
            ["Candidate", "Mean dev-date AP", "Test AP", "Test F1", "Test FPR"],
            [
                [
                    name,
                    f(value["dev"]["mean_source_ap"]),
                    f(value["test"]["dev_max_f1"]["ap"]),
                    f(value["test"]["dev_max_f1"]["f1"]),
                    pct(value["test"]["dev_max_f1"]["fpr"]),
                ]
                for name, value in thermal.items()
            ],
        ),
        "",
        "This is a component benchmark using complete author-provided tracks, not an automatic detector/tracker or an RGB CLIP result. No future-context/online-alert claim follows from a whole-trajectory descriptor. The source is one physical camera; holding out dates/seasons improves the temporal split but does not establish unseen-camera generalization. Track IDs do not cross source clips; frame units are retained without inventing a frame rate.",
        "",
        "## Interpretation and next evidence needed",
        "",
        "The data expansion produces useful gains on some tested sources, while genuine new video sources expose remaining transfer failures. Independent cameras, independent event review and matched deployment normals still matter more than extracting additional adjacent frames. A next training cycle can use the newly acquired video domains with explicit source-held-out development folds, reserving another fresh source for final confirmation. The current transfer results become historical if used to guide that cycle.",
        "",
        "Sparse detector geometry is not ownership reasoning or a reliable multi-person tracker. Loitering still needs event-duration criteria and track-level evidence at deployment; the thermal study supports investigating that representation but does not solve RGB transfer. The public-sleep photo study concerns visible sleep/context and never housing status. No current candidate is established as a universal five-event alert system.",
        "",
        "## Artifacts and reproducibility",
        "",
        "- [Study protocol](../../reports/strategy_protocol_20261016.md), [workflow guide](../../docs/strategy_study.md), [study state](run.json), [verification](verification.json).",
        "- [Findings overview](../../reports/strategy_findings_20261016.md), [research model card](model_card.md) and [checkpoint/policy index](model_bundle.json).",
        "- [Source-separated numeric export](comparison.csv), [paired source-group uncertainty](group_uncertainty.json), [new-source figure](figures/new_source_ap.png), [historical AP changes](figures/historical_ap_change.png).",
        "- Per-backbone folders retain `selection.json`, every candidate head, `selected_heads.pt`, development scores and `transfer_evaluation/` predictions/thresholds. Use `scripts/predict_strategy.py` with the recorded base/interpolated weights and detector for experimental inference.",
        "- [Trajectory component results](thermal_trajectories/results.json), [source-identity audit](thermal_trajectories/source_identity_audit.json), [new data metadata](../../data/strategy_study_20261016/dataset.json), and source acquisition/review receipts under `data/strategy_sources_20261016/`.",
        "",
        "Grouped bootstrap uses 2,000 paired draws, seed 20261015. Camera/source support is too small for meaningful intervals in some cohorts; photographer/title groups do not guarantee person independence. No seed replication or latency experiment is added by this exploratory screen. All visual features use float32 with TF32 disabled.",
        "",
    ]
    (RUN / "report.md").write_text("\n".join(lines))
    (RUN / "report_data.json").write_text(
        json.dumps(
            {
                "primary_rows": rows,
                "families": families,
                "routes": routes,
                "numeric_rows": len(exported),
            },
            indent=2,
        )
        + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = RUN / "figures"
    figures.mkdir(exist_ok=True)
    sources = list(NAMES)[:4]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, (model, label) in zip(axes, MODELS.items()):
        x = np.arange(4)
        for offset, family, color, legend in [
            (-0.17, "previous_bundle", "#8799a5", "Previous bundle"),
            (0.17, "selected", "#007e80", "New dev-selected route"),
        ]:
            axis.bar(
                x + offset,
                [all_metrics[model, source, family]["raw"]["ap"] for source in sources],
                0.32,
                label=legend,
                color=color,
            )
        axis.scatter(
            x,
            [
                all_metrics[model, source, "selected"]["raw"]["positive"]
                / all_metrics[model, source, "selected"]["raw"]["n"]
                for source in sources
            ],
            marker="_",
            s=300,
            color="#172d40",
            label="Positive prevalence",
        )
        axis.set_xticks(
            x,
            [
                "Smoking\nphoto holdout\n(n=224)",
                "Smoking\nHMDB clips\n(n=906)",
                "Public sleep*\nnew authors\n(n=28)",
                "Unattended*\nMEVA action\n(n=27)",
            ],
        )
        axis.set_ylim(0, 1.05)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Average precision")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3)
    fig.suptitle(
        "New held-out photos and source transfer — fixed development selections"
    )
    fig.text(
        0.5,
        0.015,
        "* Public-sleep labels are provisional; MEVA measures an abandonment-action proxy. The photo holdout shares its collection with training data.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.87])
    fig.savefig(figures / "new_source_ap.png", dpi=180)
    fig.savefig(figures / "new_source_ap.pdf")
    plt.close(fig)
    sources = [
        "smoking_cctv",
        "mivia_iwdd",
        "street_scene",
        "commons_visible_sleeping",
        "unattended_reviewed",
    ]
    fig, axis = plt.subplots(figsize=(10, 4.5))
    x = np.arange(5)
    for offset, (model, label) in zip([-0.17, 0.17], MODELS.items()):
        axis.bar(
            x + offset,
            [
                all_metrics[model, s, "selected"]["raw"]["ap"]
                - all_metrics[model, s, "previous_bundle"]["raw"]["ap"]
                for s in sources
            ],
            0.32,
            label=label,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(
        x,
        ["Smoking CCTV", "Waste dumping", "Loitering", "Public sleep*", "Unattended*"],
    )
    axis.set_ylabel("AP change from previous bundle")
    axis.legend()
    axis.set_title("Historical diagnostics — previously inspected holdouts")
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figures / "historical_ap_change.png", dpi=180)
    plt.close(fig)
    print(
        "SOURCE-SEPARATED STRATEGY REPORT COMPLETE",
        len(exported),
        "numeric operating-point rows",
        flush=True,
    )


if __name__ == "__main__":
    main()
