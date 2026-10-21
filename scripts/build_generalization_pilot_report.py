"""Build the 20 October pilot report from immutable measured outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/generalization_pilot_20261020"
DATA = ROOT / "data/generalization_pilot_20261020"
OUT = ROOT / "reports/generalization_pilot_20261020"
MODELS = ("mobileclip2-s2", "siglip2-b16-224")
SHORT = {MODELS[0]: "MobileCLIP", MODELS[1]: "SigLIP2"}
SOURCES = {
    "smoking_cctv": "CCTV",
    "stanford40": "Stanford40",
    "kaggle_sujaykapadnis_smoking": "Kaggle photos",
    "xrf55_rgb_public": "XRF55 proxy",
}
FAMILIES = ("old_only", "new_only", "mixed")


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percent(v):
    return f"{v * 100:.1f}%"


def mdtable(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        ]
    )


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(events, smoke):
    directory = OUT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    modes = ("full", "item_tiles", "all_tiles")
    colors = ("#187985", "#c77a27", "#a94247")
    for ax, group, title in zip(
        axes,
        ("avss_primary", "historical_provisional"),
        ("AVSS published primary (2 events)", "ABODA provisional (5 events)"),
    ):
        cells = [
            next(
                r
                for r in events["aggregate"]
                if r["mode"] == mode and r["evaluation_group"] == group
            )
            for mode in modes
        ]
        ax.bar(range(3), [r["event_recall"] for r in cells], color=colors)
        for i, r in enumerate(cells):
            ax.text(
                i,
                r["event_recall"] + 0.03,
                f"{r['detected_events']}/{r['events']}; FP={r['normal_false_alerts']}",
                ha="center",
                fontsize=9,
            )
        ax.set_xticks(range(3), ("Full", "Item tiles", "All tiles"))
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Event recall")
        ax.set_title(title, fontsize=11)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"stream_comparison.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), constrained_layout=True)
    for ax, model in zip(axes, MODELS):
        values = np.array(
            [
                [
                    smoke[model]["results"][f + "/" + source]["operating_points"][
                        "dev_fpr_budget"
                    ]["ap"]
                    for source in SOURCES
                ]
                for f in FAMILIES
            ]
        )
        ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        for i in range(3):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > 0.65 else "black",
                )
        ax.set_xticks(range(4), SOURCES.values(), rotation=20, ha="right")
        ax.set_yticks(range(3), ("Old only", "New only", "Mixed"))
        ax.set_title(SHORT[model] + " held-out AP")
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"source_ap.{suffix}", dpi=180)
    plt.close(fig)


def tex_escape(s):
    s = (
        s.replace("–", "--")
        .replace("—", "---")
        .replace("“", "``")
        .replace("”", "''")
        .replace("’", "'")
    )
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
    return "".join(replacements.get(c, c) for c in s)


def text_tex(s):
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)
    return tex_escape(s.replace("**", "").replace("`", ""))


def latex(markdown):
    output = [
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[margin=20mm]{geometry}",
        r"\usepackage{lmodern}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{graphicx,longtable,booktabs,array,listings}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\emergencystretch}{4em}",
        r"\lstset{basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible}",
        r"\title{Public-source generalization pilot}",
        r"\author{Five-anomaly evaluation project}",
        r"\date{20 October 2026}",
        r"\begin{document}\maketitle\tableofcontents\newpage",
    ]
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            i += 1
            continue
        if line.startswith("## "):
            output.append(r"\section{" + text_tex(line[3:]) + "}")
        elif line.startswith("### "):
            output.append(r"\subsection{" + text_tex(line[4:]) + "}")
        elif line.startswith("!["):
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            assert match
            path = Path(match[2]).with_suffix(".pdf")
            output += [
                r"\begin{figure}[htbp]\centering",
                r"\includegraphics[width=\linewidth]{" + str(path) + "}",
                r"\caption{" + text_tex(match[1]) + "}",
                r"\end{figure}",
            ]
        elif line.startswith("```"):
            output.append(r"\begin{lstlisting}")
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                output.append(lines[i])
                i += 1
            output.append(r"\end{lstlisting}")
        elif line.startswith("| "):
            table = []
            while i < len(lines) and lines[i].startswith("| "):
                cells = [v.strip() for v in lines[i].strip("| ").split("|")]
                if not all(re.fullmatch(r"[-:]+", c) for c in cells):
                    table.append(cells)
                i += 1
            n = len(table[0])
            width = 1 / n
            output += [
                r"{\scriptsize\setlength{\tabcolsep}{3pt}",
                r"\begin{longtable}{@{}"
                + (
                    r">{\raggedright\arraybackslash}p{\dimexpr "
                    + f"{width:.5f}"
                    + r"\linewidth-6pt\relax}"
                )
                * n
                + "@{}}",
                r"\toprule",
            ]
            for j, row in enumerate(table):
                output.append(" & ".join(text_tex(c) for c in row) + r" \\")
                if j == 0:
                    output.append(r"\midrule\endhead")
            output += [r"\bottomrule\end{longtable}}"]
            continue
        elif line.startswith("- "):
            output.append(r"\begin{itemize}")
            while i < len(lines) and (
                lines[i].startswith("- ") or not lines[i].strip()
            ):
                if lines[i].startswith("- "):
                    output.append(r"\item " + text_tex(lines[i][2:]))
                i += 1
            output.append(r"\end{itemize}")
            continue
        else:
            output.append(text_tex(line))
        i += 1
    return "\n".join(output + [r"\end{document}", ""])


def main():
    verification = load(RUN / "verification.json")
    native = load(RUN / "native_feature_validation.json")
    assert verification["status"] == native["status"] == "passed"
    assert load(RUN / "unit_tests.json")["exit_code"] == 0
    OUT.mkdir(parents=True, exist_ok=True)
    events = load(RUN / "results.json")
    smoke = {m: load(RUN / "smoking" / m / "results.json") for m in MODELS}
    selections = {m: load(RUN / "smoking" / m / "selection.json") for m in MODELS}
    smol = {
        n: load(RUN / "smolvlm_transfer" / n / "results.json")
        for n in ("original", "lora")
    }
    figures(events, smoke)
    event_rows = []
    for r in events["rows"]:
        event_rows.append(
            {k: r[k] for k in ("mode", "recording_id", "evaluation_group")}
            | r["metrics"]
            | {"diagnostic_" + k: v for k, v in r["diagnostics"].items()}
        )
    write_csv(OUT / "streams.csv", event_rows)
    write_csv(OUT / "stream_aggregate.csv", events["aggregate"])
    cells = []
    for model in MODELS:
        for key, value in smoke[model]["results"].items():
            family, source = key.split("/")
            for point, metric in value["operating_points"].items():
                cells.append(
                    {
                        "model": model,
                        "family": family,
                        "source": source,
                        "operating_point": point,
                    }
                    | metric
                )
    write_csv(OUT / "smoking_all_operating_points.csv", cells)
    candidates = []
    selected_rows = []
    for model in MODELS:
        for name, c in selections[model]["candidates"].items():
            for source, metric in c["development"].items():
                candidates.append(
                    {
                        "model": model,
                        "candidate": name,
                        "source": source,
                        "selected": name in selections[model]["selected"].values(),
                        "training_examples": c["training_examples"],
                        "converged": c["converged"],
                    }
                    | metric
                )
        for family, name in selections[model]["selected"].items():
            c = selections[model]["candidates"][name]
            selected_rows.append(
                [
                    SHORT[model],
                    family.replace("_", " "),
                    c["recipe"].replace("_", " "),
                    c["C"],
                    c["training_examples"],
                    f"{c['thresholds']['dev_fpr_budget']:.5f}",
                ]
            )
    write_csv(OUT / "candidate_development.csv", candidates)
    subject_rows, action_rows = [], []
    action_table = []
    for model in MODELS:
        for entry in smoke[model]["new_source_breakdown"]:
            if "subject" in entry:
                subject_rows.append(
                    {
                        "model": model,
                        "family": entry["family"],
                        "subject_alias": entry["subject"],
                    }
                    | entry["metrics"]
                )
            else:
                action_rows.append({"model": model} | entry)
        for family in FAMILIES:
            values = {
                r["action"]: r["positive_prediction_rate"]
                for r in smoke[model]["new_source_breakdown"]
                if r["family"] == family and "action" in r
            }
            action_table.append(
                [
                    SHORT[model],
                    family.replace("_", " "),
                    *(
                        percent(values[a])
                        for a in ("smoking", "drinking", "eating", "using_phone")
                    ),
                ]
            )
    write_csv(OUT / "new_source_subjects.csv", subject_rows)
    write_csv(OUT / "new_source_actions.csv", action_rows)
    primary_table = []
    for r in events["aggregate"]:
        primary_table.append(
            [
                r["evaluation_group"]
                .replace("historical_provisional", "ABODA provisional")
                .replace("avss_", "AVSS "),
                r["mode"].replace("_", " "),
                f"{r['detected_events']}/{r['events']}",
                r["normal_false_alerts"],
                r["duplicate_alerts"],
                r["unknown_time_alerts"],
                f"{r['normal_seconds'] / 60:.2f}",
            ]
        )
    smoking_table = []
    references = []
    for model in MODELS:
        for key, value in smoke[model]["results"].items():
            family, source = key.split("/")
            point = "raw" if family == "pretrained_text" else "dev_fpr_budget"
            c = value["operating_points"][point]
            row = [
                SHORT[model],
                family.replace("_", " "),
                SOURCES[source],
                c["n"],
                f"{c['ap']:.4f}",
                percent(c["recall"]),
                percent(c["fpr"]),
                f"{c['tp']}/{c['positive']}; {c['fp']}/{c['negative']}",
            ]
            (smoking_table if family in FAMILIES else references).append(row)
    smol_table = []
    for name, value in smol.items():
        for calibration in ("new_source_calibrated", "original19oct_cutoffs"):
            c = value[calibration]["dev_fpr_budget"]
            smol_table.append(
                [
                    name,
                    "New-source dev"
                    if calibration == "new_source_calibrated"
                    else "19 Oct dev",
                    f"{c['ap']:.4f}",
                    percent(c["recall"]),
                    percent(c["fpr"]),
                    f"{c['tp']}/{c['positive']}; {c['fp']}/{c['negative']}",
                ]
            )
    conclusions = []
    for model in MODELS:
        for source in ("smoking_cctv", "xrf55_rgb_public"):
            old = smoke[model]["results"]["old_only/" + source]["operating_points"][
                "dev_fpr_budget"
            ]
            mixed = smoke[model]["results"]["mixed/" + source]["operating_points"][
                "dev_fpr_budget"
            ]
            conclusions.append(
                f"- {SHORT[model]}, {SOURCES[source]}: old-only to mixed AP {old['ap']:.4f} to {mixed['ap']:.4f}; recall {percent(old['recall'])} to {percent(mixed['recall'])}; FPR {percent(old['fpr'])} to {percent(mixed['fpr'])}."
            )
    primary_full = next(
        r
        for r in events["aggregate"]
        if r["mode"] == "full" and r["evaluation_group"] == "avss_primary"
    )
    document = f"""# Public-source generalization pilot — 20 October 2026

This is an addendum to the completed 19 October technical runthrough. It records actual new public-data acquisition, 36 classifier fits, new-source SmolVLM transfer, and the full-stream tiling experiment. All measured outputs were produced on the local CUDA runtime without pacing or a latency benchmark. Earlier experiments and their reports are preserved.

## Findings and decision

Full-stream tiling regresses the primary unattended-event result: full-frame detection finds 2/2 published AVSS events with zero reviewed normal-time alerts; both tiled modes find 1/2 and add one normal-time alert. The earlier CAVIAR box-recall gain does not establish a better complete event detector. Retain the full-frame baseline for the next prospective comparison.

The smoking data-addition results are:

{chr(10).join(conclusions)}

These comparisons use family-specific development-selected heads and frozen development cutoffs. Historical test cohorts have already been inspected in earlier studies. The fresh XRF55 test measures scripted action discrimination on a new scene and three held-out numeric subject groups. It cannot establish improvement on genuine CCTV smoking events. No model is automatically promoted, and no five-objective deployment winner is established.

SmolVLM's original-to-prior-LoRA AP on this proxy source falls from 0.3855 to 0.3573. With newly frozen source-development budget cutoffs, original recall/FPR are 61.7%/36.7%; LoRA recall/FPR are 35.0%/18.3%. Both older budget cutoffs predict no positives. This is further evidence of scene-dependent ranking and calibration, within the limits of the source-action labels.

## Public data collection and label compatibility

The [author XRF55 repository](https://github.com/airslab2020/XRF55-repo) identifies the action mapping and public RGB release. The [public Kaggle RGB/depth/IR release](https://www.kaggle.com/datasets/airslab2020/xrf55-rgb-depth-ir), version 6, supplies the selected RGB clips. The release README records CC BY-NC 4.0; retain the source's attribution and use conditions. Local receipts establish downloaded-file identity, not author-issued checksums.

Downloaded: **740 clips, 886,570,077 bytes**, representing four publisher primary-action labels: using phone (03), drinking (13), eating (14), smoking (15). There are 185 class-15 clips and 555 controls. Every clip has 75 frames at 15 fps (about five seconds). These short correlated repetitions do not constitute an hour of reviewed normal CCTV exposure.

A visual check restricted to training examples (scene 1, numeric subject 01, actions 03/13/15, frames 10/40/65) found a masked actor and a smoking-labelled hand-to-mouth gesture. A cigarette was not independently verified. The source is therefore retained as a **weak smoking-action/gesture proxy**. The original class IDs are preserved; no subjective relabelling of held-out examples was used. Publisher primary-action labels are not exhaustive five-anomaly negatives.

{mdtable(["Split", "Scene IDs", "Numeric subject groups", "Clips", "Positive / controls", "Purpose"], [["Train", "1 and 2", "15 (17 scene-subject groups)", 340, "85 / 255", "Weak supervised fitting"], ["Development", "3", "2", 160, "40 / 120", "Selection and cutoffs"], ["Test", "4", "3", 240, "60 / 180", "Frozen source transfer"]])}

Numeric subject IDs are conservatively treated as global aliases across scenes. Train/dev/test aliases do not overlap. Scene IDs denote source blocks; they do not prove distinct cameras. Training uses repetitions 1, 4, 7, 10, 13; development/test use all 20 repetitions. The split supersedes the publisher repetition-only split and was fixed before downloading and model scoring. Test diversity is one scene and three subject groups, regardless of its 240 clips.

Preparation decodes sixteen ordered uniform views (indices rounded across 0–74), resizes to a maximum edge of 768 pixels, and saves JPEG quality 92: **11,840 model-input views**. Encoded-file SHA-256 and canonical decoded RGB hashes are saved. No exact decoded pixels cross the new splits or overlap the historical smoking inputs. This does not rule out perceptual similarity or correlated actions within subjects.

Acquisition originally received 404 responses because slashes in individual-file paths were not percent-encoded. The saved transport amendment percent-encodes the entire filename; labels, requested files and splits are unchanged. A separate pre-score amendment aligns ancillary eight-view SmolVLM frame metadata with its selected model inputs.

Other public leads were checked rather than silently counted as new target evidence. PETS original/linked hosts and a CDnet mirror were unavailable during the probes. The AOD survey package already acquired AVSS. One abandoned-luggage repository contained synthetic/Photoshop variants and rendered method outputs; those were rejected as real-event tests. NTU RGB+D's cited taxonomy lacks smoking; Kyoto LMA provides an incompatible depth/skeleton release; the cited bilayer-smoking repository did not supply downloadable annotated clips. The source audit records URLs, observations and exclusions. No independent human reviewer or additional local CCTV was available.

## Controlled CLIP smoking experiments

The two pinned pretrained encoders are frozen: MobileCLIP2-S2 and SigLIP2-B/16-224. This round fits **36 logistic-regression heads, with zero new visual-backbone fine-tunes**. Both frozen pretrained encoders and all new features use float32 CUDA, TF32 disabled. The previous six visual CLIP fine-tunes and three SmolVLM LoRA training runs remain separate historical work.

Each encoder has three training families: old-only (2,812 examples), new-only (340 weak proxy clips), mixed (3,152). Each family tries two representations and C in 0.1, 1, 10: 18 candidates per encoder. Logistic regression uses LBFGS, at most 1,000 iterations and tolerance 1e-6; every fitted candidate converged. Weights balance annotated class, source and recording/author group. Exported coefficients and all selection scores are float32.

The full representation concatenates the last view, mean, elementwise maximum and last-minus-first embedding, followed by L2 normalization. Full-plus-crop concatenates normalized full/crop representations and normalizes again. Regions are predicted automatically by the same fixed COCO Faster R-CNN ResNet50 FPN v2 used previously (confidence 0.5, person/backpack/handbag/suitcase); they do not use smoking labels or ground-truth boxes. The highest-confidence contextual person and item regions are combined, with 20% padding; missing regions fall back to the full image. New data use sixteen views; old samples retain their existing views. This is a clip classifier, not an onset/offset detector.

All candidates use the **same combined development cohort of 1,474 cases across four sources**. A single global cutoff must yield empirical FPR at most 5% in each development source. Within each family the selection order is worst-source recall, mean-source recall, worst-source AP, mean-source AP. Both model selections are frozen before held-out scoring. Raw zero and pooled development maximum-F1 cutoffs are also saved.

Because the same combined development set is used, old-only receives new-source calibration and new-only receives old-source development feedback. They differ in classifier training data, not in all forms of source supervision. The new-source test scene and subjects remain held out. The family comparison includes development selection among the same two representations and C values; it does not isolate a single fixed head architecture.

{mdtable(["Encoder", "Family", "Selected representation", "C", "Train n", "Global dev cutoff"], selected_rows)}

Each encoder scores **1,916 held-out cases**: 646 CCTV, 806 Stanford40, 224 Kaggle photos and 240 new-source proxy clips. The first 1,676 are retrospective historical tests; these are not 1,916 independent events or people. Five prediction families per encoder are retained: the three controlled training families, the genuine pretrained contrastive text-margin control, and the frozen historical native strategy.

### Held-out results at the every-source development budget cutoff

AP evaluates ranking and does not depend on the displayed cutoff. Recall and FPR use each family's frozen single global development cutoff. TP/P and FP/N make the denominators explicit. A 5% development constraint is not a guarantee of 5% test FPR.

{mdtable(["Encoder", "Family", "Source", "n", "AP", "Recall", "FPR", "TP/P; FP/N"], smoking_table)}

![Controlled old/new/mixed held-out ranking by source; historical and fresh proxy evidence remain distinct](figures/source_ap.png)

### Original and historical reference controls

The pretrained text score is the mean full-image contrastive binary margin over each clip, with raw zero cutoff. The historical 16 October selected strategy uses its native last-frame smoking input and original development thresholds. It therefore differs from the sixteen-view controlled data-addition recipes. Neither is silently renamed as a newly trained backbone.

{mdtable(["Encoder", "Reference", "Source", "n", "AP", "Recall", "FPR", "TP/P; FP/N"], references)}

All three operating points, all 36 candidate development tables, per-new-subject metrics and per-negative-action prediction rates are preserved in machine-readable outputs. The three-subject test is too small for a strong population-generalization claim; repetition counts must not be treated as independent people.

### Where the new-source errors occur

Each action has 60 held-out clips. These are the positive prediction rates at the same frozen development-budget cutoffs. For publisher smoking they represent recall; for the three control actions they represent false-positive proportions within that action, subject to the weak primary-action labels.

{mdtable(["Encoder", "Family", "Smoking", "Drinking", "Eating", "Phone"], action_table)}

All new-source control errors in the controlled CLIP families occur on drinking clips. Mixed SigLIP's per-subject smoking recalls are 10%, 0%, and 55% for numeric aliases 03, 04, and 13; each has 20 positive clips and 60 controls. The aggregate 21.7% recall therefore conceals large subject variation. These observed error breakdowns are retrospective diagnostics; they do not trigger a new test-fitted threshold or classifier selection.

## SmolVLM original-versus-LoRA source transfer

This is **transfer evaluation, with no new SmolVLM training**. Compare the pinned original SmolVLM-500M-Instruct against the previous development-selected smoking adapter (seed 42, learning rate 0.0001). The existing adapter changes 4,341,760 language projection parameters; vision remains frozen. No new-source test labels select the adapter or prompt.

Both routes receive the same eight uniformly spaced frames selected from the sixteen-view manifest. The validated native prompt supports up to eight images. This within-SmolVLM comparison is matched; its eight-view input is different from the CLIP sixteen-view recipe. Frozen visual tokens are reused without changing the vision tower. The fixed question includes the existing smoking definition. Scores are the float32 first-answer Yes-minus-No logits, rather than explanations or generated class labels.

Both routes score 160 development cases and freeze new-source maximum-F1 and 5%-FPR cutoffs before either route scores the 240 test cases. The 19 October original thresholds are retained separately to reveal transfer calibration failure. No fresh-source route substitution is made.

{mdtable(["Route", "Cutoff calibration", "AP", "Recall", "FPR", "TP/P; FP/N"], smol_table)}

Native-image inference validates one test case per route against the visual-token cache at an absolute margin tolerance of 0.002. Per-case Yes/No token mass is also retained; a binary margin measures a constrained comparison, not necessarily a confident free-form answer.

## Full-stream unattended-item test

Ten existing recordings are processed from their beginnings to EOF at 2 Hz, with 2,512 unique source images and 12,560 detector views. Seven ABODA videos represent two conservative scene/camera blocks; three AVSS recordings represent one conservative source block. These are three source/camera blocks, not ten independently sampled cameras. All footage was previously inspected. The ABODA intervals remain provisional assistant-derived annotations. Published AVSS EASY/MEDIUM are primary; HARD remains a separate right-censored diagnostic.

The pinned Grounding DINO tiny model uses the unchanged query “person . bag . backpack . suitcase . cardboard box . package .”, box threshold 0.25 and text threshold 0.20, with maximum item box area 20% of the full frame. Four 60%-width/height corner tiles overlap around the center. Coordinates are remapped using actual integer crop offsets and normalized to the full image. Class-wise greedy NMS uses IoU 0.5. Exactly the same raw five views underlie every arm.

- Full: full-frame people and items only.
- Item tiles: full-frame people plus full/tiled item proposals; this isolates item proposal coverage.
- All tiles: full/tiled people and items.

Each recording/arm starts with a fresh causal tracker: new confidence 0.25, low confidence 0.20, maximum association gap two seconds, and at least two seconds of owner proximity evidence. The same existing state rule requires known ownership, at least 30 seconds of item stationarity and at least 0.5 seconds of owner absence. Alerts require two seconds persistence, one second release, ten seconds cooldown, and maximum sample gap 1.1 seconds. There is no annotation-driven frame selection, threshold search, state retuning or event-model training in this arm.

The complete recording is processed chronologically without future observations. Event recall uses the first emitted alarm within each positive interval; duplicates are additional alarms within an already detected event. Normal false alerts use only reviewed normal intervals. Positive intervals take precedence at shared boundaries. Alarms during unreviewed time remain unknown, not automatic false positives or correct detections. Alarm delay is measured from annotated event onset, not runtime inference latency.

{mdtable(["Evidence group", "Arm", "Detected", "Normal FP", "Duplicates", "Unknown", "Normal minutes"], primary_table)}

![Tiling reduces full-stream event recall and introduces normal-time alerts](figures/stream_comparison.png)

The full-frame primary delays are 24.72 seconds (EASY) and 44.91 seconds (MEDIUM), median 34.82 seconds. The primary normal exposure is {primary_full["normal_seconds"]:.3f} seconds (3.43 minutes). Zero observed normal alerts yields a one-sided 95% Poisson upper bound of **{primary_full["false_rate_one_sided_95pct_upper_per_hour"]:.2f} alerts/hour** under that rate model; this is far too weak to validate operational reliability. It also does not address camera/domain dependence. HARD's normal exposure is 118.32 seconds and ABODA's is 359 seconds, reported separately.

Earlier primary tables started scoring at t >= 20 seconds, giving 165.739 normal seconds and a 65.07/hour zero-alert bound. Starting at zero adds forty seconds of the same existing recordings, not new cameras or newly collected normal footage. Per-recording aligned t >= 20 metrics are included in results.json for direct comparison. The full-frame arm still detects 2/2 primary events in that aligned exposure.

More proposals are not necessarily better tracks. On MEDIUM, unique emitted item-track IDs increase from 284 (full) to 497 (either tile arm); its event is missed with tiles. On EASY, counts increase from 162 to 371 and a reviewed normal false alarm appears. These are tracker proposal/identity diagnostics, not ground-truth identity-switch rates. They are consistent with more fragmented or spurious associations, but no owner/track reference annotations establish the exact error mechanism. Preserve the failure cases for association and ownership supervision, rather than choosing new rules on these same test recordings.

## Verification, artifacts and reproduction

The stream verifier recomputes all **7,536** tracker/state decisions, every emitted alarm and each per-stream metric from raw detector output; it also checks aggregate counts and 2,512 encoded/decoded image identities. The smoking verifier reconstructs all candidate development scores, cutoffs, convergence flags and all six selected family choices, then reconstructs the selected test and reference scores. It checks all 11,840 new decoded inputs, cross-split aliases/pixels, cache normalization and manifest identity. SmolVLM dev thresholds and test operating points are replayed for both routes.

Native CUDA CLIP validation re-encodes full and crop views for three fixed cases per encoder, covering train/dev/test and both labels, then compares selected head margins against cached features. SmolVLM separately checks its native image path. Verification does not turn weak action labels into verified CCTV truth. The unit suite previously passed 192 tests, including five new tile-coordinate/fusion invariants; test execution output is preserved separately.

Primary local artifacts:

- `data/generalization_pilot_20261020/source_audit.json`: public-source decisions and exclusions.
- `data/generalization_pilot_20261020/xrf55/`: acquisition protocol/amendment, receipts, label audit, manifest, videos, frames and feature caches.
- `results/generalization_pilot_20261020/protocol.json`, `results.json`, `traces.jsonl`, `alerts.jsonl`: frozen stream settings and all measured arms.
- `results/generalization_pilot_20261020/smoking/`: shared protocol, all heads, per-candidate dev predictions, frozen selections, test predictions and source/subject/action metrics.
- `results/generalization_pilot_20261020/smolvlm_transfer/`: original/adapter protocol, calibration, test predictions, token-cache identities and native-path validation.
- `results/generalization_pilot_20261020/verification.json` and `native_feature_validation.json`: replay and live CUDA checks.
- This report's CSVs contain per-stream metrics, aggregate events, all held-out operating points and all development candidates. Figures are supplied as PNG and vector PDF. `runthrough.tex` is the compilable source for the PDF addendum.

Commands and runtime dependencies are in `docs/generalization_pilot.md`. The CPU environment handles convex classifier fitting and numerical verification; CUDA handles neural feature extraction and inference. Accuracy runs are unpaced and no new CPU latency experiment is run.

## Consequences for the five objectives

- Smoking: additional public clips supply an honest weak-label/data-source ablation. Prioritize visible-cigarette confirmation, distant CCTV examples and matched drinking/eating/phone hard negatives. Preserve source separation and a single development-only operating threshold.
- Unattended items: retain the full-frame causal baseline. The next training target is coherent item/person association, owner evidence and long-gap recovery, with reviewed owner/item histories and longer normal streams. More box recall alone failed this test.
- Littering: the previous IWDD and transfer evidence remains unchanged in this pilot. Collect or locate independently annotated release/disposal events and matched putting-down/picking-up controls before claiming street-litter generalization.
- Loitering: previous transfer and seed regressions remain unresolved. Duration, trajectory and scene-zone supervision require longer person tracks and camera/day-separated validation; isolated action clips are insufficient.
- Rough sleeping: apparent public sleeping/resting remains a visual proxy. Housing status cannot be inferred from appearance. Add independently reviewed temporal resting/wake/movement controls and new locations without broadening the label claim.

The next acceptance test should be frozen before any new independent event footage is scored: independent source/camera blocks, explicit onset/offset and owner/item labels where relevant, matched negative exposure, per-source recall/FPR and alert burden with denominators. Under a Poisson model, zero alerts in about three reviewed normal camera-hours gives an upper rate near one/hour; source coverage and a longer observation period are still needed. Neither 740 repeated lab clips nor 3.43 minutes of primary normal video solves that evidence gap.
"""
    (OUT / "report.md").write_text(document)
    (OUT / "runthrough.tex").write_text(latex(document))
    (RUN / "report.md").write_text(
        document.replace(
            "(figures/", "(../../reports/generalization_pilot_20261020/figures/"
        )
    )
    source_paths = [
        RUN / "verification.json",
        RUN / "native_feature_validation.json",
        RUN / "results.json",
        DATA / "source_audit.json",
        DATA / "xrf55/preparation.json",
        Path(__file__),
        RUN / "protocol.json",
        RUN / "smoking/protocol.json",
        RUN / "smolvlm_transfer/protocol.json",
        RUN / "unit_tests.json",
        RUN / "unit_tests.log",
        DATA / "xrf55/samples.jsonl",
        DATA / "xrf55/input_identities.jsonl",
        DATA / "xrf55/detections.jsonl",
        DATA / "xrf55/detections.metadata.json",
        ROOT / "docs/generalization_pilot.md",
    ]
    source_paths += [
        RUN / "smoking" / m / name
        for m in MODELS
        for name in ("results.json", "selection.json")
    ]
    source_paths += sorted((DATA / "xrf55/features").glob("*/*"))
    source_paths += [
        ROOT / "scripts" / name
        for name in (
            "run_generalization_pilot.py",
            "fetch_xrf55_pilot.py",
            "train_source_generalization.py",
            "extract_source_pilot_features.py",
            "evaluate_smolvlm_source_pilot.py",
            "verify_generalization_pilot.py",
            "validate_source_pilot_gpu.py",
        )
    ]
    source_paths += [RUN / "smolvlm_transfer" / n / "results.json" for n in smol]
    (OUT / "source_manifest.json").write_text(
        json.dumps(
            {
                "date": "2026-10-20",
                "status": "built_from_verified_outputs",
                "sources_sha256": {
                    str(p.relative_to(ROOT)): digest(p) for p in source_paths
                },
                "documents_sha256": {
                    p.name: digest(p)
                    for p in (OUT / "report.md", OUT / "runthrough.tex")
                },
                "exports_sha256": {
                    str(p.relative_to(OUT)): digest(p)
                    for p in sorted(
                        list(OUT.glob("*.csv")) + list((OUT / "figures").glob("*"))
                    )
                    if p.is_file()
                },
                "no_new_training_in_reporting": True,
            },
            indent=2,
        )
        + "\n"
    )
    print("Built report, LaTeX, CSV tables, and figures", OUT)


if __name__ == "__main__":
    main()
