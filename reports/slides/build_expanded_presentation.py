"""Build the expanded accuracy briefing from completed, verified saved results.

Run with the presentation dependencies and repository root on PYTHONPATH.
This builder performs no inference and preserves the earlier presentation.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from pptx import Presentation

from reports.slides import build_presentation as d

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "results/expanded_accuracy_20261013"
STEM = "model_evaluation_expanded_20261014"
SUMMARY_STEM = "summary_slide_expanded_20261014"
SHORT = ["MobileCLIP2-S2", "SigLIP2-B/16-224", "SmolVLM-500M", "MiniCPM-V 4.0"]


def new(category, title, subtitle, source, notes):
    slide = d.new(category, title, subtitle, source, notes)
    for item in slide["items"]:
        if item["kind"] == "text" and "EVALUATION 13 OCT 2026" in item["value"]:
            item["value"] = item["value"].replace(
                "EVALUATION 13 OCT 2026", "EXPANDED EVALUATION 14 OCT 2026"
            )
    return slide


def brief(slide, y, value, color=d.TEAL):
    d.banner(slide, y, value, color=color, size=14, h=57)


def build_content():
    d.SLIDES.clear()
    d.SOURCES.clear()
    quality = d.read(RUN / "quality_summary.json")
    audit = d.read(RUN / "verification.json")
    state = d.read(RUN / "state.json")
    assert quality["status"] == state["status"] == "complete"
    assert quality["completed_summaries"] == 40
    assert quality["completed_predictions"] == audit["completed_predictions"] == 55156
    assert all(audit["checks"].values()) and state["no_latency_experiments"]
    rows = {(row["cohort"], row["split"], row["model"]): row for row in quality["rows"]}

    def result(cohort, model, split="test"):
        return rows[cohort, split, model]

    for path in [
        RUN / "report.md",
        RUN / "accuracy_comparison.csv",
        ROOT / "data/expanded_20261013/review_decisions.json",
        ROOT / "data/expanded_20261013/cohorts.json",
        ROOT / "scripts/expand_accuracy.py",
        ROOT / "scripts/run_expanded_accuracy.py",
        ROOT / "scripts/report_expanded_accuracy.py",
    ]:
        d.SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for run in state["runs"]:
        for model in state["models"]:
            for name in ["summary.json", "predictions.jsonl"]:
                path = ROOT / run["output"] / model / name
                with path.open("rb") as stream:
                    d.SOURCES[str(path.relative_to(ROOT))] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()

    s = new(
        "Expanded findings / accuracy only",
        "Broader coverage changes the findings",
        "Four image models; provisional unattended labels; weak anomaly diagnostics; retained smoking/VadCLIP",
        "quality_summary.json + raw GPU predictions",
        "The expanded study was finalized on 14 October 2026 Asia/Jakarta. Cohort/run folders retain the 20261013 preparation identifier. All 40 model/cohort/split summaries completed, with 55,156 new predictions. Inputs are protocol evaluations, not independent events: windows share frame endpoints, and person/loitering queries share images. No additional latency experiment was run. MiniCPM has the highest raw unattended frame-test F1, but only 34.7% recall. The new event labels are provisional assistant timeline review, without independent human verification. The other three requested anomalies were attempted as explicitly limited diagnostics. Smoking and supervised offline VadCLIP evidence are reused, not rerun.",
    )
    for x, value, label, detail in [
        (42, "55,156", "New GPU predictions", "40 completed model/split runs"),
        (340, "1,770", "Unattended endpoints", "36 previously; 49.2× increase"),
        (638, "962", "Held-out inputs / protocol", "242 positive + 720 normal"),
    ]:
        d.stat(s, x, 168, 280, 144, value, label, detail)
    d.card(
        s,
        42,
        331,
        427,
        156,
        "MiniCPM: strongest raw frame F1",
        "F1 0.479 • recall 34.7% • FPR 3.47%\nMany positive endpoints are still missed.",
        size=16,
    )
    d.card(
        s,
        491,
        331,
        427,
        156,
        "Three anomalies now attempted",
        "6,667 littering; 1,675 loitering; 111 pose scenes. Weak labels limit accuracy claims.",
        color=d.ORANGE,
        size=16,
    )

    s = new(
        "Coverage / what increased",
        "Use the available data at its valid level",
        "13,789 input/protocol evaluations per model; shared images and endpoints are counted explicitly",
        "data/expanded_20261013/cohorts.json",
        "The unattended frames and 20-second windows each have 808 dev and 962 test inputs. Both protocols share 1,770 endpoints. Littering contributes 6,667 unique decoded images after removing 17 aliases from the usable collector rows; augmentations remain and are not independent source events. Loitering and person-presence are different queries on the same 1,675 person-labelled images. Rough sleeping uses 111 full synthetic lying/fall scenes as weak candidates. The oracle pose diagnostic uses 121 box crops from the same pose source, split by original image into 73 dev and 48 test crops. The prior smoking study already uses 3,278 local development frames and is retained separately.",
    )
    d.table(
        s,
        42,
        165,
        [275, 92, 92, 417],
        ["Study", "Dev", "Test", "What the labels support"],
        [
            [
                "Unattended: frames",
                "808",
                "962",
                "Provisional event/control comparison",
            ],
            [
                "Unattended: 20 s windows",
                "808",
                "962",
                "Same endpoints, four-frame sheets",
            ],
            [
                "Littering candidates",
                "6,667",
                "—",
                "Weak action labels; target-prediction rate",
            ],
            ["Loitering candidates", "1,675", "—", "Person boxes; duration unverified"],
            ["Person presence", "1,675", "—", "Same images; grounding diagnostic"],
            ["Rough-sleeping candidates", "111", "—", "Synthetic lying/fall scenes"],
            ["Lying vs standing crops", "73", "48", "Oracle boxes; posture diagnostic"],
        ],
        row_h=39,
        font=13.5,
    )
    d.text(
        s,
        45,
        480,
        870,
        17,
        "More sampled inputs do not imply more independent events or cameras.",
        12,
        d.GRAY,
    )

    s = new(
        "Methodology / source review",
        "Recover events from source timelines",
        "Assistant review precedes model inference; independent human verification remains necessary",
        "review_decisions.json + review/ boards + CAVIAR XML",
        "Source videos were reviewed chronologically and event/control intervals were chosen before the new inference. CAVIAR XML roles supplement visual timeline evidence, rather than equating an object box with abandonment. Frames are lossless native-resolution PNGs at actual source frame indices, sampled approximately 2 fps; source SHA, frame index and timestamps are saved. The 18 included recordings comprise eight CAVIAR and ten ABODA recordings, contributing event intervals and/or clear controls. Exact duplicate raw/additional video copies are not counted twice. Behind-chair occlusion, owners remaining nearby, occluded ABODA placements and a pre-existing item without owner history are excluded as positives. This is assistant-generated provisional annotation, not independent human ground truth.",
    )
    d.card(
        s,
        42,
        170,
        280,
        250,
        "1. Inspect source videos",
        "Chronological review boards\n~2 fps native frames\nCAVIAR role/XML checks\n\n18 included recordings; raw/additional duplicates removed.",
        size=16,
    )
    d.card(
        s,
        340,
        170,
        280,
        250,
        "2. Freeze intervals",
        "Choose visible departure and clear normal controls.\n\nExclude occlusion, nearby owners and items lacking placement history.",
        size=16,
    )
    d.card(
        s,
        638,
        170,
        280,
        250,
        "3. Record provenance",
        "Source SHA and frame index\nReviewed interval decisions\nDecoded image identity\n\nNo inferred labels from model responses.",
        size=16,
    )
    brief(
        s,
        439,
        "Labels are provisional assistant review. These results are an event study, not a verified deployment benchmark.",
        d.ORANGE,
    )

    s = new(
        "Methodology / holdout design",
        "Related camera recordings stay together",
        "Camera-family holdout avoids mixing adjacent views and recordings between development and test",
        "Unattended manifest group_id + cohort metadata",
        "Development uses CAVIAR's INRIA camera, ABODA lobby and ABODA entrance families: 808 endpoints, 255 positive and 553 normal. Test uses the ABODA walkway and classroom families: 962 endpoints, 242 positive and 720 normal, seven recordings, five positive source intervals. Related classroom views are conservatively held together. The camera/source clusters, rather than adjacent frame samples, are the bootstrap unit. There are only two test families, so uncertainty estimates are exploratory and cannot establish broad camera generalization. CLIP thresholds are chosen from development and applied unchanged to test. All weak exports remain development-only because event labels and/or source identities are insufficient. Source-image identities keep all pose crops from one original image in the same split.",
    )
    d.card(
        s,
        42,
        170,
        427,
        239,
        "Development: 3 camera families",
        "CAVIAR: INRIA\nABODA: lobby + entrance\n\n808 endpoints: 255 positive / 553 normal\nFit CLIP thresholds on these inputs.",
        size=17,
    )
    d.card(
        s,
        491,
        170,
        427,
        239,
        "Test: 2 held-out families",
        "ABODA: walkway + classroom\n7 recordings; 5 positive intervals\n\n962 endpoints: 242 positive / 720 normal\nApply frozen thresholds once.",
        color=d.ORANGE,
        size=17,
    )
    brief(
        s,
        435,
        "The unit of independence is the source/camera group. 962 adjacent endpoints are not 962 independent events.",
    )

    s = new(
        "Testing approach / GPU quality",
        "Run quality evaluation without pacing",
        "RTX 5000 Ada; four isolated model workers; all eligible inputs; no additional latency pilot",
        "state.json commands + per-model summary.json",
        "All four models ran on the RTX 5000 Ada with CUDA. Accuracy was unpaced, without a default host RAM cap. The four model workers ran concurrently with eight CPU threads each. CLIP inference used batch 64, float32 weights; generative models used single-input calls and unquantized bfloat16 weights. The shared input policy letterboxes and bounds the image to 768 pixels before native model processing. Generation is deterministic with max_new_tokens 64. MiniCPM keeps the previous one-slice configuration. There is one measured prediction per input, no latency repetitions and no warmup calls; this is a quality experiment. Four-frame 2×2 contact sheets use only past context over 20 seconds, versus six seconds in the earlier study. This is not each model's native multi-frame video API, and sheets reduce spatial detail per frame. Exact commands, revisions and raw responses are saved.",
    )
    d.table(
        s,
        42,
        169,
        [280, 176, 164, 256],
        ["Model", "CUDA dtype", "Batch", "Evaluation input"],
        [
            [SHORT[0], "float32", "64", "Shared image / contact sheet"],
            [SHORT[1], "float32", "64", "Shared image / contact sheet"],
            [SHORT[2], "bfloat16", "1", "Deterministic 64-token reply"],
            [SHORT[3], "bfloat16", "1", "64-token reply; one slice"],
        ],
        row_h=44,
        font=15,
    )
    d.text(
        s,
        45,
        397,
        870,
        59,
        "Shared preprocessing: letterbox, maximum side 768 px. Windows: four chronological frames spanning 20 seconds, with no future context. Saved revisions and raw outputs make the comparison reviewable.",
        16,
        d.GRAY,
    )
    d.text(
        s,
        45,
        470,
        870,
        20,
        "No pacing sleeps or accuracy RAM cap; concurrent runs are not a fair latency comparison.",
        13,
        d.TEAL,
        True,
    )

    s = new(
        "Evaluation / what each number means",
        "Keep event quality and diagnostics separate",
        "Abstentions remain in positive denominators; score ranking is separate from categorical decisions",
        "metrics.py + verification.json + calibration.json",
        "Precision is TP/(TP+FP), recall is TP/(TP+FN), and FPR is FP/all labelled normal inputs. Positive abstentions count as missed positives; unanswered normal cases are not silently counted as true negatives. Coverage is always shown when incomplete. AP and AUROC require genuine continuous scores, supplied only for CLIP; no ranking metric is invented for hard VLM labels. Unattended test prevalence is 242/962 = 25.2%, a useful AP reference. Positive-only weak candidate sets have no event precision or FPR; only target-prediction rates and response coverage are reported. The person-presence task has genuine person annotations but no absent-person controls. Source/camera bootstrap uses 2,000 draws, with only two unattended test families. Raw predictions were reloaded and all 40 saved metric summaries matched recomputation exactly.",
    )
    d.card(
        s,
        42,
        170,
        280,
        254,
        "Reviewed event comparison",
        "Precision / recall / F1 / FPR\nCLIP score AP / AUROC\n\nFrozen development thresholds\nCamera-group uncertainty",
        size=16,
    )
    d.card(
        s,
        340,
        170,
        280,
        254,
        "Weak candidate diagnostics",
        "Target-prediction rate\nAbstentions + errors\n\nNo event precision or FPR\nNo verified anomaly recall",
        color=d.ORANGE,
        size=16,
    )
    d.card(
        s,
        638,
        170,
        280,
        254,
        "Oracle pose component",
        "Lying versus standing crops\nSource-image split\n\nGround-truth boxes are free\nOnly four test negatives",
        size=16,
    )
    brief(
        s,
        439,
        "All 55,156 saved predictions were checked; recomputed metrics match all 40 summaries.",
    )

    s = new(
        "Findings / unattended frames",
        "MiniCPM has the highest raw frame F1",
        "Provisional held-out labels: 242 positive + 720 normal endpoints; two camera families",
        "unattended_expanded_frame/test/*/summary.json",
        "All rows are raw categorical decisions on the same 962 inputs. MiniCPM has TP84 FP25 FN158 TN695: precision 77.1%, recall 34.7%, F1 .479 and FPR 3.47%. Its earlier constant-normal answer does not persist on this broader sample, but it still misses nearly two thirds of positives. MobileCLIP has 75.2% recall at 74.6% FPR; SmolVLM has 63.6% recall at 93.8% FPR. SigLIP has zero categorical recall with 2.64% FPR, while its continuous scores have AUROC .675 and AP .309. Positive prevalence is .252. All frame-test models have 100% response coverage. This is a descriptive provisional-label result, not a statistically established model winner or deployment validation.",
    )
    frame = [result("unattended_expanded_frame", n) for n in d.NAMES]
    d.table(
        s,
        42,
        170,
        [274, 124, 124, 106, 124, 124],
        ["Model", "Precision", "Recall", "F1", "FPR", "AP"],
        [
            [
                label,
                d.p(r["precision"]),
                d.p(r["recall"]),
                d.metric(r["f1"]),
                d.p(r["fpr"]),
                d.metric(r["ap"]),
            ]
            for label, r in zip(SHORT, frame)
        ],
        row_h=48,
        font=15,
    )
    d.text(
        s,
        45,
        418,
        870,
        42,
        "MiniCPM: 84 TP / 25 FP / 158 FN / 695 TN. Higher-recall alternatives also flag most normal endpoints. SigLIP score ranking exceeds its raw decision quality.",
        15,
        d.GRAY,
    )
    d.text(
        s,
        45,
        474,
        870,
        20,
        "AP reference prevalence: 0.252. VLM AP is unavailable. Raw frame response coverage is 100%.",
        12.5,
        d.TEAL,
    )

    s = new(
        "Findings / past-only windows",
        "More temporal context has mixed effects",
        "Same 962 endpoint labels; four-frame sheets cover the preceding 20 seconds",
        "unattended_expanded_window/test/*/summary.json",
        "The 20-second window test uses the same endpoint labels as the frame study. MiniCPM recall falls from 34.7% to 10.7%, and FPR falls from 3.47% to .14%, with 26TP1FP216FN719TN. SmolVLM recall rises to 96.3% with 97.4% FPR; 14 unanswered inputs give 98.54% coverage. MobileCLIP recall is 76.0% with 68.5% FPR. SigLIP raw recall is only 1.65%, but window scores have AP .325 and AUROC .663. The earlier contact sheets used six seconds, so these results combine data expansion and changed temporal context. Contact sheets also reduce spatial resolution per frame. This experiment does not establish native video capability.",
    )
    window = [result("unattended_expanded_window", n) for n in d.NAMES]
    d.table(
        s,
        42,
        170,
        [274, 124, 124, 106, 124, 124],
        ["Model", "Precision", "Recall", "F1", "FPR", "AP"],
        [
            [
                label,
                d.p(r["precision"]),
                d.p(r["recall"]),
                d.metric(r["f1"]),
                d.p(r["fpr"]),
                d.metric(r["ap"]),
            ]
            for label, r in zip(SHORT, window)
        ],
        row_h=48,
        font=15,
    )
    d.text(
        s,
        45,
        414,
        870,
        62,
        "MiniCPM trades recall for fewer false alarms. SmolVLM alarms on almost every input (14 unanswered). The old 6-second sheets and new 20-second sheets are different protocols.",
        16,
        d.GRAY,
    )

    s = new(
        "Findings / frozen CLIP thresholds",
        "Development false-alarm budgets transfer poorly",
        "Thresholds fitted on 553 normal + 255 positive development endpoints; frozen for test",
        "unattended_expanded_frame/window/test/calibration.json",
        "CLIP operating points are fitted using only development data. All eight rows are shown, including maximum-development-F1 and empirical 5% development-FPR budget selections. The latter is not a population guarantee. MobileCLIP budget-test FPR rises to 20.3% for frames and 35.1% for windows. SigLIP's window budget gives 7.0% recall at 5.14% FPR, while maximum-dev-F1 gives 55.4% recall at 30% FPR. MobileCLIP's maximum-dev-F1 frame threshold predicts every test endpoint anomalous, with F1 .402 simply reflecting prevalence. Test labels were not used to refit thresholds. Small held-out camera count limits claims about transfer.",
    )
    calibrated = []
    for protocol in ["frame", "window"]:
        calibration = d.read(
            RUN / f"unattended_expanded_{protocol}/test/calibration.json"
        )
        for model, label in zip(d.NAMES[:2], ["MobileCLIP", "SigLIP"]):
            m = calibration["models"][model]
            for op, op_label in [
                ("dev_max_f1", "Max dev F1"),
                ("dev_fpr_budget", "Dev 5% FPR"),
            ]:
                b = m["operating_points"][op]["all_samples"]["binary"]
                calibrated.append(
                    [
                        protocol,
                        label,
                        op_label,
                        d.metric(m["thresholds"][op]),
                        d.p(b["recall"]),
                        d.p(b["false_positive_rate"]),
                        d.metric(b["f1"]),
                    ]
                )
    d.table(
        s,
        42,
        165,
        [92, 143, 193, 106, 116, 116, 110],
        ["Input", "Model", "Dev selection", "Threshold", "Recall", "FPR", "F1"],
        calibrated,
        row_h=34,
        font=13,
    )
    d.text(
        s,
        45,
        480,
        870,
        17,
        "An empirical 5% development constraint does not ensure 5% false positives on another camera.",
        12,
        d.ORANGE,
        True,
    )

    s = new(
        "Findings / reviewed intervals",
        "Detecting an interval can still mean many alarms",
        "Screen: at least one raw categorical alarm inside each reviewed source/label interval",
        "interval_alarm_screen.json + predictions.jsonl",
        "The screen groups the held-out predictions by source recording and ground-truth label and counts any raw categorical alarm. There are five positive source intervals and seven normal source intervals. This is not a streaming detector: no persistence filter, no event delay, and no alerts/hour estimate. Any-alarm counts are sensitive to interval length. MobileCLIP/SmolVLM frame predictions touch all five positive intervals but also all seven normal intervals. MiniCPM frame predictions touch three positive and three normal intervals; its window predictions touch four positive intervals and one normal interval, despite lower endpoint recall. Endpoint coverage and interval coverage therefore answer different questions. Source intervals are still clustered in only two camera families.",
    )
    interval_rows = []
    for protocol in ["frame", "window"]:
        for model, label in zip(d.NAMES, SHORT):
            e = quality["intervals"][protocol + "/" + model]
            interval_rows.append(
                [
                    protocol,
                    label,
                    f"{e['detected_positive_intervals']}/{e['positive_intervals']}",
                    f"{e['normal_intervals_with_alarm']}/{e['normal_intervals']}",
                ]
            )
    d.table(
        s,
        42,
        165,
        [108, 286, 219, 263],
        ["Input", "Model", "Positive intervals hit", "Normal intervals with alarm"],
        interval_rows,
        row_h=34,
        font=14,
    )
    d.text(
        s,
        45,
        480,
        870,
        17,
        "Any-alarm counts depend on interval length and do not establish operational event detection.",
        12,
        d.ORANGE,
    )

    s = new(
        "Findings / three additional anomalies",
        "The remaining datasets are usable as diagnostics",
        "Reported values are target-prediction rates; verified event recall, precision and FPR remain unavailable",
        "accuracy_comparison.csv / positive_only_diagnostic rows",
        "Each value is the number of target answers divided by all candidate inputs, including unanswered inputs. No verified anomaly recall or event precision/FPR is claimed. Littering provides 6,667 generic action-labelled candidates, some augmented; boxes can describe dropping, dumping, placing, throwing or tossing rather than rubbish disposal. Loitering provides 1,675 person-labelled images, without verified duration. Rough sleeping uses 111 synthetic lying/fall scenes without established public sleeping or bedding evidence. Low loitering/rough-sleeping rates may reflect correct rejection of insufficient evidence. High rates, such as SigLIP's rough-sleeping 95.5%, do not validate event recognition. SmolVLM abstains on 924 littering, 319 loitering and 13 rough-sleeping candidates. MiniCPM abstains on 24 rough-sleeping candidates.",
    )
    weak_rows = []
    for cohort, label, count in [
        ("littering_weak", "Littering", "6,667"),
        ("loitering_weak", "Loitering", "1,675"),
        ("rough_sleeping_weak", "Rough sleeping", "111"),
    ]:
        weak_rows.append(
            [
                label,
                count,
                *[
                    d.p(result(cohort, n, "dev")["target_prediction_rate"])
                    for n in d.NAMES
                ],
            ]
        )
    d.table(
        s,
        42,
        170,
        [174, 90, 153, 153, 153, 153],
        ["Candidate query", "Inputs", "MobileCLIP", "SigLIP", "SmolVLM", "MiniCPM"],
        weak_rows,
        row_h=48,
        font=15,
    )
    d.text(
        s,
        45,
        381,
        870,
        89,
        "Littering: generic action boxes, no matched normals.\nLoitering: person boxes do not establish duration.\nRough sleeping: lying/fall posture does not establish sleeping outdoors.\nHigh prediction rates cannot establish event quality; low rates may reject unsupported claims.",
        15,
        d.GRAY,
    )

    s = new(
        "Findings / person grounding",
        "Seeing a person does not establish loitering",
        "Two definitions queried on the same 1,675 person-labelled development images",
        "person_presence/dev + loitering_weak/dev summaries",
        "This separate grounding diagnostic uses the actual person boxes as person-presence evidence. It is positive-only, so it estimates positive hit rate, not precision or false-positive rate. The original loitering definition remains unchanged and requires duration evidence absent in the static export. SmolVLM detects person presence on 92.7% and answers loitering on .12%; MiniCPM detects person presence on 96.1% and answers loitering on 4.60%. The gap does not demonstrate loitering failure. SigLIP answers person presence on 17.4% and loitering on 84.5%, exposing strong definition/decision sensitivity without determining which unverified loitering labels are correct. There are no absent-person controls, and camera provenance is uncertain. Person and loitering images are the same inputs, not additional independent data.",
    )
    d.table(
        s,
        42,
        170,
        [294, 286, 296],
        ["Model", "Person-presence positive hit", "Loitering target-prediction rate"],
        [
            [
                label,
                d.p(result("person_presence", n, "dev")["target_prediction_rate"]),
                d.p(result("loitering_weak", n, "dev")["target_prediction_rate"]),
            ]
            for n, label in zip(d.NAMES, SHORT)
        ],
        row_h=48,
        font=16,
    )
    brief(
        s,
        424,
        "SmolVLM and MiniCPM usually recognize people. Duration and event evidence still require temporal labels.",
    )

    s = new(
        "Findings / posture component",
        "Oracle crops identify a promising component",
        "Synthetic lying-versus-standing diagnostic; 44 lying + 4 standing test crops",
        "lying_pose_oracle/test/*/summary.json",
        "The diagnostic supplies source bounding boxes free of charge and asks lying versus standing, not rough sleeping. Source-image grouping keeps mixed scenes together across splits. There are 111 lying and ten standing crops overall, with 67/6 in dev and 44/4 in test. SigLIP is correct on all 48 test crops. MiniCPM finds 40/44 lying crops and all four standing controls, F1 .952; it abstains on four positive crops. SmolVLM finds 26/44 lying crops and all standing controls, F1 .743; it abstains on 18 positive crops. MobileCLIP finds seven lying crops and incorrectly labels all four standing crops lying, F1 .255. Four negative controls are insufficient for a stable false-positive estimate. Oracle posture success is not validation of rough sleeping or of a deployed detector/crop pipeline.",
    )
    pose = [result("lying_pose_oracle", n) for n in d.NAMES]
    d.table(
        s,
        42,
        170,
        [274, 139, 139, 116, 124, 84],
        ["Model", "Precision", "Recall", "F1", "FPR", "Missed"],
        [
            [
                label,
                d.p(r["precision"]),
                d.p(r["recall"]),
                d.metric(r["f1"]),
                d.p(r["fpr"]),
                str(44 - round(r["recall"] * 44)),
            ]
            for label, r in zip(SHORT, pose)
        ],
        row_h=48,
        font=15,
    )
    brief(
        s,
        423,
        "SigLIP: 48/48 correct; MiniCPM: 40/44 lying + 4/4 standing. Only four controls; boxes are supplied.",
        d.ORANGE,
    )

    smoke = {
        n: d.read(
            ROOT
            / "results/gpu_all_models_20261013/smoking_frame/dev"
            / n
            / "summary.json"
        )
        for n in d.NAMES
    }
    vad = d.read(ROOT / "results/gpu_all_models_20261013/vadclip/adapter/eval.json")
    vb = vad["metrics"][vad["selected_branch"]]
    s = new(
        "Retained evidence / smoking and VadCLIP",
        "Smoking remains a difficult task",
        "Prior GPU evidence reused; image models and supervised offline VadCLIP have different protocols",
        "gpu_all_models_20261013/smoking_frame + vadclip/adapter",
        "The prior smoking comparison already uses all 3,278 development frames, 1,422 positive and 1,856 normal, across 16 known groups. Raw smoking recall is .56% MobileCLIP, zero SigLIP, 5.41% SmolVLM and zero MiniCPM. The smoking source groups were historically inspected, so this is development evidence, not a fresh held-out test. VadCLIP is a separately trained supervised offline adaptation with future/full-sequence context and different training/test inputs. Its source split has nine train, four dev and three test sessions; test has 1,056 frames, 560 positive and 496 normal. The dev-selected dual_mean branch has AP .633, AUROC .587, recall 51.8%, F1 .565 and FPR 35.7%. The visual/dual_max branches predict all positive; a higher F1 from that trivial pattern is not better ranking. Neither protocol is pooled with unattended/proxy results, and VadCLIP is not evaluated for the other anomalies. No smoking or VadCLIP rerun was needed for this expansion.",
    )
    d.table(
        s,
        42,
        170,
        [235, 86, 86, 86],
        ["Image model / dev", "Recall", "F1", "FPR"],
        [
            [
                label,
                d.p(smoke[n]["metrics"]["binary"]["recall"]),
                d.metric(smoke[n]["metrics"]["binary"]["f1"]),
                d.p(smoke[n]["metrics"]["binary"]["false_positive_rate"]),
            ]
            for n, label in zip(d.NAMES, SHORT)
        ],
        row_h=43,
        font=13,
    )
    d.card(
        s,
        553,
        170,
        365,
        237,
        "VadCLIP: supervised offline",
        f"1,056 source-split test frames\nAP {vb['aupr']:.3f}; AUROC {vb['auroc']:.3f}\nRecall {vb['recall'] * 100:.1f}%; FPR {vb['false_positive_rate'] * 100:.1f}%\nF1 {vb['f1']:.3f}; selected dual_mean\n\nDifferent training and temporal context.",
        color=d.ORANGE,
        size=15,
    )
    brief(
        s,
        437,
        "3,278 smoking development frames are already used. VadCLIP is a separate supervised smoking study.",
    )

    s = new(
        "Interpretation / next evaluation",
        "Choose the next experiment by failure mode",
        "No deployment winner across all five anomalies is established by these source datasets",
        "Expanded report + response counts + retained smoking evidence",
        "MiniCPM is worth pursuing for conservative unattended frame decisions, but misses many positives and its window endpoint recall drops. SigLIP has good oracle pose discrimination and some unattended score ranking, yet raw event decisions and frozen thresholds transfer poorly. MobileCLIP's high unattended FPR and poor oracle standing discrimination require label/prompt/crop development checks before deployment evaluation. SmolVLM's near-universal unattended window alarms and response abstentions are material issues. Across all new inputs, SmolVLM has 1,457 abstentions (10.6%) and MiniCPM has 45 (.33%); there are zero runtime errors. The next evidence should independently verify the provisional intervals, add deployment-domain cameras with real target events and matched negatives, and test detector plus crops/native temporal inputs under a frozen protocol. Repeated inspection means these test sources become development data for future tuning; a fresh source holdout is needed after changes. Do not infer deployment robustness from weak prediction rates or perfect four-negative pose results.",
    )
    d.table(
        s,
        42,
        169,
        [199, 312, 365],
        ["Model", "Useful observed evidence", "Issue to resolve before deployment"],
        [
            [
                "MiniCPM",
                "Best raw unattended frame F1",
                "Misses 65% of positive endpoints",
            ],
            [
                "SigLIP",
                "Strong oracle posture discrimination",
                "Raw decisions / threshold transfer",
            ],
            [
                "MobileCLIP",
                "High unattended endpoint recall",
                "High false alarms; prompt/crop sensitivity",
            ],
            [
                "SmolVLM",
                "High person-grounding hit rate",
                "Near-universal alarms; 1,457 abstentions",
            ],
        ],
        row_h=48,
        font=14,
    )
    d.text(
        s,
        45,
        412,
        870,
        75,
        "Next: independently verify the event labels; add real littering, duration and sleeping events with matched controls; include fresh cameras. Evaluate detector crops and native temporal inputs as development experiments, then freeze a fresh test.",
        16,
        d.GRAY,
    )

    s = new(
        "Evidence / reproducibility",
        "Updated results are traceable and reviewable",
        "Raw inference artifacts are preserved; this deck uses the new expanded accuracy report",
        "expanded_source_manifest_20261014.json + verification.json",
        "Read results/expanded_accuracy_20261013/report.md for the full accuracy findings and qualifications, accuracy_comparison.csv for the qualified numeric export, quality_summary.json for slide inputs, state.json for exact GPU commands, verification.json for saved-prediction metric audit, and per-model summaries/predictions.jsonl for responses. data/expanded_20261013/cohorts.json, review_decisions.json and review boards hold preparation provenance. scripts/expand_accuracy.py prepares source review and cohorts; run_expanded_accuracy.py ran the complete CUDA evaluation; report_expanded_accuracy.py regenerates the qualified report. Do not rerun the inference script into the completed exclusive output path. This presentation can be regenerated with PYTHONPATH containing the repository root and presentation dependencies, using reports/slides/build_expanded_presentation.py. Requirements are in reports/slides/requirements.txt. All 144 existing tests passed, and all 40 metric summaries match saved-prediction recomputation. Source file SHA256 hashes and the complete common layout are recorded in expanded_source_manifest_20261014.json. PDF is rendered independently with embedded fonts from the same measured layout as the editable PPTX. The old 20261013 deck remains historical. No additional latency results are included.",
    )
    d.card(
        s,
        42,
        170,
        427,
        247,
        "Results and source evidence",
        "Expanded accuracy report + numeric CSV\n40 model summaries + 55,156 predictions\nFrozen CLIP thresholds\nReviewed source intervals + camera groups\n\nPaths and source hashes are in the notes.",
        size=16,
    )
    d.card(
        s,
        491,
        170,
        427,
        247,
        "Validation and presentation",
        "144 existing tests passed\n40/40 metric recomputations matched\nAll runs: CUDA, unpaced, zero errors\n\nEditable PowerPoint + vector PDF\nNo additional latency evaluation",
        size=16,
    )
    brief(
        s,
        437,
        "Primary limitations: provisional event labels, two test camera families, weak event proxies and four pose controls.",
        d.ORANGE,
    )
    assert len(d.SLIDES) == 16


def main():
    build_content()
    d.check_geometry(d.SLIDES)
    standalone = deepcopy(d.SLIDES[0])
    for stem, slides in [(STEM, d.SLIDES), (SUMMARY_STEM, [standalone])]:
        pptx_path = OUT / (stem + ".pptx")
        d.render_pptx(slides, pptx_path)
        presentation = Presentation(pptx_path)
        presentation.core_properties.title = "Expanded CLIP & VLM accuracy evaluation"
        presentation.core_properties.subject = (
            "Findings, methodology and accuracy evaluation; 14 October 2026"
        )
        presentation.save(pptx_path)
        d.render_pdf(slides, OUT / (stem + ".pdf"))
    notes = [
        "# Expanded evaluation speaker notes",
        "",
        "Finalized 14 October 2026 (Asia/Jakarta). Paths are relative to the repository root.",
        "",
    ]
    for i, slide in enumerate(d.SLIDES, 1):
        notes += [f"## {i:02d}. {slide['title']}", "", slide["notes"], ""]
    (OUT / "expanded_speaker_notes_20261014.md").write_text("\n".join(notes))
    (OUT / "expanded_source_manifest_20261014.json").write_text(
        json.dumps(
            {
                "finalized_date": "2026-10-14",
                "timezone": "Asia/Jakarta",
                "slide_count": len(d.SLIDES),
                "sources_sha256": d.SOURCES,
                "builders_sha256": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__), OUT / "build_presentation.py"]
                },
                "layout": d.SLIDES,
                "pdf_rendering": "Independent vector PDF from the same measured layout, with embedded fonts.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Created {len(d.SLIDES)} expanded slides and standalone summary (editable PPTX + PDF)."
    )


if __name__ == "__main__":
    main()
