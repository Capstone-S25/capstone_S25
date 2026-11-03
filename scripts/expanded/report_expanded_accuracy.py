"""Summarize expanded accuracy studies without presenting proxy labels as events."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/expanded_accuracy_20261013"
DATA = ROOT / "data/expanded_20261013"
NAMES = ["mobileclip2-s2", "siglip2-b16-224", "smolvlm-500m", "minicpm-v4"]
LABELS = ["MobileCLIP2-S2", "SigLIP2-B/16-224", "SmolVLM-500M", "MiniCPM-V 4.0"]


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


def pct(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def summary(name, split, model):
    path = RUN / name / split / model / "summary.json"
    if not path.exists():
        return None
    try:
        value = read(path)
    except json.JSONDecodeError:
        return None
    return value if value["status"] == "ok" else None


def episodes(name, model, threshold=None):
    """Any alarm within a reviewed interval; no persistence or event-delay claim."""
    directory = RUN / name / "test"
    prediction_path = directory / model / "predictions.jsonl"
    if not (directory / model / "summary.json").exists():
        return None
    manifest = ROOT / "data/expanded_20261013/cohorts" / (name + ".csv")
    source_by_path = {}
    from anomaly_eval.paths import resolve_path

    for row in csv.DictReader(manifest.open()):
        if row["split"] == "test":
            source_by_path[str(resolve_path(row["path"], manifest.parent))] = row[
                "source"
            ]
    by_source = {}
    for line in prediction_path.read_text().splitlines():
        row = json.loads(line)
        if row["repeat"] != 0:
            continue
        source = source_by_path[str(resolve_path(row["path"]))]
        key = (source, row["label"])
        alarm = row.get("status") == "ok" and (
            row["anomaly_score"] >= threshold
            if threshold is not None
            else row["prediction"] == "unattended_items"
        )
        by_source[key] = by_source.get(key, False) or alarm
    positives = [
        alarm for (_, label), alarm in by_source.items() if label == "unattended_items"
    ]
    normals = [alarm for (_, label), alarm in by_source.items() if label == "normal"]
    return {
        "positive_intervals": len(positives),
        "detected_positive_intervals": sum(positives),
        "normal_intervals": len(normals),
        "normal_intervals_with_alarm": sum(normals),
        "rule": "at least one categorical/threshold alarm within each reviewed source/label interval; no persistence filtering",
    }


def main():
    state = read(RUN / "state.json")
    cohorts = read(DATA / "cohorts.json")
    total = sum(c["samples"] for c in cohorts)
    measured = 0
    complete = 0
    data_rows = []
    lines = [
        "# Expanded GPU accuracy evaluation",
        "",
        "Results finalized 14 October 2026 (Asia/Jakarta); cohort/run identifiers retain their preparation date, 20261013.",
        "",
        f"Run status: **{state['status']}**. Planned: **{total:,} input/protocol evaluations per model**, **{total * 4:,} predictions** across seven cohorts. These include repeated images queried for different target definitions; they are not that many independent events.",
        "",
        "No additional CPU or latency experiment was run. This report focuses on quality. Previous smoking/VadCLIP results are retained as separate baseline evidence.",
        "",
        "The unattended-item study is **provisional**: labels were set by assistant visual review of source timelines before model inference, with XML checks for CAVIAR positive intervals. They have not been independently human verified. The three other anomaly datasets support clearly labelled diagnostics rather than verified event-accuracy claims.",
        "",
        "## Coverage increase",
        "",
        "Unattended items grew from 36 endpoints to **1,770 endpoints**, a **49.2× increase**. Its frame test cohort grew from 16 to **962 inputs**, with **242 positive and 720 normal** inputs. There are five positive test recordings, compared with one previously. All related views/recordings in the ABODA classroom and walkway families are held together for test: **two test camera families**, not 962 independent events. Development has 808 inputs (255 positive / 553 normal) from three camera families.",
        "",
        "The same 1,770 endpoints also form 20-second, four-frame, past-only window images. The earlier windows covered six seconds. Frame and window inputs are separate protocols with shared event endpoints; the new window results cannot isolate a data-volume effect from the changed temporal context.",
        "",
        "All 18 included recordings contribute reviewed intervals or controls. Uncertain positive cases were excluded, including an occluded CAVIAR bag, an owner who remains in the scene, occluded street placements and a pre-existing item without a visible deposition/owner history. Raw/additional duplicate video copies were not counted twice.",
        "",
        table(
            ["Cohort", "Dev inputs", "Test inputs", "Evidence / interpretation"],
            [
                [
                    c["name"],
                    sum(c["counts"]["dev"].values()),
                    sum(c["counts"]["test"].values()),
                    "Provisional reviewed event/control labels"
                    if c["name"].startswith("unattended")
                    else "Synthetic pose oracle crops; source-image holdout"
                    if c["name"] == "lying_pose_oracle"
                    else "Positive-only person annotation diagnostic"
                    if c["name"] == "person_presence"
                    else "Positive-only weak event candidate diagnostic",
                ]
                for c in cohorts
            ],
        ),
        "",
        "All eligible local inputs are retained after byte/pixel deduplication. Littering includes augmented images; those views are not independent source events. The whole weak export remains development-only. Camera/source identities are uncertain for the person exports, so they also remain development-only.",
        "",
        "## Test approach",
        "",
        "All four models run on the RTX 5000 Ada GPU, with four isolated concurrent model workers, eight CPU threads per worker, CLIP batch size 64 and VLM single-input calls. CLIP weights use float32; unquantized VLM weights use automatic bfloat16. There is one measured prediction per input, zero pacing sleeps, no default accuracy RAM cap and no latency repetitions. Shared letterbox preprocessing bounds inputs to 768 pixels; generation is deterministic with a 64-token budget. MiniCPM retains the one-slice baseline. Model revisions, source hashes and raw responses are saved.",
        "",
        "The unattended-item label definitions match the preceding GPU comparison. Thresholds are fitted only on development camera families and frozen for test. Failures/abstentions remain in quality denominators. Cluster bootstrap uses camera/source groups; the small test group count limits uncertainty estimates.",
        "",
        "## Unattended items: raw categorical test results",
        "",
        "These scores compare the models against the provisional labels. They do not constitute an independently verified deployment test.",
        "",
    ]
    quality = []
    for protocol in ["frame", "window"]:
        for model, label in zip(NAMES, LABELS):
            s = summary("unattended_expanded_" + protocol, "test", model)
            if s is None:
                continue
            b = s["metrics"]["binary"]
            quality.append(
                [
                    protocol,
                    label,
                    pct(b["precision"]),
                    pct(b["recall"]),
                    f(b["f1"]),
                    pct(b["false_positive_rate"]),
                    f(b["average_precision"]),
                    f(b["roc_auc"]),
                    pct(s["metrics"]["prediction_coverage"]),
                ]
            )
    lines += [
        table(
            [
                "Input",
                "Model",
                "Precision",
                "Recall",
                "F1",
                "FPR",
                "AP",
                "AUROC",
                "Coverage",
            ],
            quality,
        )
        if quality
        else "Inference pending.",
        "",
        "AP/AUROC are only supplied for genuine continuous CLIP scores. Hard VLM labels have no invented ranking metric. SmolVLM/MiniCPM response patterns and abstentions are visible below.",
        "",
        "On the expanded frame test, **MiniCPM has the highest raw F1 (0.479)**, with 77.1% precision, 34.7% recall and 3.47% FPR (84 TP, 25 FP, 158 FN, 695 TN). Its earlier constant-normal response does not persist on this broader cohort. It still misses nearly two thirds of positive endpoints. This is a descriptive result on provisional labels and two held-out camera families, not a statistically established deployment winner.",
        "",
        "MobileCLIP and SmolVLM flag many positive frames but also 74.6% and 93.8% of normal frames. SigLIP's raw categorical frame decision finds no positives, although its score ranking has AUROC 0.675 and AP 0.309, against a positive prevalence of 0.252. Raw decisions and score ranking answer different questions.",
        "",
        "Twenty-second contact sheets do not improve every model: MiniCPM's endpoint recall falls to 10.7% while FPR falls to 0.14%; SmolVLM reaches 96.3% recall with 97.4% FPR and 14 unanswered inputs. Contact sheets change temporal evidence and spatial detail together; these results do not establish the performance of native video inference.",
        "",
        "## Development-calibrated operating points",
        "",
    ]
    calibrated = []
    episode_rows = []
    episode_data = {}
    for protocol in ["frame", "window"]:
        name = "unattended_expanded_" + protocol
        p = RUN / name / "test/calibration.json"
        if p.exists():
            calibration = read(p)
            for model, label in zip(NAMES[:2], LABELS[:2]):
                for op, point in calibration["models"][model][
                    "operating_points"
                ].items():
                    b = point["all_samples"]["binary"]
                    threshold = calibration["models"][model]["thresholds"][op]
                    calibrated.append(
                        [
                            protocol,
                            label,
                            op,
                            f(threshold),
                            pct(b["precision"]),
                            pct(b["recall"]),
                            f(b["f1"]),
                            pct(b["false_positive_rate"]),
                        ]
                    )
        for model, label in zip(NAMES, LABELS):
            e = episodes(name, model)
            if e is None:
                continue
            episode_data[protocol + "/" + model] = e
            episode_rows.append(
                [
                    protocol,
                    label,
                    f"{e['detected_positive_intervals']}/{e['positive_intervals']}",
                    f"{e['normal_intervals_with_alarm']}/{e['normal_intervals']}",
                ]
            )
    lines += [
        table(
            [
                "Input",
                "Model",
                "Dev selection",
                "Threshold",
                "Test precision",
                "Test recall",
                "Test F1",
                "Test FPR",
            ],
            calibrated,
        )
        if calibrated
        else "Calibration pending.",
        "",
        "A 5% empirical development FPR constraint is not a population guarantee, particularly after a camera/domain change. Results at both the maximum-dev-F1 and FPR-budget points are retained.",
        "",
        "The budget point gives MobileCLIP 20.3% frame-test FPR and 35.1% window-test FPR, showing poor transfer of the development operating point. SigLIP's window budget point gives 7.0% recall at 5.14% FPR. Its maximum-dev-F1 point gives 55.4% recall at 30.0% FPR. No test labels were used to refit these thresholds.",
        "",
        "## Reviewed-interval alarm screen",
        "",
        "This screen counts whether at least one raw categorical alarm occurs inside each positive or normal reviewed source interval. It is sensitive to interval length and uses no persistence filtering; it is not a streaming event detector or an alerts/hour measure. There are only five positive intervals and seven normal intervals in the held-out set.",
        "",
        table(
            [
                "Input",
                "Model",
                "Positive intervals detected",
                "Normal intervals with ≥1 alarm",
            ],
            episode_rows,
        )
        if episode_rows
        else "Pending.",
        "",
        "## Other anomaly candidates and grounding diagnostics",
        "",
        "For the three weak cohorts below, the reported number is the **model target-prediction rate** on candidate inputs. It is not verified anomaly recall. No event precision, FPR, AP or F1 is inferred from an all-positive proxy-labelled set. Low rough-sleeping/loitering rates may be appropriate rejection of unsupported posture/duration evidence; high rates do not establish event detection.",
        "",
    ]
    weak = []
    for cohort in [
        "littering_weak",
        "loitering_weak",
        "rough_sleeping_weak",
        "person_presence",
    ]:
        for model, label in zip(NAMES, LABELS):
            s = summary(cohort, "dev", model)
            if s is None:
                continue
            target = next(k for k in s["metrics"]["per_class"] if k != "normal")
            rate = (
                s["prediction_counts"].get(target, 0) / s["metrics"]["labeled_samples"]
            )
            weak.append(
                [
                    cohort,
                    label,
                    s["metrics"]["labeled_samples"],
                    pct(rate),
                    s["abstentions"],
                    s["errors"],
                ]
            )
    lines += [
        table(
            [
                "Diagnostic",
                "Model",
                "Inputs",
                "Target-prediction rate",
                "Abstentions",
                "Errors",
            ],
            weak,
        )
        if weak
        else "Pending.",
        "",
        "Littering boxes name generic dropping/dumping/placing/throwing/tossing actions, not uniformly rubbish disposal. Loitering exports have person boxes but no reviewed duration events. The rough-sleeping source is synthetic fall/pose data, with no established bedding/public sleeping evidence. Person-presence detection on the same loitering images is a separate grounding check using genuine person annotations.",
        "",
        "These sources are useful for checking model behaviour. On the same 1,675 person-labelled images, SmolVLM/MiniCPM predict person presence on 92.7%/96.1% but loitering on only 0.12%/4.60%. Visible people do not establish duration, so this gap is not evidence of loitering failure. SigLIP predicts loitering on 84.5% and rough sleeping on 95.5% of their weak candidates; those high rates cannot be interpreted as event accuracy. SmolVLM abstains on 924 littering candidates (13.9%) and 319 loitering candidates (19.0%); unanswered inputs remain in the rate denominator.",
        "",
        "## Lying versus standing: oracle pose diagnostic",
        "",
        "This new study uses source bounding boxes to crop 111 lying/fallen people and 10 standing people. Original image identities stay together across splits, including mixed scenes. Test contains 44 lying and four standing crops. Bounding boxes are provided free to the diagnostic and cannot be assumed available in deployment. This is a synthetic posture task, not rough-sleeping event accuracy.",
        "",
    ]
    pose = []
    for model, label in zip(NAMES, LABELS):
        s = summary("lying_pose_oracle", "test", model)
        if s is None:
            continue
        b = s["metrics"]["binary"]
        pose.append(
            [
                label,
                pct(b["precision"]),
                pct(b["recall"]),
                f(b["f1"]),
                pct(b["false_positive_rate"]),
                f(b["average_precision"]),
                f(b["roc_auc"]),
            ]
        )
    lines += [
        table(["Model", "Precision", "Recall", "F1", "FPR", "AP", "AUROC"], pose)
        if pose
        else "Pending.",
        "",
        "Four negative test crops are insufficient for a stable false-positive estimate. Any favourable oracle result only identifies a component worth investigating with a real detector and event labels.",
        "",
        "SigLIP classifies all 48 test crops correctly; MiniCPM finds 40/44 lying crops and all four standing controls (F1 0.952). SmolVLM finds 26/44 lying crops and abstains on the other 18; its four standing controls are correct. This localizes a promising posture component without validating rough-sleeping detection.",
        "",
        "## Response diagnostics and completion",
        "",
    ]
    diagnostics = []
    for c in cohorts:
        for split in ["dev", "test"]:
            for model, label in zip(NAMES, LABELS):
                s = summary(c["name"], split, model)
                if s is None:
                    continue
                complete += 1
                measured += s["completed_inferences"]
                diagnostics.append(
                    [
                        c["name"] + "/" + split,
                        label,
                        str(s["prediction_counts"]),
                        s["abstentions"],
                        s["errors"],
                    ]
                )
                b = s["metrics"]["binary"]
                kind = (
                    "provisional_event"
                    if c["name"].startswith("unattended")
                    else "oracle_pose"
                    if c["name"] == "lying_pose_oracle"
                    else "positive_only_diagnostic"
                )
                target = next(k for k in s["metrics"]["per_class"] if k != "normal")
                data_rows.append(
                    {
                        "cohort": c["name"],
                        "split": split,
                        "model": model,
                        "kind": kind,
                        "inputs": s["metrics"]["labeled_samples"],
                        "target_prediction_rate": s["prediction_counts"].get(target, 0)
                        / s["metrics"]["labeled_samples"],
                        "precision": b["precision"]
                        if kind != "positive_only_diagnostic"
                        else None,
                        "recall": b["recall"]
                        if kind != "positive_only_diagnostic"
                        else None,
                        "f1": b["f1"] if kind != "positive_only_diagnostic" else None,
                        "fpr": b["false_positive_rate"]
                        if kind != "positive_only_diagnostic"
                        else None,
                        "ap": b["average_precision"]
                        if kind != "positive_only_diagnostic"
                        else None,
                        "auroc": b["roc_auc"]
                        if kind != "positive_only_diagnostic"
                        else None,
                        "coverage": s["metrics"]["prediction_coverage"],
                        "abstentions": s["abstentions"],
                        "errors": s["errors"],
                        "qualification": c["qualification"],
                    }
                )
    lines += [
        f"Completed summaries available: **{complete}/40**; measured predictions available: **{measured:,}/{total * 4:,}**.",
        "",
        table(
            ["Cohort / split", "Model", "Predicted labels", "Abstentions", "Errors"],
            diagnostics,
        ),
        "",
        "## Retained smoking and VadCLIP evidence",
        "",
        "The earlier [full GPU smoking comparison](../gpu_all_models_20261013/report.md) remains unchanged: 3,278 development frames per image model. Raw smoking recalls were 0.56% MobileCLIP, 0% SigLIP, 5.41% SmolVLM and 0% MiniCPM. It is development evidence, not a fresh test. VadCLIP remains a separate supervised offline smoking reanalysis (dual_mean AP 0.633, AUROC 0.587, recall 51.8%, FPR 35.7%). Neither is pooled with the new target/proxy cohorts.",
        "",
        "## Artifacts and limits",
        "",
        "- [Expanded study state](state.json), [accuracy-only numeric export](accuracy_comparison.csv), [review decisions](../../data/expanded_20261013/review_decisions.json) and [prepared cohort metadata](../../data/expanded_20261013/cohorts.json).",
        "- [Updated PDF slides](../../reports/slides/model_evaluation_expanded_20261014.pdf), [editable PowerPoint](../../reports/slides/model_evaluation_expanded_20261014.pptx), [one-slide summary](../../reports/slides/summary_slide_expanded_20261014.pdf) and [saved-prediction verification](verification.json).",
        "- Source-video review images and interval metadata are retained under `data/expanded_20261013/review/`. New results do not overwrite prior inference artifacts.",
        "- The 144 existing tests passed; production manifest loading validates each new cohort for content/source leakage and hashes. Native GPU inference and saved-prediction analysis provide the live integration checks.",
        "- Larger input counts do not imply more independent events. Unattended labels need independent human review; proxy tasks need target-event labels and matched negatives. Native video/image slicing, detector failures, domain shift and operational alert policy remain separate questions.",
        "",
        "The expanded runs increase coverage and test whether earlier constant answers persist on more varied inputs. A final model choice still requires verified target events and acceptable false alerts on fresh deployment-domain recordings.",
        "",
    ]
    (RUN / "report.md").write_text("\n".join(lines))
    if data_rows:
        with (RUN / "accuracy_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data_rows[0]))
            writer.writeheader()
            writer.writerows(data_rows)
    (RUN / "interval_alarm_screen.json").write_text(
        json.dumps(episode_data, indent=2) + "\n"
    )
    (RUN / "quality_summary.json").write_text(
        json.dumps(
            {
                "finalized_date": "2026-10-14",
                "timezone": "Asia/Jakarta",
                "status": state["status"],
                "planned_inputs_per_model": total,
                "planned_predictions": total * 4,
                "completed_summaries": complete,
                "completed_predictions": measured,
                "cohorts": cohorts,
                "rows": data_rows,
                "intervals": episode_data,
            },
            indent=2,
        )
        + "\n"
    )
    print("Updated", RUN / "report.md", f"({complete}/40 summaries)", flush=True)


if __name__ == "__main__":
    main()
