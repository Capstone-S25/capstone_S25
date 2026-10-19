"""Produce source-separated findings, event tables and score timelines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]


def f(value, digits=3):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    data = args.data.resolve()
    run = args.run.resolve()
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    clip_rows = []
    event_rows = []
    results = {}
    selections = {}
    for model in ["mobileclip2-s2", "siglip2-b16-224"]:
        directory = run / model
        results[model] = json.loads((directory / "results.json").read_text())
        selections[model] = json.loads((directory / "joint_selection.json").read_text())
        for task, candidates in results[model]["clip_metrics"].items():
            primary = selections[model]["tasks"][task]["primary"]["name"]
            for name, cohorts in candidates.items():
                for source, cohort in cohorts.items():
                    for op, metrics in cohort["operating_points"].items():
                        clip_rows.append(
                            {
                                "model": model,
                                "task": task,
                                "candidate": name,
                                "primary": name == primary,
                                "source": source,
                                "operating_point": op,
                                "groups": cohort["groups"],
                                **metrics,
                            }
                        )
        for task, candidates in results[model]["event_metrics"].items():
            for name, cohorts in candidates.items():
                for cohort, values in cohorts.items():
                    metrics = {
                        k: v
                        for k, v in values["aggregate"].items()
                        if k != "delays_seconds"
                    }
                    event_rows.append(
                        {
                            "model": model,
                            "task": task,
                            "candidate": name,
                            "primary": name
                            == selections[model]["tasks"][task]["primary"]["name"],
                            "cohort": cohort,
                            **metrics,
                        }
                    )
    for filename, rows in [("comparison.csv", clip_rows), ("events.csv", event_rows)]:
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with (run / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    info = json.loads((data / "dataset.json").read_text())
    guard = json.loads((data / "source_guard.json").read_text())
    coverage = json.loads((run / "error_audit/pilot_region_coverage.json").read_text())[
        "summary"
    ]["test"]
    old_component = json.loads((run / "error_audit/track_components.json").read_text())[
        "aggregate"
    ]
    new_component = json.loads(
        (run / "error_audit/grounded_track_components.json").read_text()
    )["aggregate"]
    old_recall = next(
        r["recall"] for r in old_component if r["kind"] == "item" and r["iou"] == 0.5
    )
    new_recall = next(
        r["recall"] for r in new_component if r["kind"] == "item" and r["iou"] == 0.5
    )
    lines = [
        "# Causal video-event pilot — 16 October 2026",
        "",
        "This study audits all five objectives and trains new smoking/unattended-item heads over frozen pretrained MobileCLIP2-S2 and SigLIP2-B/16-224 encoders. It is not another full visual-backbone fine-tuning sweep. No latency tests were run.",
        "",
        "## What changed",
        "",
        f"- Prepared {info['recordings']} existing video recordings and {guard['rows']:,} modeling windows/static examples after excluding {guard['excluded']} known shared-film training clips. Counts include highly correlated overlapping windows and unknown periods, not independent events.",
        "- HMDB now contributes 530 training clips; its grouped 205-clip evaluation is retrospective. Movie/web actions do not establish CCTV transfer, and primary action labels are non-exhaustive smoking controls.",
        "- Added three author-hosted AVSS2007 videos and published temporal unattended-object annotations. EASY/MEDIUM form the primary fresh subset; HARD is a separately reported censored diagnostic. Camera identities are not verified, so all are conservatively one source block.",
        "- Compared scene and multiple-region linear heads, causal GRUs, automatic track geometry, training-person supervision, training-only hard-negative mining, and open-vocabulary detector geometry. Three seeds are retained for learned nonlinear/instance heads.",
        "- Epochs and routes maximize worst development-source AP, with mean AP breaking ties. Alert thresholds/rules freeze using development recordings before fresh scoring.",
        "",
        "## Error audit and component findings",
        "",
        "- Reviewed 117 deliberately selected error cases across the five objectives. The review is provisional assistant triage; it is not independent human verification or a representative estimate of error causes.",
        f"- CCTV smokers have median largest annotated person height {coverage['median_largest_smoker_height_at_224']:.1f} pixels at 224 input. The fraction of positive frames with less than half of every target in the selected crop falls from {coverage['prior_under_half_target_coverage_fraction']:.1%} to {coverage['pilot_under_half_target_coverage_fraction']:.1%}. This is geometric coverage, not recognition accuracy; detector resolution and crop policy both change.",
        f"- On source CAVIAR training-camera item boxes, track recall at IoU 0.5 changes from {old_recall:.1%} to {new_recall:.1%} with Grounding DINO. The annotations omit other bags, so unmatched item tracks cannot be interpreted as proven false positives.",
        "- Coughing, drinking and eating are useful smoking controls. Loitering and unattended state require identified history. Static sleeping photographs remain insufficient to validate persistent public sleep.",
        "",
        "## Smoking — previously inspected recordings",
        "",
        "| Backbone | Source | Cases | Pilot scene baseline AP | Selected pilot AP | Previous bundle AP on same cases | Recall / FPR at development 5% frame-FPR-budget point |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    smoking_chart = []
    route_notes = []
    for model in results:
        primary = selections[model]["tasks"]["smoking"]["primary"]["name"]
        old = ROOT / "results/strategy_study_20261016" / model / "transfer_evaluation"
        prior = {
            r["id"]: r
            for tag in ["historical", "hmdb"]
            for r in map(
                json.loads,
                (old / (tag + "_predictions.jsonl")).read_text().splitlines(),
            )
            if r["family"] == "selected" and r["task"] == "smoking"
        }
        predictions = [
            r
            for r in map(
                json.loads, (run / model / "predictions.jsonl").read_text().splitlines()
            )
            if r["candidate"] == primary and r["task"] == "smoking"
        ]
        for source in ["smoking_cctv", "hmdb51_action_transfer"]:
            chosen = [r for r in predictions if r["source"] == source]
            assert all(r["id"] in prior for r in chosen)
            old_ap = average_precision_score(
                [r["target"] for r in chosen], [prior[r["id"]]["score"] for r in chosen]
            )
            metrics = results[model]["clip_metrics"]["smoking"][primary][source][
                "operating_points"
            ]["dev_fpr_budget"]
            baseline = results[model]["clip_metrics"]["smoking"]["scene_linear"][
                source
            ]["operating_points"]["dev_fpr_budget"]["ap"]
            lines.append(
                f"| {model} | {source} | {len(chosen)} | {baseline:.3f} | {metrics['ap']:.3f} | {old_ap:.3f} | {metrics['recall']:.1%} / {metrics['fpr']:.1%} |"
            )
            smoking_chart.append((model, source, baseline, metrics["ap"]))
        route_notes.append(f"Selected {model} smoking route: `{primary}`.")
    lines.extend([""] + route_notes)
    lines.extend(
        [
            "",
            "The previous bundle used eight HMDB views and no HMDB training; the pilot uses sixteen views and grouped HMDB adaptation. This comparison combines data and method changes. The pilot scene baseline shares its training/data/views with the new heads. CCTV case IDs/labels are unchanged and that cohort remains retrospective.",
            "",
            "## Unattended events — primary fresh subset",
            "",
            "These are **two annotated events**, not hundreds of independent samples. Overlapping decision windows are used to compute frame-state AP separately. The global rule uses development-only thresholds, two-second persistence, one-second release, and ten-second cooldown. Alerts match only if they start inside the labelled event; an early continuing alert does not become a true positive later.",
            "",
            "| Backbone | Fixed route | Protocol | Detected events | Normal false / repeated alerts | Reviewed normal minutes | False alerts / normal camera-hour | Median detection delay (s) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in results:
        name = selections[model]["tasks"]["unattended_items"]["primary"]["name"]
        for cohort, label in [
            ("fresh_avss_clean_primary", "Global rule"),
            (
                "fresh_avss_clean_known_30_second_rule",
                "Known AVSS 30-second state gate",
            ),
            ("fresh_avss_clean_state_rule_only", "State gate alone"),
        ]:
            metrics = results[model]["event_metrics"]["unattended_items"][name][cohort][
                "aggregate"
            ]
            lines.append(
                f"| {model} | `{name}` | {label} | {metrics['detected_events']}/{metrics['events']} | {metrics['normal_false_alerts']} / {metrics['duplicate_alerts']} | {metrics['normal_seconds'] / 60:.2f} | {f(metrics['false_alerts_per_normal_camera_hour'], 1)} | {f(metrics['median_delay_seconds'], 1)} |"
            )
    lines.extend(
        [
            "",
            "The known-criteria gate requires the same detected item to be stationary for 30 seconds and its previously associated owner absent for at least 0.5 seconds. It is a separately specified source-criteria diagnostic, not a test-fitted threshold. Its owner relation is a proximity heuristic and can fail. The state-only prediction is independent of the CLIP backbone; duplicate rows make the common comparison explicit.",
            "",
            "Zero false alerts over a few minutes cannot establish a reliable one-alert-per-hour operating point. Global transfer, known source criteria, and camera calibration are separate protocols. HARD has a censored annotation tail and is not included in this primary table.",
            "",
            "## Disjoint normal-recording calibration — historical diagnostic",
            "",
            "The complete classroom video7 and walkway video2 recordings are reserved for calibration in this comparison and excluded from both evaluation arms. Only their existing reviewed normal periods provide offsets. These are scene groups with provisional labels, not fresh cameras or a well-powered calibration set. AVSS calibration is unavailable because there is no independent normal recording and camera identity is unverified.",
            "",
            "| Backbone | Protocol on identical remaining recordings | Events detected | Normal false alerts | Normal minutes | False alerts / normal camera-hour |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in results:
        name = selections[model]["tasks"]["unattended_items"]["primary"]["name"]
        comparison = results[model]["calibration_comparison"]["unattended_items"][name]
        for key in ["global_same_recordings", "calibrated_disjoint_normal_recording"]:
            m = comparison[key]["aggregate"]
            lines.append(
                f"| {model} | {key} | {m['detected_events']}/{m['events']} | {m['normal_false_alerts']} | {m['normal_seconds'] / 60:.2f} | {f(m['false_alerts_per_normal_camera_hour'], 1)} |"
            )
    lines.extend(
        [
            "",
            "## Limits and next evidence needed",
            "",
            "- Public source diversity and independent event counts remain small despite substantial downloads and many frames. Smoking still lacks a new independently labelled surveillance-video source; the CCTV evaluation is static and retrospective.",
            "- Historical unattended intervals and public-sleep labels remain provisional assistant review. MEVA is an abandonment-action proxy; it does not certify persistent unattended state. Its metrics are kept separate.",
            "- Known film-part aliases are purged, exact decoded-pixel overlap is checked, and title/recording groups remain separated. Actor/movie identities beyond supplied names are not independently established.",
            "- Hard-negative runs can select a checkpoint before mining begins at epoch four; epoch metadata expose this. Mining is not claimed effective merely because its run was configured.",
            "- Existing CLIP backbone weights remain unchanged in this pilot. New weights are region/temporal heads; detector weights are pinned public pretrained checkpoints.",
            "- Littering, loitering and public sleep have an error audit but no newly trained temporal detector in this first pilot. Their next step is event/track annotation and a compatible detector rather than treating posture or rubbish presence as ground truth.",
            "",
            "## Artifacts",
            "",
            "- `comparison.csv`: all source-separated operating points for every retained seed/candidate.",
            "- `events.csv`: strict event metrics, including censored/all-three and known-criteria diagnostics.",
            "- Per-backbone `joint_selection.json`, checkpoints, development predictions, `event_policies.json`, raw `predictions.jsonl`, and `results.json`.",
            "- `error_audit/`: source score shifts, crop coverage, component audits, original-resolution review flags and the independent-review queue.",
            "- Data `input_identities.jsonl`, `data_verification.json`, `source_guard.json`, canonical verified/modeling manifests and AVSS source receipt.",
            "- `verification.json` and `unit_tests.log` document the final replay/checks when complete.",
            "",
        ]
    )
    lines[2] += (
        " The pilot gives mixed recognition changes and does not establish reliable camera transfer. MobileCLIP improves modestly on CCTV smoking while SigLIP regresses. The most useful fresh-source result is the explicit item-state rule; its two-event result and roughly 35-second median delay require substantially more validation."
    )
    lines.insert(
        lines.index("## Limits and next evidence needed") + 2,
        "- Historical event recall uses conservative provisional onset windows. Alerts in unreviewed transition gaps are unknown, not established false alarms; some precede a positive interval. Independent boundary review is needed before treating those cases as certain detection failures.",
    )
    lines.insert(
        lines.index("## Disjoint normal-recording calibration — historical diagnostic")
        - 1,
        "The alert-count column separates false alerts during normal footage from repeated alerts during an event. The SigLIP classifier plus state gate still repeats one alert; the state rule alone does not.",
    )
    text = "\n".join(lines)
    (run / "report.md").write_text(text)
    summary = args.summary
    if summary is not None:
        summary.write_text(
            text.replace(
                "## Artifacts",
                "## Artifacts\n\n[Full run and tables](../results/video_pilot_20261016/report.md).\n",
            )
        )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(smoking_chart))
    ax.bar(
        x - 0.18,
        [r[2] for r in smoking_chart],
        0.36,
        label="Scene baseline, same pilot data",
    )
    ax.bar(
        x + 0.18,
        [r[3] for r in smoking_chart],
        0.36,
        label="Development-selected pilot",
    )
    ax.set_xticks(
        x,
        [
            r[0].replace("-b16-224", "")
            + "\n"
            + r[1].replace("_action_transfer", "").replace("smoking_", "")
            for r in smoking_chart
        ],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Average precision")
    ax.set_title("Retrospective smoking evaluation; source-separated")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "smoking_sources.png", dpi=180)
    plt.close(fig)
    records = {
        r["id"]: r
        for r in map(
            json.loads, (data / "fresh_recordings.jsonl").read_text().splitlines()
        )
        if not r["right_censored_annotation"]
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=False)
    for i, model in enumerate(results):
        name = selections[model]["tasks"]["unattended_items"]["primary"]["name"]
        policies = json.loads((run / model / "event_policies.json").read_text())
        cutoff = policies["unattended_items"]["event_policies"][name]["threshold"]
        predictions = [
            r
            for r in map(
                json.loads, (run / model / "predictions.jsonl").read_text().splitlines()
            )
            if r["task"] == "unattended_items" and r["candidate"] == name
        ]
        windows = {
            r["id"]: r
            for r in map(
                json.loads,
                (data / "verified_fresh_avss.jsonl").read_text().splitlines(),
            )
        }
        for j, (key, record) in enumerate(sorted(records.items())):
            values = [r for r in predictions if r["recording_id"] == key]
            values.sort(key=lambda r: windows[r["id"]]["timestamps"][-1])
            ax = axes[i, j]
            times = [windows[r["id"]]["timestamps"][-1] for r in values]
            ax.plot(
                times, [r["score"] for r in values], lw=1, label="Causal window score"
            )
            ax.axhline(
                cutoff, color="black", ls="--", lw=1, label="Frozen dev threshold"
            )
            for a, b, label in record["intervals"]:
                if label == "unattended_items":
                    ax.axvspan(a, b, color="tab:green", alpha=0.18)
            event = results[model]["event_metrics"]["unattended_items"][name][
                "fresh_avss_clean_primary"
            ]
            rr = next(r for r in event["per_recording"] if r["recording_id"] == key)
            for time in rr["alerts"]:
                ax.axvline(time, color="tab:red", alpha=0.6, lw=1)
            ax.set_title(model + " / " + key.removeprefix("avss:"))
            ax.set_xlabel("Encoded video time (seconds)")
            ax.set_ylabel("Classifier logit")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Fresh-source timelines: green = author event; red = emitted global alert"
    )
    fig.tight_layout()
    fig.savefig(figures / "fresh_event_timelines.png", dpi=180)
    plt.close(fig)
    print(
        "VIDEO REPORT WRITTEN",
        len(clip_rows),
        "operating points",
        len(event_rows),
        "event summaries",
        flush=True,
    )


if __name__ == "__main__":
    main()
