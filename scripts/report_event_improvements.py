"""Replay new metrics and publish the qualified improvement study report."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from anomaly_eval.training.event_memory import ItemEventMemory
from anomaly_eval.training.strategies import overlap
from scripts.run_event_improvements import (
    combined_scores,
    source_budget_threshold,
    source_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/event_improvements_20261017"
DATA = ROOT / "data/event_improvements_20261017"


def read(name):
    return json.loads((OUT / name).read_text())


def jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    selection, scores, events, component = [
        read(n)
        for n in [
            "smoking_selection.json",
            "smoking_results.json",
            "event_results.json",
            "component_results.json",
        ]
    ]
    verified = 0
    pilot_data = ROOT / "data/video_pilot_20261016"
    row_map = {r["id"]: r for r in jsonl(pilot_data / "modeling_windows.jsonl")}
    for model, sel in selection.items():
        prior = ROOT / "results/video_pilot_20261016" / model
        stored = json.loads((prior / "development_predictions.json").read_text())
        dev = {
            k.split("/", 1)[1]: v for k, v in stored.items() if k.startswith("smoking/")
        }
        dev.update(
            json.loads((prior / "person_selection.json").read_text())[
                "development_predictions"
            ]
        )
        ids = next(iter(dev.values()))["ids"]
        assert all(v["ids"] == ids for v in dev.values())
        dev_rows = [row_map[k] for k in ids]
        assert all(r["split"] == "dev" for r in dev_rows)
        raw_scores = {
            k: np.asarray(v["scores"], dtype=np.float64) for k, v in dev.items()
        }
        cs = sel["candidates"]
        for c in cs.values():
            s = combined_scores(c["members"], raw_scores)
            assert source_budget_threshold(dev_rows, s) == c["threshold"]
            assert source_metrics(dev_rows, s, c["threshold"]) == c["development"]
        assert sel["selected"] == max(
            cs,
            key=lambda k: tuple(
                cs[k][v] for v in ["worst_recall", "mean_recall", "worst_ap", "mean_ap"]
            ),
        )
        assert all(
            v["fpr"] <= 0.05 for c in cs.values() for v in c["development"].values()
        )
        rows = jsonl(OUT / (model + "_smoking_predictions.jsonl"))
        for name in {r["candidate"] for r in rows}:
            subset = [r for r in rows if r["candidate"] == name]
            replay = source_metrics(
                subset, [r["score"] for r in subset], cs[name]["threshold"]
            )
            assert replay == scores[model][name]["sources"]
        verified += len(rows)
    traces = jsonl(OUT / "event_traces.jsonl")
    grouped = defaultdict(list)
    for r in traces:
        grouped[(r["recording"], r["policy"])].append(r)
    alert_count = 0
    for values in grouped.values():
        memory = ItemEventMemory()
        for r in values:
            alarms = memory.update(r["time"], r["tracks"])
            assert alarms == r["alerts"]
            alert_count += len(alarms)
    references = {r["path"]: r for r in jsonl(DATA / "component_frames.jsonl")}
    # Rebuild filtering and NMS from serialized views independently of scoring.
    raw_views = defaultdict(list)
    for row in jsonl(DATA / "tiled_detections.jsonl"):
        raw_views[row["path"]].append(row)
    recomputed = defaultdict(lambda: [0, 0])
    for r in jsonl(OUT / "component_predictions.jsonl"):
        assert len(raw_views[r["path"]]) == 5
        detections = [
            d
            for view in raw_views[r["path"]]
            if r["mode"] != "full" or np.allclose(view["tile"], [0, 0, 1, 1])
            for d in view["detections"]
            if d["label"] == 1001
        ]
        candidates = sorted(detections, key=lambda d: -d["confidence"])
        merged = []
        while candidates:
            best = candidates.pop(0)
            merged.append(best)
            candidates = [
                d for d in candidates if overlap(best["box"], d["box"]) <= 0.5
            ]
        assert merged == r["detections"], (r["path"], r["mode"])
        g = references[r["path"]]["boxes"]
        p = r["detections"]
        for cutoff in [0.3, 0.5]:
            matched = 0
            if p:
                cost = np.asarray([[1 - overlap(a, b["box"]) for b in p] for a in g])
                a, b = linear_sum_assignment(cost)
                matched = sum(cost[i, j] <= 1 - cutoff for i, j in zip(a, b))
            v = recomputed[(r["source"], r["mode"], cutoff)]
            v[0] += matched
            v[1] += len(g)
    for r in component["aggregate"]:
        a, b = recomputed[(r["source"], r["mode"], r["iou"])]
        assert (int(a), b) == (r["matched"], r["reference_boxes"])
    text = [
        "# Event and small-item improvements — 17 October 2026",
        "",
        "Completed development-only smoking operating-point/ensemble selection, causal item-state/alert-memory ablations, and a GPU full-image versus tiled small-object detector comparison. No new backbone training or latency benchmark was performed. Historical smoking, CAVIAR and AVSS sources were already inspected; all reported policy changes on those sources are retrospective.",
        "",
        "## Smoking: source-aware low-FPR selection",
        "",
        "Each encoder compares 14 saved heads, four equal-weight averages of sigmoid scores within three-seed families, and one equal-weight all-head score average: **19 policies per encoder / 38 total**. Sigmoid values are score transforms, not calibrated probabilities. A single cutoff must meet <=5% empirical FPR in every development source. Selection maximizes worst-source recall, then mean recall, then worst/mean AP. Both encoders’ selections freeze before loading held-out scores. The saved heads and backbone tensors remain unchanged. This reuses the same development set used for previous model selection, so it is exploratory and may overfit that set.",
        "",
        "| Backbone / policy | CCTV AP | CCTV recall / FPR | HMDB AP | HMDB recall / FPR |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for model, values in scores.items():
        sel = selection[model]["selected"]
        for name in ["previous_primary_original_policy", sel]:
            c = values[name]["sources"]["smoking_cctv"]
            h = values[name]["sources"]["hmdb51_action_transfer"]
            text.append(
                f"| {model} / {'previous' if name.startswith('previous') else 'new: ' + sel} | {c['ap']:.3f} | {c['recall']:.1%} / {c['fpr']:.1%} | {h['ap']:.3f} | {h['recall']:.1%} / {h['fpr']:.1%} |"
            )
    text += [
        "",
        "SigLIP’s all-head ensemble improves retrospective HMDB ranking and recall at the same observed FPR. Its CCTV recall remains very low. MobileCLIP retains the same head with a more conservative threshold, sacrificing recall. Per-source development constraints do **not** guarantee the same budget on a new camera. HMDB movie/web clips are not CCTV generalization. The AP-selected alternative is retained in the JSON rather than silently substituted after testing.",
        "",
        "## Stationarity smoothing and per-item event memory",
        "",
        "The original proximity-owner tracker remains the control. The new variant uses a trailing three-center median and requires one second of sustained displacement to reset stationarity; pending movement blocks an alert. A per-item memory uses spatial one-to-one association across short track-ID gaps and emits once until sustained owner return, movement, or a 30-second absence expires memory. The memory does not invent ownership or extend stationary evidence across missing tracks. All rule settings are fixed engineering choices, not fitted on test labels. The same 30-second stationary/.5-second owner-away criterion and two-second alert persistence apply.",
        "",
        "| Policy, AVSS primary (retrospective) | Events | Normal false / repeat alerts | Median onset delay |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, v in events["aggregate"]["avss_primary_retrospective"].items():
        text.append(
            f"| {name} | {v['detected_events']}/{v['events']} | {v['normal_false_alerts']} / {v['duplicate_alerts']} | {v['median_delay_seconds']:.1f} s |"
        )
    upper = events["aggregate"]["avss_primary_retrospective"][
        "robust_tracks_item_memory"
    ]["normal_far_upper_95_per_hour"]
    text += [
        "",
        f"Neither smoothing nor memory improves the two-event primary result; smoothing adds about 0.25 seconds to median delay. Zero false alerts in only 165.74 seconds of labelled normal exposure gives a **one-sided 95% Poisson upper bound of {upper:.1f} false alerts/hour**, assuming a homogeneous Poisson process; it is not a camera-transfer guarantee. At least three hours with zero false alerts would be needed just for that conditional upper bound to fall below one/hour.",
        "",
        "On the five historical test events the fixed source-criteria state rule detects 4/5, with zero normal false alerts and one unknown-time alert. Original global debounce already has zero repeated alerts; smoothing plus item memory also has zero. On the separately censored AVSS HARD diagnostic, item memory increases repeats from one to two. It is not promoted as an accuracy improvement. Unit tests verify independent simultaneous items, track-fragment memory, causality and reset semantics; these synthetic checks are not real-world event results. Historical provisional onset windows do not share a consistent 30-second source definition, so their state-rule diagnostic must remain separate from classifier metrics.",
        "",
        "## Small-object detection and new data",
        "",
        "Downloaded the pinned author archive for [the abandoned-object release](https://github.com/hangsuuuu/video-of-abandoned-object-detection). Three entries explicitly alias existing ABODA videos and are excluded from new-data counts. The other three are 7.6–14.6-second highway videos with small debris boxes; they lack owner histories and 30-second event evidence. They are a component stress test, not three new verified unattended-item events. Source boxes/timestamps are unchanged, frame sizes checked, actual encoded FPS used, and ten positive annotated frames per bundled video are sampled. Out-of-stream timestamps are exposed in the preparation audit; no onset is rescaled.",
        "",
        "A fixed full-image plus four overlapping 60%-size tiles is compared with full-image Grounding DINO using the same pinned public weights, generic query, score thresholds and NMS. Source boxes never select inference tiles. The 166 annotated frames produce 830 GPU image views in float32 with TF32 off and no pacing.",
        "",
        "| Reference source | IoU | Full-image recall | Full + tiles recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source in ["caviar_training_component", "author_highway_debris_component"]:
        for cutoff in [0.3, 0.5]:
            vs = {
                r["mode"]: r
                for r in component["aggregate"]
                if r["source"] == source and r["iou"] == cutoff
            }
            text.append(
                f"| {source} | {cutoff} | {vs['full']['matched']}/{vs['full']['reference_boxes']} ({vs['full']['recall']:.1%}) | {vs['full_plus_tiles']['matched']}/{vs['full_plus_tiles']['reference_boxes']} ({vs['full_plus_tiles']['recall']:.1%}) |"
            )
    text += [
        "",
        "References are partial item annotations. Unmatched detector boxes cannot establish false positives, so no item-precision or event-accuracy claim is made. Adjacent frames are correlated. The CAVIAR cohort is training-source evidence; the new highway collection is an out-of-domain stress test. Tiling has not yet been validated as a full event-tracking replacement.",
        "",
        "## Evaluation status and next action",
        "",
        "Retain the earlier released bundles as baselines. The SigLIP ensemble is a candidate for a newly frozen camera test, not an automatic replacement. Keep the simple state rule until stronger evidence favors the added tracker complexity. Review and label event ownership/boundaries and matched normal footage from independent target cameras. Littering needs pedestrian disposal actions plus benign placing/carrying; loitering needs complete tracks and operational-duration labels; public sleep needs time persistence and rest/lying controls. More adjacent frames and more hyperparameter trials do not supply that evidence.",
        "",
        "## Reproduction and verification",
        "",
        "Run `scripts/run_event_improvements.py` in a new output directory to replay ensembles and state ablations. Run `scripts/prepare_item_component.py` once after the pinned author source acquisition, then `scripts/benchmark_item_tiles.py` with the existing GPU runtime and `scripts/report_event_improvements.py` for independent score/alert/component-metric replay. The completed result directory refuses overwrites for scored studies. CPU postprocessing reuses actual saved GPU predictions; only the new tiled detector requires fresh encoder inference.",
        "",
        f"Verified **{verified:,} smoking predictions**, all 38 development-policy constraints/selections, **{len(traces):,} causal trace decisions** and {alert_count} emitted memory alerts. Component reference matches are recomputed from saved merged predictions. Source/code hashes, acquisition receipt, pixel inputs and raw detector views are retained. Earlier failed preparation attempts were corrected before the completed run; no labels, model weights or selected thresholds were changed in response to held-out results.",
        "",
        "Artifacts: `smoking_selection.json`, `smoking_results.json`, per-encoder smoking predictions, `event_results.json`, `event_traces.jsonl`, `component_results.json`, `component_predictions.jsonl`, `protocol.json`, `verification.json`, and data `component_protocol.json`, `component_frames.jsonl`, `tiled_detections.jsonl`, `receipt.json`.",
        "",
    ]
    report = "\n".join(text)
    (OUT / "report.md").write_text(report)
    (ROOT / "reports/event_improvements_findings_20261017.md").write_text(report)
    verification = {
        "status": "complete",
        "smoking_predictions_replayed": verified,
        "development_policies_checked": sum(
            len(v["candidates"]) for v in selection.values()
        ),
        "trace_decisions_replayed": len(traces),
        "memory_alerts_replayed": alert_count,
        "component_metric_cells_recomputed": len(component["aggregate"]),
        "artifact_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in OUT.iterdir()
            if p.suffix in {".json", ".jsonl"} and p.name != "verification.json"
        },
        "no_new_event_validation_claim": "AVSS/historical sources retrospective; highway data component only",
    }
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(
        "VERIFIED",
        verified,
        "predictions;",
        len(traces),
        "trace decisions;",
        len(component["aggregate"]),
        "component cells",
    )


if __name__ == "__main__":
    main()
