"""Freeze dev-only smoking ensembles and replay causal item-alert improvements."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit

from anomaly_eval.training.clip import binary_metrics
from anomaly_eval.training.event_memory import (
    ItemEventMemory,
    RobustStationarityTracker,
    poisson_upper_rate,
)
from anomaly_eval.training.video import event_metrics, make_alerts, unattended_state
from scripts.video.evaluate_video_pilot import intervals_for

ROOT = Path(__file__).resolve().parents[2]


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_budget_threshold(rows, scores, budget=0.05):
    """Smallest strict cutoff satisfying each supplied dev source's FPR budget."""
    cutoffs = []
    for source in sorted({r["source"] for r in rows}):
        normal = sorted(
            [
                float(s)
                for r, s in zip(rows, scores)
                if r["source"] == source and r["target"] == 0
            ],
            reverse=True,
        )
        if not normal:
            raise ValueError(f"Missing development normals: {source}")
        allowed = int(np.floor(len(normal) * budget))
        cutoffs.append(np.nextafter(normal[allowed], np.inf))
    return float(max(cutoffs))


def source_metrics(rows, scores, threshold):
    return {
        source: binary_metrics(
            [r["target"] for r in rows if r["source"] == source],
            [float(s) for r, s in zip(rows, scores) if r["source"] == source],
            threshold,
        )
        for source in sorted({r["source"] for r in rows})
    }


def combined_scores(definition, components):
    # Fixed, equal probability averaging. No fitted stacking coefficients.
    return np.mean([expit(components[k]) for k in definition], axis=0)


def smoking(output, previous, data):
    row_map = {r["id"]: r for r in read_rows(data / "modeling_windows.jsonl")}
    selections = {}
    # Both policies freeze for both encoders before any held-out scores load.
    for model in ["mobileclip2-s2", "siglip2-b16-224"]:
        directory = previous / model
        dev = json.loads((directory / "development_predictions.json").read_text())
        components = {
            k.split("/", 1)[1]: v for k, v in dev.items() if k.startswith("smoking/")
        }
        components.update(
            json.loads((directory / "person_selection.json").read_text())[
                "development_predictions"
            ]
        )
        ids = next(iter(components.values()))["ids"]
        rows = [row_map[k] for k in ids]
        assert all(r["split"] == "dev" for r in rows)
        assert all(v["ids"] == ids for v in components.values())
        scores = {
            k: np.asarray(v["scores"], dtype=np.float64) for k, v in components.items()
        }
        definitions = {k: [k] for k in scores}
        families = defaultdict(list)
        for k in scores:
            if "_seed_" in k:
                families[k.split("_seed_")[0]].append(k)
        definitions.update({k + "_ensemble": sorted(v) for k, v in families.items()})
        definitions["all_head_ensemble"] = sorted(scores)
        candidates = {}
        for name, members in definitions.items():
            s = combined_scores(members, scores)
            threshold = source_budget_threshold(rows, s)
            m = source_metrics(rows, s, threshold)
            candidates[name] = {
                "members": members,
                "threshold": threshold,
                "development": m,
                "worst_recall": min(v["recall"] for v in m.values()),
                "mean_recall": float(np.mean([v["recall"] for v in m.values()])),
                "worst_ap": min(v["ap"] for v in m.values()),
                "mean_ap": float(np.mean([v["ap"] for v in m.values()])),
            }
        selections[model] = {
            "candidates": candidates,
            "selected": max(
                candidates,
                key=lambda k: tuple(
                    candidates[k][v]
                    for v in ["worst_recall", "mean_recall", "worst_ap", "mean_ap"]
                ),
            ),
            "ap_selected_ensemble": max(
                candidates,
                key=lambda k: (candidates[k]["worst_ap"], candidates[k]["mean_ap"]),
            ),
            "dev_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        }
    save(output / "smoking_selection.json", selections)
    results = {}
    for model, selection in selections.items():
        by_candidate = defaultdict(dict)
        for r in read_rows(previous / model / "predictions.jsonl"):
            if r["task"] == "smoking":
                by_candidate[r["candidate"]][r["id"]] = r
        ids = sorted(next(iter(by_candidate.values())))
        rows = [next(iter(by_candidate.values()))[k] for k in ids]
        assert all(r["split"] in {"test", "external"} for r in rows)
        components = {
            k: np.asarray([v[i]["score"] for i in ids]) for k, v in by_candidate.items()
        }
        primary = json.loads((previous / model / "joint_selection.json").read_text())[
            "tasks"
        ]["smoking"]["primary"]
        names = {
            selection["selected"],
            selection["ap_selected_ensemble"],
            primary["name"],
        }
        results[model] = {}
        prediction_rows = []
        for name in sorted(names):
            c = selection["candidates"][name]
            scores = combined_scores(c["members"], components)
            results[model][name] = {
                "threshold": c["threshold"],
                "sources": source_metrics(rows, scores, c["threshold"]),
            }
            for r, s in zip(rows, scores):
                prediction_rows.append(
                    {
                        **{k: r[k] for k in ["id", "source", "target", "split"]},
                        "candidate": name,
                        "score": float(s),
                    }
                )
        results[model]["previous_primary_original_policy"] = {
            "sources": source_metrics(
                rows,
                components[primary["name"]],
                primary["thresholds"]["dev_fpr_budget"],
            ),
            "threshold": primary["thresholds"]["dev_fpr_budget"],
            "candidate": primary["name"],
        }
        (output / (model + "_smoking_predictions.jsonl")).write_text(
            "".join(json.dumps(r) + "\n" for r in prediction_rows)
        )
    save(output / "smoking_results.json", results)


def event_replay(output, data):
    recordings = [
        r
        for n in ["recordings.jsonl", "fresh_recordings.jsonl"]
        for r in read_rows(data / n)
        if r["mode"] == "continuous"
    ]
    detections = {
        r["path"]: r["detections"]
        for r in read_rows(data / "grounding_detections.jsonl")
    }
    original = {
        r["path"]: r["tracks"] for r in read_rows(data / "grounding_tracks_all.jsonl")
    }
    results = defaultdict(dict)
    traces = []
    for r in recordings:
        tracker = RobustStationarityTracker(new_threshold=0.25, low_threshold=0.2)
        memories = {
            k: ItemEventMemory()
            for k in ["original_tracks_item_memory", "robust_tracks_item_memory"]
        }
        emitted = defaultdict(list)
        times, old_scores, robust_scores = [], [], []
        for f in r["frames"]:
            new = tracker.update(f["time"], detections[f["path"]])
            old = original[f["path"]]
            # Warm tracking from start, but preserve the previous 20 s decision warmup.
            if f["time"] < 20:
                continue
            times.append(f["time"])
            old_scores.append(int(unattended_state(old)))
            robust_scores.append(
                int(
                    any(
                        t["kind"] == "item"
                        and t["owner"] is not None
                        and t["stationary"] >= 30
                        and t["owner_absence"] >= 0.5
                        and not t.get("motion_pending", False)
                        for t in new
                    )
                )
            )
            for name, tracks in [
                ("original_tracks_item_memory", old),
                ("robust_tracks_item_memory", new),
            ]:
                alarms = memories[name].update(f["time"], tracks)
                emitted[name].extend(a["time"] for a in alarms)
                traces.append(
                    {
                        "recording": r["id"],
                        "time": f["time"],
                        "policy": name,
                        "alerts": alarms,
                        "tracks": tracks,
                    }
                )
        emitted["original_tracks_global_debounce"] = make_alerts(times, old_scores, 0.5)
        emitted["robust_tracks_global_debounce"] = make_alerts(
            times, robust_scores, 0.5
        )
        events, normal = intervals_for(r)
        group = (
            "avss_censored"
            if r.get("right_censored_annotation")
            else "avss_primary_retrospective"
            if r["id"].startswith("avss:")
            else "historical_" + r["split"]
        )
        for name, alarms in emitted.items():
            m = event_metrics(alarms, events, normal)
            results[group].setdefault(name, []).append(
                {"recording": r["id"], "alerts": alarms, **m}
            )
    aggregates = {}
    for group, policies in results.items():
        aggregates[group] = {}
        for name, values in policies.items():
            total = {
                k: sum(r[k] for r in values)
                for k in [
                    "events",
                    "detected_events",
                    "normal_false_alerts",
                    "duplicate_alerts",
                    "unknown_time_alerts",
                    "normal_seconds",
                ]
            }
            delays = [x for r in values for x in r["delays_seconds"]]
            total.update(
                median_delay_seconds=float(np.median(delays)) if delays else None,
                delays_seconds=delays,
                normal_far_upper_95_per_hour=poisson_upper_rate(
                    total["normal_false_alerts"], total["normal_seconds"]
                )
                if total["normal_seconds"]
                else None,
            )
            aggregates[group][name] = total
    save(
        output / "event_results.json",
        {"aggregate": aggregates, "per_recording": results},
    )
    (output / "event_traces.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in traces)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/event_improvements_20261017"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "run.json").exists():
        raise ValueError("Completed output is immutable")
    data = ROOT / "data/video_pilot_20261016"
    save(
        args.output / "protocol.json",
        {
            "date": "2026-10-17",
            "status": "frozen_before_new_policy_scores",
            "smoking": "14 previous heads + four within-family seed ensembles + all-head ensemble, each encoder; equal sigmoid averages; single threshold meeting <=5% FPR in every dev source; select worst then mean dev source recall, then worst/mean AP; report AP-selected alternative and previous primary",
            "events": "Original fixed Grounding DINO boxes, original and three-center median/sustained-motion stationarity; same 30s item and .5s absent owner gate, 2s persistence. Per-item spatial alert memory: 5s owner return or 3s motion to rearm, 30s missing memory expiry. No test fitting.",
            "qualification": "All smoking/historical/AVSS results retrospective: prior results inspected. Additional outdoor author data are component evidence only until label and time alignment audit. No latency experiment; no backbone weight updates.",
            "original_data": str(data.relative_to(ROOT)),
            "source_sha256": {
                n: hashlib.sha256((data / n).read_bytes()).hexdigest()
                for n in [
                    "modeling_windows.jsonl",
                    "grounding_detections.jsonl",
                    "grounding_tracks_all.jsonl",
                    "fresh_recordings.jsonl",
                ]
            },
            "code_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    ROOT / "anomaly_eval/training/event_memory.py",
                ]
            },
        },
    )
    smoking(args.output, ROOT / "results/video_pilot_20261016", data)
    event_replay(args.output, data)
    save(
        args.output / "run.json",
        {
            "status": "complete",
            "no_latency_tests": True,
            "evaluation": "retrospective; development-only model/threshold selection",
        },
    )
    print("COMPLETE", args.output)


if __name__ == "__main__":
    main()
