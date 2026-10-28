"""Evaluate frozen pilot heads and strict event alerts; no test threshold fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.training.clip import binary_metrics
from anomaly_eval.training.video import event_metrics, make_alerts, unattended_state
from scripts.video.train_grounded_pilot import grounded_features
from scripts.video.train_person_pilot import instance_arrays, score_person
from scripts.video.train_video_pilot import batch_arrays, save, score_checkpoint


def intervals_for(record):
    normal = []
    events = []
    for a, b, label in record["intervals"]:
        a = max(20.0, a)
        b = min(b, record["duration"])
        if a > b:
            continue
        (normal if label == "normal" else events).append((a, b))
    return events, normal


def summarize_events(rows, scores, recordings, threshold, excluded=(), offsets=None):
    by_record = defaultdict(list)
    for row, score in zip(rows, scores):
        if row["recording_id"] in recordings and row["recording_id"] not in excluded:
            by_record[row["recording_id"]].append((row, float(score)))
    result = []
    for key, values in sorted(by_record.items()):
        record = recordings[key]
        values.sort(key=lambda x: x[0]["timestamps"][-1])
        times = [r["timestamps"][-1] for r, s in values]
        cutoff = threshold + (offsets or {}).get(record["group_id"], 0.0)
        alerts = make_alerts(times, [s for r, s in values], cutoff)
        events, normal = intervals_for(record)
        metrics = event_metrics(alerts, events, normal)
        result.append(
            {
                "recording_id": key,
                "group_id": record["group_id"],
                "threshold": float(cutoff),
                "alerts": alerts,
                **metrics,
            }
        )
    counts = {
        k: sum(r[k] for r in result)
        for k in [
            "events",
            "detected_events",
            "normal_false_alerts",
            "duplicate_alerts",
            "unknown_time_alerts",
            "normal_seconds",
            "reviewed_seconds",
        ]
    }
    delays = [d for r in result for d in r["delays_seconds"]]
    aggregate = {
        **counts,
        "recordings": len(result),
        "camera_or_scene_groups": len({r["group_id"] for r in result}),
        "event_recall": counts["detected_events"] / counts["events"]
        if counts["events"]
        else None,
        "false_alerts_per_normal_camera_hour": counts["normal_false_alerts"]
        * 3600
        / counts["normal_seconds"]
        if counts["normal_seconds"]
        else None,
        "unmatched_alerts_per_reviewed_camera_hour": (
            counts["normal_false_alerts"] + counts["duplicate_alerts"]
        )
        * 3600
        / counts["reviewed_seconds"]
        if counts["reviewed_seconds"]
        else None,
        "median_delay_seconds": float(np.median(delays)) if delays else None,
        "delays_seconds": delays,
    }
    return {"aggregate": aggregate, "per_recording": result}


def select_event_threshold(rows, scores, recordings):
    best = None
    selected = None
    for threshold in np.r_[np.nextafter(float(max(scores)), np.inf), np.unique(scores)]:
        metrics = summarize_events(rows, scores, recordings, float(threshold))[
            "aggregate"
        ]
        fph = metrics["false_alerts_per_normal_camera_hour"]
        if fph is None or fph > 1.0:
            continue
        key = (
            metrics["event_recall"] or 0.0,
            -metrics["duplicate_alerts"],
            -metrics["normal_false_alerts"],
            float(threshold),
        )
        if best is None or key > best:
            best = key
            selected = (float(threshold), metrics)
    if selected is None:
        raise ValueError("Development needs reviewed normal exposure")
    return {
        "threshold": selected[0],
        "development": selected[1],
        "selection": "maximum development event recall subject to <=1 normal false alert/camera-hour; then fewer duplicates/false alerts; highest threshold tie-break",
        "rule": {
            "persistence_seconds": 2.0,
            "release_seconds": 1.0,
            "cooldown_seconds": 10.0,
            "max_score_gap_seconds": 1.1,
        },
        "qualification": "Short development normal exposure cannot establish a reliable per-hour guarantee",
    }


def bootstrap_ap(rows, scores):
    from sklearn.metrics import average_precision_score

    groups = sorted({r["group_id"] for r in rows})
    if any(r["source"] != "hmdb51_action_transfer" for r in rows):
        return {
            "interval": None,
            "reason": "Insufficient independently identified cameras/recordings; photo identities are not supplied",
            "groups": len(groups),
        }
    if len(groups) < 5:
        return {
            "interval": None,
            "reason": "fewer than five independent supplied groups",
            "groups": len(groups),
        }
    rng = np.random.default_rng(20261016)
    positions = {
        g: [i for i, r in enumerate(rows) if r["group_id"] == g] for g in groups
    }
    labels = np.asarray([r["target"] for r in rows])
    scores = np.asarray(scores)
    values = []
    for _ in range(500):
        ids = [
            i
            for g in rng.choice(groups, len(groups), replace=True)
            for i in positions[g]
        ]
        if len(set(labels[ids])) == 2:
            values.append(average_precision_score(labels[ids], scores[ids]))
    return {
        "interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "groups": len(groups),
        "valid_draws": len(values),
        "method": "500 supplied-recording/title-group bootstrap resamples; not frame bootstrap",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["mobileclip2-s2", "siglip2-b16-224"], required=True
    )
    args = parser.parse_args()
    data = args.data.resolve()
    output = args.run.resolve() / args.model
    if (output / "results.json").exists():
        raise ValueError("Completed evaluation is immutable")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selection = json.loads((output / "joint_selection.json").read_text())
    rows = [
        json.loads(s)
        for s in (data / "modeling_windows.jsonl").read_text().splitlines()
    ]
    records = {
        r["id"]: r
        for r in map(json.loads, (data / "recordings.jsonl").read_text().splitlines())
        if r["mode"] == "continuous"
    }
    with np.load(data / "features" / args.model / "features.npz") as archive:
        features = {k: archive[k] for k in archive.files}
    grounded = grounded_features(data, features)
    grounded_tracks = {
        r["path"]: r["tracks"]
        for r in map(
            json.loads, (data / "grounding_tracks_all.jsonl").read_text().splitlines()
        )
    }
    # Freeze all development-fitted event policies before scoring fresh labels.
    policies = {}
    candidate_heads = {}
    dev_scores = {}
    for task in ["smoking", "unattended_items"]:
        candidates = selection["tasks"][task]["candidates"]
        family_best = {}
        for c in candidates:
            family = c["name"].split("_seed_")[0]
            key = lambda c: (
                c["selection"]["worst_source_ap"],
                c["selection"]["mean_source_ap"],
            )
            if family not in family_best or key(c) > key(family_best[family]):
                family_best[family] = c
        policies[task] = {
            "primary": selection["tasks"][task]["primary"]["name"],
            "families": {k: v["name"] for k, v in family_best.items()},
        }
        if task != "unattended_items":
            continue
        dev = [r for r in rows if r["task"] == task and r["split"] == "dev"]
        arrays = batch_arrays(dev, features)
        grounded_arrays = batch_arrays(dev, grounded)
        policies[task]["event_policies"] = {}
        for candidate in family_best.values():
            name = candidate["name"]
            head = torch.load(
                output / task / (name + ".pt"), weights_only=True, map_location="cpu"
            )
            scores = score_checkpoint(
                head,
                grounded_arrays
                if head.get("geometry_source") == "grounding_dino"
                else arrays,
            )
            dev_scores[name] = scores
            candidate_heads[task, name] = head
            policies[task]["event_policies"][name] = select_event_threshold(
                dev, scores, records
            )
            print(
                "EVENT THRESHOLD FROZEN",
                args.model,
                name,
                policies[task]["event_policies"][name]["threshold"],
                flush=True,
            )
    policies["unattended_items"]["known_source_criteria"] = {
        "source": "AVSS2007 author annotations",
        "stationary_seconds": 30.0,
        "owner_absence_seconds": 0.5,
        "same_item_required": True,
        "detector": "Grounding DINO tiny, fixed query/thresholds",
        "protocol": "separate from unchanged global-rule transfer; no test fitting",
    }
    save(output / "event_policies.json", policies)
    # Fresh input labels become visible to evaluation only after choices freeze.
    fresh = [
        json.loads(s)
        for s in (data / "verified_fresh_avss.jsonl").read_text().splitlines()
    ]
    fresh_records = {
        r["id"]: r
        for r in map(
            json.loads, (data / "fresh_recordings.jsonl").read_text().splitlines()
        )
    }
    clean_fresh = {
        k: r for k, r in fresh_records.items() if not r["right_censored_annotation"]
    }
    censored_fresh = {
        k: r for k, r in fresh_records.items() if r["right_censored_annotation"]
    }
    results = {}
    predictions = []
    event_results = {}
    calibration_results = {}
    calibration_ids = {
        "aboda:classroom": "aboda_video7",
        "aboda:walkway": "aboda_video2",
    }
    for task in ["smoking", "unattended_items"]:
        test = [
            r
            for r in rows
            if r["task"] == task and r["split"] in {"test", "external", "transfer"}
        ]
        if task == "unattended_items":
            test += fresh
        arrays = batch_arrays(test, features)
        grounded_arrays = (
            batch_arrays(test, grounded) if task == "unattended_items" else None
        )
        instances = instance_arrays(test, features) if task == "smoking" else None
        results[task] = {}
        event_results[task] = {}
        calibration_results[task] = {}
        for candidate in selection["tasks"][task]["candidates"]:
            name = candidate["name"]
            head = candidate_heads.get((task, name))
            if head is None:
                head = torch.load(
                    output / task / (name + ".pt"),
                    weights_only=True,
                    map_location="cpu",
                )
            scores = (
                score_person(head, instances)
                if head["kind"] == "person_mil"
                else score_checkpoint(
                    head,
                    grounded_arrays
                    if head.get("geometry_source") == "grounding_dino"
                    else arrays,
                )
            )
            predictions.extend(
                {
                    "id": r["id"],
                    "recording_id": r["recording_id"],
                    "group_id": r["group_id"],
                    "task": task,
                    "source": r["source"],
                    "split": r["split"],
                    "target": r["target"],
                    "candidate": name,
                    "score": float(s),
                }
                for r, s in zip(test, scores)
            )
            by_source = defaultdict(list)
            for i, r in enumerate(test):
                if r["target"] >= 0:
                    by_source[r["source"]].append(i)
                    if r["recording_id"] in clean_fresh:
                        by_source["avss2007_clean_alignment_primary"].append(i)
                    elif r["recording_id"] in censored_fresh:
                        by_source["avss2007_right_censored_diagnostic"].append(i)
            cohorts = {}
            for source, ids in by_source.items():
                selected = [test[i] for i in ids]
                ss = scores[ids]
                if len({r["target"] for r in selected}) < 2:
                    continue
                cohorts[source] = {
                    "n": len(ids),
                    "positive": sum(r["target"] for r in selected),
                    "groups": len({r["group_id"] for r in selected}),
                    "operating_points": {
                        op: binary_metrics([r["target"] for r in selected], ss, t)
                        for op, t in candidate["thresholds"].items()
                    },
                }
                if name == policies[task]["primary"]:
                    cohorts[source]["ap_group_bootstrap"] = bootstrap_ap(selected, ss)
            results[task][name] = cohorts
            if (
                task != "unattended_items"
                or name not in policies[task]["event_policies"]
            ):
                continue
            cutoff = policies[task]["event_policies"][name]["threshold"]
            event_results[task][name] = {
                "historical": summarize_events(test, scores, records, cutoff),
                "fresh_avss": summarize_events(test, scores, fresh_records, cutoff),
                "fresh_avss_clean_primary": summarize_events(
                    test, scores, clean_fresh, cutoff
                ),
                "fresh_avss_right_censored_diagnostic": summarize_events(
                    test, scores, censored_fresh, cutoff
                ),
            }
            source_state = np.asarray(
                [
                    unattended_state(grounded_tracks.get(r["views"][-1], []))
                    for r in test
                ]
            )
            gated = np.where(source_state, scores, -1e30)
            event_results[task][name]["fresh_avss_clean_known_30_second_rule"] = (
                summarize_events(test, gated, clean_fresh, cutoff)
            )
            event_results[task][name]["fresh_avss_clean_state_rule_only"] = (
                summarize_events(test, source_state.astype(float), clean_fresh, 0.5)
            )
            # Camera/scene offsets use only a disjoint normal calibration recording.
            dev = [r for r in rows if r["task"] == task and r["split"] == "dev"]
            ds = dev_scores[name]
            normal_dev = [float(s) for r, s in zip(dev, ds) if r["target"] == 0]
            reference = float(np.quantile(normal_dev, 0.95))
            offsets = {}
            cal_meta = {}
            for group, rec_id in calibration_ids.items():
                values = [
                    float(s)
                    for r, s in zip(test, scores)
                    if r["recording_id"] == rec_id and r["target"] == 0
                ]
                if not values:
                    continue
                offsets[group] = float(np.quantile(values, 0.95) - reference)
                cal_meta[group] = {
                    "recording_id": rec_id,
                    "normal_decisions": len(values),
                    "normal_95th_percentile": float(np.quantile(values, 0.95)),
                    "offset": offsets[group],
                    "positive_labels_used": False,
                }
            eligible = [i for i, r in enumerate(test) if r["group_id"] in offsets]
            cr = [test[i] for i in eligible]
            cs = scores[eligible]
            calibration_results[task][name] = {
                "calibration": cal_meta,
                "reference_development_normal_95th_percentile": reference,
                "global_same_recordings": summarize_events(
                    cr, cs, records, cutoff, excluded=calibration_ids.values()
                ),
                "calibrated_disjoint_normal_recording": summarize_events(
                    cr,
                    cs,
                    records,
                    cutoff,
                    excluded=calibration_ids.values(),
                    offsets=offsets,
                ),
                "fresh_avss_calibration": {
                    "status": "unavailable",
                    "reason": "no disjoint normal recording and unverified physical camera identities",
                },
            }
        print("EVALUATED", args.model, task, len(test), "decisions", flush=True)
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in predictions)
    )
    save(
        output / "results.json",
        {
            "model": args.model,
            "clip_metrics": results,
            "event_metrics": event_results,
            "calibration_comparison": calibration_results,
            "policies_sha256": hashlib.sha256(
                (output / "event_policies.json").read_bytes()
            ).hexdigest(),
            "fresh_cohort_qualification": json.loads(
                (data / "fresh_cohort.json").read_text()
            ),
            "limitations": [
                "historical labels provisional",
                "HMDB adaptation test retrospective",
                "three fresh AVSS sequences in one conservative source group",
                "short normal exposure; no reliable low false-alert-rate claim",
                "strict alert-onset event matching",
                "no independent human verification of historical labels",
            ],
        },
    )


if __name__ == "__main__":
    main()
