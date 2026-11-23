"""Export a round-3 selection as a runtime bundle (`ensemble-20261114`).

    export_round3.py --run r3_final --smoking-channels r3_detg --normal-cap 0.02 \
        --detector allg --output ../../results/f1_study_20261111/bundle_v10

Differences from the round-2 bundle:

* a member can be a zero-shot text head (fixed text vectors of the SigLIP2
  encoder, no fitting) or the smoking detector's confidence;
* smoking has channels: each is a small ensemble with its own threshold, and
  the window's score is the largest margin over a channel's threshold (flag
  at 0). Other tasks keep one ensemble and one threshold;
* thresholds are also capped by the share of ordinary new-site footage they
  flag (fit-side cameras only), at two levels: the bundle carries a
  "balanced" (5%) and a "low_false_alarm" (2%) set for the same heads.
"""

from __future__ import annotations

import os

os.environ["F1_ROUND3"] = "1"

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import heads
import numpy as np
import round3_report as report
import zero_shot
from common import OUT, ROOT, TASKS, evaluate, load_rows, macro
from export import BACKBONES
from sklearn.linear_model import LogisticRegression

STAGES = {"full": "full", "crop": "crop:yolo"}


def members_of(task, saved, arrays, text, prefix):
    out = []
    weights = saved.get("weights") or [1.0] * len(saved["detail"])
    for member, weight in zip(saved["detail"], weights):
        if weight <= 0:
            continue
        if "recipes" in member:
            exported = []
            for r in member["recipes"]:
                key = f"{prefix}/{member['tag']}/{r['name']}"
                arrays[key + "/weight"] = np.stack([h[0] for h in r["heads"]]).astype(
                    np.float32
                )
                arrays[key + "/bias"] = np.array([h[1] for h in r["heads"]], np.float32)
                exported.append(
                    {k: r[k] for k in ["recipe", "name", "C", "mean", "std", "oof_ap"]}
                    | {"arrays": key}
                )
            out.append(
                {
                    "tag": member["tag"],
                    "recipes": exported,
                    "mean": member["mean"],
                    "std": member["std"],
                }
            )
            continue
        kind, _, name = member["tag"].partition(":")
        entry = {
            "tag": member["tag"],
            "kind": kind,
            "mean": member["mean"],
            "std": member["std"],
        }
        if kind == "zeroshot":
            key = f"{prefix}/zeroshot"
            arrays[key + "/positive"] = text[task][0].astype(np.float32)
            arrays[key + "/negative"] = text[task][1].astype(np.float32)
            entry |= {
                "backbone": "siglip2-b",
                "stages": [STAGES[n] for n in name.split("+")],
                "arrays": key,
                "prompts": {
                    "behaviour": zero_shot.PROMPTS[task][0],
                    "ordinary": zero_shot.PROMPTS[task][1],
                },
            }
        elif kind != "detector":
            raise ValueError(f"Cannot export member {member['tag']}")
        out.append(entry)
    return out


def operating_point(args, by_id, normal_cap, cross_cap, fpr_cap=None):
    """Thresholds, fused smoking scores and calibrations for one cap setting."""
    with open(OUT / "runs" / f"{args.run}_round3.pkl", "rb") as f:
        saved = pickle.load(f)
    names = [args.run, *args.smoking_channels]
    if args.smoking_channels:
        share = len(names)
        saved = report.fuse(
            saved,
            names,
            by_id,
            normal_cap / share if normal_cap else None,
            cross_cap / share,
            fpr_cap=fpr_cap,
        )
    thresholds, info = report.thresholds_for(
        saved, by_id, normal_cap, cross_cap, fpr_cap=fpr_cap
    )
    if args.smoking_channels:
        thresholds["smoking"] = 0.0
    calibration = {}
    for task in TASKS:
        s, ids = saved[task]["scores"], saved[task]["ids"]
        core = [by_id[i] for i in ids["own"] + ids["used"]]
        y = np.array([r["target"] for r in core])
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(
            np.concatenate([s["own"], s["used"]])[:, None],
            y,
            sample_weight=heads.weights(core),
        )
        slope = float(platt.coef_[0, 0])
        if slope <= 0:
            print(f"{task}: out-of-fold calibration slope {slope:.2f}; none exported")
            calibration[task] = None
            continue
        calibration[task] = {"slope": slope, "intercept": float(platt.intercept_[0])}
    held = [by_id[i] for task in TASKS for i in saved[task]["ids"]["test"]]
    scores = [v for task in TASKS for v in saved[task]["scores"]["test"].tolist()]
    results = evaluate(held, scores, thresholds)
    recall, flagged = report.summary(report.site_metrics(saved, by_id, thresholds))
    cross = float(
        np.mean(
            [(saved[t]["scores"]["aux_test"] >= thresholds[t]).mean() for t in TASKS]
        )
    )
    fit_recall = {}
    for task in TASKS:
        s, ids = saved[task]["scores"], saved[task]["ids"]
        core = [by_id[i] for i in ids["own"] + ids["used"]]
        y = np.array([r["target"] for r in core])
        v = np.concatenate([s["own"], s["used"]])
        fit_recall[task] = (
            float((v[y == 1] >= thresholds[task]).mean()) if y.any() else None
        )
    return {
        "fit_recall": fit_recall,
        "saved": saved,
        "thresholds": thresholds,
        "info": info,
        "calibration": calibration,
        "summary": {
            "macro_f1": macro(results),
            "macro_ap": macro(results, "ap"),
            "sources": results,
            "cross_task_flagged": cross,
            "new_site_recall": recall,
            "new_site_ordinary_flagged": flagged,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--smoking-channels", nargs="*", default=[])
    parser.add_argument("--normal-cap", type=float, default=0.02)
    parser.add_argument("--balanced-cap", type=float, default=0.05)
    parser.add_argument("--cross-cap", type=float, default=0.05)
    parser.add_argument("--detector", default="allg")
    parser.add_argument(
        "--detector-cap",
        type=float,
        default=0.0,
        help="Ordinary fit-side footage the detector channel may flag (0 = none)",
    )
    parser.add_argument(
        "--high-recall-cap",
        type=float,
        default=0.10,
        help="high_recall point: share of own negatives, other behaviours' windows "
        "and ordinary footage each task may flag out of fold",
    )
    parser.add_argument(
        "--min-fit-recall",
        type=float,
        default=0.05,
        help="Refuse a threshold that flags less than this share of the task's own "
        "fitting positives out of fold (it would never fire)",
    )
    parser.add_argument(
        "--allow-no-fire",
        nargs="*",
        default=[],
        help="Tasks knowingly shipped with a threshold that does not fire",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    by_id = {r["id"]: r for r in load_rows()}
    report.DETECTOR_CAP = args.detector_cap
    # Three operating points from the same heads. "balanced" (the default) and
    # "low_false_alarm" start at the out-of-fold max-F1 threshold and cap the
    # ordinary fit-side footage a task may flag at 5% and 2%. "high_recall"
    # goes as low as flagging 10% of the task's own negatives, of other
    # behaviours' windows and of ordinary footage allows (never above
    # balanced). All are set on fitting-side scores only.
    hr = args.high_recall_cap
    settings = {
        "balanced": (args.balanced_cap, args.cross_cap, None),
        "low_false_alarm": (args.normal_cap, args.cross_cap, None),
        "high_recall": (hr, hr, hr),
    }
    caps = {k: v[0] for k, v in settings.items()}
    points = {k: operating_point(args, by_id, *v) for k, v in settings.items()}
    default = "balanced"
    silent = [
        f"{task} at {name} (flags {r:.0%} of its fitting positives)"
        for name, point in points.items()
        for task, r in point["fit_recall"].items()
        if r is not None and r < args.min_fit_recall and task not in args.allow_no_fire
    ]
    if silent:
        raise SystemExit(
            "Threshold would never fire: "
            + "; ".join(silent)
            + ". Fix the fit, or pass --allow-no-fire to ship it knowingly."
        )
    with open(OUT / "runs" / f"{args.run}_round3.pkl", "rb") as f:
        primary = pickle.load(f)
    channel_runs = {args.run: primary["smoking"]}
    for name in args.smoking_channels:
        with open(OUT / "runs" / f"{name}_round3.pkl", "rb") as f:
            channel_runs[name] = pickle.load(f)["smoking"]
    text = zero_shot.text_features("google/siglip2-base-patch16-224")
    backbones = ["mobileclip2-b", "siglip2-b"]
    bundle = {
        "tasks": {},
        "backbones": {b: BACKBONES[b] for b in backbones},
        "selection": {"run": args.run, "smoking_channels": args.smoking_channels},
        "cross_task_cap": args.cross_cap,
        "ordinary_footage_caps": caps,
        "detector_ordinary_cap": args.detector_cap,
        # As scored in the study (smoking_scores.py, suffix g).
        "smoking_detector": {"square_padding": True},
        "operating_points": list(points),
        "default_operating_point": default,
        # Share of each task's fitting positives its threshold flags, out of
        # fold. A task listed in no_fire_tasks was shipped knowing it is ~0.
        "fit_recall": {k: v["fit_recall"] for k, v in points.items()},
        "no_fire_tasks": sorted(args.allow_no_fire),
    }
    arrays = {}
    for task in TASKS:
        ids = primary[task]["ids"]
        core = [by_id[i] for i in ids["own"] + ids["used"]]
        entry = {
            "threshold": points[default]["thresholds"][task],
            "thresholds": {k: v["thresholds"][task] for k, v in points.items()},
            "calibration": points[default]["calibration"][task],
            "calibrations": {k: v["calibration"][task] for k, v in points.items()},
            "fitted_rows": len(core),
            "fitted_sources": sorted({r["source"] for r in core}),
        }
        if entry["calibration"] is None:
            # Out-of-fold scores ran backwards; ship no percentage at all
            # rather than one that falls as the score rises.
            del entry["calibration"], entry["calibrations"]
        if task == "smoking" and args.smoking_channels:
            entry["channels"] = []
            for name, run in channel_runs.items():
                members = members_of(task, run, arrays, text, f"{task}/{name}")
                fused = {
                    k: v["saved"]["smoking"]["fused"][name]["threshold"]
                    for k, v in points.items()
                }
                entry["channels"].append(
                    {
                        "name": "smoking detector"
                        if members[0].get("kind") == "detector"
                        else "scene and person views",
                        "run": name,
                        "members": members,
                        "threshold": fused[default],
                        "thresholds": fused,
                    }
                )
        else:
            entry["members"] = members_of(task, primary[task], arrays, text, task)
        bundle["tasks"][task] = entry
        print(
            task, {k: round(v, 3) for k, v in entry["thresholds"].items()}, flush=True
        )
    bundle["held_out"] = points[default]["summary"] | {
        "by_operating_point": {k: v["summary"] for k, v in points.items()}
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "heads.npz", **arrays)
    bundle["heads_sha256"] = hashlib.sha256(
        (args.output / "heads.npz").read_bytes()
    ).hexdigest()
    (args.output / "ensemble.json").write_text(json.dumps(bundle, indent=1) + "\n")
    shutil.copyfile(
        OUT / "smoking_detector/runs" / args.detector / "weights/last.pt",
        args.output / "smoking_detector.pt",
    )
    saved = points[default]["saved"]
    reference = {
        task: [
            {
                "id": i,
                "views": by_id[i]["views"],
                "reference": by_id[i].get("reference"),
                "group_id": by_id[i]["group_id"],
                "score": float(score),
            }
            for i, score in zip(
                saved[task]["ids"]["test"], saved[task]["scores"]["test"]
            )
        ]
        for task in TASKS
    }
    (args.output / "parity_reference.json").write_text(
        json.dumps({"root": str(ROOT), "tasks": reference})
    )
    for name, value in points.items():
        s = value["summary"]
        print(
            f"{name}: held-out macro F1 {s['macro_f1']:.3f}  macro AP {s['macro_ap']:.3f}  "
            f"cross-task flagged {s['cross_task_flagged']:.3f}  new-site recall "
            f"{s['new_site_recall']:.3f}  ordinary footage flagged "
            f"{s['new_site_ordinary_flagged']:.3f}"
        )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
