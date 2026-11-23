"""Round 4 report: the released ensemble on the Drive's new labelled videos.

Reads results/f1_study_20261111/round4/windows.jsonl (round4_drive.py) and the
released thresholds, and prints per set: windows, AP, and at each operating
point precision, recall, F1, the share of negative windows flagged, false
alarms per hour of negative footage, and the share of labelled events caught
(any of its positive windows flagged). 95% intervals resample whole cameras.
Usage: round4_report.py [--bundle DIR] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/f1_study_20261111/round4"
BUNDLE = ROOT / "dist/capstone-cpu-deployment-20261114/models/ensemble-20261114"
SHIFT_CAMERAS = {"02", "05", "08", "11"}  # as in round4_sources.py


def thresholds(bundle):
    spec = json.loads((bundle / "ensemble.json").read_text())
    return {
        op: {t: float(v["thresholds"][op]) for t, v in spec["tasks"].items()}
        for op in ["balanced", "low_false_alarm"]
    }


def metrics(rows, threshold, step):
    y = np.array([r["target"] for r in rows])
    s = np.array([r["score"] for r in rows])
    flag = s >= threshold
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    negative_hours = (tp + fp + fn + tn - tp - fn) * step / 3600
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged_negatives": fp / (fp + tn) if fp + tn else None,
        "false_alarms_per_hour": fp / negative_hours if negative_hours else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def events_caught(rows, threshold, gap):
    """Contiguous positive windows of one video form an event."""
    by_video = collections.defaultdict(list)
    for r in rows:
        if r["target"] == 1:
            by_video[r["video"]].append(r)
    caught = total = 0
    for items in by_video.values():
        items.sort(key=lambda r: r["t"])
        run = [items[0]]
        for r in items[1:] + [None]:
            if r is not None and r["t"] - run[-1]["t"] <= gap:
                run.append(r)
                continue
            total += 1
            caught += any(x["score"] >= threshold for x in run)
            if r is not None:
                run = [r]
    return caught, total


def ap(rows):
    y = [r["target"] for r in rows]
    return (
        float(average_precision_score(y, [r["score"] for r in rows]))
        if 0 < sum(y) < len(y)
        else None
    )


def auc(rows):
    y = [r["target"] for r in rows]
    return (
        float(roc_auc_score(y, [r["score"] for r in rows]))
        if 0 < sum(y) < len(y)
        else None
    )


def bootstrap(rows, threshold, step, n=1000, seed=0):
    by_camera = collections.defaultdict(list)
    for r in rows:
        by_camera[r["camera"]].append(r)
    cameras = list(by_camera)
    rng = np.random.default_rng(seed)
    f1s, aps = [], []
    for _ in range(n):
        sample = [r for c in rng.choice(cameras, len(cameras)) for r in by_camera[c]]
        f1s.append(metrics(sample, threshold, step)["f1"])
        value = ap(sample)
        if value is not None:
            aps.append(value)
    ci = lambda v: (
        [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
    )
    return {"f1": ci(f1s), "ap": ci(aps), "cameras": len(cameras)}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--bundle", type=Path, default=BUNDLE)
    ap_.add_argument("--windows", type=Path, default=OUT / "windows.jsonl")
    ap_.add_argument("--step", type=float, default=2.0)
    ap_.add_argument("--json", type=Path, default=OUT / "report_released.json")
    a = ap_.parse_args()
    rows = [json.loads(line) for line in a.windows.open()]
    by_set = collections.defaultdict(list)
    for r in rows:
        by_set[r["set"]].append(r)
    th = thresholds(a.bundle)
    report = {}
    print(
        "| Set | Task | Videos | Windows (+) | Positive share | AUC | AP | Flag-all F1 | Op | Precision | Recall | F1 [95% CI] | Neg. flagged | FA/h | Events caught |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, items in by_set.items():
        task = items[0]["task"]
        entry = {
            "task": task,
            "videos": len({r["video"] for r in items}),
            "windows": len(items),
            "positives": sum(r["target"] for r in items),
            "ap": ap(items),
            "auc": auc(items),
        }
        prevalence = entry["positives"] / entry["windows"]
        entry["prevalence"], entry["flag_all_f1"] = (
            prevalence,
            2 * prevalence / (1 + prevalence),
        )
        for op in th:
            t = th[op][task]
            m = metrics(items, t, a.step)
            caught, total = events_caught(items, t, gap=a.step * 1.5)
            m["events_caught"], m["events"] = caught, total
            m["ci"] = bootstrap(items, t, a.step)
            entry[op] = m
            f = lambda v, p=3: "–" if v is None else f"{v:.{p}f}"
            ci = m["ci"]["f1"]
            print(
                f"| {name} | {task} | {entry['videos']} | {entry['windows']} ({entry['positives']}) | {prevalence:.2f} | {f(entry['auc'])} | {f(entry['ap'])} | {entry['flag_all_f1']:.3f} | {op} | "
                f"{f(m['precision'])} | {f(m['recall'])} | {f(m['f1'])} [{f(ci[0], 2)}–{f(ci[1], 2)}] | "
                f"{f(m['flagged_negatives'])} | {f(m['false_alarms_per_hour'], 0)} | {caught}/{total} |"
            )
        report[name] = entry
    # Threshold only: the released heads, with the threshold set on the
    # fitting-side ICVL cameras so that at most 5% of their negative windows
    # are flagged (the release's false-alarm cap), applied to the held-out ones.
    print(
        "\n| Set | Fit-side threshold (released) | Held-out AUC | Released: recall / neg. flagged / precision | Refitted: recall / neg. flagged / precision |"
    )
    print("|---|---|---|---|---|")
    for name, items in by_set.items():
        if not name.startswith("icvl_"):
            continue
        task = items[0]["task"]
        fit = [r for r in items if r["camera"] not in SHIFT_CAMERAS]
        shift = [r for r in items if r["camera"] in SHIFT_CAMERAS]
        negatives = sorted(r["score"] for r in fit if r["target"] == 0)
        if not shift or len(negatives) < 20:
            continue
        refit = float(np.quantile(negatives, 0.95)) + 1e-9
        before = metrics(shift, th["balanced"][task], a.step)
        after = metrics(shift, refit, a.step)
        report[name]["threshold_only"] = {
            "threshold": refit,
            "fit_negatives": len(negatives),
            "shift_auc": auc(shift),
            "shift_before": before,
            "shift_after": after,
        }
        g = lambda m: (
            f"{m['recall'] or 0:.2f} / {m['flagged_negatives'] or 0:.2f} / {m['precision'] or 0:.2f}"
        )
        print(
            f"| {name} | {refit:.2f} ({th['balanced'][task]:.2f}) | {auc(shift):.3f} | {g(before)} | {g(after)} |"
        )
    a.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
