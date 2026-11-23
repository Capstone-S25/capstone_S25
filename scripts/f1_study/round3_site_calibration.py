"""Would calibrating on a camera's own ordinary footage help?

For every held-out camera (benchmark test rows and new-site shift rows) that
has ordinary windows, half of them (by time order within the camera) play the
role of "footage recorded before going live": the task's threshold is raised,
for that camera only, until at most `--share` of them are flagged. The other
half and all of the camera's positives are then scored with the released
threshold and with the calibrated one.

    round3_site_calibration.py "r3_final|r3_detg@n05d0"
"""

from __future__ import annotations

import os

os.environ["F1_ROUND3"] = "1"

import argparse
import pickle
from collections import defaultdict

import numpy as np
import round3_report as report
from common import OUT, TASKS, f1_at, load_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec")
    parser.add_argument("--share", type=float, default=0.02)
    parser.add_argument("--minimum", type=int, default=20)
    args = parser.parse_args()
    by_id = {r["id"]: r for r in load_rows()}
    run, normal_cap, cross_cap, channels = report.parse(args.spec)
    with open(OUT / "runs" / f"{run}_round3.pkl", "rb") as f:
        saved = pickle.load(f)
    if channels:
        share = len(channels) + 1
        saved = report.fuse(
            saved, [run, *channels], by_id, normal_cap / share, cross_cap / share
        )
    thresholds, _ = report.thresholds_for(saved, by_id, normal_cap, cross_cap)
    if channels:
        thresholds["smoking"] = 0.0
    print(
        "| Task: camera | Ordinary windows (calibration / check) | Positives | "
        "Flagged before | Flagged after | Recall before | Recall after |"
    )
    print("|---|---|---|---|---|---|---|")
    totals = defaultdict(list)
    for task in TASKS:
        s, ids = saved[task]["scores"], saved[task]["ids"]
        cameras = defaultdict(list)
        for part in ["test", "shift"]:
            for i, score in zip(ids[part], s[part]):
                row = by_id[i]
                if row.get("label_uncertain"):
                    continue
                cameras[row["source"], row["group_id"]].append((row, float(score)))
        merged = defaultdict(list)
        for (source, group), items in cameras.items():
            # Photo collections have no camera; clips of a staged set neither.
            fixed = source in {
                "smoking_cctv",
                "unattended_reviewed",
                "street_scene",
                "meva",
                "virat",
                "avenue",
                "xrf55",
            }
            if fixed:
                merged[source, group] = items
        for (source, group), items in sorted(merged.items()):
            ordinary = [(r, v) for r, v in items if r["target"] == 0]
            positive = np.array([v for r, v in items if r["target"] == 1])
            if len(ordinary) < 2 * args.minimum:
                continue
            ordinary.sort(key=lambda x: x[0]["views"][-1])
            half = len(ordinary) // 2
            calibration = np.array([v for _, v in ordinary[:half]])
            check = np.array([v for _, v in ordinary[half:]])
            base = thresholds[task]
            site = max(base, float(np.quantile(calibration, 1 - args.share)) + 1e-9)
            before, after = (check >= base).mean(), (check >= site).mean()
            recall = (
                ((positive >= base).mean(), (positive >= site).mean())
                if len(positive)
                else (None, None)
            )
            cell = lambda v: "—" if v is None else f"{v:.0%}"
            print(
                f"| {task}: {group} | {half} / {len(check)} | {len(positive)} | "
                f"{before:.0%} | {after:.0%} | {cell(recall[0])} | {cell(recall[1])} |"
            )
            totals["before"].append(before)
            totals["after"].append(after)
            if len(positive):
                y = np.r_[np.ones(len(positive)), np.zeros(len(check))]
                both = np.r_[positive, check]
                totals["f1_before"].append(f1_at(y, both, base)["f1"])
                totals["f1_after"].append(f1_at(y, both, site)["f1"])
                totals["recall_before"].append(recall[0])
                totals["recall_after"].append(recall[1])
    print()
    print(
        f"{len(totals['before'])} task-camera pairs: ordinary windows flagged "
        f"{np.mean(totals['before']):.1%} -> {np.mean(totals['after']):.1%}; "
        f"{len(totals['f1_before'])} pairs with positives: recall "
        f"{np.mean(totals['recall_before']):.1%} -> {np.mean(totals['recall_after']):.1%}, "
        f"F1 {np.mean(totals['f1_before']):.3f} -> {np.mean(totals['f1_after']):.3f}"
    )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
