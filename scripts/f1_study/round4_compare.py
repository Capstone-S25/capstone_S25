"""Round 4: held-out new-site rows, before and after fitting the ICVL cameras.

    round4_compare.py "r4_base|r4_detg_base@n05d0" "r4_icvl|r4_detg_icvl@n05d0"

Thresholds and smoking's channel fusion are round3_report's (fitting side
only). Per shift source: positive share, AUC, AP, and at the threshold the
recall, share of negatives flagged and precision. AUC intervals resample
whole cameras (or videos, for single-camera sets).
"""

from __future__ import annotations

import os

os.environ["F1_ROUND3"] = "1"
os.environ["F1_ROUND4"] = "1"

import collections
import json
import pickle
import sys

import numpy as np
import round3_report as rr
from common import OUT, TASKS, load_rows
from sklearn.metrics import average_precision_score, roc_auc_score

NEW = {
    "icvl_smoking",
    "icvl_littering",
    "icvl_loitering",
    "iitb_loitering",
    "aboda",
    "d2cdnet",
    "chad",
}


def team_test(row):
    """Under F1_SPLIT=ian, keep only rows on the team split's test side."""
    if os.environ.get("F1_SPLIT") != "ian":
        return True
    source, video = row["source"], row.get("video", "")
    if source.startswith("icvl_"):
        return row["camera"] in {"02", "10"}
    if source == "aboda":
        return video in {"video2", "video3", "video5", "video7", "video8", "video10"}
    if source == "d2cdnet":
        return video in {"PETS2006", "abandonedBox", "tramstop"}
    if source == "chad":
        return video.split("_")[0] in {"2", "4"}
    return True


def scored(spec, by_id):
    run, normal_cap, cross_cap, channels = rr.parse(spec)
    with open(OUT / "runs" / f"{run}_round3.pkl", "rb") as f:
        saved = pickle.load(f)
    if channels:
        share = len(channels) + 1
        saved = rr.fuse(
            saved,
            [run, *channels],
            by_id,
            normal_cap / share if normal_cap else None,
            cross_cap / share,
        )
    thresholds, _ = rr.thresholds_for(saved, by_id, normal_cap, cross_cap)
    if channels:
        thresholds["smoking"] = 0.0
    return saved, thresholds


def auc_ci(y, s, groups, n=500, seed=0):
    rng = np.random.default_rng(seed)
    units = sorted(set(groups))
    index = collections.defaultdict(list)
    for i, g in enumerate(groups):
        index[g].append(i)
    values = []
    for _ in range(n):
        idx = [i for g in rng.choice(units, len(units)) for i in index[g]]
        if 0 < y[idx].sum() < len(idx):
            values.append(roc_auc_score(y[idx], s[idx]))
    return (
        [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        if values
        else None
    )


def main():
    specs = sys.argv[1:]
    by_id = {r["id"]: r for r in load_rows()}
    result = {}
    for spec in specs:
        saved, thresholds = scored(spec, by_id)
        for task in TASKS:
            ids, s = saved[task]["ids"]["shift"], saved[task]["scores"]["shift"]
            for source in sorted({by_id[i]["source"] for i in ids} & NEW):
                idx = [
                    k
                    for k, i in enumerate(ids)
                    if by_id[i]["source"] == source and team_test(by_id[i])
                ]
                if not idx:
                    continue
                rows = [by_id[ids[k]] for k in idx]
                y, sc = np.array([r["target"] for r in rows]), s[idx]
                flag = sc >= thresholds[task]
                tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
                two = 0 < y.sum() < len(y)
                units = [
                    r.get("camera") if r["source"].startswith("icvl") else r["video"]
                    for r in rows
                ]
                result.setdefault((task, source), {})[spec] = {
                    "n": len(y),
                    "positive": int(y.sum()),
                    "auc": float(roc_auc_score(y, sc)) if two else None,
                    "auc_ci": auc_ci(y, sc, units) if two else None,
                    "ap": float(average_precision_score(y, sc)) if two else None,
                    "recall": tp / y.sum() if y.sum() else None,
                    "flagged_negatives": fp / (y == 0).sum()
                    if (y == 0).sum()
                    else None,
                    "precision": tp / (tp + fp) if tp + fp else None,
                }
    f = lambda v, p=2: "–" if v is None else f"{v:.{p}f}"
    print(
        "| Held-out source | n (+) | "
        + " | ".join(
            f"{s}: AUC [95% CI] / recall / neg. flagged / precision" for s in specs
        )
        + " |"
    )
    print("|---|---|" + "---|" * len(specs))
    for (task, source), by_spec in result.items():
        first = next(iter(by_spec.values()))
        cells = []
        for spec in specs:
            m = by_spec[spec]
            ci = m["auc_ci"] or [None, None]
            cells.append(
                f"{f(m['auc'], 3)} [{f(ci[0])}–{f(ci[1])}] / {f(m['recall'])} / {f(m['flagged_negatives'])} / {f(m['precision'])}"
            )
        print(
            f"| {task}: {source} | {first['n']} ({first['positive']}) | "
            + " | ".join(cells)
            + " |"
        )
    name = "_vs_".join(s.split("|")[0] + "@" + s.split("@")[-1] for s in specs)
    (OUT / "round4").mkdir(exist_ok=True)
    (OUT / "round4" / f"compare_{name}.json").write_text(
        json.dumps({f"{t}|{s}": v for (t, s), v in result.items()}, indent=1)
    )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
