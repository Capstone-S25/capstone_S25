"""Thresholds and tables for round-3 runs.

    round3_report.py r3_release r3_release@n02 r3_site@n02

`run@nXX` adds a normal-footage cap: the threshold is raised until at most
XX% of the task's fit-side ordinary windows are flagged (out-of-fold scores
when those windows were fitted, final-head scores when they were not).
`run@cYY` changes the cross-task cap (default 5%).
`runA|runB@n02` flags smoking when either run's smoking score passes that
run's own threshold (channels), with the caps shared between them.
`...@n05d0` sets the detector channel's ordinary cap separately (0 = no
fit-side ordinary window flagged).

Every threshold uses fitting-side scores only. The benchmark's held-out
sources and the new-site "shift" rows are scored afterwards.
"""

from __future__ import annotations

import os

os.environ["F1_ROUND3"] = "1"

import argparse
import pickle
import re

import numpy as np
from common import OUT, TASKS, evaluate, f1_at, load_rows, macro, save_json, table
from heads import macro_source_threshold
from sklearn.metrics import average_precision_score

NORMAL_SOURCES = {"virat", "avenue", "meva", "xrf55"}
DETECTOR_CAP = None


def parse(spec):
    run, _, options = spec.partition("@")
    normal = re.search(r"n(\d+)", options)
    cross = re.search(r"c(\d+)", options)
    strict = re.search(r"d(\d+)", options)
    global DETECTOR_CAP
    DETECTOR_CAP = float(strict.group(1)) / 100 if strict else None
    run, *channels = run.split("|")
    return (
        run,
        float(normal.group(1)) / 100 if normal else None,
        float(cross.group(1)) / 100 if cross else 0.05,
        channels,
    )


def competent(core, scores, floor=0.75):
    """Rows of the sources a channel can separate (AUC >= floor), plus the
    single-class pool. A channel with no skill on a source would otherwise
    set its threshold by that source's base rate."""
    from sklearn.metrics import roc_auc_score

    keep = []
    for source in sorted({r["source"] for r in core}):
        idx = [i for i, r in enumerate(core) if r["source"] == source]
        y = np.array([core[i]["target"] for i in idx])
        if (
            source == "single-class sources"
            or len(set(y)) < 2
            or roc_auc_score(y, scores[idx]) >= floor
        ):
            keep += idx
    keep = sorted(keep) or list(range(len(core)))
    return [core[i] for i in keep], scores[keep]


def thresholds_for(
    saved, by_id, normal_cap, cross_cap, tasks=TASKS, channel=False, fpr_cap=None
):
    """Per task: the out-of-fold macro-source max-F1 threshold, raised to the
    cross-task and ordinary-footage caps. With `fpr_cap` (the high-recall
    point) the starting value is instead allowed down to the score that flags
    that share of the task's own fitting negatives, so the caps decide."""
    out, info = {}, {}
    # Round 4: sources fitted but left out of threshold selection
    # (F1_THRESHOLD_WITHOUT="icvl_smoking icvl_littering ...").
    skip = set(os.environ.get("F1_THRESHOLD_WITHOUT", "").split())
    for task in tasks:
        s, ids = saved[task]["scores"], saved[task]["ids"]
        core = [by_id[i] for i in ids["own"] + ids["used"]]
        scores = np.concatenate([s["own"], s["used"]])
        if skip:
            keep = [k for k, r in enumerate(core) if r["source"] not in skip]
            core, scores = [core[k] for k in keep], scores[keep]
        # Sources with one class only (a site of smokers, ordinary footage)
        # have no F1 of their own; together they form one more source.
        both = {
            source
            for source in {r["source"] for r in core}
            if len({r["target"] for r in core if r["source"] == source}) == 2
        }
        core = [
            r if r["source"] in both else {**r, "source": "single-class sources"}
            for r in core
        ]
        if channel:
            core, scores = competent(core, scores)
        base, _ = macro_source_threshold(core, scores)
        if fpr_cap is not None:
            negatives = [v for r, v in zip(core, scores) if r["target"] == 0]
            if negatives:
                base = min(base, float(np.quantile(negatives, 1 - fpr_cap)))
        value = base
        if len(s["aux"]) and cross_cap < 1:
            value = max(value, float(np.quantile(s["aux"], 1 - cross_cap)))
        after_cross = value
        normals = [
            score
            for key in ["used", "unused"]
            for i, score in zip(ids[key], s[key])
            if by_id[i]["source"] in NORMAL_SOURCES and by_id[i]["target"] == 0
        ]
        if normal_cap is not None and normals:
            # A cap of 0 means the highest ordinary score: nothing flagged.
            value = max(value, float(np.quantile(normals, 1 - normal_cap)) + 1e-9)
        out[task] = value
        info[task] = {
            "max_f1": base,
            "after_cross_task_cap": after_cross,
            "threshold": value,
            "fit_normal_windows": len(normals),
            "fit_normal_flagged": float(np.mean(np.array(normals) >= value))
            if normals
            else None,
        }
    return out, info


def fuse(saved, channels, by_id, normal_cap, cross_cap, task="smoking", fpr_cap=None):
    """OR of channels: each keeps its own threshold; the fused score is the
    largest margin over a channel's threshold, flagged at 0."""
    margins, detail = [], {}
    for position, name in enumerate(channels):
        with open(OUT / "runs" / f"{name}_round3.pkl", "rb") as f:
            other = pickle.load(f)
        cap = normal_cap
        if position and DETECTOR_CAP is not None:
            # The detector's false alarms are camera-specific (one fixture
            # that looks like a head scores on every frame), and only a few
            # cameras calibrate it. `d0` therefore asks for no flag at all on
            # the fit-side ordinary footage; `d1` for at most 1%.
            cap = DETECTOR_CAP
        t, info = thresholds_for(
            other, by_id, cap, cross_cap, tasks=[task], channel=True, fpr_cap=fpr_cap
        )
        detail[name] = info[task]
        # Runs may split the fit-side rows differently (fitted or only
        # scored), so match rows by id.
        margin = {
            i: v - t[task]
            for part, ids in other[task]["ids"].items()
            for i, v in zip(ids, other[task]["scores"][part])
        }
        margins.append(
            {
                part: np.array([margin[i] for i in ids])
                for part, ids in saved[task]["ids"].items()
            }
        )
    saved[task] = {
        **saved[task],
        "scores": {k: np.max([m[k] for m in margins], 0) for k in margins[0]},
        "fused": detail,
    }
    return saved


def fitting_f1(saved, by_id, thresholds):
    """Selection view: mean F1 over fitting-side sources, per task, with
    single-class sources pooled into one. Every score is out of sample
    (out-of-fold for fitted rows, final heads for fit-side rows a run did not
    fit), and every run is judged on the same rows. No held-out row."""
    out = {}
    for task in TASKS:
        s, ids = saved[task]["scores"], saved[task]["ids"]
        core = [by_id[i] for i in ids["own"] + ids["used"] + ids["unused"]]
        core = [r for r in core if not r.get("label_uncertain")]
        keep = {r["id"] for r in core}
        scores = np.array(
            [
                v
                for part in ["own", "used", "unused"]
                for i, v in zip(ids[part], s[part])
                if i in keep
            ]
        )
        both = {
            source
            for source in {r["source"] for r in core}
            if len({r["target"] for r in core if r["source"] == source}) == 2
        }
        names = np.array(
            [r["source"] if r["source"] in both else "single-class" for r in core]
        )
        y = np.array([r["target"] for r in core])
        values = []
        for name in sorted(set(names)):
            idx = names == name
            if len(set(y[idx])) == 2:
                values.append(f1_at(y[idx], scores[idx], thresholds[task])["f1"])
        out[task] = float(np.mean(values))
    return out


def site_metrics(saved, by_id, thresholds):
    """Per (task, source) on shift rows: recall, false-alarm rate, AP."""
    out = {}
    for task in TASKS:
        rows = [by_id[i] for i in saved[task]["ids"]["shift"]]
        scores = saved[task]["scores"]["shift"]
        for source in sorted({r["source"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["source"] == source]
            y = np.array([rows[i]["target"] for i in idx])
            s = scores[idx]
            m = f1_at(y, s, thresholds[task])
            out[task, source] = {
                "n": len(y),
                "positive": int(y.sum()),
                "recall": m["recall"] if y.sum() else None,
                "fpr": m["fpr"] if (y == 0).sum() else None,
                "f1": m["f1"] if 0 < y.sum() < len(y) else None,
                "ap": float(average_precision_score(y, s))
                if 0 < y.sum() < len(y)
                else None,
                "uncertain": bool(rows[idx[0]].get("label_uncertain")),
                "groups": [rows[i]["group_id"] for i in idx],
                "flag": (s >= thresholds[task]).tolist(),
                "y": y.tolist(),
            }
    return out


def summary(site):
    """Mean recall over positive sources and mean false-alarm rate over
    ordinary-footage sources with certain labels (task-source cells)."""
    recall = [v["recall"] for v in site.values() if v["recall"] is not None]
    fpr = [
        v["fpr"]
        for (task, source), v in site.items()
        if v["fpr"] is not None and source in NORMAL_SOURCES and not v["uncertain"]
    ]
    return float(np.mean(recall)), float(np.mean(fpr))


def cell(value):
    return "—" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--save", help="Write the first spec's thresholds/scores")
    args = parser.parse_args()
    by_id = {r["id"]: r for r in load_rows()}
    bench, sites, lines, selection = {}, {}, [], []
    for spec in args.specs:
        run, normal_cap, cross_cap, channels = parse(spec)
        with open(OUT / "runs" / f"{run}_round3.pkl", "rb") as f:
            saved = pickle.load(f)
        if channels:
            # Each channel takes an equal share of the false-alarm caps.
            share = len(channels) + 1
            saved = fuse(
                saved,
                [run, *channels],
                by_id,
                normal_cap / share if normal_cap else None,
                cross_cap / share,
            )
        thresholds, info = thresholds_for(saved, by_id, normal_cap, cross_cap)
        if channels:
            thresholds["smoking"] = 0.0
            info["smoking"] = {"threshold": 0.0, "channels": saved["smoking"]["fused"]}
        held_rows, held_scores, cross = [], [], []
        for task in TASKS:
            held_rows += [by_id[i] for i in saved[task]["ids"]["test"]]
            held_scores += saved[task]["scores"]["test"].tolist()
            cross.append(
                float((saved[task]["scores"]["aux_test"] >= thresholds[task]).mean())
            )
        bench[spec] = evaluate(held_rows, held_scores, thresholds)
        sites[spec] = site_metrics(saved, by_id, thresholds)
        recall, fpr = summary(sites[spec])
        fitting = fitting_f1(saved, by_id, thresholds)
        selection.append(
            f"{spec}: fitting-side F1 "
            + ", ".join(f"{t[:5]} {fitting[t]:.3f}" for t in TASKS)
        )
        lines.append(
            f"{spec}: benchmark macro F1 {macro(bench[spec]):.3f}, macro AP "
            f"{macro(bench[spec], 'ap'):.3f}, cross-task flagged {np.mean(cross):.3f} | "
            f"new sites: recall {recall:.3f}, ordinary footage flagged {fpr:.3f} | thresholds "
            + ", ".join(f"{t[:5]} {thresholds[t]:.2f}" for t in TASKS)
        )
        save_json(
            OUT
            / "runs"
            / f"{spec.replace('@', '_').replace('|', '+')}_november_scores.json",
            {
                "ids": [r["id"] for r in held_rows],
                "scores": held_scores,
                "thresholds": thresholds,
            },
        )
        save_json(
            OUT / "runs" / f"{spec.replace('@', '_').replace('|', '+')}_site.json",
            {
                "thresholds": info,
                "site": {f"{t}|{s}": v for (t, s), v in sites[spec].items()},
                "benchmark": bench[spec],
            },
        )
    print(table(bench, fields=("ap", "f1", "fpr")))
    print()
    keys = list(next(iter(sites.values())))
    print(
        "| New-site evidence | n (+) | "
        + " | ".join(f"{s} recall / flagged / AP" for s in args.specs)
        + " |"
    )
    print("|" + "---|" * (2 + len(args.specs)))
    for task, source in keys:
        first = sites[args.specs[0]][task, source]
        note = " (unlabelled)" if first["uncertain"] else ""
        print(
            f"| {task}: {source}{note} | {first['n']} ({first['positive']}) | "
            + " | ".join(
                " / ".join(
                    cell(sites[s][task, source][k]) for k in ["recall", "fpr", "ap"]
                )
                for s in args.specs
            )
            + " |"
        )
    print()
    print("\n".join(lines))
    print()
    print("\n".join(selection))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
