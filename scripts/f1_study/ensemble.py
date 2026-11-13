"""Average the standardized scores of several runs' selected heads per task.

Each member is standardized with the mean/std of its own out-of-fold scores,
so the ensemble threshold is again chosen on out-of-fold data only.
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
from common import OUT, TASKS, TEST_SOURCES, evaluate, load_rows, save_json, table
from heads import macro_source_threshold, source_ap


def zscore(oof, test):
    mu, sd = oof.mean(), oof.std() + 1e-9
    return (oof - mu) / sd, (test - mu) / sd


def member(tag, task, top=1):
    """Average of the top `top` recipes (best C each) by out-of-fold AP."""
    with open(OUT / "runs" / f"{tag}.pkl", "rb") as f:
        saved = pickle.load(f)
    if ("november", task) not in saved:
        return None
    out = saved["november", task]
    if top == 1:
        z = zscore(out["oof_scores"], out["test_scores"])
        return z[0], z[1], out["recipe"]
    best = {}
    for c in out["candidates"]:
        if c["recipe"] not in best or c["oof_ap"] > best[c["recipe"]]["oof_ap"]:
            best[c["recipe"]] = c
    chosen = sorted(best.values(), key=lambda c: -c["oof_ap"])[:top]
    zs = [zscore(c["oof"], c["test"]) for c in chosen]
    oof = np.mean([z[0] for z in zs], 0)
    test = np.mean([z[1] for z in zs], 0)
    z = zscore(oof, test)
    return z[0], z[1], "+".join(c["recipe"] for c in chosen)


def combine(tags, rows, name, top=1):
    held_rows, held_scores, thresholds, info = [], [], {}, {}
    for task in TASKS:
        pool = [r for r in rows if r["task"] == task and r["split"] in {"train", "dev"}]
        test = [
            r
            for r in rows
            if r["task"] == task
            and r["split"] in {"test", "external"}
            and (task, r["source"]) in TEST_SOURCES
        ]
        members = [m for m in (member(t, task, top) for t in tags) if m is not None]
        oof = np.mean([m[0] for m in members], 0)
        scores = np.mean([m[1] for m in members], 0)
        thresholds[task], oof_f1 = macro_source_threshold(pool, oof)
        info[task] = {
            "members": [m[2] for m in members],
            "oof_ap": source_ap(pool, oof)[0],
            "oof_macro_f1": oof_f1,
        }
        held_rows += test
        held_scores += scores.tolist()
    return (
        evaluate(held_rows, held_scores, thresholds),
        info,
        (held_rows, held_scores, thresholds),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--top", type=int, default=1)
    args = parser.parse_args()
    rows = load_rows()
    results, info, (held, scores, thresholds) = combine(
        args.members, rows, args.tag, args.top
    )
    label = f"{args.tag}:november"
    save_json(
        OUT / "runs" / f"{args.tag}.json",
        {"results": {label: results}, "choices": info},
    )
    save_json(
        OUT / "runs" / f"{args.tag}_november_scores.json",
        {"ids": [r["id"] for r in held], "scores": scores, "thresholds": thresholds},
    )
    for task, value in info.items():
        print(task, f"oof AP {value['oof_ap']:.3f}", value["members"])
    print(table({label: results}, fields=("ap", "f1", "fpr")))


if __name__ == "__main__":
    main()
