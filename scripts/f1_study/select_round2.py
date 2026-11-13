"""Choose each task's round-2 variant on out-of-fold data only.

For every candidate run (round2.py outputs) and task:

* threshold = the out-of-fold macro-source max-F1 threshold, raised if needed
  so that at most `--cap` of the other tasks' out-of-fold windows are flagged;
* the task keeps the candidate whose out-of-fold macro-source F1 at that
  threshold is highest.

Held-out sources are only scored afterwards.
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
from common import OUT, TASKS, evaluate, f1_at, load_rows, macro, save_json, table
from heads import macro_source_threshold, source_ap


def source_f1(rows, scores, threshold):
    values = []
    for source in sorted({r["source"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source"] == source]
        y = np.array([rows[i]["target"] for i in idx])
        if len(set(y)) == 2:
            values.append(f1_at(y, np.asarray(scores)[idx], threshold)["f1"])
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--cap", type=float, default=0.05)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--fixed", help="Use this run for every task (no selection)")
    args = parser.parse_args()
    by_id = {r["id"]: r for r in load_rows()}
    saved = {}
    for run in args.runs:
        with open(OUT / "runs" / f"{run}_round2.pkl", "rb") as f:
            saved[run] = pickle.load(f)
    held_rows, held_scores, thresholds, info = [], [], {}, {}
    for task in TASKS:
        options = {}
        for run, value in saved.items():
            out = value[task]
            own = [by_id[i] for i in out["own_ids"]]
            threshold, _ = macro_source_threshold(own, out["oof"])
            capped = threshold
            if len(out["aux_oof"]):
                capped = max(
                    threshold, float(np.quantile(out["aux_oof"], 1 - args.cap))
                )
            options[run] = {
                "threshold": capped,
                "raised": capped > threshold,
                "oof_f1": source_f1(own, out["oof"], capped),
                "oof_ap": source_ap(own, out["oof"])[0],
                "cross_task_oof_fpr": float((out["aux_oof"] >= capped).mean()),
                "cross_task_test_fpr": float((out["aux_test_scores"] >= capped).mean()),
                "cross_task_test_fpr_by_task": {
                    name: float(
                        (
                            out["aux_test_scores"][
                                np.array(out["aux_test_task"]) == name
                            ]
                            >= capped
                        ).mean()
                    )
                    for name in sorted(set(out["aux_test_task"]))
                },
            }
        best = args.fixed or max(options, key=lambda run: options[run]["oof_f1"])
        out = saved[best][task]
        thresholds[task] = options[best]["threshold"]
        info[task] = {"chosen": best, **options[best], "options": options}
        held_rows += [by_id[i] for i in out["test_ids"]]
        held_scores += out["test_scores"].tolist()
        print(
            task,
            "->",
            best,
            {
                run: f"{o['oof_f1']:.3f}{'^' if o['raised'] else ''}"
                for run, o in options.items()
            },
            f"cross-task oof {options[best]['cross_task_oof_fpr']:.3f}",
            f"test {options[best]['cross_task_test_fpr']:.3f}",
            {
                k: round(v, 3)
                for k, v in options[best]["cross_task_test_fpr_by_task"].items()
            },
        )
    results = evaluate(held_rows, held_scores, thresholds)
    label = f"{args.tag}:november"
    cross = float(np.mean([info[t]["cross_task_test_fpr"] for t in TASKS]))
    save_json(
        OUT / "runs" / f"{args.tag}.json",
        {
            "results": {label: results},
            "choices": info,
            "macro_f1": macro(results),
            "macro_ap": macro(results, "ap"),
            "mean_cross_task_test_fpr": cross,
        },
    )
    save_json(
        OUT / "runs" / f"{args.tag}_november_scores.json",
        {
            "ids": [r["id"] for r in held_rows],
            "scores": held_scores,
            "thresholds": thresholds,
        },
    )
    print(table({label: results}, fields=("ap", "f1", "fpr")))
    print(f"mean cross-task test FPR {cross:.3f}")


if __name__ == "__main__":
    main()
