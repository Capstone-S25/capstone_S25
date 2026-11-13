"""Export the round-2 selection (select_round2.py) as a runtime bundle.

Same bundle layout as export.py. Per task it refits the selected variant
(optionally with other tasks' windows as negatives), checks the scores against
the study run, applies the cross-task false-alarm cap to the threshold, and
adds a Platt calibration of the ensemble score for display.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import heads
import numpy as np
import round2
from common import OUT, ROOT, TASKS, evaluate, load_rows, macro
from export import BACKBONES
from sklearn.linear_model import LogisticRegression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, help="select_round2.py tag")
    parser.add_argument(
        "--shares", nargs="+", required=True, help="run=aux share, e.g. r2c_aux40=0.4"
    )
    parser.add_argument(
        "--members", nargs="+", default=["v8_siglip2-b", "v8_mobileclip2-b"]
    )
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--cap", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shares = {k: float(v) for k, v in (item.split("=") for item in args.shares)}
    selection = json.loads((OUT / "runs" / f"{args.selection}.json").read_text())
    rows = load_rows()
    backbones = sorted(m.split("_", 1)[1] for m in args.members)
    recipes = round2.named_recipes(backbones)
    bundle = {
        "tasks": {},
        "backbones": {b: BACKBONES[b] for b in backbones},
        "top": args.top,
        "members": args.members,
        "selection": args.selection,
        "cross_task_cap": args.cap,
    }
    arrays, held, held_scores, thresholds, parity = {}, [], [], {}, {}
    for task in TASKS:
        run = selection["choices"][task]["chosen"]
        with open(OUT / "runs" / f"{run}_round2.pkl", "rb") as f:
            study = pickle.load(f)[task]
        chosen = {
            m: [(name, recipes[name]) for name in round2.released(m, task, args.top)]
            for m in args.members
        }
        out = round2.fit_task(rows, task, chosen, shares[run])
        for key in ["oof", "test_scores"]:
            if np.corrcoef(out[key], study[key])[0, 1] < 0.999:
                raise ValueError(f"Refit differs from the study: {task} {key}")
        threshold, oof_f1 = heads.macro_source_threshold(out["own"], out["oof"])
        capped = max(threshold, float(np.quantile(out["aux_oof"], 1 - args.cap)))
        # Probability for display: logistic fit of the label on the out-of-fold
        # ensemble score, with the same class/source/group balancing as the heads.
        y = np.array([r["target"] for r in out["own"]])
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(
            out["oof"][:, None], y, sample_weight=heads.weights(out["own"])
        )
        members = []
        for member in out["detail"]:
            exported = []
            for r in member["recipes"]:
                key = f"{task}/{member['tag']}/{r['name']}"
                arrays[key + "/weight"] = np.stack([h[0] for h in r["heads"]]).astype(
                    np.float32
                )
                arrays[key + "/bias"] = np.array([h[1] for h in r["heads"]], np.float32)
                exported.append(
                    {k: r[k] for k in ["recipe", "name", "C", "mean", "std", "oof_ap"]}
                    | {"arrays": key}
                )
            members.append(
                {
                    "tag": member["tag"],
                    "recipes": exported,
                    "mean": member["mean"],
                    "std": member["std"],
                }
            )
        bundle["tasks"][task] = {
            "members": members,
            "threshold": capped,
            "uncapped_threshold": threshold,
            "oof_macro_source_f1": oof_f1,
            "study_run": run,
            "other_task_negative_share": shares[run],
            "cross_task_oof_fpr": float((out["aux_oof"] >= capped).mean()),
            "calibration": {
                "slope": float(platt.coef_[0, 0]),
                "intercept": float(platt.intercept_[0]),
            },
        }
        thresholds[task] = capped
        held += out["test"]
        held_scores += out["test_scores"].tolist()
        parity[task] = {
            r["id"]: float(s) for r, s in zip(out["test"], out["test_scores"])
        }
        print(
            task,
            run,
            f"threshold {capped:.3f} (uncapped {threshold:.3f})",
            f"study {selection['choices'][task]['threshold']:.3f}",
            flush=True,
        )
    results = evaluate(held, held_scores, thresholds)
    bundle["held_out"] = {
        "macro_f1": macro(results),
        "macro_ap": macro(results, "ap"),
        "sources": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "heads.npz", **arrays)
    bundle["heads_sha256"] = hashlib.sha256(
        (args.output / "heads.npz").read_bytes()
    ).hexdigest()
    (args.output / "ensemble.json").write_text(json.dumps(bundle, indent=1) + "\n")
    rows_by_id = {r["id"]: r for r in rows}
    reference = {
        task: [
            {
                "id": i,
                "views": rows_by_id[i]["views"],
                "reference": rows_by_id[i].get("reference"),
                "group_id": rows_by_id[i]["group_id"],
                "score": s,
            }
            for i, s in scores.items()
        ]
        for task, scores in parity.items()
    }
    (args.output / "parity_reference.json").write_text(
        json.dumps({"root": str(ROOT), "tasks": reference})
    )
    print(
        f"held-out macro F1 {macro(results):.3f}  macro AP {macro(results, 'ap'):.3f}"
    )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
