"""Export the selected ensemble as a runtime bundle, with parity references.

Per task and backbone, the top recipes by out-of-fold AP are refitted with the
same group folds (logistic regression is deterministic), checked against the
saved study scores, and written as fold heads plus standardization constants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import run_heads
from common import OUT, ROOT, TASKS, TEST_SOURCES, evaluate, load_rows, macro
from heads import cross_fit, folds, macro_source_threshold
from representations import matrix, recipe_name

BACKBONES = {
    "siglip2-b": {"kind": "siglip", "source": "google/siglip2-base-patch16-224"},
    "mobileclip2-b": {"kind": "open_clip", "source": "timm/MobileCLIP2-B-OpenCLIP"},
    "mobileclip2-l14": {
        "kind": "open_clip",
        "source": "timm/MobileCLIP2-L-14-OpenCLIP",
    },
    "s2": {"kind": "open_clip", "source": "timm/MobileCLIP2-S2-OpenCLIP"},
}


def recipe_lookup(backbones, smoke_detector):
    run_heads.BOXES[:] = ["frcnn", "yolo"]
    run_heads.SMOKE_DETECTOR = smoke_detector
    table = {}
    for b in backbones + ["none"]:
        for r in run_heads.recipes_for(b):
            table[recipe_name(r)] = r
    return table


def zscore_constants(values):
    return float(values.mean()), float(values.std() + 1e-9)


def main():
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1):
        run()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--members", nargs="+", required=True, help="run tags, e.g. v4_siglip2-b"
    )
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--smoke-detector", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows()
    backbones = sorted(
        {t.split("_", 1)[1] for t in args.members} - {"none", "smokedet"}
    )
    lookup = recipe_lookup(backbones, args.smoke_detector)
    bundle = {
        "tasks": {},
        "backbones": {b: BACKBONES[b] for b in backbones},
        "top": args.top,
        "members": args.members,
    }
    arrays, held, held_scores, thresholds, parity = {}, [], [], {}, {}
    for task in TASKS:
        pool = [r for r in rows if r["task"] == task and r["split"] in {"train", "dev"}]
        test = [
            r
            for r in rows
            if r["task"] == task
            and r["split"] in {"test", "external"}
            and (task, r["source"]) in TEST_SOURCES
        ]
        fold_sets = [folds(pool, seed=s) for s in range(3)]
        members = []
        member_oof, member_test = [], []
        for tag in args.members:
            with open(OUT / "runs" / f"{tag}.pkl", "rb") as f:
                saved = pickle.load(f)
            if ("november", task) not in saved:
                continue
            out = saved["november", task]
            best = {}
            for c in out["candidates"]:
                if c["recipe"] not in best or c["oof_ap"] > best[c["recipe"]]["oof_ap"]:
                    best[c["recipe"]] = c
            chosen = sorted(best.values(), key=lambda c: -c["oof_ap"])[: args.top]
            recipes, zs_oof, zs_test = [], [], []
            for c in chosen:
                recipe = lookup[c["recipe"]]
                oof, heads = cross_fit(matrix(pool, recipe), pool, c["C"], fold_sets)
                test_scores = np.mean(
                    [matrix(test, recipe) @ h[0] + h[1] for h in heads], 0
                )
                # Weakly regularized lbfgs fits stop at slightly different points
                # under multi-threaded BLAS; the exported heads are the reference.
                if (
                    np.corrcoef(oof, c["oof"])[0, 1] < 0.999
                    or np.corrcoef(test_scores, c["test"])[0, 1] < 0.999
                ):
                    raise ValueError(
                        f"Refit differs from the study: {task} {c['recipe']}"
                    )
                mu, sd = zscore_constants(oof)
                key = f"{task}/{tag}/{c['recipe']}"
                arrays[key + "/weight"] = np.stack([h[0] for h in heads]).astype(
                    np.float32
                )
                arrays[key + "/bias"] = np.array([h[1] for h in heads], np.float32)
                recipes.append(
                    {
                        "recipe": recipe,
                        "name": c["recipe"],
                        "C": c["C"],
                        "mean": mu,
                        "std": sd,
                        "oof_ap": c["oof_ap"],
                        "arrays": key,
                    }
                )
                zs_oof.append((oof - mu) / sd)
                zs_test.append((test_scores - mu) / sd)
            avg_oof, avg_test = np.mean(zs_oof, 0), np.mean(zs_test, 0)
            mu2, sd2 = zscore_constants(avg_oof)
            members.append({"tag": tag, "recipes": recipes, "mean": mu2, "std": sd2})
            member_oof.append((avg_oof - mu2) / sd2)
            member_test.append((avg_test - mu2) / sd2)
        oof = np.mean(member_oof, 0)
        scores = np.mean(member_test, 0)
        threshold, oof_f1 = macro_source_threshold(pool, oof)
        bundle["tasks"][task] = {
            "members": members,
            "threshold": threshold,
            "oof_macro_source_f1": oof_f1,
        }
        thresholds[task] = threshold
        held += test
        held_scores += scores.tolist()
        parity[task] = {r["id"]: float(s) for r, s in zip(test, scores)}
        print(
            task,
            "members",
            [len(m["recipes"]) for m in members],
            f"threshold {threshold:.3f}",
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
    # Rows with their view and reference paths, for runtime parity checks.
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
    main()
