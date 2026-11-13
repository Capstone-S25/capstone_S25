"""Screening classifier, Gemma, and their combinations on the same windows.

Uses the windows scored by gemma_video.py (the pipeline's selected frames) and
the screening scores of a round-2 run. HMDB and held-out-camera negatives were
subsampled for the Gemma run, so counts are re-weighted to the full held-out
sets before computing F1.

Rules (none uses test labels):
  screening   score >= its out-of-fold threshold
  gemma       Gemma answers Yes (margin > 0)
  agree       both of the above (the application's alert rule)
  either      at least one
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict

import numpy as np
from common import OUT, TEST_SOURCES, load_rows, save_json
from sklearn.metrics import average_precision_score


def weighted(y, pred, w):
    tp = float(w[(pred) & (y == 1)].sum())
    fp = float(w[(pred) & (y == 0)].sum())
    fn = float(w[(~pred) & (y == 1)].sum())
    tn = float(w[(~pred) & (y == 0)].sum())
    return {
        "f1": 2 * tp / max(1e-9, 2 * tp + fp + fn),
        "precision": tp / max(1e-9, tp + fp),
        "recall": tp / max(1e-9, tp + fn),
        "fpr": fp / max(1e-9, fp + tn),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="r2c_base")
    parser.add_argument("--gemma", default="e4b")
    parser.add_argument("--condition", default="selected@280")
    args = parser.parse_args()
    rows = load_rows()
    full = Counter(
        (r["source"], r["target"])
        for r in rows
        if r["split"] in {"test", "external"}
        and (r["task"], r["source"]) in TEST_SOURCES
    )
    manifest = [
        json.loads(line)
        for line in (OUT / "gemma_video/manifest.jsonl").read_text().splitlines()
    ]
    sampled = Counter((r["source"], r["target"]) for r in manifest)
    gemma = json.loads((OUT / f"gemma_video/scores_{args.gemma}.json").read_text())[
        args.condition
    ]["scores"]
    with open(OUT / "runs" / f"{args.run}_round2.pkl", "rb") as f:
        saved = pickle.load(f)
    info = json.loads((OUT / "runs" / f"{args.run}.json").read_text())["choices"]
    screening, thresholds = {}, {}
    for task, out in saved.items():
        thresholds[task] = info[task]["threshold"]
        screening.update(zip(out["test_ids"], out["test_scores"]))
    by_source = defaultdict(list)
    for r in manifest:
        by_source[r["source"]].append(r)
    report = {}
    for source, part in by_source.items():
        task = part[0]["task"]
        y = np.array([r["target"] for r in part])
        w = np.array(
            [full[source, r["target"]] / sampled[source, r["target"]] for r in part]
        )
        s = np.array([screening[r["id"]] for r in part]) - thresholds[task]
        g = np.array([gemma[r["id"]] for r in part])
        rules = {
            "screening": s >= 0,
            "gemma": g > 0,
            "agree": (s >= 0) & (g > 0),
            "either": (s >= 0) | (g > 0),
        }
        report[TEST_SOURCES[task, source]] = {
            "task": task,
            "n": len(part),
            "screening_ap": float(average_precision_score(y, s, sample_weight=w)),
            "gemma_ap": float(average_precision_score(y, g, sample_weight=w)),
            # Rank fusion needs no shared scale: average the two ranks.
            "fused_ap": float(
                average_precision_score(
                    y,
                    np.argsort(np.argsort(s)) + np.argsort(np.argsort(g)),
                    sample_weight=w,
                )
            ),
            **{name: weighted(y, pred, w) for name, pred in rules.items()},
        }
    save_json(OUT / "gemma_video" / f"cascade_{args.run}_{args.gemma}.json", report)
    print(
        "| Source | Screening AP | Gemma AP | Fused AP | "
        + " | ".join(f"{r} F1 (P/R)" for r in ["screening", "gemma", "agree", "either"])
        + " |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name, v in report.items():
        print(
            f"| {name} | {v['screening_ap']:.3f} | {v['gemma_ap']:.3f} | {v['fused_ap']:.3f} | "
            + " | ".join(
                f"{v[r]['f1']:.3f} ({v[r]['precision']:.2f}/{v[r]['recall']:.2f})"
                for r in ["screening", "gemma", "agree", "either"]
            )
            + " |"
        )
    by_task = defaultdict(lambda: defaultdict(list))
    for v in report.values():
        for r in ["screening", "gemma", "agree", "either"]:
            by_task[r][v["task"]].append(v[r]["f1"])
    print(
        "macro F1 over these three tasks:",
        {
            r: round(float(np.mean([np.mean(x) for x in t.values()])), 3)
            for r, t in by_task.items()
        },
    )


if __name__ == "__main__":
    main()
