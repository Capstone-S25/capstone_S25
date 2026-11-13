"""Paired bootstrap of macro F1 (and per-source F1) between two saved systems.

Resamples groups (recordings, authors, videos) within each held-out source; a
source with fewer than five groups is resampled by row, which understates its
uncertainty (the single-camera CCTV and two-camera unattended sets).
"""

import json
import sys
from collections import defaultdict

import numpy as np
from common import OUT, TEST_SOURCES, load_rows


def load(spec):
    tag, protocol = spec.split(":")
    if tag == "baseline":
        return None
    data = json.loads((OUT / "runs" / f"{tag}_{protocol}_scores.json").read_text())
    return dict(zip(data["ids"], data["scores"])), data["thresholds"]


def baseline_scores():
    import baseline as b
    from baseline import STUDY  # noqa: F401
    from common import read_jsonl

    folder = b.STUDY / "mobileclip2-s2" / "transfer_evaluation"
    policies = json.loads((folder / "frozen_policies.json").read_text())
    thresholds = {
        t: policies[t]["selected"]["thresholds"]["dev_max_f1"] for t in policies
    }
    scores = {}
    for name in ["historical", "hmdb", "meva", "photo", "rough"]:
        for p in read_jsonl(folder / f"{name}_predictions.jsonl"):
            if p.get("family") == "selected":
                scores[p["id"]] = p["score"]
    return scores, thresholds


def f1(y, pred):
    tp = (pred & (y == 1)).sum()
    return 2 * tp / max(1, 2 * tp + (pred & (y == 0)).sum() + (~pred & (y == 1)).sum())


def main(a, b, n=2000, seed=0):
    rows = {r["id"]: r for r in load_rows()}
    systems = [
        baseline_scores() if s.startswith("baseline") else load(s) for s in (a, b)
    ]
    common = set(systems[0][0]) & set(systems[1][0])
    sources = defaultdict(lambda: defaultdict(list))
    for i in common:
        r = rows[i]
        key = (r["task"], r["source"])
        if key in TEST_SOURCES:
            sources[key][r["group_id"]].append(i)
    rng = np.random.default_rng(seed)
    per_task = defaultdict(list)
    for key in sources:
        per_task[key[0]].append(key)
    draws = np.zeros((n, 2))
    source_draws = defaultdict(list)
    for k in range(n):
        task_values = [[], []]
        for task, keys in per_task.items():
            vals = [[], []]
            for key in keys:
                groups = list(sources[key].values())
                if len(groups) >= 5:
                    pick = [
                        i
                        for g in rng.integers(0, len(groups), len(groups))
                        for i in groups[g]
                    ]
                else:
                    flat = [i for g in groups for i in g]
                    pick = [flat[j] for j in rng.integers(0, len(flat), len(flat))]
                y = np.array([rows[i]["target"] for i in pick])
                for s, (scores, thr) in enumerate(systems):
                    v = f1(y, np.array([scores[i] for i in pick]) >= thr[task])
                    vals[s].append(v)
                source_draws[key].append(vals[1][-1] - vals[0][-1])
            for s in range(2):
                task_values[s].append(np.mean(vals[s]))
        draws[k] = [np.mean(task_values[0]), np.mean(task_values[1])]
    diff = draws[:, 1] - draws[:, 0]
    lo, hi = np.quantile(diff, [0.025, 0.975])
    print(
        f"macro F1 {a}: {draws[:, 0].mean():.3f}  {b}: {draws[:, 1].mean():.3f}  "
        f"diff {diff.mean():+.3f} [95% {lo:+.3f}, {hi:+.3f}]  P(diff>0)={np.mean(diff > 0):.3f}"
    )
    for key, name in TEST_SOURCES.items():
        if key in source_draws:
            d = np.array(source_draws[key])
            lo, hi = np.quantile(d, [0.025, 0.975])
            print(f"  {name:<32} diff {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
