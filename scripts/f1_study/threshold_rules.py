"""Choose a threshold rule using only out-of-fold training data, then score test.

For each rule and task, the threshold is set on all-but-one group fold of the
ensemble's out-of-fold scores and scored (macro-source F1) on the held-out
fold. The rule with the best average across tasks is applied to all data.
Test labels are only read for the final report.
"""

from __future__ import annotations

import argparse

import numpy as np
from common import TASKS, TEST_SOURCES, evaluate, load_rows, macro
from ensemble import member
from heads import folds, macro_source_threshold


def f1(y, s, t):
    p = s >= t
    tp = (p & (y == 1)).sum()
    return 2 * tp / max(1, 2 * tp + (p & (y == 0)).sum() + (~p & (y == 1)).sum())


def by_source(rows, scores):
    out = {}
    for r, s in zip(rows, scores):
        out.setdefault(r["source"], []).append((r["target"], s))
    return {k: tuple(map(np.asarray, zip(*v))) for k, v in out.items()}


def rule_max_f1(rows, s):
    return macro_source_threshold(rows, s)[0]


def rule_youden(rows, s):
    cand = np.unique(np.quantile(s, np.linspace(0, 1, 801)))
    total = np.zeros(len(cand))
    for y, v in by_source(rows, s).values():
        if len(set(y)) < 2:
            continue
        p = v[None] >= cand[:, None]
        total += (p & (y == 1)).sum(1) / (y == 1).sum() - (p & (y == 0)).sum(1) / (
            y == 0
        ).sum()
    return float(cand[np.argmax(total)])


def rule_balanced_f1(rows, s):
    """Max F1 after reweighting each source to 50% prevalence."""
    cand = np.unique(np.quantile(s, np.linspace(0, 1, 801)))
    total = np.zeros(len(cand))
    for y, v in by_source(rows, s).values():
        if len(set(y)) < 2:
            continue
        p = v[None] >= cand[:, None]
        tpr = (p & (y == 1)).sum(1) / (y == 1).sum()
        fpr = (p & (y == 0)).sum(1) / (y == 0).sum()
        total += 2 * tpr / np.maximum(1e-9, 2 * tpr + fpr + (1 - tpr))
    return float(cand[np.argmax(total)])


def rule_bagged(rows, s, n=50):
    rng = np.random.default_rng(0)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(rows), len(rows))
        sub = [rows[i] for i in idx]
        if len({r["target"] for r in sub}) < 2:
            continue
        values.append(macro_source_threshold(sub, s[idx])[0])
    return float(np.median(values))


RULES = {
    "max_f1": rule_max_f1,
    "youden": rule_youden,
    "balanced_f1": rule_balanced_f1,
    "bagged_max_f1": rule_bagged,
}


def inner(rows, oof, rule):
    scores = []
    for test in folds(rows, seed=7):
        train = np.setdiff1d(np.arange(len(rows)), test)
        t = rule([rows[i] for i in train], oof[train])
        vals = [
            f1(y, v, t)
            for y, v in by_source([rows[i] for i in test], oof[test]).values()
            if len(set(y)) == 2
        ]
        if vals:
            scores.append(np.mean(vals))
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    rows = load_rows()
    data = {}
    for task in TASKS:
        pool = [r for r in rows if r["task"] == task and r["split"] in {"train", "dev"}]
        test = [
            r
            for r in rows
            if r["task"] == task
            and r["split"] in {"test", "external"}
            and (task, r["source"]) in TEST_SOURCES
        ]
        ms = [
            m
            for m in (member(t, task, args.top) for t in args.members)
            if m is not None
        ]
        data[task] = (
            pool,
            test,
            np.mean([m[0] for m in ms], 0),
            np.mean([m[1] for m in ms], 0),
        )
    inner_scores = {
        name: {t: inner(p, o, rule) for t, (p, _, o, _) in data.items()}
        for name, rule in RULES.items()
    }
    for name, v in inner_scores.items():
        print(
            f"inner-fold F1 {name:<14} mean {np.mean(list(v.values())):.3f}  "
            + " ".join(f"{t[:6]}={x:.3f}" for t, x in v.items())
        )
    chosen = max(inner_scores, key=lambda n: np.mean(list(inner_scores[n].values())))
    print("chosen rule:", chosen)
    for name, rule in RULES.items():
        held, scores, thr = [], [], {}
        for task, (pool, test, oof, ts) in data.items():
            thr[task] = rule(pool, oof)
            held += test
            scores += ts.tolist()
        res = evaluate(held, scores, thr)
        print(
            f"test {name:<14} macro F1 {macro(res):.3f}  "
            + " ".join(
                f"{k.split(':')[1].strip()[:10]}={v['f1']:.2f}" for k, v in res.items()
            )
        )


if __name__ == "__main__":
    main()
