"""Cross-fitted linear heads with group-held-out thresholds.

October protocol: fit on train, choose the recipe by dev source AP, choose the
threshold by dev max F1, predict with the train-only head.

November protocol: pool train and dev, split them into folds that keep every
camera/recording/author group whole, choose the recipe by out-of-fold source
AP and the threshold by out-of-fold macro-source F1, then predict with the
average of the fold heads. Held-out cameras therefore set the threshold, which
is the situation a deployed camera is in.
"""

from __future__ import annotations

import sys
import warnings
from collections import defaultdict

import numpy as np
from common import ROOT, best_f1_threshold
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(ROOT))
from anomaly_eval.training.data import group_class_weights

C_GRID = [0.03, 0.1, 0.3, 1, 3, 10]


def weights(rows):
    w = np.asarray(group_class_weights(rows), dtype=np.float64)
    return w * len(w) / w.sum()


def fit(x, y, w, c):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LogisticRegression(C=c, max_iter=1000, tol=1e-4)
        model.fit(x, y, sample_weight=w)
    return model.coef_[0].astype(np.float32), float(model.intercept_[0])


def score(head, x):
    return x @ head[0] + head[1]


def folds(rows, k=5, seed=0):
    y = np.array([r["target"] for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    k = min(k, len(set(groups)))
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return [test for _, test in splitter.split(np.zeros(len(y)), y, groups)]


def source_ap(rows, scores):
    by = defaultdict(list)
    for r, s in zip(rows, scores):
        by[r["source"]].append((r["target"], s))
    values = {}
    for source, pairs in by.items():
        y, s = map(np.asarray, zip(*pairs))
        if len(set(y)) == 2:
            values[source] = float(average_precision_score(y, s))
    return float(np.mean(list(values.values()))), values


def macro_source_threshold(rows, scores):
    """One task threshold maximizing the mean F1 across development sources."""
    scores = np.asarray(scores, float)
    sources = sorted({r["source"] for r in rows})
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 801)))
    total = np.zeros(len(candidates))
    for source in sources:
        idx = [i for i, r in enumerate(rows) if r["source"] == source]
        y = np.array([rows[i]["target"] for i in idx])
        if len(set(y)) < 2:
            continue
        s = scores[idx]
        pred = s[None, :] >= candidates[:, None]
        tp = (pred & (y == 1)).sum(1)
        fp = (pred & (y == 0)).sum(1)
        fn = (~pred & (y == 1)).sum(1)
        total += 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    i = int(np.argmax(total))
    return float(candidates[i]), float(total[i] / len(sources))


def cross_fit(x, rows, c, fold_sets):
    """Out-of-fold scores (averaged over repeats) and every fold head."""
    y = np.array([r["target"] for r in rows])
    w = weights(rows)
    oof = np.zeros(len(rows))
    heads = []
    for test_sets in fold_sets:
        for test in test_sets:
            train = np.setdiff1d(np.arange(len(rows)), test)
            if len(set(y[train])) < 2:
                continue
            head = fit(x[train], y[train], w[train], c)
            oof[test] += score(head, x[test]) / len(fold_sets)
            heads.append(head)
    return oof, heads


def november(task_rows, matrices, test_matrices, repeats=3, n_jobs=20):
    """task_rows: train+dev rows. matrices/test_matrices: name -> array."""
    fold_sets = [folds(task_rows, seed=s) for s in range(repeats)]
    jobs = [(name, c) for name in matrices for c in C_GRID]
    fits = Parallel(n_jobs=min(n_jobs, len(jobs)))(
        delayed(cross_fit)(matrices[name], task_rows, c, fold_sets) for name, c in jobs
    )
    candidates = []
    for (name, c), (oof, heads) in zip(jobs, fits):
        mean_ap, per_source = source_ap(task_rows, oof)
        candidates.append(
            {
                "recipe": name,
                "C": c,
                "oof_ap": mean_ap,
                "oof_source_ap": per_source,
                "oof": oof,
                "heads": heads,
            }
        )
    for d in candidates:
        d["test"] = np.mean(
            [score(h, test_matrices[d["recipe"]]) for h in d["heads"]], 0
        )
    best = max(candidates, key=lambda d: d["oof_ap"])
    threshold, oof_f1 = macro_source_threshold(task_rows, best["oof"])
    test = best["test"]
    return {
        "recipe": best["recipe"],
        "C": best["C"],
        "oof_ap": best["oof_ap"],
        "oof_source_ap": best["oof_source_ap"],
        "oof_macro_f1": oof_f1,
        "threshold": threshold,
        "test_scores": test,
        "oof_scores": best["oof"],
        "heads": best["heads"],
        "candidates": [
            {k: v for k, v in d.items() if k != "heads"} for d in candidates
        ],
    }


def october(task_rows, matrices, test_matrices, n_jobs=20):
    """Train-only fit, dev selection and dev max-F1 threshold."""
    train = [i for i, r in enumerate(task_rows) if r["split"] == "train"]
    dev = [i for i, r in enumerate(task_rows) if r["split"] == "dev"]
    tr_rows = [task_rows[i] for i in train]
    dev_rows = [task_rows[i] for i in dev]
    y = np.array([r["target"] for r in tr_rows])
    w = weights(tr_rows)
    jobs = [(name, c) for name in matrices for c in [0.1, 1, 10]]
    heads = Parallel(n_jobs=min(n_jobs, len(jobs)))(
        delayed(fit)(matrices[name][train], y, w, c) for name, c in jobs
    )
    best = None
    for (name, c), head in zip(jobs, heads):
        dev_scores = score(head, matrices[name][dev])
        ap, _ = source_ap(dev_rows, dev_scores)
        if best is None or ap > best[0]:
            best = (ap, name, c, head, dev_scores)
    ap, name, c, head, dev_scores = best
    threshold, _ = best_f1_threshold([r["target"] for r in dev_rows], dev_scores)
    return {
        "recipe": name,
        "C": c,
        "dev_ap": ap,
        "threshold": threshold,
        "test_scores": score(head, test_matrices[name]),
    }
