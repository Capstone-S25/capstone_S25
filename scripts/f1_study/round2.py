"""Second-round head experiments on top of the released recipes.

Every task's head runs on every window in deployment, so a smoking head also
sees unattended-item footage. This script

* measures cross-task false alarms: how often a task's head fires on the
  held-out windows of the other four tasks;
* optionally adds the other tasks' train/dev windows as extra negatives
  ("--aux-share"), with whole cameras still held out in every fold;
* optionally raises the threshold until at most "--aux-fpr" of the other
  tasks' out-of-fold windows are flagged;
* lets extra backbones/recipes join the ensemble ("--members", "--extra").

Recipes default to the three per backbone that the released run selected.
"""

from __future__ import annotations

import argparse
import pickle
import warnings

import heads
import numpy as np
import representations
import run_heads
from common import (
    OUT,
    REFERENCE_TASKS,
    TASKS,
    TEST_SOURCES,
    evaluate,
    f1_at,
    load_rows,
    macro,
    reference_frame,
    save_json,
    table,
)
from joblib import Parallel, delayed
from representations import matrix, recipe_name
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

HELD = {"test", "external"}
# Dropping a bag of rubbish and leaving a bag behind look the same, so each
# task's positives are not used as the other's negatives.
CONFUSABLE = {("littering", "unattended_items"), ("unattended_items", "littering")}
# Movie clips cut between shots, so they cannot stand in for a fixed camera
# when a head compares the window with the camera's reference frame.
NOT_FIXED_CAMERA = {"hmdb51_action_transfer"}


def named_recipes(backbones):
    run_heads.FROZEN_ONLY = True
    run_heads.SMOKE_DETECTOR = True
    return {recipe_name(r): r for b in backbones for r in run_heads.recipes_for(b)}


def released(tag, task, top):
    with open(OUT / "runs" / f"{tag}.pkl", "rb") as f:
        out = pickle.load(f)["november", task]
    best = {}
    for c in out["candidates"]:
        if c["recipe"] not in best or c["oof_ap"] > best[c["recipe"]]["oof_ap"]:
            best[c["recipe"]] = c
    return [
        c["recipe"] for c in sorted(best.values(), key=lambda c: -c["oof_ap"])[:top]
    ]


def as_negative(row, task):
    """Another task's window, presented to `task`'s head as a normal window."""
    views = row["views"]
    if row["target"] == 1 and (task, row["task"]) in CONFUSABLE:
        return None
    if task in REFERENCE_TASKS and row["source"] in NOT_FIXED_CAMERA:
        return None
    out = {**row, "task": task, "target": 0, "aux": row["task"]}
    out["source"] = f"aux:{row['task']}:{row['source']}"
    if task in REFERENCE_TASKS:
        if len(views) < 4:
            return None  # stills have no history to compare with
        idx = np.rint(np.linspace(0, len(views) - 1, 4)).astype(int)
        out["views"] = [views[i] for i in idx]
        out["reference"] = row.get("reference") or reference_frame(views[0])
    return out


def split_rows(rows, task):
    own = [r for r in rows if r["task"] == task and r["split"] in {"train", "dev"}]
    test = [
        r
        for r in rows
        if r["task"] == task
        and r["split"] in HELD
        and (task, r["source"]) in TEST_SOURCES
    ]
    aux, aux_test = [], []
    for r in rows:
        if r["task"] == task:
            continue
        if r["split"] in {"train", "dev"}:
            aux.append(as_negative(r, task))
        elif r["split"] in HELD and (r["task"], r["source"]) in TEST_SOURCES:
            aux_test.append(as_negative(r, task))
    return own, test, [r for r in aux if r], [r for r in aux_test if r]


def sample_weights(own, aux, share):
    w = heads.weights(own)
    if not aux or share <= 0:
        return w
    y = np.array([r["target"] for r in own])
    extra = heads.weights(aux)
    extra *= share / (1 - share) * w[y == 0].sum() / extra.sum()
    return np.concatenate([w, extra])


def cross_fit(x, y, w, groups, c, seeds=3):
    from sklearn.model_selection import StratifiedGroupKFold

    oof, fitted = np.zeros(len(y)), []
    for seed in range(seeds):
        k = min(5, len(set(groups)))
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        for train, test in splitter.split(np.zeros(len(y)), y, groups):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = LogisticRegression(C=c, max_iter=1000, tol=1e-4)
                model.fit(x[train], y[train], sample_weight=w[train])
            head = (model.coef_[0].astype(np.float32), float(model.intercept_[0]))
            oof[test] += heads.score(head, x[test]) / seeds
            fitted.append(head)
    return oof, fitted


def fit_task(rows, task, recipes_by_member, share, n_jobs=20):
    """Return standardized ensemble scores for own/aux pool and test rows."""
    own, test, aux, aux_test = split_rows(rows, task)
    train_rows = own + (aux if share > 0 else [])
    y = np.array([r["target"] for r in train_rows])
    groups = np.array([r["group_id"] for r in train_rows])
    w = sample_weights(own, aux if share > 0 else [], share)
    n = len(own)
    jobs, mats = [], {}
    for member, recipes in recipes_by_member.items():
        for name, recipe in recipes:
            mats[name] = {
                "train": matrix(train_rows, recipe),
                "aux": matrix(aux, recipe) if aux else None,
                "test": matrix(test, recipe),
                "aux_test": matrix(aux_test, recipe) if aux_test else None,
            }
            jobs += [(member, name, c) for c in heads.C_GRID]
    fits = Parallel(n_jobs=min(n_jobs, len(jobs)))(
        delayed(cross_fit)(mats[name]["train"], y, w, groups, c) for _, name, c in jobs
    )
    best = {}
    for (member, name, c), (oof, fitted) in zip(jobs, fits):
        ap = heads.source_ap(own, oof[:n])[0]
        if (member, name) not in best or ap > best[member, name]["ap"]:
            best[member, name] = {"ap": ap, "C": c, "oof": oof, "heads": fitted}

    def apply(fitted, x):
        return np.mean([heads.score(h, x) for h in fitted], 0)

    member_scores, chosen, detail = [], {}, []
    for member, recipes in recipes_by_member.items():
        parts, exported = [], []
        for name, recipe in recipes:
            b = best[member, name]
            mu, sd = b["oof"][:n].mean(), b["oof"][:n].std() + 1e-9
            exported.append(
                {
                    "name": name,
                    "recipe": recipe,
                    "C": b["C"],
                    "mean": float(mu),
                    "std": float(sd),
                    "oof_ap": b["ap"],
                    "heads": b["heads"],
                }
            )
            aux_pool = (
                b["oof"][n:]
                if share > 0
                else apply(b["heads"], mats[name]["aux"])
                if aux
                else np.zeros(0)
            )
            parts.append(
                [
                    (b["oof"][:n] - mu) / sd,
                    (aux_pool - mu) / sd,
                    (apply(b["heads"], mats[name]["test"]) - mu) / sd,
                    (apply(b["heads"], mats[name]["aux_test"]) - mu) / sd
                    if aux_test
                    else np.zeros(0),
                ]
            )
            chosen[name] = {"C": b["C"], "oof_ap": b["ap"]}
        mean = [np.mean([p[i] for p in parts], 0) for i in range(4)]
        mu, sd = mean[0].mean(), mean[0].std() + 1e-9
        member_scores.append([(m - mu) / sd for m in mean])
        detail.append(
            {"tag": member, "mean": float(mu), "std": float(sd), "recipes": exported}
        )
    scores = [np.mean([m[i] for m in member_scores], 0) for i in range(4)]
    return {
        "own": own,
        "test": test,
        "aux": aux,
        "aux_test": aux_test,
        "oof": scores[0],
        "aux_oof": scores[1],
        "test_scores": scores[2],
        "aux_test_scores": scores[3],
        "chosen": chosen,
        "detail": detail,
    }


def by_task(rows, scores, threshold):
    out = {}
    for name in sorted({r["aux"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["aux"] == name]
        out[name] = float((scores[idx] >= threshold).mean())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--members", nargs="+", default=["v8_siglip2-b", "v8_mobileclip2-b"]
    )
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--aux-share", type=float, default=0.0)
    parser.add_argument("--aux-fpr", type=float, default=None)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument(
        "--detector", default="", help="Smoking detector fold suffix, e.g. x"
    )
    args = parser.parse_args()
    representations.DETECTOR_SUFFIX = args.detector
    rows = load_rows()
    backbones = [m.split("_", 1)[1] for m in args.members]
    recipes = named_recipes(backbones)
    held_rows, held_scores, thresholds, info, saved = [], [], {}, {}, {}
    for task in args.tasks:
        chosen = {
            m: [(name, recipes[name]) for name in released(m, task, args.top)]
            for m in args.members
        }
        out = fit_task(rows, task, chosen, args.aux_share)
        threshold, oof_f1 = heads.macro_source_threshold(out["own"], out["oof"])
        if args.aux_fpr is not None and len(out["aux_oof"]):
            threshold = max(
                threshold, float(np.quantile(out["aux_oof"], 1 - args.aux_fpr))
            )
        thresholds[task] = threshold
        y_aux = np.zeros(len(out["aux_test"]), int)
        info[task] = {
            "threshold": threshold,
            "oof_ap": heads.source_ap(out["own"], out["oof"])[0],
            "oof_macro_f1": oof_f1,
            "recipes": out["chosen"],
            "cross_task_oof_fpr": float((out["aux_oof"] >= threshold).mean()),
            "cross_task_test_fpr": f1_at(y_aux, out["aux_test_scores"], threshold)[
                "fpr"
            ],
            "cross_task_test_fpr_by_task": by_task(
                out["aux_test"], out["aux_test_scores"], threshold
            ),
            "cross_task_test_n": len(y_aux),
        }
        held_rows += out["test"]
        held_scores += out["test_scores"].tolist()
        saved[task] = {
            k: out[k] for k in ["oof", "aux_oof", "test_scores", "aux_test_scores"]
        } | {
            "own_ids": [r["id"] for r in out["own"]],
            "test_ids": [r["id"] for r in out["test"]],
            "aux_test_ids": [r["id"] for r in out["aux_test"]],
            "aux_test_task": [r["aux"] for r in out["aux_test"]],
        }
        print(
            task,
            f"oof AP {info[task]['oof_ap']:.3f} F1 {oof_f1:.3f}",
            f"cross-task FPR oof {info[task]['cross_task_oof_fpr']:.3f}",
            f"test {info[task]['cross_task_test_fpr']:.3f}",
            info[task]["cross_task_test_fpr_by_task"],
            flush=True,
        )
    results = evaluate(held_rows, held_scores, thresholds)
    label = f"{args.tag}:november"
    cross = float(np.mean([info[t]["cross_task_test_fpr"] for t in info]))
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
    with open(OUT / "runs" / f"{args.tag}_round2.pkl", "wb") as f:
        pickle.dump(saved, f)
    print(table({label: results}, fields=("ap", "f1", "fpr")))
    print(f"mean cross-task test FPR {cross:.3f}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
