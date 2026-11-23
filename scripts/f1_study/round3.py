"""Round 3: fit the released recipes with or without new-site footage, and
score footage from sites no head has seen.

Rows come from the October manifests plus `round3_sources.py`. For each task:

* the benchmark's train/dev rows are fitted exactly as in round 2;
* `--site-fit` adds the task's "site_fit" rows to the fitting pool (whole
  cameras per fold);
* `--recipes-from` / `--add-recipe` change the representation;
* every "site_shift" row is scored by the final heads and never fitted.

Scores are saved; `round3_report.py` picks thresholds and prints the tables.
"""

from __future__ import annotations

import os

os.environ["F1_ROUND3"] = "1"

import argparse
import pickle

import heads
import numpy as np
import round2
from common import OUT, TASKS, load_rows
from joblib import Parallel, delayed
from representations import matrix

# Per-task share of other tasks' windows among the negatives, as released.
RELEASED_SHARE = {"loitering": 0.4}
# Fit-side sources of one behaviour that serve as negatives for the others.
OTHER_SITE_NEGATIVES = {"xrf55", "smoking_site2"}
# Added after r3_final (the release) was fitted; --no-other-site-negatives
# reproduces the release.
OTHER_SITE = True


def site(rows, task, side):
    return [r for r in rows if r["task"] == task and r["split"] == f"site_{side}"]


def trainable(row):
    """Unlabelled footage is scored but not used as a certain negative for
    the heads it could plausibly contain a positive for."""
    return not (row.get("label_uncertain") and row["task"] == "loitering")


def precomputed(spec, parts, task):
    """A member scored elsewhere: "persons:<tag>" (person_head.py) or
    "zeroshot:<kind>[+<kind>]" (zero_shot.py; kinds are averaged) or
    "detector:<suffix>[@far]" (the smoking detector's confidence; "@far"
    counts only boxes at CCTV distance)."""
    import json

    kind, _, name = spec.partition(":")
    if kind == "persons":
        scores = json.loads((OUT / "runs" / f"persons_{name}.json").read_text())[
            "scores"
        ]
        return {k: np.array([scores[r["id"]] for r in v]) for k, v in parts.items()}
    if kind == "detector":
        # Top "smoking" confidence of the detector that has not seen the row.
        import representations

        name, _, gate = name.partition("@")
        representations.DETECTOR_SUFFIX = name
        value = (
            representations.smoking_detector_far
            if gate
            else lambda r: float(representations.smoking_detector_features(r)[0])
        )
        return {k: np.array([value(r) for r in v]) for k, v in parts.items()}
    table = _zeroshot()
    return {
        k: np.array(
            [
                np.mean([table[n][f"{r['id']}|{task}"] for n in name.split("+")])
                for r in v
            ]
        )
        for k, v in parts.items()
    }


_ZEROSHOT = {}


def _zeroshot():
    import json

    if not _ZEROSHOT:
        _ZEROSHOT.update(
            json.loads((OUT / "runs" / "zeroshot_siglip2-b.json").read_text())
        )
    return _ZEROSHOT


def fit_task(
    rows,
    task,
    recipes_by_member,
    share,
    site_fit,
    without=(),
    persons=None,
    member_weights=None,
    n_jobs=20,
):
    own, test, aux, aux_test = round2.split_rows(rows, task)
    own = [r for r in own if r["source"] not in without]
    if share > 0 and OTHER_SITE:
        # New-site footage of other behaviours also shows people who are not
        # doing this one (workers smoking, people phoning indoors).
        aux = aux + [
            n
            for r in rows
            if r["task"] != task
            and r["split"] == "site_fit"
            and r["source"] in OTHER_SITE_NEGATIVES
            for n in [round2.as_negative(r, task)]
            if n
        ]
    fit_rows, shift = site(rows, task, "fit"), site(rows, task, "shift")
    usable = lambda r: site_fit and trainable(r) and r["source"] not in without
    used = [r for r in fit_rows if usable(r)]
    unused = [r for r in fit_rows if not usable(r)]
    core = own + used
    train_rows = core + (aux if share > 0 else [])
    y = np.array([r["target"] for r in train_rows])
    groups = np.array([r["group_id"] for r in train_rows])
    w = round2.sample_weights(core, aux if share > 0 else [], share)
    n, m = len(own), len(core)
    parts = {
        "aux": aux,
        "test": test,
        "aux_test": aux_test,
        "shift": shift,
        "unused": unused,
    }
    jobs, mats = [], {}
    for member, recipes in recipes_by_member.items():
        for name, recipe in recipes:
            mats[name] = {"train": matrix(train_rows, recipe)} | {
                k: matrix(v, recipe) if v else None for k, v in parts.items()
            }
            jobs += [(member, name, c) for c in heads.C_GRID]
    fits = Parallel(n_jobs=min(n_jobs, len(jobs)))(
        delayed(round2.cross_fit)(mats[name]["train"], y, w, groups, c)
        for _, name, c in jobs
    )
    best = {}
    for (member, name, c), (oof, fitted) in zip(jobs, fits):
        ap = heads.source_ap(core, oof[:m])[0]
        if (member, name) not in best or ap > best[member, name]["ap"]:
            best[member, name] = {"ap": ap, "C": c, "oof": oof, "heads": fitted}

    def apply(fitted, x):
        if x is None:
            return np.zeros(0)
        return np.mean([heads.score(h, x) for h in fitted], 0)

    keys = ["own", "used", "aux", "test", "aux_test", "shift", "unused"]
    member_scores, detail = [], []
    for member, recipes in recipes_by_member.items():
        collected, exported = [], []
        for name, recipe in recipes:
            b = best[member, name]
            mu, sd = b["oof"][:m].mean(), b["oof"][:m].std() + 1e-9
            raw = {
                "own": b["oof"][:n],
                "used": b["oof"][n:m],
                "aux": b["oof"][m:]
                if share > 0
                else apply(b["heads"], mats[name]["aux"]),
                "test": apply(b["heads"], mats[name]["test"]),
                "aux_test": apply(b["heads"], mats[name]["aux_test"]),
                "shift": apply(b["heads"], mats[name]["shift"]),
                "unused": apply(b["heads"], mats[name]["unused"]),
            }
            collected.append({k: (v - mu) / sd for k, v in raw.items()})
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
        mean = {k: np.mean([c[k] for c in collected], 0) for k in keys}
        base = np.concatenate([mean["own"], mean["used"]])
        mu, sd = base.mean(), base.std() + 1e-9
        member_scores.append({k: (v - mu) / sd for k, v in mean.items()})
        detail.append(
            {"tag": member, "mean": float(mu), "std": float(sd), "recipes": exported}
        )
    ids = {
        "own": own,
        "used": used,
        "aux": aux,
        "test": test,
        "aux_test": aux_test,
        "shift": shift,
        "unused": unused,
    }
    for spec in persons or []:
        raw = precomputed(spec, ids, task)
        # Standardize like the fitted members: on the fitted rows.
        base = np.concatenate([raw["own"], raw["used"]])
        mu, sd = base.mean(), base.std() + 1e-9
        member_scores.append({k: (v - mu) / sd for k, v in raw.items()})
        detail.append({"tag": spec, "mean": float(mu), "std": float(sd)})
    weights = member_weights or [1.0] * len(member_scores)
    scores = {
        k: np.average([s[k] for s in member_scores], 0, weights=weights) for k in keys
    }
    ids = {
        "own": own,
        "used": used,
        "aux": aux,
        "test": test,
        "aux_test": aux_test,
        "shift": shift,
        "unused": unused,
    }
    return {
        "scores": scores,
        "ids": {k: [r["id"] for r in v] for k, v in ids.items()},
        "aux_task": [r["aux"] for r in aux],
        "aux_test_task": [r["aux"] for r in aux_test],
        "detail": detail,
        "weights": list(weights),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--members", nargs="+", default=["v8_siglip2-b", "v8_mobileclip2-b"]
    )
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--site-fit", action="store_true")
    parser.add_argument(
        "--aux-share",
        type=float,
        default=None,
        help="Same share for every task (default: as released, loitering 0.4)",
    )
    parser.add_argument(
        "--share",
        nargs="*",
        default=[],
        help="Per-task share of other behaviours' windows, e.g. littering=0.25",
    )
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument(
        "--variant",
        default="",
        help="Representation change from round3_recipes.VARIANTS",
    )
    parser.add_argument(
        "--without", nargs="*", default=[], help="Sources left out of fitting"
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="task=member, e.g. smoking=zeroshot:full+crop smoking=persons:site",
    )
    parser.add_argument(
        "--detector", default="", help="Smoking detector suffix (f = with site data)"
    )
    parser.add_argument(
        "--no-fitted",
        nargs="*",
        default=[],
        help="Tasks that use their extra members alone",
    )
    parser.add_argument(
        "--no-other-site-negatives",
        action="store_true",
        help="As the release: no other behaviours' new-site rows as negatives",
    )
    args = parser.parse_args()
    global OTHER_SITE
    OTHER_SITE = not args.no_other_site_negatives
    import representations

    representations.DETECTOR_SUFFIX = args.detector
    rows = load_rows()
    backbones = [m.split("_", 1)[1] for m in args.members]
    recipes = round2.named_recipes(backbones)
    saved = {}
    for task in args.tasks:
        chosen = {
            m: [(name, recipes[name]) for name in round2.released(m, task, args.top)]
            for m in args.members
        }
        if args.variant:
            import round3_recipes

            chosen = round3_recipes.VARIANTS[args.variant](task, chosen)
        share = (
            RELEASED_SHARE.get(task, 0.0) if args.aux_share is None else args.aux_share
        )
        share = float(dict(v.split("=") for v in args.share).get(task, share))
        extra = [e.split("=", 1)[1] for e in args.extra if e.split("=")[0] == task]
        saved[task] = fit_task(
            rows,
            task,
            chosen,
            share,
            args.site_fit,
            without=args.without,
            persons=extra,
            member_weights=[0.0] * len(chosen) + [1.0] * len(extra)
            if task in args.no_fitted
            else None,
        )
        s = saved[task]["scores"]
        print(task, {k: len(v) for k, v in s.items()}, flush=True)
    with open(OUT / "runs" / f"{args.tag}_round3.pkl", "wb") as f:
        pickle.dump(saved, f)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
