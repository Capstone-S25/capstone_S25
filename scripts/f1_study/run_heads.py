"""Compare head-fitting protocols and backbones on the shared held-out sources."""

from __future__ import annotations

import argparse
import pickle

import heads
import numpy as np
from common import (
    OUT,
    REFERENCE_TASKS,
    TASKS,
    TEST_SOURCES,
    evaluate,
    load_rows,
    save_json,
    table,
)
from representations import matrix, recipe_name

OCTOBER_ALPHAS = {"mobileclip2-s2": [0.0, 1.0], "siglip2-b16-224": [0.0, 1.0]}
FROZEN_ONLY = False
SMOKE_DETECTOR = False


BOXES = ["frcnn", "yolo"]


def recipes_for(backbone):
    if backbone == "smokedet":
        return [{"backbone": "none", "smoke_detector": True, "detector_scale": 1.0}]
    if backbone == "none":
        out = [
            {"backbone": "none", "boxes": b, "geometry": True, "dwell": True}
            for b in BOXES
        ] + [{"backbone": "none", "boxes": b, "geometry": True} for b in BOXES]
        return out + [{**r, "offset": True} for r in out]
    out = []
    for alpha in [0.0] if FROZEN_ONLY else OCTOBER_ALPHAS.get(backbone, [0.0]):
        base = {"backbone": backbone, "alpha": alpha}
        out += [{**base}, {**base, "ordered": True}, {**base, "endpoint": True}]
        october = backbone in OCTOBER_ALPHAS
        for boxes in ["frcnn"] if october else BOXES:
            b = {**base, "boxes": boxes}
            out.append({**b, "geometry": True})
            if alpha == 0.0:
                out += [{**b, "crop": True}, {**b, "crop": True, "geometry": True}]
            if not october:
                out += [
                    {**b, "dwell": True, "geometry": True},
                    {**b, "dwell": True, "geometry": True, "reference": True},
                    {**b, "reference": True, "geometry": True},
                    {**b, "crop": True, "reference": True},
                    {**b, "crop": True, "reference": True, "geometry": True},
                ]
        if not october:
            out.append({**base, "reference": True})
            for boxes in BOXES:
                b = {**base, "boxes": boxes, "offset": True}
                out += [
                    {**b, "geometry": True},
                    {**b, "geometry": True, "dwell": True},
                    {**b, "crop": True, "geometry": True},
                ]
            out.append({**base, "offset": True})
            if SMOKE_DETECTOR:
                for boxes in BOXES:
                    out += [
                        {**base, "boxes": boxes, "crop": True, "smoke_detector": True},
                        {**base, "smoke_detector": True},
                    ]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbones", nargs="+", required=True)
    parser.add_argument("--protocols", nargs="+", default=["october", "november"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--boxes", nargs="+", default=["frcnn", "yolo"])
    parser.add_argument("--smoke-detector", action="store_true")
    parser.add_argument(
        "--frozen-only",
        action="store_true",
        help="Only pretrained features (no train-split fine-tuned backbone)",
    )
    args = parser.parse_args()
    global FROZEN_ONLY, SMOKE_DETECTOR
    FROZEN_ONLY = args.frozen_only
    SMOKE_DETECTOR = args.smoke_detector
    BOXES[:] = args.boxes
    rows = load_rows()
    recipes = [r for b in args.backbones for r in recipes_for(b)]
    results, choices, saved = {}, {}, {}
    for protocol in args.protocols:
        held_rows, held_scores, thresholds = [], [], {}
        for task in args.tasks:
            pool_rows = [
                r for r in rows if r["task"] == task and r["split"] in {"train", "dev"}
            ]
            test_rows = [
                r
                for r in rows
                if r["task"] == task
                and r["split"] in {"test", "external"}
                and (task, r["source"]) in TEST_SOURCES
            ]
            usable = [
                r
                for r in recipes
                if not (r.get("reference") or r.get("dwell") or r.get("offset"))
                or task in REFERENCE_TASKS
            ]
            if task != "smoking":
                usable = [r for r in usable if not r.get("smoke_detector")]
            if task not in REFERENCE_TASKS:
                usable = [
                    r
                    for r in usable
                    if r["backbone"] != "none" or r.get("smoke_detector")
                ]
            if not usable:
                continue
            mats = {recipe_name(r): matrix(pool_rows, r) for r in usable}
            test_mats = {recipe_name(r): matrix(test_rows, r) for r in usable}
            fn = heads.november if protocol == "november" else heads.october
            out = fn(pool_rows, mats, test_mats)
            thresholds[task] = out["threshold"]
            held_rows += test_rows
            held_scores += out["test_scores"].tolist()
            choices.setdefault(protocol, {})[task] = {
                k: v
                for k, v in out.items()
                if k not in {"test_scores", "oof_scores", "heads", "candidates"}
            }
            out.pop("heads", None)
            saved[protocol, task] = out
            print(protocol, task, choices[protocol][task], flush=True)
        label = f"{args.tag}:{protocol}"
        results[label] = evaluate(held_rows, held_scores, thresholds)
        save_json(
            OUT / "runs" / f"{args.tag}_{protocol}_scores.json",
            {
                "ids": [r["id"] for r in held_rows],
                "scores": held_scores,
                "thresholds": thresholds,
            },
        )
    save_json(
        OUT / "runs" / f"{args.tag}.json", {"results": results, "choices": choices}
    )
    with open(OUT / "runs" / f"{args.tag}.pkl", "wb") as f:
        pickle.dump(saved, f)
    print(table(results, fields=("ap", "f1", "fpr")))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
