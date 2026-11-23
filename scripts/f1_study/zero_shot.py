"""Zero-shot text scores from the SigLIP2 encoder already in the ensemble.

No fitting: a row's score is the similarity of its frames (or person crops)
to descriptions of the behaviour minus the similarity to descriptions of
ordinary scenes. Nothing here has seen any site, so it is a site-neutral
member for the ensemble. Output: `runs/zeroshot_<name>.json` (id -> score).
"""

from __future__ import annotations

import os

os.environ["F1_ROUND3"] = "1"

import argparse

import numpy as np
import torch
from common import OUT, TASKS, load_rows, save_json
from person_head import Persons
from representations import image_features

PROMPTS = {
    "smoking": (
        [
            "a person smoking a cigarette",
            "a person holding a cigarette to their mouth",
            "someone smoking, cigarette in hand",
            "a person exhaling cigarette smoke",
        ],
        [
            "a person standing",
            "a person walking",
            "a person talking on a phone",
            "a person drinking from a cup",
            "a person eating",
            "people in a room",
            "an empty scene",
        ],
    ),
    "littering": (
        [
            "a person throwing rubbish on the ground",
            "a person dumping a bag of waste",
            "someone dropping litter on the street",
            "a person tossing trash away",
        ],
        [
            "a person walking",
            "a person standing",
            "people passing by",
            "an empty street",
            "a person carrying a bag",
        ],
    ),
    "loitering": (
        [
            "a person loitering, standing around for a long time",
            "someone lingering in one place with no purpose",
            "a person hanging around and waiting",
        ],
        [
            "people walking past",
            "an empty street",
            "pedestrians crossing",
            "cars driving",
        ],
    ),
    "rough_sleeping": (
        [
            "a homeless person sleeping on the ground",
            "a person sleeping on a bench outdoors",
            "someone lying asleep on the pavement with a blanket",
        ],
        ["a person sitting", "a person standing", "an empty bench", "people walking"],
    ),
    "unattended_items": (
        [
            "an unattended bag left on the floor",
            "abandoned luggage with nobody near it",
            "a suitcase left alone in a hallway",
        ],
        [
            "a person carrying a bag",
            "an empty hallway",
            "people walking",
            "a person standing next to a bag",
        ],
    ),
}


def text_features(name):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    out = {}
    with torch.inference_mode():
        for task, (positive, negative) in PROMPTS.items():
            tokens = processor(
                text=positive + negative,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
            )
            e = model.get_text_features(**tokens)
            e = torch.nn.functional.normalize(e, dim=-1).numpy()
            out[task] = (e[: len(positive)], e[len(positive) :])
    return out


def score(x, text):
    """x: (n, d) unit vectors -> behaviour minus ordinary, best prompt each."""
    positive, negative = text
    return (x @ positive.T).max(1) - (x @ negative.T).max(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--backbone", default="siglip2-b")
    args = parser.parse_args()
    rows = load_rows()
    text = text_features(args.model)
    full = image_features(args.backbone, "full")
    crop = image_features(args.backbone, "crop-yolo")
    persons = Persons(args.backbone)
    out = {"full": {}, "crop": {}, "persons": {}}
    for row in rows:
        # Every head sees every window in deployment, so score all tasks.
        for task in TASKS:
            key = f"{row['id']}|{task}"
            views = row["views"]
            out["full"][key] = float(
                np.mean(score(np.stack([full[p] for p in views]), text[task]))
            )
            out["crop"][key] = float(
                np.mean(score(np.stack([crop[p] for p in views]), text[task]))
            )
            if task == "smoking":
                out["persons"][key] = float(
                    np.mean(
                        [score(persons.view(p)[0], text[task]).max() for p in views]
                    )
                )
    save_json(OUT / "runs" / f"zeroshot_{args.backbone}.json", out)
    # Quick look: AP per held-out or shift source, own task.
    from sklearn.metrics import average_precision_score

    held = [
        r for r in rows if r["split"] in {"test", "external", "site_shift", "site_fit"}
    ]
    for kind, table in out.items():
        print("==", kind)
        for task in TASKS:
            mine = [r for r in held if r["task"] == task]
            normal = [r for r in mine if r["source"] in {"virat", "avenue", "meva"}]
            n_scores = np.array(
                [table.get(f"{r['id']}|{task}", np.nan) for r in normal]
            )
            for source in sorted(
                {r["source"] for r in mine} - {"virat", "avenue", "meva"}
            ):
                part = [r for r in mine if r["source"] == source]
                s = np.array([table.get(f"{r['id']}|{task}", np.nan) for r in part])
                if np.isnan(s).all():
                    continue
                y = np.array([r["target"] for r in part])
                ap = (
                    average_precision_score(y, s)
                    if 0 < y.sum() < len(y)
                    else float("nan")
                )
                # positives of this source against ordinary footage of new cameras
                both = np.r_[s[y == 1], n_scores]
                yy = np.r_[np.ones(int(y.sum())), np.zeros(len(n_scores))]
                ap_n = (
                    average_precision_score(yy, both)
                    if len(n_scores) and y.sum()
                    else float("nan")
                )
                print(
                    f"{task:17s}{source:32s} n {len(y):4d} pos {int(y.sum()):4d} AP {ap:.3f}  base {y.mean():.2f} | vs ordinary AP {ap_n:.3f} base {yy.mean() if len(yy) else 0:.2f}"
                )


if __name__ == "__main__":
    main()
