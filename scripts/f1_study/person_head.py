"""Person-level smoking head: score each detected person, take the highest.

Instances are the person crops of `persons.py` (the whole frame stands in
when no person was detected). Labels:

* negative frame: every person is negative;
* CCTV frames with head boxes: a person is positive when a "smoking" head box
  sits inside the person box, negative when the site labels non-smokers
  (site 1), otherwise left out (site 2 labels smokers only);
* other positive frames (photos, clips): the largest person is positive, the
  others are left out.

Heads are cross-fitted with whole cameras/authors/subjects held out. A row's
score is the mean over its views of the highest person score. Output:
`runs/persons_<tag>.json` with out-of-fold scores for fitted rows and
fold-mean scores for every other row (all tasks, so that other tasks'
windows can be checked for false alarms).
"""

from __future__ import annotations

import os

os.environ["F1_ROUND3"] = "1"

import argparse
import json
import warnings
from collections import defaultdict

import heads
import numpy as np
from common import OUT, ROOT, load_rows, save_json
from joblib import Parallel, delayed
from representations import image_features
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

SITE1 = ROOT / "data/raw/datasets/smoking/smoking-video.v1i.coco"
SITE2 = ROOT / "data/raw/datasets/smoking/smoking_cctv_image.v1i.coco"


def head_boxes():
    """path -> [(label, cx, cy)] in 0-1 fractions; label 1 smoking, 0 not."""
    out = {}
    for root in [SITE1, SITE2]:
        for split in ["train", "valid", "test"]:
            file = root / split / "_annotations.coco.json"
            if not file.exists():
                continue
            coco = json.loads(file.read_text())
            names = {c["id"]: c["name"].lower() for c in coco["categories"]}
            images = {i["id"]: i for i in coco["images"]}
            for a in coco["annotations"]:
                name = names[a["category_id"]]
                if name not in {"smoking", "not smoking"}:
                    continue
                image = images[a["image_id"]]
                x, y, w, h = a["bbox"]
                path = str((root / split / image["file_name"]).relative_to(ROOT))
                out.setdefault(path, []).append(
                    (
                        int(name == "smoking"),
                        (x + w / 2) / image["width"],
                        (y + h / 2) / image["height"],
                    )
                )
    return out


class Persons:
    def __init__(self, backbone):
        value = np.load(OUT / "features" / f"{backbone}_persons.npz")
        features = value["features"].astype(np.float32)
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
        self.features, self.boxes = features, value["boxes"]
        self.index = {
            p: (int(a), int(b))
            for p, a, b in zip(
                value["paths"].tolist(), value["start"][:-1], value["start"][1:]
            )
        }
        self.full = image_features(backbone, "full")

    def view(self, path):
        """(features, boxes) of one frame; the frame itself if nobody found."""
        a, b = self.index.get(path, (0, 0))
        if b > a:
            return self.features[a:b], self.boxes[a:b]
        return self.full[path][None], np.array([[0, 0, 1, 1]], np.float32)


def instances(rows, persons, annotated, labelled):
    """Stack instances; returns x, y (or -1 = left out), row index, view index."""
    xs, ys, row_of, view_of = [], [], [], []
    for i, row in enumerate(rows):
        for v, path in enumerate(row["views"]):
            x, boxes = persons.view(path)
            if not labelled or row["target"] == 0:
                y = np.zeros(len(x), int)
            elif path in annotated:
                y = np.full(len(x), 0 if "smoking-video" in path else -1)
                for label, cx, cy in annotated[path]:
                    inside = (
                        (boxes[:, 0] <= cx)
                        & (cx <= boxes[:, 2])
                        & (boxes[:, 1] <= cy)
                        & (cy <= boxes[:, 3])
                    )
                    if label == 1:
                        y[inside] = 1
            else:
                y = np.full(len(x), -1)
                y[0] = 1
            xs.append(x)
            ys.append(y)
            row_of += [i] * len(x)
            view_of += [v] * len(x)
    return np.concatenate(xs), np.concatenate(ys), np.array(row_of), np.array(view_of)


def row_scores(scores, row_of, view_of, n):
    """Mean over views of the best instance."""
    best = defaultdict(lambda: -np.inf)
    for s, r, v in zip(scores, row_of, view_of):
        best[r, v] = max(best[r, v], s)
    per_row = defaultdict(list)
    for (r, _), s in best.items():
        per_row[r].append(s)
    return np.array([np.mean(per_row[r]) if per_row[r] else np.nan for r in range(n)])


def fit(x, y, w, c):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = LogisticRegression(C=c, max_iter=1000, tol=1e-4)
        model.fit(x, y, sample_weight=w)
    return model.coef_[0].astype(np.float32), float(model.intercept_[0])


def cross_fit(x, y, w, row_of, view_of, rows, c, seeds=3):
    target = np.array([r["target"] for r in rows])
    groups = np.array([r["group_id"] for r in rows])
    keep = y >= 0
    oof, fitted = np.zeros(len(rows)), []
    for seed in range(seeds):
        k = min(5, len(set(groups)))
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        for train, test in splitter.split(np.zeros(len(rows)), target, groups):
            in_train = np.isin(row_of, train) & keep
            head = fit(x[in_train], y[in_train], w[in_train], c)
            in_test = np.isin(row_of, test)
            scored = row_scores(
                x[in_test] @ head[0] + head[1],
                row_of[in_test],
                view_of[in_test],
                len(rows),
            )
            oof[test] += scored[test] / seeds
            fitted.append(head)
    return oof, fitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--backbone", default="siglip2-b")
    parser.add_argument("--task", default="smoking")
    parser.add_argument("--site-fit", action="store_true")
    parser.add_argument(
        "--without", nargs="*", default=[], help="Sources left out of fitting"
    )
    args = parser.parse_args()
    rows = load_rows()
    splits = {"train", "dev"} | ({"site_fit"} if args.site_fit else set())
    train = [
        r
        for r in rows
        if r["task"] == args.task
        and r["split"] in splits
        and r["source"] not in args.without
    ]
    fitted_ids = {r["id"] for r in train}
    rest = [r for r in rows if r["id"] not in fitted_ids]
    persons = Persons(args.backbone)
    annotated = head_boxes()
    x, y, row_of, view_of = instances(train, persons, annotated, labelled=True)
    row_w = heads.weights(train)
    per_row = np.bincount(row_of[y >= 0], minlength=len(train)).clip(min=1)
    w = row_w[row_of] / per_row[row_of]
    print(
        len(train),
        "rows",
        len(x),
        "instances",
        {int(k): int((y == k).sum()) for k in (-1, 0, 1)},
        flush=True,
    )
    fits = Parallel(n_jobs=6)(
        delayed(cross_fit)(x, y, w, row_of, view_of, train, c) for c in heads.C_GRID
    )
    best = None
    for c, (oof, fitted) in zip(heads.C_GRID, fits):
        ap, by_source = heads.source_ap(train, oof)
        print(
            f"C {c:g} oof AP {ap:.3f}", {k: round(v, 3) for k, v in by_source.items()}
        )
        if best is None or ap > best[0]:
            best = (ap, c, oof, fitted)
    ap, c, oof, fitted = best
    coef = np.mean([h[0] for h in fitted], 0)
    bias = float(np.mean([h[1] for h in fitted]))
    scores = dict(zip([r["id"] for r in train], oof.tolist()))
    xr, _, rr, vr = instances(rest, persons, annotated, labelled=False)
    scores.update(
        zip(
            [r["id"] for r in rest],
            row_scores(xr @ coef + bias, rr, vr, len(rest)).tolist(),
        )
    )
    save_json(
        OUT / "runs" / f"persons_{args.tag}.json",
        {
            "task": args.task,
            "backbone": args.backbone,
            "C": c,
            "oof_ap": ap,
            "site_fit": args.site_fit,
            "without": args.without,
            "coef": coef.tolist(),
            "bias": bias,
            "scores": scores,
        },
    )
    print("saved", args.tag, "C", c, "oof AP", round(ap, 3))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
