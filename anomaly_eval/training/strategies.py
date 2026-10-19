"""Scene, automatic-region and ordered-view representations for strategy screens."""

from __future__ import annotations

import numpy as np


def normalize(value):
    return value / max(float(np.linalg.norm(value)), 1e-12)


def represent(views, ordered=False):
    views = np.asarray(views, dtype=np.float32)
    if ordered:
        indices = np.rint(np.linspace(0, len(views) - 1, 4)).astype(int)
        return normalize(
            np.concatenate([views[indices].ravel(), views.mean(0), views.max(0)])
        )
    return normalize(
        np.concatenate([views[-1], views.mean(0), views.max(0), views[-1] - views[0]])
    )


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = lambda c: max(0, c[2] - c[0]) * max(0, c[3] - c[1])
    return intersection / max(area(a) + area(b) - intersection, 1e-12)


def geometry(ordered_detections):
    """Automatic sparse person/item geometry; never an ownership ground truth."""
    frames = []
    previous = []
    for found in ordered_detections:
        persons = [d["box"] for d in found if d["label"] == 1]
        bags = [d["box"] for d in found if d["label"] in {27, 31, 33}]
        centers = lambda boxes: np.array(
            [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes]
        )
        if persons and bags:
            distance = float(
                np.linalg.norm(
                    centers(persons)[:, None] - centers(bags)[None], axis=-1
                ).min()
            )
        else:
            distance = 1.5
        persistence = max((overlap(a, b) for a in persons for b in previous), default=0)
        frames.append(
            [
                min(len(persons), 8) / 8,
                min(len(bags), 4) / 4,
                float(bool(persons)),
                float(bool(bags)),
                distance,
                persistence,
            ]
        )
        previous = persons
    return represent(frames)


def source_ap(rows, scores):
    from sklearn.metrics import average_precision_score

    values = {}
    for source in sorted({r["source"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["source"] == source]
        y = [rows[i]["target"] for i in indices]
        if len(set(y)) == 2:
            values[source] = float(
                average_precision_score(y, np.asarray(scores)[indices])
            )
    if not values:
        raise ValueError("No development source has both labels")
    return {
        "mean_source_ap": float(np.mean(list(values.values()))),
        "worst_source_ap": min(values.values()),
        "source_ap": values,
    }
