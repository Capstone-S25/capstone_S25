"""Per-image feature caches and per-sample representations."""

from __future__ import annotations

import json
import sys
from functools import cache

import numpy as np
from common import DATA, OUT, ROOT

sys.path.insert(0, str(ROOT))
from anomaly_eval.training.strategies import geometry

OCTOBER = {"mobileclip2-s2", "siglip2-b16-224"}
TAGS = ["main", "historical", "hmdb", "meva", "rough"]


@cache
def image_features(backbone, stage, alpha=0.0):
    """path -> L2-normalized vector for one backbone and view type."""
    if backbone in OCTOBER:
        found = {}
        for tag in TAGS:
            path = DATA / "features" / backbone / f"{tag}_{stage}_alpha_{alpha:g}.npz"
            if path.exists():
                value = np.load(path)
                found.update(zip(value["paths"].tolist(), value["features"]))
        return found
    value = np.load(OUT / "features" / f"{backbone}_{stage}.npz")
    features = value["features"].astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return dict(zip(value["paths"].tolist(), features))


@cache
def detections(source="frcnn"):
    """frcnn: October Faster R-CNN boxes (>=0.5). yolo: the tracker's YOLO26s (>=0.35)."""
    found = {}
    if source == "yolo":
        for line in (OUT / "yolo26s_detections.jsonl").read_text().splitlines():
            row = json.loads(line)
            found[row["path"]] = [
                d for d in row["detections"] if d["confidence"] >= 0.35
            ]
        return found
    for name in ["detections.jsonl", "rough_detections.jsonl"]:
        for line in (DATA / name).read_text().splitlines():
            row = json.loads(line)
            found[row["path"]] = row["detections"]
    for line in (OUT / "frcnn_reference_detections.jsonl").read_text().splitlines():
        row = json.loads(line)
        found.setdefault(row["path"], row["detections"])
    return found


@cache
def smoking_detector_scores():
    return json.loads((OUT / "smoking_detector_scores.json").read_text())


def smoking_detector_features(row):
    """Detector that never saw this row's camera (cross-fitted for c4/c2)."""
    fold = {"smoking_camera:c4": "c2", "smoking_camera:c2": "c4"}.get(
        row["group_id"], "all"
    )
    values = np.array([smoking_detector_scores()[fold][p] for p in row["views"]])
    return np.array(
        [values[:, 0].max(), values[:, 1].max(), min(values[:, 2].max(), 4) / 4],
        np.float32,
    )


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda c: max(0.0, c[2] - c[0]) * max(0.0, c[3] - c[1])
    return inter / max(area(a) + area(b) - inter, 1e-12)


def stays(boxes_by_view, others_by_view=None):
    """How firmly endpoint objects stayed put over the whole window.

    For each object in the last view, its best IoU and centre shift (in object
    heights) to the same class in every earlier view; then the strongest such
    object, how many stayed, and (optionally) its distance to other objects.
    """
    last = boxes_by_view[-1]
    if not last:
        return [0.0, 0.0, 0.0, 2.0, 0.0, 1.5]
    per = []
    for box in last:
        h = max(box[3] - box[1], 1e-3)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        ious, shifts = [], []
        for earlier in boxes_by_view[:-1]:
            if not earlier:
                ious.append(0.0)
                shifts.append(2.0)
                continue
            ious.append(max(iou(box, e) for e in earlier))
            shifts.append(
                min(
                    min(
                        np.hypot((e[0] + e[2]) / 2 - cx, (e[1] + e[3]) / 2 - cy) / h,
                        2.0,
                    )
                    for e in earlier
                )
            )
        alone = 1.5
        if others_by_view is not None and others_by_view[-1]:
            alone = min(
                np.hypot((o[0] + o[2]) / 2 - cx, (o[1] + o[3]) / 2 - cy)
                for o in others_by_view[-1]
            )
        per.append(
            (
                min(ious),
                float(np.mean(ious)),
                max(shifts),
                float(np.mean(shifts)),
                alone,
            )
        )
    best = max(per, key=lambda v: v[0])
    stayed = sum(v[0] > 0.3 for v in per)
    return [best[0], best[1], min(stayed, 4) / 4, best[2], best[3], best[4]]


def dwell(found_by_view):
    persons = [[d["box"] for d in f if d["label"] == 1] for f in found_by_view]
    bags = [[d["box"] for d in f if d["label"] in {27, 31, 33}] for f in found_by_view]
    return np.asarray(stays(persons) + stays(bags, persons), dtype=np.float32)


def normalize(x):
    return x / max(float(np.linalg.norm(x)), 1e-12)


def pool(views, ordered=False):
    views = np.asarray(views, dtype=np.float32)
    if ordered:
        idx = np.rint(np.linspace(0, len(views) - 1, 4)).astype(int)
        return normalize(
            np.concatenate([views[idx].ravel(), views.mean(0), views.max(0)])
        )
    return normalize(
        np.concatenate([views[-1], views.mean(0), views.max(0), views[-1] - views[0]])
    )


def matrix(rows, recipe):
    """recipe: dict(backbone, alpha, ordered, endpoint, crop, geometry, dwell, reference, offset)."""
    if recipe.get("offset"):
        # Window features minus the same features of the camera's own reference
        # frame: a linear head then scores departure from the empty scene.
        plain = {k: v for k, v in recipe.items() if k != "offset"}
        refs = [
            {**row, "views": [row["reference"]] * len(row["views"])} for row in rows
        ]
        return matrix(rows, plain) - matrix(refs, plain)
    boxes_only = recipe["backbone"] == "none"
    full = (
        None
        if boxes_only
        else image_features(recipe["backbone"], "full", recipe.get("alpha", 0.0))
    )
    boxes = recipe.get("boxes", "frcnn")
    crop = (
        image_features(
            recipe.get("crop_backbone", recipe["backbone"]),
            "crop" if boxes == "frcnn" else "crop-yolo",
            0.0,
        )
        if recipe.get("crop")
        else None
    )
    found = detections(boxes) if recipe.get("geometry") or recipe.get("dwell") else None
    out = []
    for row in rows:
        views = row["views"]
        if boxes_only:
            parts = []
        elif recipe.get("endpoint"):
            parts = [full[views[-1]]]
        else:
            parts = [pool([full[p] for p in views], recipe.get("ordered", False))]
        if crop is not None:
            parts.append(pool([crop[p] for p in views], recipe.get("ordered", False)))
        if found is not None and recipe.get("geometry"):
            parts.append(geometry([found[p] for p in views]))
        if recipe.get("dwell"):
            parts.append(
                dwell([found[p] for p in views])
                * (1.0 if boxes_only else recipe.get("dwell_scale", 0.25))
            )
        if recipe.get("smoke_detector"):
            parts.append(
                smoking_detector_features(row) * recipe.get("detector_scale", 0.5)
            )
        if recipe.get("reference"):
            # Change against the camera's own earlier scene (label-free).
            ref = full[row["reference"]] if "reference" in row else full[views[0]]
            stack = np.stack([full[p] for p in views])
            parts.append(
                normalize(np.concatenate([stack[-1] - ref, stack.mean(0) - ref]))
            )
        if boxes_only:
            out.append(np.concatenate(parts))
        else:
            out.append(normalize(np.concatenate(parts)) if len(parts) > 1 else parts[0])
    return np.stack(out).astype(np.float32)


def recipe_name(recipe):
    bits = [recipe["backbone"]]
    if recipe.get("alpha"):
        bits.append(f"a{recipe['alpha']:g}")
    bits.append(
        "endpoint"
        if recipe.get("endpoint")
        else "ordered"
        if recipe.get("ordered")
        else "pool"
    )
    if recipe.get("crop"):
        bits.append(
            "crop"
            if recipe.get("crop_backbone", recipe["backbone"]) == recipe["backbone"]
            else f"crop:{recipe['crop_backbone']}"
        )
    if recipe.get("geometry"):
        bits.append("geometry")
    if recipe.get("reference"):
        bits.append("reference")
    if recipe.get("dwell"):
        bits.append("dwell")
    if recipe.get("offset"):
        bits.append("offset")
    if recipe.get("smoke_detector"):
        bits.append("smokedet")
    if recipe.get("boxes", "frcnn") != "frcnn":
        bits.append(recipe["boxes"])
    return "/".join(bits)
