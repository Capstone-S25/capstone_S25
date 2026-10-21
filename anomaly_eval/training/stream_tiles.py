"""Fixed-view detection fusion for controlled continuous-video comparisons."""

from __future__ import annotations

import math

from anomaly_eval.training.strategies import overlap

TILES = (
    (0.0, 0.0, 1.0, 1.0),
    (0.0, 0.0, 0.6, 0.6),
    (0.4, 0.0, 1.0, 0.6),
    (0.0, 0.4, 0.6, 1.0),
    (0.4, 0.4, 1.0, 1.0),
)
MODES = ("full", "item_tiles", "all_tiles")


def tile_pixels(tile, width, height):
    if width <= 0 or height <= 0 or tuple(tile) not in TILES:
        raise ValueError("Invalid image size or unregistered tile")
    return tuple(round(v * s) for v, s in zip(tile, (width, height, width, height)))


def global_box(box, pixel_tile, width, height):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("Invalid detector box")
    result = [
        max(0.0, min(1.0, (float(v) + offset) / size))
        for v, offset, size in zip(
            box,
            (pixel_tile[0], pixel_tile[1], pixel_tile[0], pixel_tile[1]),
            (width, height, width, height),
        )
    ]
    if result[2] <= result[0] or result[3] <= result[1]:
        return None
    return result


def fuse(views, mode, nms_iou=0.5):
    """Class-wise NMS; item-only tiles keep the full-image person proposals."""
    if mode not in MODES:
        raise ValueError("Unregistered fusion arm")
    if len(views) != len(TILES) or {tuple(v["tile"]) for v in views} != set(TILES):
        raise ValueError("Need exactly the five registered views, without aliases")
    selected = []
    for view in views:
        full = tuple(view["tile"]) == TILES[0]
        for detection in view["detections"]:
            if detection["label"] not in {1, 1001}:
                raise ValueError("Unknown detection class")
            if (
                full
                or mode == "all_tiles"
                or (mode == "item_tiles" and detection["label"] == 1001)
            ):
                selected.append(detection)
    selected.sort(key=lambda r: (-r["confidence"], r["label"], tuple(r["box"])))
    kept = []
    for row in selected:
        if not any(
            row["label"] == k["label"] and overlap(row["box"], k["box"]) > nms_iou
            for k in kept
        ):
            kept.append(row)
    return kept
