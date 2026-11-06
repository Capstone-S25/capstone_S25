"""Inference feature representations and fitted head evaluation."""

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn


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

    def area(c):
        return max(0, c[2] - c[0]) * max(0, c[3] - c[1])

    return intersection / max(area(a) + area(b) - intersection, 1e-12)


def geometry(ordered_detections):
    """Automatic sparse person/item geometry; never an ownership ground truth."""
    frames = []
    previous = []
    for found in ordered_detections:
        persons = [d["box"] for d in found if d["label"] == 1]
        bags = [d["box"] for d in found if d["label"] in {27, 31, 33}]

        def centers(boxes):
            return np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes])

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


def matrix(rows, recipe, features, detections):
    alpha = float(recipe.rsplit("a", 1)[1])
    output = []
    for row in rows:
        if recipe.startswith("endpoint"):
            full = features["full", alpha][row["views"][-1]]
        else:
            full = represent(
                [features["full", alpha][p] for p in row["views"]],
                ordered="ordered" in recipe,
            )
        if "crop" in recipe:
            local = represent(
                [features["crop", 0][p] for p in row["views"]],
                ordered="ordered" in recipe,
            )
            full = normalize(np.concatenate([full, local]))
        if "geometry" in recipe:
            full = normalize(
                np.concatenate([full, geometry([detections[p] for p in row["views"]])])
            )
        output.append(full)
    return np.stack(output).astype(np.float32)


def contextual_box(box, padding=0.2):
    x1, y1, x2, y2 = box
    dx, dy = ((x2 - x1) * padding, (y2 - y1) * padding)
    return [max(0, x1 - dx), max(0, y1 - dy), min(1, x2 + dx), min(1, y2 + dy)]


class Views:
    def __init__(self, paths, encoder, detections=None):
        self.paths, self.encoder, self.detections = (paths, encoder, detections)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        name = self.paths[i]
        if self.detections is None:
            return self.encoder.image_tensor(name)
        found = self.detections[name]
        boxes = []
        for labels in [{1}, {27, 31, 33}]:
            values = [d for d in found if d["label"] in labels]
            if values:
                boxes.append(
                    contextual_box(max(values, key=lambda d: d["confidence"])["box"])
                )
        if not boxes:
            return self.encoder.image_tensor(name)
        with Image.open(name) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            box = [
                min((b[0] for b in boxes)),
                min((b[1] for b in boxes)),
                max((b[2] for b in boxes)),
                max((b[3] for b in boxes)),
            ]
            w, h = im.size
            im = im.crop(tuple((v * s for v, s in zip(box, [w, h, w, h]))))
            canvas = Image.new("RGB", (max(im.size), max(im.size)), (0, 0, 0))
            canvas.paste(
                im, ((canvas.width - im.width) // 2, (canvas.height - im.height) // 2)
            )
            canvas.thumbnail((768, 768))
            if self.encoder.name == "mobileclip2-s2":
                return self.encoder.processor(canvas)
            return self.encoder.processor(images=canvas, return_tensors="pt")[
                "pixel_values"
            ][0]


def predict(head, values):
    if head["kind"] == "linear":
        return (values @ head["weight"].numpy().T + head["bias"].numpy())[:, 0]
    model = nn.Sequential(
        nn.Linear(head["input_dimension"], 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )
    model.load_state_dict(head["state_dict"])
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(values))[:, 0].numpy()


def temporal_indices(timestamps, count):
    if timestamps is None:
        return np.rint(np.linspace(0, count - 1, 4)).astype(int).tolist()
    times = np.asarray(timestamps)
    targets = np.linspace(max(times[0], times[-1] - 20), times[-1], 4)
    return [int(np.argmin(abs(times - value))) for value in targets]


def temporal_support(task, timestamps, count):
    if task == "littering":
        return count >= 8
    if task not in {"loitering", "unattended_items"}:
        return True
    valid = bool(
        count >= 4
        and timestamps is not None
        and (len(timestamps) == count)
        and (len(set(timestamps)) >= 4)
        and (timestamps[-1] - timestamps[0] >= 19.5)
    )
    if not valid:
        return False
    indices = temporal_indices(timestamps, count)
    selected = [timestamps[i] for i in indices]
    return (
        len(set(indices)) == 4
        and selected[-1] - selected[0] >= 19.5
        and (selected[0] >= timestamps[-1] - 20.5)
    )
