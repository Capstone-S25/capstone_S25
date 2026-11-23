"""Person-centred embeddings: one vector per detected person, not per frame.

A cigarette is a few pixels of a 224-px frame embedding. Cropping each
detected person (YOLO26s box, 15% padding, letterboxed) gives the encoder the
whole 224 px for one person, the way recent smoking detectors work
(person -> upper-body crop -> classifier).

Output `features/<name>_persons.npz`: `paths`, `start` (first crop of each
path; `start[i+1] - start[i]` crops belong to path i), `boxes` and `features`.
"""

from __future__ import annotations

import os

os.environ["F1_ROUND3"] = "1"

import argparse
import time

import numpy as np
import torch
from common import OUT, ROOT, load_rows
from extract import letterbox, load
from PIL import Image, ImageOps
from representations import detections
from torch.utils.data import DataLoader, Dataset

MAX_PERSONS = 6


def person_boxes(found, limit=MAX_PERSONS, padding=0.15):
    """The largest confident people in a frame, padded, as 0-1 fractions."""
    people = [d for d in found if d["label"] == 1 and d["confidence"] >= 0.35]
    people.sort(
        key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
    )
    out = []
    for d in people[:limit]:
        x1, y1, x2, y2 = d["box"]
        dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
        out.append([max(0, x1 - dx), max(0, y1 - dy), min(1, x2 + dx), min(1, y2 + dy)])
    return out


class Crops(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, box = self.items[i]
        with Image.open(ROOT / name) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        w, h = image.size
        image = image.crop(tuple(v * s for v, s in zip(box, [w, h, w, h])))
        image = letterbox(image)
        image.thumbnail((1024, 1024))
        return self.transform(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="siglip:google/siglip2-base-patch16-224")
    parser.add_argument("--name", default="siglip2-b")
    parser.add_argument("--tasks", nargs="+", default=["smoking"])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = [r for r in load_rows() if r["task"] in args.tasks]
    paths = sorted({p for r in rows for p in r["views"]})
    found = detections("yolo")
    items, start, boxes = [], [0], []
    for name in paths:
        for box in person_boxes(found.get(name, [])):
            items.append((name, box))
            boxes.append(box)
        start.append(len(items))
    encode, transform = load(args.model)
    loader = DataLoader(
        Crops(items, transform), batch_size=128, num_workers=16, pin_memory=True
    )
    values, began = [], time.time()
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            values.append(encode(batch.cuda(non_blocking=True)).float().cpu().numpy())
            if i % 50 == 0:
                print(
                    i * 128, "/", len(items), f"{time.time() - began:.0f}s", flush=True
                )
    out = OUT / "features" / f"{args.name}_persons.npz"
    np.savez(
        out,
        paths=np.array(paths),
        start=np.array(start),
        boxes=np.array(boxes, np.float32),
        features=np.concatenate(values).astype(np.float16),
    )
    print("COMPLETE", out, len(paths), "images", len(items), "persons", flush=True)


if __name__ == "__main__":
    main()
