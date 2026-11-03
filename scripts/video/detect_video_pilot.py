"""Shared low-score automatic detections for the causal video pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import pil_to_tensor

ROOT = Path(__file__).resolve().parents[2]


class Frames(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(ROOT / self.paths[index]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            return self.paths[index], pil_to_tensor(im).float() / 255, (w, h)


def collate(batch):
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    data = args.data.resolve()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = [
        json.loads(s)
        for name in ["windows.jsonl", "fresh_avss.jsonl"]
        for s in (data / name).read_text().splitlines()
    ]
    paths = {p for r in rows for p in r["views"]}
    for name in ["recordings.jsonl", "fresh_recordings.jsonl"]:
        paths.update(
            f["path"]
            for r in map(json.loads, (data / name).read_text().splitlines())
            for f in r["frames"]
        )
    output = data / "detections.jsonl"
    done = (
        {r["path"] for r in map(json.loads, output.read_text().splitlines())}
        if output.exists()
        else set()
    )
    weights = (
        ROOT / "data/models/detection/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    )
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None, min_size=640, max_size=1024
    )
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    model.eval().cuda()
    pending = sorted(paths - done)
    loader = DataLoader(
        Frames(pending),
        batch_size=args.batch_size,
        num_workers=8,
        collate_fn=collate,
        pin_memory=True,
    )
    with output.open("a") as stream, torch.inference_mode():
        for i, batch in enumerate(loader):
            predictions = model([im.cuda(non_blocking=True) for _, im, _ in batch])
            for (name, _, (w, h)), prediction in zip(batch, predictions):
                boxes = prediction["boxes"].cpu().tolist()
                labels = prediction["labels"].cpu().tolist()
                scores = prediction["scores"].cpu().tolist()
                found = [
                    {
                        "box": [x / s for x, s in zip(b, [w, h, w, h])],
                        "label": label,
                        "confidence": score,
                    }
                    for b, label, score in zip(boxes, labels, scores)
                    if label in {1, 27, 31, 33} and score >= 0.2
                ]
                stream.write(json.dumps({"path": name, "detections": found}) + "\n")
            stream.flush()
            if i % 40 == 0:
                print(
                    "DETECTED",
                    len(done) + min((i + 1) * args.batch_size, len(pending)),
                    "/",
                    len(paths),
                    flush=True,
                )
    (data / "detector.json").write_text(
        json.dumps(
            {
                "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "model": "Faster R-CNN ResNet50 FPN v2 COCO",
                "min_size": 640,
                "max_size": 1024,
                "low_threshold": 0.2,
                "new_track_threshold": 0.5,
                "images": len(paths),
                "precision": "float32; TF32 disabled",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
