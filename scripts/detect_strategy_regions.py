"""Cache fixed automatic detections; no anomaly labels or ground-truth boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import pil_to_tensor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/strategy_study_20261016"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, nargs="+", default=[DATA / "samples.jsonl"]
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache", type=Path, default=DATA / "detections.jsonl")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    rows = [
        json.loads(line)
        for manifest in args.manifest
        for line in manifest.read_text().splitlines()
    ]
    images = sorted({p for r in rows for p in r["views"]})
    path = args.cache
    done = (
        {json.loads(line)["path"] for line in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    weights = (
        ROOT / "data/models/detection/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    )
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    model.eval().to("cuda")
    pending = [p for p in images if p not in done]
    with path.open("a") as output, torch.inference_mode():
        for start in range(0, len(pending), args.batch_size):
            batch_paths = pending[start : start + args.batch_size]
            tensors, dimensions = [], []
            for name in batch_paths:
                with Image.open(ROOT / name) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((1280, 1280))
                    dimensions.append(im.size)
                    tensors.append(
                        pil_to_tensor(im).to("cuda", dtype=torch.float32) / 255
                    )
            predictions = model(tensors)
            for name, (w, h), prediction in zip(batch_paths, dimensions, predictions):
                detections = []
                for box, label, score in zip(
                    prediction["boxes"].cpu(),
                    prediction["labels"].cpu(),
                    prediction["scores"].cpu(),
                ):
                    if int(label) not in {1, 27, 31, 33} or float(score) < 0.5:
                        continue
                    detections.append(
                        {
                            "box": [
                                float(x) / scale for x, scale in zip(box, [w, h, w, h])
                            ],
                            "label": int(label),
                            "confidence": float(score),
                        }
                    )
                output.write(
                    json.dumps({"path": name, "detections": detections}) + "\n"
                )
            output.flush()
            if start % 200 == 0:
                print(
                    "detected",
                    len(images) - len(pending) + start + len(batch_paths),
                    "/",
                    len(images),
                    flush=True,
                )
    metadata = {
        "detector": "torchvision Faster R-CNN ResNet50 FPN v2 COCO",
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "threshold": 0.5,
        "coordinate_policy": "normalized EXIF-transposed RGB",
        "class_ids": {"person": 1, "backpack": 27, "handbag": 31, "suitcase": 33},
        "anomaly_labels_used": False,
        "images": len(images),
    }
    metadata_path = (
        DATA / "detector.json"
        if args.cache == DATA / "detections.jsonl"
        else args.cache.with_suffix(".metadata.json")
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print("DETECTIONS COMPLETE", len(images), flush=True)


if __name__ == "__main__":
    main()
