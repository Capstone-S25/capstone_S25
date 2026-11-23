"""Faster R-CNN boxes for the new-site loitering and unattended-item windows
(views and reference frames), in the October record format."""

import json
import os

os.environ["F1_ROUND3"] = "1"

import torch
from common import OUT, REFERENCE_TASKS, ROOT, load_rows
from PIL import Image, ImageOps
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import pil_to_tensor

WEIGHTS = (
    ROOT
    / ".local/worktrees/vision-app-integration/models/detectors/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
)


def main():
    rows = [
        r
        for r in load_rows()
        if r["split"].startswith("site_") and r["task"] in REFERENCE_TASKS
    ]
    paths = sorted({p for r in rows for p in [*r["views"], r["reference"]]})
    target = OUT / "round3/frcnn_detections.jsonl"
    done = set()
    if target.exists():
        done = {json.loads(line)["path"] for line in target.read_text().splitlines()}
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    model.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location="cpu"))
    model.eval().cuda()
    with target.open("a") as f, torch.inference_mode():
        for name in paths:
            if name in done:
                continue
            with Image.open(ROOT / name) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((1280, 1280))
                w, h = im.size
                x = pil_to_tensor(im).cuda().float() / 255
            p = model([x])[0]
            found = [
                {
                    "box": [float(v) / s for v, s in zip(b, [w, h, w, h])],
                    "label": int(c),
                    "confidence": float(q),
                }
                for b, c, q in zip(
                    p["boxes"].cpu(), p["labels"].cpu(), p["scores"].cpu()
                )
                if int(c) in {1, 27, 31, 33} and float(q) >= 0.5
            ]
            f.write(json.dumps({"path": name, "detections": found}) + "\n")
    print("done", len(paths))


if __name__ == "__main__":
    main()
