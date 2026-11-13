"""October-identical Faster R-CNN boxes for the per-video reference frames."""

import json

import torch
from common import OUT, ROOT, load_rows
from PIL import Image, ImageOps
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import pil_to_tensor

WEIGHTS = (
    ROOT
    / ".local/worktrees/vision-app-integration/models/detectors/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
)

rows = load_rows()
paths = sorted({r["reference"] for r in rows if "reference" in r})
model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
model.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location="cpu"))
model.eval().cuda()
with open(OUT / "frcnn_reference_detections.jsonl", "w") as f, torch.inference_mode():
    for name in paths:
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
            for b, c, q in zip(p["boxes"].cpu(), p["labels"].cpu(), p["scores"].cpu())
            if int(c) in {1, 27, 31, 33} and float(q) >= 0.5
        ]
        f.write(json.dumps({"path": name, "detections": found}) + "\n")
print("done", len(paths))
