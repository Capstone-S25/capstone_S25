"""YOLO26s person/bag boxes for every benchmark image (the tracker's detector).

Stored in the Faster R-CNN record format (COCO-91 labels, normalized boxes) so
crops and geometry can be built from the detector the runtime already runs.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, ROOT, load_rows

COCO91 = {0: 1, 24: 27, 26: 31, 28: 33}


def main():
    rows = load_rows()
    paths = sorted(
        {p for r in rows for p in r["views"]}
        | {r["reference"] for r in rows if "reference" in r}
    )
    model = YOLO(str(ROOT / "data/models/stage1/yolo26s.pt"))
    out = OUT / "yolo26s_detections.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["path"] for line in out.read_text().splitlines()}
    todo = [p for p in paths if p not in done]
    with out.open("a") as f:
        for start in range(0, len(todo), 64):
            batch = todo[start : start + 64]
            images = []
            for name in batch:
                with Image.open(ROOT / name) as im:
                    images.append(
                        np.asarray(ImageOps.exif_transpose(im).convert("RGB"))[
                            :, :, ::-1
                        ]
                    )
            results = model.predict(
                images,
                imgsz=640,
                conf=0.1,
                classes=list(COCO91),
                verbose=False,
                device=0,
            )
            for name, r in zip(batch, results):
                h, w = r.orig_shape
                found = [
                    {
                        "box": [float(v) / s for v, s in zip(b, [w, h, w, h])],
                        "label": COCO91[int(c)],
                        "confidence": float(q),
                    }
                    for b, c, q in zip(
                        r.boxes.xyxy.cpu().numpy(),
                        r.boxes.cls.cpu().numpy(),
                        r.boxes.conf.cpu().numpy(),
                    )
                ]
                f.write(json.dumps({"path": name, "detections": found}) + "\n")
            if start % 3200 == 0:
                print(start, "/", len(todo), flush=True)
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
