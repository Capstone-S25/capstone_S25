"""Run the three camera-held-out smoking detectors on every smoking image.

Output per image and detector: top smoking / not-smoking confidences and the
number of smoking boxes >= 0.25. Rows later read the detector that did not
see their camera (c4 rows -> detector c2, c2 rows -> detector c4, all other
rows -> detector all).
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, ROOT, load_rows

RUNS = OUT / "smoking_detector/runs"


def main():
    rows = [r for r in load_rows() if r["task"] == "smoking"]
    paths = sorted({p for r in rows for p in r["views"]})
    result = {}
    for fold in ["c4", "c2", "all"]:
        model = YOLO(str(RUNS / fold / "weights/last.pt"))
        values = {}
        for start in range(0, len(paths), 64):
            batch = paths[start : start + 64]
            images = []
            for name in batch:
                with Image.open(ROOT / name) as im:
                    images.append(
                        np.asarray(ImageOps.exif_transpose(im).convert("RGB"))[
                            :, :, ::-1
                        ]
                    )
            for name, r in zip(
                batch,
                model.predict(images, imgsz=640, conf=0.05, verbose=False, device=0),
            ):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy()
                smoke, other = conf[cls == 0], conf[cls == 1]
                values[name] = [
                    float(smoke.max()) if len(smoke) else 0.0,
                    float(other.max()) if len(other) else 0.0,
                    float((smoke >= 0.25).sum()),
                ]
        result[fold] = values
        print(fold, "done", flush=True)
    (OUT / "smoking_detector_scores.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
