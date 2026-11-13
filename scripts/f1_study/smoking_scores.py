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
    """--suffix x scores the detectors trained with the second CCTV site;
    --other-tasks adds the other tasks' images for the full detector only."""
    target = OUT / "smoking_detector_scores.json"
    suffix = sys.argv[sys.argv.index("--suffix") + 1] if "--suffix" in sys.argv else ""
    result = json.loads(target.read_text()) if target.exists() else {}
    rows = load_rows()
    smoking = sorted({p for r in rows if r["task"] == "smoking" for p in r["views"]})
    others = sorted({p for r in rows for p in r["views"]} - set(smoking))
    plan = {"all" + suffix: smoking + others}
    if "--other-tasks" not in sys.argv:
        plan = {"c4" + suffix: smoking, "c2" + suffix: smoking, **plan}
    for fold, wanted in plan.items():
        paths = [p for p in wanted if p not in result.get(fold, {})]
        if not paths:
            continue
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
        result.setdefault(fold, {}).update(values)
        print(fold, "done", flush=True)
    target.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
