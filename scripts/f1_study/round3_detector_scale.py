"""Smoking detector at different input sizes on new-site frames.

Reports, per detector and size, the image-level top "smoking" confidence:
AP on CCTV camera c1 (held-out camera of site 1), recall on the second site's
shift images and the share of ordinary frames flagged, at the confidence that
flags 2% of the fit-side ordinary frames.
"""

import os

os.environ["F1_ROUND3"] = "1"
import json
import sys

import numpy as np
from common import OUT, ROOT, load_rows
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score
from ultralytics import YOLO

RUNS = OUT / "smoking_detector/runs"


def top_conf(model, paths, size):
    out = {}
    for start in range(0, len(paths), 32):
        batch = paths[start : start + 32]
        images = []
        for name in batch:
            with Image.open(ROOT / name) as im:
                images.append(
                    np.asarray(ImageOps.exif_transpose(im).convert("RGB"))[:, :, ::-1]
                )
        for name, r in zip(
            batch, model.predict(images, imgsz=size, conf=0.02, verbose=False, device=0)
        ):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            out[name] = float(conf[cls == 0].max()) if (cls == 0).any() else 0.0
    return out


def main():
    folds, sizes = sys.argv[1].split(","), [int(v) for v in sys.argv[2].split(",")]
    rows = [r for r in load_rows() if r["task"] == "smoking"]
    sets = {
        "c1": [
            r for r in rows if r["source"] == "smoking_cctv" and r["split"] == "test"
        ],
        "site2": [
            r
            for r in rows
            if r["source"] == "smoking_site2" and r["split"] == "site_shift"
        ],
        "normal_fit": [
            r
            for r in rows
            if r["source"] in {"virat", "meva"} and r["split"] == "site_fit"
        ],
        "normal_shift": [
            r
            for r in rows
            if r["source"] in {"virat", "meva", "avenue"} and r["split"] == "site_shift"
        ],
    }
    paths = sorted({r["views"][-1] for v in sets.values() for r in v})
    result = {}
    for fold in folds:
        model = YOLO(str(RUNS / fold / "weights/last.pt"))
        for size in sizes:
            conf = top_conf(model, paths, size)
            s = {
                k: np.array([conf[r["views"][-1]] for r in v]) for k, v in sets.items()
            }
            y = np.array([r["target"] for r in sets["c1"]])
            cut = float(np.quantile(s["normal_fit"], 0.98))
            cut = max(cut, 0.05)
            result[f"{fold}@{size}"] = {
                "c1_ap": float(average_precision_score(y, s["c1"])),
                "cut": cut,
                "c1_recall": float((s["c1"][y == 1] >= cut).mean()),
                "c1_fpr": float((s["c1"][y == 0] >= cut).mean()),
                "site2_recall": float((s["site2"] >= cut).mean()),
                "normal_shift_flagged": float((s["normal_shift"] >= cut).mean()),
                "site2_recall@0.25": float((s["site2"] >= 0.25).mean()),
                "normal_shift@0.25": float((s["normal_shift"] >= 0.25).mean()),
            }
            print(
                fold,
                size,
                {k: round(v, 3) for k, v in result[f"{fold}@{size}"].items()},
                flush=True,
            )
    (OUT / "round3" / f"detector_scale_{'_'.join(folds)}.json").write_text(
        json.dumps(result, indent=1)
    )


if __name__ == "__main__":
    main()
