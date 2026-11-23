"""Fine-tune YOLO26s on CCTV smoking/not-smoking head boxes, camera-held-out.

Cameras: c4 = train, c2 = dev, c1 = test (never used here). Three detectors:
  c4   trained on c4, scores c2 rows (out-of-camera for cross-fitting)
  c2   trained on c2, scores c4 rows
  all  trained on c4 + c2, scores every held-out row

--extra adds the smoking boxes of a second CCTV site (about 40 cameras, all
images contain smoking) to every fold's training set; outputs get an "x"
suffix (c4x, c2x, allx).
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, ROOT

SOURCE = ROOT / "data/raw/datasets/smoking/smoking-video.v1i.coco"
EXTRA = ROOT / "data/raw/datasets/smoking/smoking_cctv_image.v1i.coco"
CLASSES = {"Smoking": 0, "Not Smoking": 1, "smoking": 0}
WORK = OUT / "smoking_detector"


def camera(name):
    return name.split("_")[1]


def add_extra(root):
    for split in ["train", "valid", "test"]:
        coco = json.loads((EXTRA / split / "_annotations.coco.json").read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        by_image = {}
        for a in coco["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)
        for image in coco["images"]:
            link = root / "images/train" / image["file_name"]
            if not link.exists():
                os.symlink(EXTRA / split / image["file_name"], link)
            w, h = image["width"], image["height"]
            lines = []
            for a in by_image.get(image["id"], []):
                if names[a["category_id"]] not in CLASSES:
                    continue
                x, y, bw, bh = a["bbox"]
                lines.append(
                    f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                )
            (
                root / "labels/train" / (Path(image["file_name"]).stem + ".txt")
            ).write_text("\n".join(lines))


def place(source, target, side=640):
    """Copy a frame for training, shrunk to the training size: decoding
    full-HD frames every epoch made training several times slower."""
    from PIL import Image, ImageOps

    target = target.with_suffix(".jpg")
    if target.exists():
        return
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((side, side))
        image.save(target, quality=95)


def add_site_fit(root, half=None, background=600, hard_negatives=False):
    """Second-site smokers from the fit-side cameras only, plus ordinary
    footage as label-free background: frames of the fit-side ordinary
    windows and a sample of the other tasks' training frames. Without
    background a detector that has only seen smokers on HD cameras calls
    every head on an HD camera a smoker."""
    import random

    os.environ["F1_ROUND3"] = "1"
    from common import detector_half, load_rows

    rows = [
        r
        for r in load_rows()
        if r["split"] != "site_fit"
        or half is None
        or detector_half(r, "camera" if hard_negatives else "video") == half
    ]
    fit = {
        Path(r["views"][0]).name
        for r in rows
        if r["source"] == "smoking_site2" and r["split"] == "site_fit"
    }
    for kind in ["images", "labels"]:
        (root / kind / "train").mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid", "test"]:
        coco = json.loads((EXTRA / split / "_annotations.coco.json").read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        by_image = {}
        for a in coco["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)
        for image in coco["images"]:
            if image["file_name"] not in fit:
                continue
            place(
                EXTRA / split / image["file_name"],
                root / "images/train" / image["file_name"],
            )
            w, h = image["width"], image["height"]
            lines = []
            for a in by_image.get(image["id"], []):
                if names[a["category_id"]] not in CLASSES:
                    continue
                x, y, bw, bh = a["bbox"]
                lines.append(
                    f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                )
            (
                root / "labels/train" / (Path(image["file_name"]).stem + ".txt")
            ).write_text("\n".join(lines))
    ordinary = sorted(
        {
            r["views"][-1]
            for r in rows
            if r["task"] == "smoking"
            and r["split"] == "site_fit"
            and r["source"] in {"virat", "meva"}
        }
    )
    if hard_negatives:
        # People phoning, drinking and eating at CCTV distance: a hand at the
        # face without a cigarette. Two mid-clip frames per clip.
        ordinary += sorted(
            {
                r["views"][k]
                for r in rows
                if r["source"] == "xrf55" and r["split"] == "site_fit"
                for k in (2, 5)
            }
        )
    others = sorted(
        {
            r["views"][-1]
            for r in rows
            if r["task"] != "smoking" and r["split"] in {"train", "dev"}
        }
    )
    random.Random(0).shuffle(others)
    for n, name in enumerate(ordinary + others[:background]):
        stem = f"background_{n:05d}"
        place(ROOT / name, root / "images/train" / (stem + ".jpg"))
        (root / "labels/train" / (stem + ".txt")).write_text("")


def build(name, cameras):
    root = WORK / f"data_{name}"
    for split in ["train", "valid"]:
        coco = json.loads((SOURCE / split / "_annotations.coco.json").read_text())
        by_image = {}
        for a in coco["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)
        names = {c["id"]: c["name"] for c in coco["categories"]}
        for image in coco["images"]:
            cam = camera(image["file_name"])
            if cam == "c1":
                continue  # test camera
            subset = "train" if cam in cameras else "val"
            img_dir, lbl_dir = root / "images" / subset, root / "labels" / subset
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            link = img_dir / image["file_name"]
            if not link.exists():
                os.symlink(SOURCE / split / image["file_name"], link)
            w, h = image["width"], image["height"]
            lines = []
            for a in by_image.get(image["id"], []):
                label = names[a["category_id"]]
                if label not in CLASSES:
                    continue
                x, y, bw, bh = a["bbox"]
                lines.append(
                    f"{CLASSES[label]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                )
            (lbl_dir / (Path(image["file_name"]).stem + ".txt")).write_text(
                "\n".join(lines)
            )
    (root / "data.yaml").write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: smoking\n  1: not_smoking\n"
    )
    return root / "data.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=["c4", "c2", "all"], required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extra", action="store_true")
    parser.add_argument(
        "--site-fit",
        action="store_true",
        help="Add fit-side second-site smokers and ordinary background (suffix f)",
    )
    parser.add_argument(
        "--no-site1",
        action="store_true",
        help="With --site-fit and --fold all: leave site 1 out (name s2f)",
    )
    parser.add_argument(
        "--hard-negatives",
        action="store_true",
        help="With --site-fit: also hand-to-face clips as background (suffix g)",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    from ultralytics import YOLO

    cameras = {"c4": {"c4"}, "c2": {"c2"}, "all": {"c4", "c2"}}[args.fold]
    base_fold = args.fold
    if args.extra:
        args.fold += "x"
    if args.site_fit:
        letter = "g" if args.hard_negatives else "f"
        args.fold = "s2" + letter if args.no_site1 else args.fold + letter
    if args.imgsz != 640:
        args.fold += str(args.imgsz)
    data = build(args.fold, set() if args.no_site1 else cameras)
    if args.extra:
        add_extra(WORK / f"data_{args.fold}")
    if args.site_fit:
        # Fold detectors take half of the fit-side new-site rows each.
        add_site_fit(
            WORK / f"data_{args.fold}",
            {"c4": "a", "c2": "b"}.get(base_fold),
            hard_negatives=args.hard_negatives,
        )
    if base_fold == "all":
        # No held-out camera remains; validate on c2 only to monitor training.
        val = WORK / f"data_{args.fold}" / "images" / "val"
        val.mkdir(parents=True, exist_ok=True)
        train_c2 = [
            p
            for p in (WORK / f"data_{args.fold}/images/train").iterdir()
            if p.name.startswith("frame_") and camera(p.name) == "c2"
        ][:200]
        for p in train_c2:
            for kind, suffix in [("images", p.suffix), ("labels", ".txt")]:
                src = WORK / f"data_{args.fold}" / kind / "train" / (p.stem + suffix)
                dst = WORK / f"data_{args.fold}" / kind / "val" / (p.stem + suffix)
                (dst.parent).mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    os.symlink(os.path.realpath(src) if kind == "images" else src, dst)
    if args.resume:
        YOLO(str(WORK / "runs" / args.fold / "weights/last.pt")).train(resume=True)
        return
    model = YOLO(str(ROOT / "data/models/stage1/yolo26s.pt"))
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=32 if args.imgsz == 640 else 16,
        device=0,
        project=str(WORK / "runs"),
        name=args.fold,
        exist_ok=True,
        workers=16,
        patience=0,
        seed=0,
        deterministic=True,
        plots=False,
        verbose=False,
    )


if __name__ == "__main__":
    main()
