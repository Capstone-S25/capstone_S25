"""Frozen image embeddings for every benchmark image: full frame and crop.

Full frames are letterboxed to a square (no scene content is cropped away).
Crops use the same label-free Faster R-CNN person/item union, with 20%
context padding, as the October crop features.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from common import OUT, ROOT, load_rows
from PIL import Image, ImageOps
from representations import detections
from torch.utils.data import DataLoader, Dataset


def contextual_box(box, padding=0.2):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return [max(0, x1 - dx), max(0, y1 - dy), min(1, x2 + dx), min(1, y2 + dy)]


def crop_box(found):
    boxes = []
    for labels in [{1}, {27, 31, 33}]:
        values = [d for d in found if d["label"] in labels]
        if values:
            boxes.append(
                contextual_box(max(values, key=lambda d: d["confidence"])["box"])
            )
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def letterbox(image):
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


class Images(Dataset):
    def __init__(self, paths, transform, crop):
        self.paths, self.transform, self.crop = paths, transform, crop
        self.found = (
            detections("yolo" if crop == "crop-yolo" else "frcnn") if crop else None
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        name = self.paths[i]
        with Image.open(ROOT / name) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        if self.crop:
            box = crop_box(self.found.get(name, []))
            if box is not None:
                w, h = image.size
                image = image.crop(tuple(v * s for v, s in zip(box, [w, h, w, h])))
        image = letterbox(image)
        image.thumbnail((1024, 1024))
        return self.transform(image)


def load(spec):
    """Returns (encode_fn, transform). spec: 'open_clip:<hub id>' or 'hf:<id>'."""
    kind, name = spec.split(":", 1)
    if kind == "open_clip":
        import open_clip

        model, _, transform = open_clip.create_model_and_transforms(f"hf-hub:{name}")
        model = model.eval().cuda()

        def encode(x):
            return model.encode_image(x)

        return encode, transform
    if kind == "siglip":
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(name, torch_dtype=torch.float32).eval().cuda()
        processor = AutoProcessor.from_pretrained(name)

        def transform(image):
            return processor(images=image, return_tensors="pt")["pixel_values"][0]

        def encode(x):
            return model.get_image_features(pixel_values=x)

        return encode, transform
    if kind == "dino":
        from transformers import AutoImageProcessor, AutoModel

        model = AutoModel.from_pretrained(name, torch_dtype=torch.float32).eval().cuda()
        processor = AutoImageProcessor.from_pretrained(name)
        size = processor.crop_size["height"] if hasattr(processor, "crop_size") else 224

        def transform(image):
            return processor(
                images=image.resize((size, size)),
                do_center_crop=False,
                do_resize=False,
                return_tensors="pt",
            )["pixel_values"][0]

        def encode(x):
            out = model(pixel_values=x).last_hidden_state
            # CLS plus mean patch token (registers excluded), a standard probe input.
            registers = getattr(model.config, "num_register_tokens", 0)
            return torch.cat([out[:, 0], out[:, 1 + registers :].mean(1)], dim=-1)

        return encode, transform
    raise ValueError(spec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stages", nargs="+", default=["full", "crop", "crop-yolo"])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = load_rows()
    paths = sorted(
        {p for r in rows for p in r["views"]}
        | {r["reference"] for r in rows if "reference" in r}
    )
    encode, transform = load(args.model)
    (OUT / "features").mkdir(parents=True, exist_ok=True)
    for stage in args.stages:
        out = OUT / "features" / f"{args.name}_{stage}.npz"
        old_paths, old_values = [], []
        if out.exists():
            old = np.load(out)
            old_paths, old_values = old["paths"].tolist(), [old["features"]]
        todo = sorted(set(paths) - set(old_paths))
        if not todo:
            print("cached", out, flush=True)
            continue
        loader = DataLoader(
            Images(todo, transform, stage if stage != "full" else None),
            batch_size=args.batch_size,
            num_workers=16,
            pin_memory=True,
        )
        values, start = [], time.time()
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                values.append(
                    encode(batch.cuda(non_blocking=True)).float().cpu().numpy()
                )
                if i % 50 == 0:
                    print(
                        args.name,
                        stage,
                        (i + 1) * args.batch_size,
                        "/",
                        len(todo),
                        f"{time.time() - start:.0f}s",
                        flush=True,
                    )
        new = np.concatenate(values).astype(np.float16)
        np.savez(
            out,
            paths=np.array(old_paths + todo),
            features=np.concatenate(old_values + [new]),
        )
        print("COMPLETE", out, f"{time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
