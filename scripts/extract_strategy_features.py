"""Float32 image representations for fixed crop and weight-interpolation studies."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from anomaly_eval.data import image_pixel_sha256
from anomaly_eval.training.clip import Encoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/strategy_study_20261016"


def contextual_box(box, padding=0.2):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return [max(0, x1 - dx), max(0, y1 - dy), min(1, x2 + dx), min(1, y2 + dy)]


class Views(Dataset):
    def __init__(self, paths, encoder, detections=None):
        self.paths, self.encoder, self.detections = paths, encoder, detections

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        name = self.paths[i]
        if self.detections is None:
            return self.encoder.image_tensor(ROOT / name)
        found = self.detections[name]
        boxes = []
        for labels in [{1}, {27, 31, 33}]:
            values = [d for d in found if d["label"] in labels]
            if values:
                boxes.append(
                    contextual_box(max(values, key=lambda d: d["confidence"])["box"])
                )
        if not boxes:
            return self.encoder.image_tensor(ROOT / name)
        with Image.open(ROOT / name) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            # Predicted contextual person/object regions; no anomaly annotations.
            box = [
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            ]
            w, h = im.size
            im = im.crop(tuple(v * s for v, s in zip(box, [w, h, w, h])))
            canvas = Image.new("RGB", (max(im.size), max(im.size)), (0, 0, 0))
            canvas.paste(
                im, ((canvas.width - im.width) // 2, (canvas.height - im.height) // 2)
            )
            canvas.thumbnail((768, 768))
            if self.encoder.name == "mobileclip2-s2":
                return self.encoder.processor(canvas)
            return self.encoder.processor(images=canvas, return_tensors="pt")[
                "pixel_values"
            ][0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", choices=["mobileclip2-s2", "siglip2-b16-224"], required=True
    )
    p.add_argument("--stage", choices=["full", "crop"], default="full")
    p.add_argument("--manifest", type=Path, default=DATA / "samples.jsonl")
    p.add_argument("--tag", default="main")
    p.add_argument("--detections", type=Path, default=DATA / "detections.jsonl")
    p.add_argument("--alphas", type=float, nargs="+", default=[0, 0.25, 0.5, 1])
    args = p.parse_args()
    if args.stage == "crop":
        args.alphas = [0]
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    paths = sorted({p for r in rows for p in r["views"]})
    expected = {p: h for r in rows for p, h in zip(r["views"], r["pixel_hashes"])}
    with ThreadPoolExecutor(12) as pool:
        for name, actual in zip(
            paths, pool.map(lambda name: image_pixel_sha256(ROOT / name), paths)
        ):
            if actual != expected[name]:
                raise ValueError("Image changed since manifest preparation: " + name)
    encoder = Encoder(args.model, "cuda").eval()
    checkpoint = torch.load(
        ROOT
        / "results/clip_finetuning_20261015/seed_42"
        / args.model
        / "finetuned_best.pt",
        weights_only=True,
        map_location="cpu",
    )
    parameters = dict(encoder.named_parameters())
    original = {
        name: parameters[name].detach().cpu().clone()
        for name in checkpoint["parameters"]
        if name.startswith("backbone.")
    }
    detections = None
    if args.stage == "crop":
        detections = {
            r["path"]: r["detections"]
            for r in map(json.loads, args.detections.read_text().splitlines())
        }
        if not all(p in detections for p in paths):
            raise ValueError("Detector pass is incomplete")
    output = DATA / "features" / args.model
    output.mkdir(parents=True, exist_ok=True)
    for alpha in args.alphas:
        out = output / f"{args.tag}_{args.stage}_alpha_{alpha:g}.npz"
        identity = {
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "model": args.model,
            "base_revision": encoder.revision,
            "alpha": alpha,
            "stage": args.stage,
            "precision": "float32; TF32 disabled",
            "paths": len(paths),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        if out.exists():
            old = json.loads(out.with_suffix(".json").read_text())
            assert all(
                old[k] == identity[k]
                for k in ["manifest_sha256", "base_revision", "alpha", "stage"]
            )
            print("cached", out.name, flush=True)
            continue
        with torch.no_grad():
            for name, base in original.items():
                parameters[name].copy_(
                    torch.lerp(base, checkpoint["parameters"][name], alpha).to("cuda")
                )
        loader = DataLoader(
            Views(paths, encoder, detections),
            batch_size=48,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        values = []
        with torch.inference_mode():
            for i, batch in enumerate(loader):
                batch = batch.to("cuda", non_blocking=True)
                value = (
                    encoder.backbone.encode_image(batch, normalize=True)
                    if args.model == "mobileclip2-s2"
                    else F.normalize(
                        encoder.backbone.get_image_features(pixel_values=batch), dim=-1
                    )
                )
                values.append(value.cpu().numpy())
                if i % 40 == 0:
                    print(
                        args.model,
                        args.stage,
                        alpha,
                        "images",
                        min((i + 1) * 48, len(paths)),
                        "/",
                        len(paths),
                        flush=True,
                    )
        np.savez_compressed(out, paths=np.array(paths), features=np.concatenate(values))
        out.with_suffix(".json").write_text(json.dumps(identity, indent=2) + "\n")
        print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
