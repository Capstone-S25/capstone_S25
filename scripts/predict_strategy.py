"""Run the exploratory development-selected strategy bundle on ordered images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import pil_to_tensor

from anomaly_eval.runtime import resolve_device
from anomaly_eval.training.clip import Encoder
from anomaly_eval.training.data import TASKS
from anomaly_eval.training.predict import temporal_indices, temporal_support
from scripts.evaluate_strategy_transfer import matrix
from scripts.extract_strategy_features import Views
from scripts.run_strategy_screen import predict

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/strategy_study_20261016"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=["mobileclip2-s2", "siglip2-b16-224"]
    )
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--timestamps", type=float, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--operating-point",
        choices=["raw", "dev_max_f1", "dev_fpr_budget"],
        default="dev_max_f1",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not all(p.is_file() for p in args.images):
        parser.error("Every image must exist")
    if args.timestamps is not None and (
        len(args.timestamps) != len(args.images)
        or not np.isfinite(args.timestamps).all()
        or any(a > b for a, b in zip(args.timestamps, args.timestamps[1:]))
    ):
        parser.error("Timestamps must be finite, chronological and match the images")
    directory = RUN / args.model
    state = json.loads((directory / "transfer_evaluation/state.json").read_text())
    if state["status"] != "complete":
        raise ValueError("Complete the matching quality evaluation first")
    provenance = json.loads(
        (ROOT / "results/clip_finetuning_20261015/provenance.json").read_text()
    )
    if (
        hashlib.sha256((ROOT / provenance["definitions_path"]).read_bytes()).hexdigest()
        != provenance["definitions_sha256"]
    ):
        raise ValueError("Target definitions changed since training")
    policy = json.loads(
        (directory / "transfer_evaluation/frozen_policies.json").read_text()
    )
    bundle = torch.load(
        directory / "selected_heads.pt", map_location="cpu", weights_only=True
    )
    if bundle["model"] != args.model:
        raise ValueError("Model/checkpoint mismatch")
    device = resolve_device(args.device)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    paths = [str(p.resolve()) for p in args.images]
    weights = (
        ROOT / "data/models/detection/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
    )
    detector = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    detector.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    detector.eval().to(device)
    detections = {}
    with torch.inference_mode():
        for name in dict.fromkeys(paths):
            with Image.open(name) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((1280, 1280))
                w, h = im.size
                image = pil_to_tensor(im).to(device, dtype=torch.float32) / 255
            value = detector([image])[0]
            detections[name] = [
                {
                    "box": [float(v) / s for v, s in zip(box, [w, h, w, h])],
                    "label": int(label),
                    "confidence": float(score),
                }
                for box, label, score in zip(
                    value["boxes"].cpu(), value["labels"].cpu(), value["scores"].cpu()
                )
                if int(label) in {1, 27, 31, 33} and float(score) >= 0.5
            ]
    del detector
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    encoder = Encoder(args.model, device).eval()
    previous = torch.load(
        ROOT
        / "results/clip_finetuning_20261015/seed_42"
        / args.model
        / "finetuned_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    parameters = dict(encoder.named_parameters())
    original = {
        name: parameters[name].detach().cpu().clone()
        for name in previous["parameters"]
        if name.startswith("backbone.")
    }
    alphas = sorted(
        {
            float(v["selected"]["selection"]["strategy"].rsplit("a", 1)[1])
            for v in policy.values()
        }
        | {0.0}
    )
    unique = list(dict.fromkeys(paths))
    full_images = torch.stack(
        [Views(unique, encoder)[i] for i in range(len(unique))]
    ).to(device)
    crop_images = torch.stack(
        [Views(unique, encoder, detections)[i] for i in range(len(unique))]
    ).to(device)
    features = {}
    with torch.inference_mode():
        for alpha in alphas:
            for name, base in original.items():
                parameters[name].copy_(
                    torch.lerp(base, previous["parameters"][name], alpha).to(device)
                )
            batch = full_images if alpha else torch.cat([full_images, crop_images])
            pooled = encoder.encode(batch, [1] * len(batch)).cpu().numpy()
            # For single views the first block equals the normalized image feature up to a scalar.
            embeddings = pooled[:, : encoder.dimension]
            embeddings /= np.maximum(
                np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
            )
            features["full", alpha] = dict(zip(unique, embeddings[: len(unique)]))
            if alpha == 0:
                features["crop", 0] = dict(zip(unique, embeddings[len(unique) :]))
    temporal = temporal_indices(args.timestamps, len(paths))
    litter = np.rint(np.linspace(0, len(paths) - 1, 8)).astype(int)
    output = {
        "model": args.model,
        "bundle": str(directory),
        "precision": "float32; TF32 disabled",
        "operating_point": args.operating_point,
        "predictions": {},
        "scope": "Exploratory scene/clip classification; sparse automatic geometry does not establish ownership; no housing-status inference",
    }
    for task in TASKS:
        chosen = policy[task]["selected"]
        inputs = (
            [paths[-1]]
            if task in {"smoking", "rough_sleeping"}
            else [paths[i] for i in litter if 0 <= i < len(paths)]
            if task == "littering"
            else [paths[i] for i in temporal]
        )
        row = {"views": inputs}
        score = float(
            predict(
                bundle["heads"][task],
                matrix([row], chosen["selection"]["strategy"], features, detections),
            )[0]
        )
        threshold = chosen["thresholds"][args.operating_point]
        supported = temporal_support(task, args.timestamps, len(paths))
        output["predictions"][task] = {
            "score": score,
            "threshold": threshold,
            "decision": ("anomaly" if score >= threshold else "normal")
            if supported
            else "unknown",
            "strategy": chosen["selection"]["strategy"],
            "classifier": chosen["selection"]["classifier"],
            "input_history_supported": supported,
        }
    text = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
