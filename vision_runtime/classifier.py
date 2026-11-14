"""Resident trained CLIP strategies, with no research-script imports."""

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

from .config import model_paths, models_root
from .evidence import (
    DEFINITIONS,
    VIEW_FOR_TASK,
    history_problem,
    unknown,
    validated_views,
)


def plan(inputs, policies):
    """Which features each selected head reads, so nothing else is computed.

    Returns frames keyed by content: full-frame features per interpolation
    weight, crop features (always the uninterpolated backbone) and detections
    for crop/geometry recipes. Byte-identical files, such as the trigger frame
    saved once per view, are computed once.
    """
    keys = {}
    for paths in inputs.values():
        for path in paths:
            if path not in keys:
                keys[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    full, crop, detect = {}, {}, {}
    for task, paths in inputs.items():
        strategy = policies[task]["selection"]["strategy"]
        alpha = float(strategy.rsplit("a", 1)[1])
        frames = full.setdefault(alpha, {})
        for path in paths[-1:] if strategy.startswith("endpoint") else paths:
            frames.setdefault(keys[path], path)
        for path in paths:
            if "crop" in strategy:
                crop.setdefault(keys[path], path)
            if "crop" in strategy or "geometry" in strategy:
                detect.setdefault(keys[path], path)
    return SimpleNamespace(keys=keys, full=full, crop=crop, detect=detect)


def by_path(keys, frames, rows):
    """Map computed rows (in frame order) back to every path with those bytes."""
    row = dict(zip(frames, rows))
    return {path: row[key] for path, key in keys.items() if key in row}


class ClipStrategy:
    """Keep encoder/detector resident; restore each interpolated backbone before every use."""

    def __init__(
        self, model="mobileclip2-s2", device="cpu", operating_point="dev_max_f1", **_
    ):
        import torch

        from .config import resolve_device
        from .encoder import Encoder

        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.torch, self.device = (torch, resolve_device(device))
        self.model, self.point = (model, operating_point)
        directory = model_paths(model).directory
        if json.loads((directory / "state.json").read_text())["status"] != "complete":
            raise ValueError("The selected strategy bundle is incomplete")
        provenance = json.loads((directory / "provenance.json").read_text())
        if (
            hashlib.sha256((DEFINITIONS).read_bytes()).hexdigest()
            != provenance["definitions_sha256"]
        ):
            raise ValueError("Anomaly definitions changed since classifier training")
        self.policy = json.loads((directory / "frozen_policies.json").read_text())
        self.bundle = torch.load(
            directory / "selected_heads.pt", map_location="cpu", weights_only=True
        )
        if self.bundle["model"] != model:
            raise ValueError("Classifier checkpoint model mismatch")
        self.encoder = Encoder(model, self.device).eval()
        self.previous = torch.load(
            directory / "finetuned_best.pt", map_location="cpu", weights_only=True
        )
        self.parameters = dict(self.encoder.named_parameters())
        self.original = {
            n: self.parameters[n].detach().cpu().clone()
            for n in self.previous["parameters"]
            if n.startswith("backbone.")
        }
        self.detector = None
        self.metadata = {
            "model": model,
            "backend": "resident_selected_strategy",
            "base_revision": self.encoder.revision,
            "device": self.device,
            "precision": "float32; TF32 disabled",
            "operating_point": operating_point,
            "bundle_sha256": hashlib.sha256(
                (directory / "selected_heads.pt").read_bytes()
            ).hexdigest(),
            "scope": "Scene/window classifier. Prior aggregate AP is not candidate-pipeline accuracy.",
        }

    def sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def _detections(self, paths):
        from PIL import Image, ImageOps
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.transforms.functional import pil_to_tensor

        if self.detector is None:
            self.detector = fasterrcnn_resnet50_fpn_v2(
                weights=None, weights_backbone=None
            )
            self.detector.load_state_dict(
                self.torch.load(
                    models_root()
                    / "detectors/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
                    weights_only=True,
                    map_location="cpu",
                )
            )
            self.detector.eval().to(self.device)
            if self.device == "cpu":
                # Same boxes (within float rounding), about 17% faster on CPU.
                self.detector.to(memory_format=self.torch.channels_last)
        detections = {}
        for name in paths:
            with Image.open(name) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((1280, 1280))
                w, h = image.size
                tensor = (
                    pil_to_tensor(image).to(self.device, dtype=self.torch.float32) / 255
                )
            result = self.detector([tensor])[0]
            detections[name] = [
                {
                    "box": [float(v) / s for v, s in zip(box, [w, h, w, h])],
                    "label": int(label),
                    "confidence": float(score),
                }
                for box, label, score in zip(
                    result["boxes"].cpu(),
                    result["labels"].cpu(),
                    result["scores"].cpu(),
                )
                if int(label) in {1, 27, 31, 33} and float(score) >= 0.5
            ]
        return detections

    def _features(self, needs):
        """Compute only the planned features and detections, at each needed weight."""
        import numpy as np

        from .features import Views

        found = self._detections(list(needs.detect.values())) if needs.detect else {}
        by_key = {key: found[path] for key, path in needs.detect.items()}
        # Frames that no crop or geometry head reads keep an empty detection list.
        detections = {path: by_key.get(key, []) for path, key in needs.keys.items()}

        def stack(frames, boxes=None):
            views = Views(list(frames.values()), self.encoder, boxes)
            return self.torch.stack([views[i] for i in range(len(views))]).to(
                self.device
            )

        crop = stack(needs.crop, detections) if needs.crop else None
        features = {}
        for alpha in sorted(set(needs.full) | ({0.0} if crop is not None else set())):
            for name, base in self.original.items():
                self.parameters[name].copy_(
                    self.torch.lerp(base, self.previous["parameters"][name], alpha).to(
                        self.device
                    )
                )
            frames = needs.full.get(alpha, {})
            parts = [stack(frames)] if frames else []
            if alpha == 0 and crop is not None:
                parts.append(crop)
            batch = self.torch.cat(parts)
            pooled = (
                self.encoder.encode(batch, [1] * len(batch))
                .cpu()
                .numpy()[:, : self.encoder.dimension]
            )
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            features["full", alpha] = by_path(needs.keys, frames, pooled)
            if alpha == 0 and crop is not None:
                features["crop", 0] = by_path(
                    needs.keys, needs.crop, pooled[len(frames) :]
                )
        return features, detections

    def classify(self, candidate):
        import numpy as np

        from .features import matrix, predict, temporal_indices, temporal_support

        views = validated_views(candidate)
        out = {}
        inputs = {}
        self.sync()
        started = time.perf_counter()
        for task in candidate["tasks"]:
            view = views[VIEW_FOR_TASK[task]]
            if history_problem(task, view, clip=True) or not temporal_support(
                task, view["times_s"], len(view["paths"])
            ):
                out[task] = unknown(
                    "Insufficient observed history for the trained CLIP input protocol"
                )
                continue
            indices = (
                [len(view["paths"]) - 1]
                if task in {"smoking", "rough_sleeping"}
                else np.rint(np.linspace(0, len(view["paths"]) - 1, 8))
                .astype(int)
                .tolist()
                if task == "littering"
                else temporal_indices(view["times_s"], len(view["paths"]))
            )
            inputs[task] = [view["paths"][i] for i in indices]
        if inputs:
            policies = {t: self.policy[t]["selected"] for t in inputs}
            with self.torch.inference_mode():
                features, detections = self._features(plan(inputs, policies))
                for task, paths in inputs.items():
                    chosen = policies[task]
                    score = float(
                        predict(
                            self.bundle["heads"][task],
                            matrix(
                                [{"views": paths}],
                                chosen["selection"]["strategy"],
                                features,
                                detections,
                            ),
                        )[0]
                    )
                    threshold = float(chosen["thresholds"][self.point])
                    decision = "anomaly" if score >= threshold else "normal"
                    out[task] = {
                        "decision": decision,
                        "score": score,
                        "threshold": threshold,
                        "strategy": chosen["selection"]["strategy"],
                        "input_history_supported": True,
                        "alert": False,
                        "scope": "scene/window classification; entity and ownership not independently verified",
                    }
        self.sync()
        return {
            "predictions": out,
            "seconds": time.perf_counter() - started,
            "metadata": self.metadata,
        }


# Newest first. 20261113 keeps the 20261111 encoders and recipes; its loitering
# heads also saw other tasks' windows as negatives and its thresholds are capped
# so that at most 5% of other tasks' out-of-fold windows are flagged. 20261114
# adds text-described (zero-shot) members, footage from more sites, a smoking
# detector with its own threshold, and a cap on ordinary footage flagged.
ENSEMBLES = ("ensemble-20261114", "ensemble-20261113", "ensemble-20261111")
ENSEMBLE = ENSEMBLES[0]
CLASSIFIERS = (*ENSEMBLES, "mobileclip2-s2", "siglip2-b16-224")


def default_classifier():
    """The newest ensemble whose folder is installed, else the newest name."""
    from .config import models_root

    for name in ENSEMBLES:
        if (models_root() / name / "ensemble.json").is_file():
            return name
    return ENSEMBLE


def load_classifier(model=None, device="cpu"):
    """An ensemble by default; the October single-backbone routes on request."""
    model = model or default_classifier()
    if model in ENSEMBLES:
        from .config import models_root
        from .ensemble import EnsembleStrategy

        return EnsembleStrategy(device=device, directory=models_root() / model)
    return ClipStrategy(model, device)
