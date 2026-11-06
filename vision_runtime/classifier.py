"""Resident trained CLIP strategies, with no research-script imports."""

import hashlib
import json
import time

from .config import model_paths, models_root
from .evidence import (
    DEFINITIONS,
    VIEW_FOR_TASK,
    history_problem,
    unknown,
    validated_views,
)


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

    def classify(self, candidate):
        import numpy as np

        from .features import Views, matrix, predict, temporal_indices, temporal_support

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
        unique = list(dict.fromkeys((p for paths in inputs.values() for p in paths)))
        if unique:
            policies = {t: self.policy[t]["selected"] for t in inputs}
            need_detection = any(
                (
                    any((s in p["selection"]["strategy"] for s in ["crop", "geometry"]))
                    for p in policies.values()
                )
            )
            with self.torch.inference_mode():
                detections = (
                    self._detections(unique)
                    if need_detection
                    else {p: [] for p in unique}
                )
                full = self.torch.stack(
                    [Views(unique, self.encoder)[i] for i in range(len(unique))]
                ).to(self.device)
                need_crop = any(
                    ("crop" in p["selection"]["strategy"] for p in policies.values())
                )
                crop = (
                    self.torch.stack(
                        [
                            Views(unique, self.encoder, detections)[i]
                            for i in range(len(unique))
                        ]
                    ).to(self.device)
                    if need_crop
                    else None
                )
                alphas = sorted(
                    {
                        float(p["selection"]["strategy"].rsplit("a", 1)[1])
                        for p in policies.values()
                    }
                    | ({0.0} if need_crop else set())
                )
                features = {}
                for alpha in alphas:
                    for name, base in self.original.items():
                        self.parameters[name].copy_(
                            self.torch.lerp(
                                base, self.previous["parameters"][name], alpha
                            ).to(self.device)
                        )
                    batch = (
                        self.torch.cat([full, crop])
                        if alpha == 0 and crop is not None
                        else full
                    )
                    pooled = (
                        self.encoder.encode(batch, [1] * len(batch))
                        .cpu()
                        .numpy()[:, : self.encoder.dimension]
                    )
                    pooled /= np.maximum(
                        np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12
                    )
                    features["full", alpha] = dict(zip(unique, pooled[: len(unique)]))
                    if alpha == 0 and crop is not None:
                        features["crop", 0] = dict(zip(unique, pooled[len(unique) :]))
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
