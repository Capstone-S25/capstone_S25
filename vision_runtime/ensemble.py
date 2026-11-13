"""Two-backbone ensemble classifier from the November F1 study.

Each task averages standardized scores from SigLIP2-B and MobileCLIP2-B
members. Every member averages its top linear heads, and every head is the
mean of 15 fold heads fitted with whole cameras/recordings held out. The task
threshold was set on those held-out (out-of-fold) scores, which is the
situation a new camera is in. See docs/f1_study.md.

Inputs per task follow the trained protocol (endpoint frame, 8 short-view
frames for littering, 4 long-view frames for loitering/unattended items).
Loitering and unattended items also compare the window with the camera's
reference frame: the first frame of the session, attached to each candidate
as ``reference_frame_path`` by the Stage 1 detector.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .config import models_root
from .evidence import VIEW_FOR_TASK, history_problem, unknown, validated_views
from .features import geometry, temporal_indices, temporal_support

PERSON, BAGS = 1, {27, 31, 33}
YOLO_TO_COCO91 = {0: 1, 24: 27, 26: 31, 28: 33}


def normalize(x):
    return x / max(float(np.linalg.norm(x)), 1e-12)


def pool(views, ordered=False):
    views = np.asarray(views, dtype=np.float32)
    if ordered:
        idx = np.rint(np.linspace(0, len(views) - 1, 4)).astype(int)
        return normalize(
            np.concatenate([views[idx].ravel(), views.mean(0), views.max(0)])
        )
    return normalize(
        np.concatenate([views[-1], views.mean(0), views.max(0), views[-1] - views[0]])
    )


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy

    def area(c):
        return max(0.0, c[2] - c[0]) * max(0.0, c[3] - c[1])

    return inter / max(area(a) + area(b) - inter, 1e-12)


def stays(boxes_by_view, others_by_view=None):
    """How firmly endpoint objects stayed in place across the whole window."""
    last = boxes_by_view[-1]
    if not last:
        return [0.0, 0.0, 0.0, 2.0, 0.0, 1.5]
    per = []
    for box in last:
        h = max(box[3] - box[1], 1e-3)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        ious, shifts = [], []
        for earlier in boxes_by_view[:-1]:
            if not earlier:
                ious.append(0.0)
                shifts.append(2.0)
                continue
            ious.append(max(iou(box, e) for e in earlier))
            shifts.append(
                min(
                    min(
                        np.hypot((e[0] + e[2]) / 2 - cx, (e[1] + e[3]) / 2 - cy) / h,
                        2.0,
                    )
                    for e in earlier
                )
            )
        alone = 1.5
        if others_by_view is not None and others_by_view[-1]:
            alone = min(
                np.hypot((o[0] + o[2]) / 2 - cx, (o[1] + o[3]) / 2 - cy)
                for o in others_by_view[-1]
            )
        per.append(
            (
                min(ious),
                float(np.mean(ious)),
                max(shifts),
                float(np.mean(shifts)),
                alone,
            )
        )
    best = max(per, key=lambda v: v[0])
    stayed = sum(v[0] > 0.3 for v in per)
    return [best[0], best[1], min(stayed, 4) / 4, best[2], best[3], best[4]]


def dwell(found_by_view):
    persons = [[d["box"] for d in f if d["label"] == PERSON] for f in found_by_view]
    bags = [[d["box"] for d in f if d["label"] in BAGS] for f in found_by_view]
    return np.asarray(stays(persons) + stays(bags, persons), dtype=np.float32)


def contextual_box(box, padding=0.2):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return [max(0, x1 - dx), max(0, y1 - dy), min(1, x2 + dx), min(1, y2 + dy)]


def crop_box(found):
    """Union of the most confident person and item boxes, with context."""
    boxes = []
    for labels in [{PERSON}, BAGS]:
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


def view_image(path, box=None):
    """EXIF-corrected RGB, optional crop, black letterbox, at most 1024 px."""
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if box is not None:
        w, h = image.size
        image = image.crop(tuple(v * s for v, s in zip(box, [w, h, w, h])))
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    canvas.thumbnail((1024, 1024))
    return canvas


class Backbone:
    """Image tower of one pretrained encoder, loaded from a local folder."""

    def __init__(self, kind, directory, device, torch):
        self.kind, self.device, self.torch = kind, device, torch
        if kind == "open_clip":
            import open_clip

            model, _, self.transform = open_clip.create_model_and_transforms(
                "local-dir:" + str(directory), device=device, precision="fp32"
            )
            self.model = model.visual.eval().requires_grad_(False)
        elif kind == "siglip":
            from transformers import SiglipImageProcessor, SiglipVisionModel

            # The slow processor, as in the study (the tokenizer is not needed).
            self.processor = SiglipImageProcessor.from_pretrained(
                str(directory), local_files_only=True
            )
            self.model = (
                SiglipVisionModel.from_pretrained(
                    str(directory), local_files_only=True, dtype=torch.float32
                )
                .to(device)
                .eval()
                .requires_grad_(False)
            )
        else:
            raise ValueError(kind)

    def tensor(self, image):
        if self.kind == "open_clip":
            return self.transform(image)
        return self.torch.from_numpy(
            np.asarray(
                self.processor(images=image, return_tensors="np")["pixel_values"][0]
            )
        )

    def encode(self, images):
        batch = self.torch.stack(images).to(self.device)
        out = (
            self.model(batch)
            if self.kind == "open_clip"
            else self.model(pixel_values=batch).pooler_output
        )
        out = out.float().cpu().numpy()
        # The heads were fitted on float16-cached embeddings; round the same way.
        out = out.astype(np.float16).astype(np.float32)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


class LRU(OrderedDict):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def get(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.size:
            self.popitem(last=False)


class EnsembleStrategy:
    """Resident two-backbone ensemble with cross-fitted heads."""

    def __init__(self, device="cpu", directory=None, threads=None, cache_size=512, **_):
        import torch

        from .config import resolve_device

        if threads:
            torch.set_num_threads(int(threads))
        self.torch, self.device = torch, resolve_device(device)
        self.directory = Path(directory or models_root() / "ensemble-20261113")
        self.name = self.directory.name
        bundle = json.loads((self.directory / "ensemble.json").read_text())
        heads_path = self.directory / "heads.npz"
        digest = hashlib.sha256(heads_path.read_bytes()).hexdigest()
        if digest != bundle["heads_sha256"]:
            raise ValueError("Ensemble heads changed since export")
        arrays = np.load(heads_path)
        self.tasks = bundle["tasks"]
        for task in self.tasks.values():
            for member in task["members"]:
                for recipe in member["recipes"]:
                    recipe["weight"] = arrays[recipe["arrays"] + "/weight"]
                    recipe["bias"] = arrays[recipe["arrays"] + "/bias"]
        self.backbones = {
            name: Backbone(spec["kind"], self.directory / name, self.device, torch)
            for name, spec in bundle["backbones"].items()
        }
        self.frcnn = self.yolo = self.smoking = None
        self.cache = LRU(cache_size)
        self.metadata = {
            "model": self.name,
            "backend": "cross_fitted_two_backbone_ensemble",
            "backbones": {k: v["source"] for k, v in bundle["backbones"].items()},
            "device": self.device,
            "precision": "float32",
            "operating_point": "out_of_fold_macro_source_max_f1"
            + (
                f", raised until at most {bundle['cross_task_cap']:.0%} of other tasks' out-of-fold windows are flagged"
                if "cross_task_cap" in bundle
                else ""
            ),
            "heads_sha256": digest,
            "held_out_macro_f1": bundle["held_out"]["macro_f1"],
            "scope": "Scene/window classifier; entity and ownership not independently verified.",
        }

    # ---- per-frame primitives, cached by file content -------------------
    def _key(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def _boxes(self, source, paths, keys):
        need = [p for p in paths if self.cache.get(("boxes", source, keys[p])) is None]
        if need:
            found = self._frcnn(need) if source == "frcnn" else self._yolo(need)
            for p in need:
                self.cache.put(("boxes", source, keys[p]), found[p])
        return {p: self.cache.get(("boxes", source, keys[p])) for p in paths}

    def _frcnn(self, paths):
        from PIL import Image, ImageOps
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.transforms.functional import pil_to_tensor

        torch = self.torch
        if self.frcnn is None:
            self.frcnn = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
            self.frcnn.load_state_dict(
                torch.load(
                    models_root()
                    / "detectors/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
                    weights_only=True,
                    map_location="cpu",
                )
            )
            self.frcnn.eval().to(self.device)
            if self.device == "cpu":
                self.frcnn.to(memory_format=torch.channels_last)
        out = {}
        for name in paths:
            with Image.open(name) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((1280, 1280))
                w, h = image.size
                tensor = pil_to_tensor(image).to(self.device, dtype=torch.float32) / 255
            result = self.frcnn([tensor])[0]
            out[name] = [
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
                if int(label) in {PERSON, *BAGS} and float(score) >= 0.5
            ]
        return out

    def _yolo(self, paths):
        from PIL import Image, ImageOps
        from ultralytics import YOLO

        if self.yolo is None:
            self.yolo = YOLO(str(models_root() / "detectors/yolo26s.pt"))
        images = []
        for name in paths:
            with Image.open(name) as im:
                images.append(
                    np.asarray(ImageOps.exif_transpose(im).convert("RGB"))[:, :, ::-1]
                )
        out = {}
        results = self.yolo.predict(
            images,
            imgsz=640,
            conf=0.1,
            classes=list(YOLO_TO_COCO91),
            verbose=False,
            device=self.device,
        )
        for name, r in zip(paths, results):
            h, w = r.orig_shape
            out[name] = [
                {
                    "box": [float(v) / s for v, s in zip(b, [w, h, w, h])],
                    "label": YOLO_TO_COCO91[int(c)],
                    "confidence": float(q),
                }
                for b, c, q in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.cls.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                )
                if float(q) >= 0.35
            ]
        return out

    def _smoking(self, paths, keys):
        """Fine-tuned YOLO26s head boxes: top smoking / not-smoking confidence
        and the number of smoking boxes >= 0.25, per frame."""
        need = [p for p in paths if self.cache.get(("smoking", keys[p])) is None]
        if need:
            from PIL import Image, ImageOps
            from ultralytics import YOLO

            if self.smoking is None:
                self.smoking = YOLO(str(self.directory / "smoking_detector.pt"))
            images = []
            for name in need:
                with Image.open(name) as im:
                    images.append(
                        np.asarray(ImageOps.exif_transpose(im).convert("RGB"))[
                            :, :, ::-1
                        ]
                    )
            results = self.smoking.predict(
                images, imgsz=640, conf=0.05, verbose=False, device=self.device
            )
            for name, r in zip(need, results):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy()
                smoke, other = conf[cls == 0], conf[cls == 1]
                self.cache.put(
                    ("smoking", keys[name]),
                    [
                        float(smoke.max()) if len(smoke) else 0.0,
                        float(other.max()) if len(other) else 0.0,
                        float((smoke >= 0.25).sum()),
                    ],
                )
        return np.array([self.cache.get(("smoking", keys[p])) for p in paths])

    def _embed(self, backbone, stage, paths, keys):
        """stage: 'full', 'crop:frcnn' or 'crop:yolo'."""
        need = [p for p in paths if self.cache.get((backbone, stage, keys[p])) is None]
        if need:
            boxes = (
                self._boxes(stage.split(":")[1], need, keys)
                if stage != "full"
                else None
            )
            model = self.backbones[backbone]
            tensors = [
                model.tensor(view_image(p, crop_box(boxes[p]) if boxes else None))
                for p in need
            ]
            for start in range(0, len(need), 16):
                values = model.encode(tensors[start : start + 16])
                for p, v in zip(need[start : start + 16], values):
                    self.cache.put((backbone, stage, keys[p]), v)
        return {p: self.cache.get((backbone, stage, keys[p])) for p in paths}

    # ---- the study's representation, for one window ------------------------
    def _vector(self, recipe, views, reference, keys):
        if recipe.get("offset"):
            plain = {k: v for k, v in recipe.items() if k != "offset"}
            return self._vector(plain, views, reference, keys) - self._vector(
                plain, [reference] * len(views), reference, keys
            )
        backbone = recipe["backbone"]
        boxes = recipe.get("boxes", "frcnn")
        if recipe.get("endpoint"):
            parts = [self._embed(backbone, "full", views[-1:], keys)[views[-1]]]
        else:
            full = self._embed(backbone, "full", views, keys)
            parts = [pool([full[p] for p in views], recipe.get("ordered", False))]
        if recipe.get("crop"):
            crop = self._embed(backbone, "crop:" + boxes, views, keys)
            parts.append(pool([crop[p] for p in views], recipe.get("ordered", False)))
        if recipe.get("geometry") or recipe.get("dwell"):
            found = self._boxes(boxes, views, keys)
            if recipe.get("geometry"):
                parts.append(geometry([found[p] for p in views]))
            if recipe.get("dwell"):
                parts.append(
                    dwell([found[p] for p in views]) * recipe.get("dwell_scale", 0.25)
                )
        if recipe.get("smoke_detector"):
            values = self._smoking(views, keys)
            parts.append(
                np.array(
                    [
                        values[:, 0].max(),
                        values[:, 1].max(),
                        min(values[:, 2].max(), 4) / 4,
                    ],
                    np.float32,
                )
                * recipe.get("detector_scale", 0.5)
            )
        if recipe.get("reference"):
            full = self._embed(backbone, "full", [*views, reference], keys)
            stack = np.stack([full[p] for p in views])
            ref = full[reference]
            parts.append(
                normalize(np.concatenate([stack[-1] - ref, stack.mean(0) - ref]))
            )
        return normalize(np.concatenate(parts)) if len(parts) > 1 else parts[0]

    def score(self, task, views, reference):
        """Standardized ensemble score for one window (threshold in self.tasks)."""
        paths = [*views, reference] if reference else list(views)
        keys = {p: self._key(p) for p in paths}
        reference = reference or views[0]
        members = []
        for member in self.tasks[task]["members"]:
            zs = []
            for r in member["recipes"]:
                x = self._vector(r["recipe"], list(views), reference, keys)
                raw = float(np.mean(r["weight"] @ x + r["bias"]))
                zs.append((raw - r["mean"]) / r["std"])
            members.append((np.mean(zs) - member["mean"]) / member["std"])
        return float(np.mean(members))

    def _probability(self, task, score):
        """Out-of-fold logistic calibration of the score, when the bundle has one.

        It assumes balanced classes and sources, so it ranks windows for display
        and is not the precision to expect on a camera.
        """
        calibration = self.tasks[task].get("calibration")
        if not calibration:
            return {}
        z = calibration["slope"] * score + calibration["intercept"]
        return {"probability": float(1 / (1 + np.exp(-np.clip(z, -30, 30))))}

    def classify(self, candidate):
        views = validated_views(candidate)
        out = {}
        started = time.perf_counter()
        reference = candidate.get("reference_frame_path")
        with self.torch.inference_mode():
            for task in candidate["tasks"]:
                view = views[VIEW_FOR_TASK[task]]
                if history_problem(task, view, clip=True) or not temporal_support(
                    task, view["times_s"], len(view["paths"])
                ):
                    out[task] = unknown(
                        "Insufficient observed history for the trained input protocol"
                    )
                    continue
                n = len(view["paths"])
                indices = (
                    [n - 1]
                    if task in {"smoking", "rough_sleeping"}
                    else np.rint(np.linspace(0, n - 1, 8)).astype(int).tolist()
                    if task == "littering"
                    else temporal_indices(view["times_s"], n)
                )
                paths = [view["paths"][i] for i in indices]
                score = self.score(task, paths, reference)
                threshold = float(self.tasks[task]["threshold"])
                out[task] = {
                    "decision": "anomaly" if score >= threshold else "normal",
                    "score": score,
                    "threshold": threshold,
                    **self._probability(task, score),
                    "strategy": self.name,
                    "reference_frame": bool(reference),
                    "input_history_supported": True,
                    "alert": False,
                    "scope": "scene/window classification; entity and ownership not independently verified",
                }
        return {
            "predictions": out,
            "seconds": time.perf_counter() - started,
            "metadata": self.metadata,
        }
