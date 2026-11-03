"""Resident backends: exact selected CLIP recipes and compact Gemma evidence reports."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

from .evidence import (
    VIEW_FOR_TASK,
    history_problem,
    parse_answer,
    prompt,
    qualify,
    spread,
    unknown,
    validated_views,
)

ROOT = Path(__file__).resolve().parents[2]


class ClipStrategy:
    """Keep encoder/detector resident; restore each interpolated backbone before every use."""

    def __init__(
        self,
        model="siglip2-b16-224",
        device="cuda:0",
        operating_point="dev_max_f1",
        **_,
    ):
        import torch

        from anomaly_eval.runtime import resolve_device
        from anomaly_eval.training.clip import Encoder

        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.torch, self.device = torch, resolve_device(device)
        self.model, self.point = model, operating_point
        directory = ROOT / "results/strategy_study_20261016" / model
        if (
            json.loads((directory / "transfer_evaluation/state.json").read_text())[
                "status"
            ]
            != "complete"
        ):
            raise ValueError("The selected strategy bundle is incomplete")
        provenance = json.loads(
            (ROOT / "results/clip_finetuning_20261015/provenance.json").read_text()
        )
        if (
            hashlib.sha256(
                (ROOT / provenance["definitions_path"]).read_bytes()
            ).hexdigest()
            != provenance["definitions_sha256"]
        ):
            raise ValueError("Anomaly definitions changed since classifier training")
        self.policy = json.loads(
            (directory / "transfer_evaluation/frozen_policies.json").read_text()
        )
        self.bundle = torch.load(
            directory / "selected_heads.pt", map_location="cpu", weights_only=True
        )
        if self.bundle["model"] != model:
            raise ValueError("Classifier checkpoint model mismatch")
        self.encoder = Encoder(model, self.device).eval()
        self.previous = torch.load(
            ROOT
            / "results/clip_finetuning_20261015/seed_42"
            / model
            / "finetuned_best.pt",
            map_location="cpu",
            weights_only=True,
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
                    ROOT
                    / "data/models/detection/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
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

        from anomaly_eval.training.predict import temporal_indices, temporal_support
        from scripts.strategy.evaluate_strategy_transfer import matrix
        from scripts.strategy.extract_strategy_features import Views
        from scripts.strategy.run_strategy_screen import predict

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
        unique = list(dict.fromkeys(p for paths in inputs.values() for p in paths))
        if unique:
            policies = {t: self.policy[t]["selected"] for t in inputs}
            need_detection = any(
                any(s in p["selection"]["strategy"] for s in ["crop", "geometry"])
                for p in policies.values()
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
                    "crop" in p["selection"]["strategy"] for p in policies.values()
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


class GemmaReasoner:
    def __init__(
        self,
        model="gemma4-12b",
        device="cuda:0",
        attention="sdpa",
        max_new_tokens=160,
        output_format="auto",
        **_,
    ):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch, self.device, self.attention = torch, device, attention
        if not (device.startswith("cuda") or device == "cpu"):
            raise ValueError("Gemma device must be CPU or CUDA")
        torch.set_num_threads(16)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        receipt_path = (
            ROOT / "results/gemma4_evaluation_20261027" / model / "checkpoint.json"
        )
        receipt = json.loads(receipt_path.read_text())
        checkpoint = Path(receipt["path"])
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Pinned checkpoint is unavailable: {checkpoint}")
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, local_files_only=True
        )
        self.processor.tokenizer.padding_side = "left"
        impl = "eager" if attention == "eager" else "sdpa"
        self.model = (
            AutoModelForMultimodalLM.from_pretrained(
                checkpoint,
                dtype=torch.bfloat16,
                device_map=device,
                attn_implementation={
                    "": "sdpa",
                    "text_config": impl,
                    "vision_config": impl,
                    "audio_config": "eager",
                },
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )
        self.maximum = max_new_tokens
        self.output_format = output_format
        self.metadata = {
            "model": model,
            "revision": receipt["revision"],
            "checkpoint_receipt_sha256": hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            "precision": "bfloat16",
            "device": device,
            "attention": attention,
            "thinking": False,
            "structured_output": "prompt_only",
            "visual_token_cap": 280,
            "scope": "New compact generative verification; not a calibrated replay of earlier binary AP.",
        }
        self.sync()

    def sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def classify(self, candidate):
        from PIL import Image, ImageOps

        views = validated_views(candidate)
        groups = {}
        out = {}
        raw = []
        for task in candidate["tasks"]:
            groups.setdefault(VIEW_FOR_TASK[task], []).append(task)
        self.sync()
        started = time.perf_counter()
        for name, tasks in groups.items():
            view = spread(views[name], 8 if name == "short" else 4)
            active = []
            for task in tasks:
                problem = history_problem(task, view)
                if problem:
                    out[task] = unknown(problem)
                else:
                    active.append(task)
            if not active:
                continue
            images = []
            for path in view["paths"]:
                with Image.open(path) as im:
                    image = ImageOps.exif_transpose(im).convert("RGB")
                    image.thumbnail((1536, 1536))
                    images.append(image.copy())
            crop = candidate.get("crop_path")
            has_crop = bool(
                crop
                and Path(crop).is_file()
                and candidate.get("candidate_origin") != "scheduled_scene_review"
            )
            if has_crop:
                with Image.open(crop) as im:
                    images.append(ImageOps.exif_transpose(im).convert("RGB").copy())
            text = prompt(
                candidate,
                active,
                view,
                has_crop,
                object_output=getattr(self, "object_output", False),
            )
            response, generation_stats = self._generate(
                images, text, active, len(view["paths"])
            )
            parsed = parse_answer(response, active, len(view["paths"]))
            for task in active:
                out[task] = {
                    **qualify(task, parsed[task], candidate, view),
                    "view": name,
                    "decision_policy": "uncalibrated_generative_verification",
                }
            raw.append(
                {
                    "view": name,
                    "tasks": active,
                    "text": response,
                    "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "frame_paths": view["paths"],
                    "frame_times_s": view["times_s"],
                    "crop_used": has_crop,
                    **generation_stats,
                }
            )
        self.sync()
        return {
            "predictions": out,
            "raw_responses": raw,
            "seconds": time.perf_counter() - started,
            "metadata": self.metadata,
        }

    def _generate(self, images, text, tasks, count):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images]
                + [{"type": "text", "text": text}],
            }
        ]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        values = self.processor(
            text=[rendered],
            images=[images],
            return_tensors="pt",
            images_kwargs={"max_soft_tokens": 280},
        )
        values = {
            k: v.to(
                device=self.device,
                dtype=self.torch.bfloat16 if v.is_floating_point() else None,
            )
            for k, v in values.items()
        }
        context = (
            sdpa_kernel(SDPBackend.MATH) if self.attention == "math" else nullcontext()
        )
        with context, self.torch.inference_mode():
            generated = self.model.generate(
                **values,
                max_new_tokens=self.maximum,
                do_sample=False,
                use_cache=True,
            )
        response = self.processor.tokenizer.decode(
            generated[0, values["input_ids"].shape[-1] :], skip_special_tokens=True
        )
        return response, {
            "input_tokens": int(values["input_ids"].shape[-1]),
            "output_tokens": int(generated.shape[-1] - values["input_ids"].shape[-1]),
        }


class GemmaGGUF(GemmaReasoner):
    """Pinned QAT model with an owned local llama.cpp server and native JSON schema."""

    object_output = True

    def __init__(
        self,
        model="gemma4-26b-a4b",
        device="cuda:0",
        max_new_tokens=160,
        output_format="auto",
        runtime_dir=None,
        **_,
    ):
        import socket
        import subprocess

        import requests

        self.requests, self.maximum = requests, max_new_tokens
        self.device, self.output_format = device, output_format
        self.process = None
        self.log = None
        receipt_path = (
            ROOT / "results/gemma4_evaluation_20261027" / model / "checkpoint.json"
        )
        receipt = json.loads(receipt_path.read_text())
        checkpoint = Path(receipt["path"])
        runtime = json.loads(
            (ROOT / "anomaly_eval/config/gemma4_runtime.json").read_text()
        )
        executable = ROOT / runtime["directory"] / runtime["binary"]
        models = [p for p in checkpoint.glob("*.gguf") if "mmproj" not in p.name]
        projectors = list(checkpoint.glob("*mmproj*.gguf"))
        if len(models) != 1 or len(projectors) != 1:
            raise ValueError("Pinned GGUF decoder/projector is unavailable")
        if device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("This GGUF profile supports CPU or cuda:0")
        directory = (
            Path(runtime_dir)
            if runtime_dir
            else ROOT / "results/reasoning_runtime" / model
        )
        directory.mkdir(parents=True, exist_ok=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        command = [
            "bash",
            str(ROOT / "scripts/gemma4_llama.sh"),
            "-m",
            str(models[0]),
            "--mmproj",
            str(projectors[0]),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--alias",
            model,
            "--ctx-size",
            "16384",
            "--parallel",
            "1",
            "--batch-size",
            "4096",
            "--ubatch-size",
            "4096",
            "--flash-attn",
            "on",
            "--n-gpu-layers",
            "0" if device == "cpu" else "99",
            "--threads",
            "16",
            "--threads-batch",
            "16",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--jinja",
            "--no-warmup",
            "--image-max-tokens",
            "280",
            "--cache-ram",
            "0",
        ]
        command += (
            ["--device", "none", "--no-mmproj-offload"]
            if device == "cpu"
            else ["--device", "CUDA0"]
        )
        self.metadata = {
            "model": model,
            "revision": receipt["revision"],
            "checkpoint_receipt_sha256": hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            "backend": "llama.cpp " + runtime["release"],
            "precision": "official Google QAT Q4_0",
            "device": device,
            "visual_token_cap": 280,
            "thinking": False,
            "structured_output": "native_json_schema"
            if output_format != "prompt"
            else "prompt_only",
            "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "scope": "New generative evidence verification; earlier binary AP is not pipeline accuracy.",
        }
        try:
            self.log = (directory / "server.log").open("a")
            self.process = subprocess.Popen(
                command, cwd=ROOT, stdout=self.log, stderr=subprocess.STDOUT
            )
            (directory / "launch.json").write_text(
                json.dumps(
                    {
                        "command": command,
                        "pid": self.process.pid,
                        "metadata": self.metadata,
                    },
                    indent=2,
                )
                + "\n"
            )
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"llama.cpp exited; inspect {directory}/server.log"
                    )
                try:
                    if requests.get(self.url + "/health", timeout=1).ok:
                        return
                except requests.RequestException:
                    pass
                time.sleep(0.1)
            raise TimeoutError("Owned model server did not become ready")
        except BaseException:
            self.close()
            raise

    def sync(self):
        pass

    def _generate(self, images, text, tasks, count):
        from anomaly_eval.gemma4.backends import data_url

        from .evidence import output_schema

        body = {
            "model": self.metadata["model"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url(image)}}
                        for image in images
                    ]
                    + [{"type": "text", "text": text}],
                }
            ],
            "max_tokens": self.maximum,
            "temperature": 0,
            "seed": 42,
            "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.output_format != "prompt":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "candidate_evidence",
                    "strict": True,
                    "schema": output_schema(tasks, count),
                },
            }
        response = self.requests.post(
            self.url + "/v1/chat/completions", json=body, timeout=(5, 110)
        )
        response.raise_for_status()
        value = response.json()
        choice = value["choices"][0]
        return choice["message"].get("content") or "", {
            "input_tokens": value.get("usage", {}).get("prompt_tokens"),
            "output_tokens": value.get("usage", {}).get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
        }

    def close(self):
        import subprocess

        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
