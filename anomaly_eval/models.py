"""CPU model adapters: generative VLMs, CLIP, and few-shot vision features."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from anomaly_eval.data import Sample
from anomaly_eval.resources import Throttler


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    checkpoint: str
    revision: str | None = None


MODELS = {
    "mobileclip2-s2": ModelSpec(
        "openclip",
        "timm/MobileCLIP2-S2-OpenCLIP",
        "ac6b37c8fc40b62b623d09ed228a6f0c9bc29fe6",
    ),
    "siglip2-b16-224": ModelSpec(
        "siglip",
        "google/siglip2-base-patch16-224",
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    ),
    "minicpm-v4": ModelSpec(
        "minicpm", "openbmb/MiniCPM-V-4", "6f69a8235885be89608f3a71ffd40695b379e1e7"
    ),
    "clip-vit-b32": ModelSpec("clip", "openai/clip-vit-base-patch32"),
    "clip-vit-b16": ModelSpec("clip", "openai/clip-vit-base-patch16"),
    "smolvlm-256m": ModelSpec("vlm", "HuggingFaceTB/SmolVLM-256M-Instruct"),
    "smolvlm-500m": ModelSpec(
        "vlm",
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        "a7da5b986cb59b408707209984f360a5f4ad7e47",
    ),
    "mobilenet-v3-small": ModelSpec("timm", "mobilenetv3_small_100.lamb_in1k"),
    "resnet18": ModelSpec("timm", "resnet18.a1_in1k"),
    "dinov2-small": ModelSpec("vision", "facebook/dinov2-small"),
}

DEFAULT_MODELS = ["mobileclip2-s2", "siglip2-b16-224", "smolvlm-500m", "minicpm-v4"]


@dataclass
class Prediction:
    label: str | None
    anomaly_score: float | None = None
    scores: dict[str, float] = field(default_factory=dict)
    raw_response: str | None = None
    generated_tokens: int | None = None
    response_format: str | None = None
    abstention_reason: str | None = None


def classification_prompt(definitions: dict[str, str]) -> str:
    descriptions = "\n".join(
        f"{label}: {description}" for label, description in definitions.items()
    )
    return (
        "Classify the visible scene using these definitions:\n"
        + descriptions
        + "\nChoose one best matching label. Use normal only when no defined anomaly is visible. "
        "Use unknown when evidence is insufficient, including missing evidence of duration or ownership. "
        'Reply only with JSON: {"label": "LABEL"}. Allowed labels: '
        + ", ".join(definitions)
        + ", unknown."
    )


def parse_vlm_response(text: str, definitions: dict[str, str]) -> Prediction:
    """Accept case-insensitive exact labels; never search for labels inside prose."""

    def exact_label(value):
        if not isinstance(value, str):
            return None, "invalid_response"
        candidate = value.strip().casefold()
        if candidate == "unknown":
            return None, "explicit_unknown"
        matches = [label for label in definitions if label.casefold() == candidate]
        if len(matches) == 1:
            return matches[0], None
        return None, "ambiguous_label" if matches else "invalid_response"

    def result(value, response_format):
        label, reason = exact_label(value)
        return Prediction(
            label,
            raw_response=text,
            response_format=response_format
            if reason in (None, "explicit_unknown")
            else "invalid",
            abstention_reason=reason,
        )

    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1]).strip()
    # Accept an exact class name, without finding label words inside prose.
    candidate = cleaned.removesuffix(".").strip()
    if (
        len(candidate) >= 2
        and candidate[0] in {"'", '"'}
        and candidate[-1] == candidate[0]
    ):
        candidate = candidate[1:-1].strip()
    label, reason = exact_label(candidate)
    if label is not None or reason == "explicit_unknown":
        return result(candidate, "plain_label")
    try:
        value = json.loads(cleaned)
    except (ValueError, TypeError):
        return result(candidate, "invalid")
    label = value.get("label") if isinstance(value, dict) else None
    return result(label, "json_label")


def load_image(path: Path, max_side: int, policy: str = "native"):
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    if policy == "letterbox":
        side = max(image.size)
        canvas = Image.new("RGB", (side, side), (127, 127, 127))
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        image = canvas
    elif policy != "native":
        raise ValueError("Unknown image policy")
    return image


class Adapter:
    def prepare(
        self,
        references: list[Sample],
        throttle: Throttler,
        check_resources: Callable[[], None],
        max_side: int,
    ) -> None:
        pass

    def predict(self, image) -> Prediction:
        raise NotImplementedError


class CLIPAdapter(Adapter):
    def __init__(self, spec: ModelSpec, definitions: dict[str, str], options: dict):
        import torch
        from transformers import AutoProcessor, CLIPModel

        self.torch = torch
        self.labels = list(definitions)
        self.processor = AutoProcessor.from_pretrained(spec.checkpoint, **options)
        self.model = (
            CLIPModel.from_pretrained(
                spec.checkpoint,
                dtype=torch.float32,
                attn_implementation="eager",
                **options,
            )
            .to("cpu")
            .eval()
        )
        tokens = self.processor(
            text=list(definitions.values()),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            self.text_features = torch.nn.functional.normalize(
                self.model.get_text_features(**tokens), dim=-1
            )

    def predict(self, image) -> Prediction:
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            features = torch.nn.functional.normalize(
                self.model.get_image_features(**inputs), dim=-1
            )
            logits = self.model.logit_scale.exp() * features @ self.text_features.T
            weights = logits.softmax(dim=-1)[0].tolist()
        scores = dict(zip(self.labels, weights))
        return Prediction(max(scores, key=scores.get), 1 - scores["normal"], scores)


class OpenCLIPAdapter(Adapter):
    """MobileCLIP2 using the OpenCLIP port of Apple's S2 checkpoint."""

    def __init__(self, spec: ModelSpec, definitions: dict[str, str], options: dict):
        import open_clip
        import torch
        from timm.utils import reparameterize_model

        self.torch = torch
        self.labels = list(definitions)
        from huggingface_hub import snapshot_download

        directory = snapshot_download(
            spec.checkpoint,
            revision=spec.revision,
            cache_dir=options.get("cache_dir"),
            local_files_only=options.get("local_files_only", False),
            allow_patterns=[
                "open_clip_config.json",
                "open_clip_model.safetensors",
                "*.json",
                "*.txt",
            ],
        )
        model_name = "local-dir:" + directory
        self.resolved_revision = spec.revision
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            device="cpu",
            precision="fp32",
            cache_dir=options.get("cache_dir"),
        )
        self.model.eval()
        # Fuse the MobileCLIP convolution branches for inference before timing.
        self.model = reparameterize_model(self.model, inplace=True)
        tokenizer = open_clip.get_tokenizer(
            model_name, cache_dir=options.get("cache_dir")
        )
        tokens = tokenizer(list(definitions.values()))
        with torch.inference_mode():
            self.text_features = self.model.encode_text(tokens, normalize=True)

    def predict(self, image) -> Prediction:
        with self.torch.inference_mode():
            features = self.model.encode_image(
                self.preprocess(image).unsqueeze(0), normalize=True
            )
            logits = self.model.logit_scale.exp() * features @ self.text_features.T
            weights = logits.softmax(dim=-1)[0].tolist()
        scores = dict(zip(self.labels, weights))
        return Prediction(max(scores, key=scores.get), 1 - scores["normal"], scores)


class SigLIPAdapter(Adapter):
    """Cache text features and preserve SigLIP's independent sigmoid scores."""

    def __init__(self, spec: ModelSpec, definitions: dict[str, str], options: dict):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.labels = list(definitions)
        self.processor = AutoProcessor.from_pretrained(
            spec.checkpoint, use_fast=False, **options
        )
        self.model = (
            AutoModel.from_pretrained(
                spec.checkpoint,
                dtype=torch.float32,
                attn_implementation="eager",
                **options,
            )
            .to("cpu")
            .eval()
        )
        tokens = self.processor(
            text=[description.lower() for description in definitions.values()],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            self.text_features = torch.nn.functional.normalize(
                self.model.get_text_features(**tokens), dim=-1
            )

    def predict(self, image) -> Prediction:
        torch = self.torch
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            features = torch.nn.functional.normalize(
                self.model.get_image_features(**inputs), dim=-1
            )
            logits = (
                self.model.logit_scale.exp() * features @ self.text_features.T
                + self.model.logit_bias
            )[0]
            values = logits.tolist()
            weights = logits.sigmoid().tolist()
        scores = dict(zip(self.labels, weights))
        raw_scores = dict(zip(self.labels, values))
        # Score the anomaly-vs-normal margin; sigmoid outputs do not sum to one.
        label = max(raw_scores, key=raw_scores.get)
        margin = (
            max(value for name, value in raw_scores.items() if name != "normal")
            - raw_scores["normal"]
        )
        return Prediction(label, margin, scores)


class VLMAdapter(Adapter):
    def __init__(
        self,
        spec: ModelSpec,
        definitions: dict[str, str],
        options: dict,
        max_new_tokens: int,
        max_side: int,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.definitions = definitions
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(spec.checkpoint, **options)
        # Bound the processor's own resize as well as the input thumbnail.
        self.processor.image_processor.size = {"longest_edge": max_side}
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                spec.checkpoint,
                dtype=torch.float32,
                attn_implementation="eager",
                **options,
            )
            .to("cpu")
            .eval()
        )
        prompt = classification_prompt(definitions)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        self.prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

    def predict(self, image) -> Prediction:
        inputs = self.processor(text=self.prompt, images=[image], return_tensors="pt")
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        new_tokens = output[:, inputs["input_ids"].shape[-1] :]
        text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        prediction = parse_vlm_response(text, self.definitions)
        prediction.generated_tokens = new_tokens.shape[-1]
        return prediction


class MiniCPMAdapter(Adapter):
    """MiniCPM-V 4.0 CPU float32 baseline, using its publisher's chat interface."""

    def __init__(
        self,
        spec: ModelSpec,
        definitions: dict[str, str],
        options: dict,
        max_new_tokens: int,
    ):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.definitions = definitions
        self.max_new_tokens = max_new_tokens
        self.prompt = classification_prompt(definitions)
        self.processor = AutoProcessor.from_pretrained(
            spec.checkpoint, trust_remote_code=True, use_fast=True, **options
        )
        self.model = (
            AutoModel.from_pretrained(
                spec.checkpoint,
                trust_remote_code=True,
                dtype=torch.float32,
                attn_implementation="sdpa",
                **options,
            )
            .to("cpu")
            .eval()
        )

    def predict(self, image) -> Prediction:
        messages = [{"role": "user", "content": [image, self.prompt]}]
        with self.torch.inference_mode():
            text = self.model.chat(
                msgs=messages,
                tokenizer=self.processor.tokenizer,
                processor=self.processor,
                sampling=False,
                num_beams=1,
                repetition_penalty=1.0,
                max_new_tokens=self.max_new_tokens,
                max_slice_nums=1,
            )
        return parse_vlm_response(text, self.definitions)


class VisionAdapter(Adapter):
    """Frozen features + nearest labeled reference per class; no weight training."""

    def __init__(self, spec: ModelSpec, definitions: dict[str, str], options: dict):
        import torch

        self.torch = torch
        self.labels = list(definitions)
        self.reference_features = None
        self.reference_labels: list[str] = []
        if spec.kind == "timm":
            import timm
            from timm.data import create_transform, resolve_model_data_config

            # These names use Hugging Face timm checkpoints, including offline cache support.
            self.model = (
                timm.create_model(
                    "hf-hub:timm/" + spec.checkpoint,
                    pretrained=True,
                    num_classes=0,
                    cache_dir=options.get("cache_dir"),
                )
                .to("cpu")
                .eval()
            )
            self.transform = create_transform(
                **resolve_model_data_config(self.model), is_training=False
            )
            self.processor = None
        else:
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(
                spec.checkpoint, use_fast=False, **options
            )
            self.model = (
                AutoModel.from_pretrained(
                    spec.checkpoint,
                    dtype=torch.float32,
                    attn_implementation="eager",
                    **options,
                )
                .to("cpu")
                .eval()
            )

    def _features(self, image):
        with self.torch.inference_mode():
            if self.processor is None:
                features = self.model(self.transform(image).unsqueeze(0))
            else:
                output = self.model(**self.processor(images=image, return_tensors="pt"))
                features = output.last_hidden_state[:, 0]
            return self.torch.nn.functional.normalize(features, dim=-1)

    def prepare(
        self,
        references: list[Sample],
        throttle: Throttler,
        check_resources: Callable[[], None],
        max_side: int,
    ) -> None:
        features = []
        for sample in references:
            check_resources()
            with throttle.operation():
                features.append(
                    self._features(
                        load_image(
                            sample.path,
                            max_side,
                            getattr(self, "image_policy", "native"),
                        )
                    )
                )
            self.reference_labels.append(sample.label)
        self.reference_features = self.torch.cat(features)

    def predict(self, image) -> Prediction:
        similarities = (self._features(image) @ self.reference_features.T)[0].tolist()
        scores = {
            label: max(
                score
                for score, ref_label in zip(similarities, self.reference_labels)
                if ref_label == label
            )
            for label in self.labels
        }
        # Cosine similarity margin, not a probability. Higher means more anomalous.
        anomaly_score = (
            max(score for label, score in scores.items() if label != "normal")
            - scores["normal"]
        )
        return Prediction(max(scores, key=scores.get), anomaly_score, scores)


def create_adapter(
    name: str,
    definitions: dict[str, str],
    cache_dir: str | None,
    offline: bool,
    max_new_tokens: int,
    max_side: int,
) -> Adapter:
    spec = MODELS[name]
    options = {"local_files_only": offline}
    if spec.revision:
        options["revision"] = spec.revision
    if cache_dir:
        options["cache_dir"] = cache_dir
    if spec.kind == "clip":
        return CLIPAdapter(spec, definitions, options)
    if spec.kind == "openclip":
        return OpenCLIPAdapter(spec, definitions, options)
    if spec.kind == "siglip":
        return SigLIPAdapter(spec, definitions, options)
    if spec.kind == "minicpm":
        return MiniCPMAdapter(spec, definitions, options, max_new_tokens)
    if spec.kind == "vlm":
        return VLMAdapter(spec, definitions, options, max_new_tokens, max_side)
    return VisionAdapter(spec, definitions, options)
