"""Pinned pretrained encoder with the original inference transforms."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F

from .config import model_paths
from .evidence import DEFINITIONS, TASKS

MODELS = {
    "mobileclip2-s2": SimpleNamespace(
        revision="ac6b37c8fc40b62b623d09ed228a6f0c9bc29fe6"
    ),
    "siglip2-b16-224": SimpleNamespace(
        revision="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
    ),
}


def load_image(path: Path, max_side: int, policy: str = "native"):
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


def pool_views(features, lengths):
    """Preserve endpoint, pooled context and temporal change, without future input."""
    groups = features.split([int(x) for x in lengths], dim=0)
    return F.normalize(
        torch.stack(
            [torch.cat([x[-1], x.mean(0), x.amax(0), x[-1] - x[0]]) for x in groups]
        ),
        dim=-1,
    )


class Encoder(nn.Module):
    def __init__(self, name, device):
        super().__init__()
        self.name = name
        self.device_name = device
        spec = MODELS[name]
        self.revision = spec.revision

        definitions = json.loads((DEFINITIONS).read_text())
        prompts = []
        for task in TASKS:
            prompts.extend(
                [
                    f"An ordinary scene with no evidence of {task.replace('_', ' ')}.",
                    definitions[task],
                ]
            )
        if name == "mobileclip2-s2":
            import open_clip

            directory = str(model_paths(name).base)
            model_name = "local-dir:" + directory
            self.backbone, _, self.processor = open_clip.create_model_and_transforms(
                model_name, device=device, precision="fp32"
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            with torch.no_grad():
                text = self.backbone.encode_text(
                    tokenizer(prompts).to(device), normalize=True
                )
            self.scale = float(self.backbone.logit_scale.exp().detach().cpu())
        else:
            from transformers import AutoModel, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                str(model_paths(name).base),
                revision=spec.revision,
                local_files_only=True,
                use_fast=False,
            )
            self.backbone = AutoModel.from_pretrained(
                str(model_paths(name).base),
                revision=spec.revision,
                local_files_only=True,
                dtype=torch.float32,
                attn_implementation="sdpa",
            ).to(device)
            tokens = self.processor(
                text=[x.lower() for x in prompts],
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                text = F.normalize(
                    self.backbone.get_text_features(
                        **{k: v.to(device) for k, v in tokens.items()}
                    ),
                    dim=-1,
                )
            self.scale = float(self.backbone.logit_scale.exp().detach().cpu())
        self.backbone.eval().requires_grad_(False)
        self.register_buffer(
            "text_direction",
            text.reshape(len(TASKS), 2, -1)[:, 1]
            - text.reshape(len(TASKS), 2, -1)[:, 0],
        )
        self.dimension = text.shape[-1]
        self.head = nn.Linear(self.dimension * 4, len(TASKS), device=device)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.weight[:, : self.dimension].copy_(
                self.text_direction * self.scale
            )

    def image_tensor(self, path, augment=False):
        image = load_image(path, 768, "letterbox")
        if augment:
            raise ValueError("Runtime does not provide training augmentation")
        if self.name == "mobileclip2-s2":
            return self.processor(image)
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]

    def encode(self, images, lengths):
        if self.name == "mobileclip2-s2":
            features = self.backbone.encode_image(images, normalize=True)
        else:
            features = F.normalize(
                self.backbone.get_image_features(pixel_values=images), dim=-1
            )
        return pool_views(features.float(), lengths)
