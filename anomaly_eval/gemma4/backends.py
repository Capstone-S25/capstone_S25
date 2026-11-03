"""Native image inputs and real answer-logit scoring, without test-label hints."""

from __future__ import annotations

import base64
import io
import math
from functools import lru_cache

from PIL import Image, ImageDraw

from anomaly_eval.gemma4.protocol import ROOT, question, view_indices
from anomaly_eval.models import load_image


@lru_cache(maxsize=512)
def _image(path):
    return load_image(ROOT / path, 1536, "native")


def images_for(row, policy):
    images = [
        _image(row["views"][i]).copy() for i in view_indices(row, policy.max_views)
    ]
    if policy.layout == "native" or len(images) == 1:
        return images
    if policy.layout != "sheet":
        raise ValueError("Unknown image layout")
    columns = 2 if len(images) <= 4 else 4
    rows = math.ceil(len(images) / columns)
    width = max(im.width for im in images)
    height = max(im.height for im in images)
    sheet = Image.new("RGB", (columns * width, rows * (height + 24)), (127, 127, 127))
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * (height + 24)
        sheet.paste(
            image,
            (x + (width - image.width) // 2, y + 24 + (height - image.height) // 2),
        )
        draw.text((x + 8, y + 5), f"Frame {index + 1}", fill=(255, 255, 255))
    return [sheet]


def hf_messages(row, policy):
    images = images_for(row, policy)
    return [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images]
            + [{"type": "text", "text": question(row, policy)}],
        }
    ], images


class TransformersBackend:
    """BF16 prefill-only classification, with OOM retry and no CPU fallback."""

    messages = staticmethod(hf_messages)

    def __init__(self, checkpoint, *, batch_size=8):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("Gemma accuracy runs require the available CUDA GPU")
        torch.set_num_threads(min(16, len(__import__("os").sched_getaffinity(0))))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self.torch = torch
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, local_files_only=True
        )
        self.processor.tokenizer.padding_side = "left"
        self.model = (
            AutoModelForMultimodalLM.from_pretrained(
                checkpoint,
                dtype=torch.bfloat16,
                device_map="cuda:0",
                attn_implementation={
                    "": "sdpa",
                    "text_config": "sdpa",
                    "vision_config": "sdpa",
                    "audio_config": "eager",
                },
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
        )
        self.yes, self.no = [
            self.processor.tokenizer.encode(word, add_special_tokens=False)
            for word in ["Yes", "No"]
        ]
        if len(self.yes) != 1 or len(self.no) != 1 or self.yes == self.no:
            raise ValueError("Gemma answer verbalizers must be distinct single tokens")
        self.yes, self.no = self.yes[0], self.no[0]
        self.metadata = {
            "backend": "transformers",
            "dtype": "bfloat16",
            "quantization": None,
            "device": torch.cuda.get_device_name(),
            "yes_token": self.yes,
            "no_token": self.no,
            "thinking": False,
            "canonical_microbatch": 1,
            "batching_note": "Canonical individual forwards avoid observed mixed-length BF16 margin drift; dispatch batches do not change an example's tensor batch",
            "model_bytes": self.model.get_memory_footprint(),
        }

    def _batch(self, rows, policy):
        torch = self.torch
        texts = []
        images = []
        for row in rows:
            messages, views = self.messages(row, policy)
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            images.append(views)
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            images_kwargs={"max_soft_tokens": policy.image_tokens},
        )
        inputs = {
            k: v.to(
                device="cuda:0", dtype=torch.bfloat16 if v.is_floating_point() else None
            )
            for k, v in inputs.items()
        }
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        with torch.inference_mode():
            logits = (
                self.model(**inputs, logits_to_keep=1, use_cache=False)
                .logits[:, -1]
                .float()
            )
            margins = (logits[:, self.yes] - logits[:, self.no]).cpu().tolist()
            masses = (
                torch.exp(
                    torch.logsumexp(logits[:, [self.yes, self.no]], dim=-1)
                    - torch.logsumexp(logits, dim=-1)
                )
                .cpu()
                .tolist()
            )
        if any(not math.isfinite(x) for x in margins + masses):
            raise ValueError("Nonfinite Gemma answer scores")
        return [
            {"score": score, "answer_token_mass": mass, "images": len(views)}
            for score, mass, views in zip(margins, masses, images)
        ]

    def score(self, rows, policy):
        """Keep each score independent of padding and other cases in a dispatch."""
        if len(rows) > 1:
            return [self._batch([row], policy)[0] for row in rows]
        try:
            return self._batch(rows, policy)
        except self.torch.OutOfMemoryError:
            self.torch.cuda.empty_cache()
            if len(rows) == 1:
                raise
            midpoint = len(rows) // 2
            return self.score(rows[:midpoint], policy) + self.score(
                rows[midpoint:], policy
            )

    def verify_batch(self, rows, policy, tolerance=0.12):
        batch = self.score(rows, policy)
        single = [self.score([row], policy)[0] for row in rows]
        error = max(abs(a["score"] - b["score"]) for a, b in zip(batch, single))
        if error > tolerance:
            raise ValueError(
                f"Batch/single BF16 margin drift {error:g} exceeds {tolerance:g}"
            )
        return {
            "examples": len(rows),
            "execution": "Canonical single-example forwards, dispatched together versus individually; mixed tensor batching is excluded",
            "max_margin_error": error,
            "tolerance": tolerance,
            "batched_scores": [r["score"] for r in batch],
            "single_scores": [r["score"] for r in single],
        }

    def close(self):
        del self.model
        self.torch.cuda.empty_cache()


def data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
