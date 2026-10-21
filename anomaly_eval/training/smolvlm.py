"""Task-conditioned SmolVLM LoRA with immutable, cached visual tokens.

Only the annotated binary objective contributes a response-token loss. The
vision tower, connector, embeddings and LM output head remain frozen. Selection
and operating points use development data, never retrospective test labels.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from torch.utils.data import Dataset

from anomaly_eval.models import MODELS, load_image
from anomaly_eval.training.clip import binary_metrics, select_points
from anomaly_eval.training.data import TASKS, group_class_weights

ROOT = Path(__file__).resolve().parents[2]
SPEC = MODELS["smolvlm-500m"]
IMAGE_SIZE = 512
VERBALIZERS = (" No", " Yes")
LORA_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def local_checkpoint():
    from huggingface_hub import snapshot_download

    return snapshot_download(
        SPEC.checkpoint, revision=SPEC.revision, local_files_only=True
    )


def processor_and_model(*, dtype=torch.float32, device="cuda"):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    directory = local_checkpoint()
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": IMAGE_SIZE}
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        directory,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa" if str(device).startswith("cuda") else "eager",
    ).to(device)
    model.config.use_cache = False
    return processor, model.eval().requires_grad_(False)


def messages(task, count):
    if task not in TASKS or not 1 <= count <= 8:
        raise ValueError("Require a defined target and one to eight ordered images")
    definitions = json.loads((ROOT / "anomaly_eval/config/anomalies.json").read_text())
    context = (
        "This image shows one moment."
        if count == 1
        else "These images show one scene or clip, ordered from earliest to latest."
    )
    question = (
        f"{context} Target: {task.replace('_', ' ')}. "
        f"Definition: {definitions[task]} "
        "Use only the provided visual evidence; do not invent duration or ownership. "
        "Does this evidence match the target? Answer Yes or No."
    )
    return [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(count)]
            + [{"type": "text", "text": question}],
        }
    ]


def response_ids(processor):
    tokens = [
        processor.tokenizer.encode(x, add_special_tokens=False) for x in VERBALIZERS
    ]
    if any(len(x) != 1 for x in tokens) or tokens[0] == tokens[1]:
        raise ValueError("The fixed binary responses must be distinct single tokens")
    return [x[0] for x in tokens]


def prompt_ids(processor, task, count):
    """Use the publisher's processor to expand image wrappers, never guess them."""
    from PIL import Image

    text = processor.apply_chat_template(
        messages(task, count), add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=[[Image.new("RGB", (32, 32), (127, 127, 127)) for _ in range(count)]],
        return_tensors="pt",
    )
    ids = inputs["input_ids"][0]
    if int((ids == processor.image_token_id).sum()) != count * processor.image_seq_len:
        raise ValueError("Image token expansion and cached visual token count disagree")
    return ids


def epoch_indices(rows, seed, epoch):
    """Visit every training example once, plus one balanced resampled pass."""
    if any(r["split"] != "train" or r["target"] not in (0, 1) for r in rows):
        raise ValueError("Optimization accepts only known training labels")
    generator = torch.Generator().manual_seed(seed * 1000 + epoch)
    weights = torch.tensor(group_class_weights(rows), dtype=torch.double)
    extra = torch.multinomial(weights, len(rows), replacement=True, generator=generator)
    indices = torch.cat([torch.arange(len(rows)), extra])
    return indices[torch.randperm(len(indices), generator=generator)].tolist()


def response_loss(logits, targets, ids):
    """Full-vocabulary CE at the first answer token; all prompt tokens are masked."""
    if targets.ndim != 1 or not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("Unknown objectives must never be negative response labels")
    labels = torch.tensor(ids, device=logits.device)[targets.long()]
    return F.cross_entropy(logits.float(), labels)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def attach_lora(model, *, rank=8):
    from peft import LoraConfig, get_peft_model

    targets = [
        name
        for name, module in model.named_modules()
        if "text_model.layers." in name
        and name.rsplit(".", 1)[-1] in LORA_MODULES
        and isinstance(module, torch.nn.Linear)
    ]
    if len(targets) != 32 * 7:
        raise ValueError("Unexpected pinned SmolVLM language-layer architecture")
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            target_modules=targets,
            bias="none",
        ),
    )
    model.peft_config["default"].base_model_name_or_path = SPEC.checkpoint
    model.peft_config["default"].revision = SPEC.revision
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (
            "lora_" not in name or "text_model" not in name
        ):
            raise ValueError("Unexpected trainable parameter outside language LoRA")
    return model


class VisualImages(Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = load_image(ROOT / self.paths[index], IMAGE_SIZE, "letterbox")
        inputs = self.processor.image_processor(images=[[image]], return_tensors="pt")
        return {
            key: value[0] for key, value in inputs.items() if torch.is_tensor(value)
        }


class TokenBatches:
    def __init__(self, rows, processor, cache, device="cuda"):
        self.rows = rows
        self.processor = processor
        self.device = device
        index = json.loads((cache / "index.json").read_text())
        self.tokens = torch.from_numpy(np.load(cache / "tokens.npy", mmap_mode="c"))
        self.indices = index["pixel_to_index"]
        self.prompts = {
            key: prompt_ids(processor, *key)
            for key in sorted({(r["task"], len(r["views"])) for r in rows})
        }

    def batch(self, indices, dtype):
        rows = [self.rows[i] for i in indices]
        prompts = [self.prompts[r["task"], len(r["views"])] for r in rows]
        inputs = self.processor.tokenizer.pad(
            [{"input_ids": x.tolist()} for x in prompts],
            padding=True,
            return_tensors="pt",
        )
        # Explicit positions make the last response invariant to left padding.
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        image_indices = [self.indices[p] for r in rows for p in r["pixel_hashes"]]
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        inputs["image_hidden_states"] = self.tokens[image_indices].to(
            device=self.device, dtype=dtype
        )
        targets = torch.tensor([r["target"] for r in rows], device=self.device)
        return inputs, targets


def evaluate_scores(rows, scores, thresholds=None):
    if len(rows) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Require one finite binary margin per annotated example")
    result = {}
    for task in TASKS:
        selected = [i for i, row in enumerate(rows) if row["task"] == task]
        if not selected:
            continue
        y = np.array([rows[i]["target"] for i in selected])
        values = np.asarray(scores)[selected]
        points = select_points(y, values) if thresholds is None else thresholds[task]
        sources = {}
        for source in sorted({rows[i]["source"] for i in selected}):
            ss = [i for i in selected if rows[i]["source"] == source]
            sources[source] = {
                point: binary_metrics(
                    [rows[i]["target"] for i in ss], np.asarray(scores)[ss], threshold
                )
                for point, threshold in points.items()
            }
        result[task] = {
            "thresholds": points,
            "operating_points": {
                point: binary_metrics(y, values, threshold)
                for point, threshold in points.items()
            },
            "sources": sources,
            "groups": len({rows[i]["group_id"] for i in selected}),
        }
    return result


def task_ap(rows, scores, task):
    values = []
    for source in sorted({r["source"] for r in rows if r["task"] == task}):
        ss = [
            i for i, r in enumerate(rows) if r["task"] == task and r["source"] == source
        ]
        y = [rows[i]["target"] for i in ss]
        if set(y) == {0, 1}:
            values.append(average_precision_score(y, np.asarray(scores)[ss]))
    if not values:
        raise ValueError("Every selection task needs a two-class development source")
    return float(np.mean(values))


def macro_ap(rows, scores):
    return float(np.mean([task_ap(rows, scores, task) for task in TASKS]))


def predict(model, batches, rows, batch_size=16, *, progress=None):
    model.eval()
    dtype = next(model.parameters()).dtype
    ids = response_ids(batches.processor)
    scores, masses = [], []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            inputs, _ = batches.batch(
                range(start, min(start + batch_size, len(rows))), dtype
            )
            logits = (
                model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1].float()
            )
            pair = logits[:, ids]
            scores.extend((pair[:, 1] - pair[:, 0]).cpu().tolist())
            masses.extend(logits.softmax(-1)[:, ids].sum(-1).cpu().tolist())
            if progress and (start // batch_size) % 25 == 0:
                print(
                    progress,
                    min(start + batch_size, len(rows)),
                    "/",
                    len(rows),
                    flush=True,
                )
    return np.array(scores, dtype=np.float64), np.array(masses)


def write_predictions(path, rows, scores, masses):
    content = []
    for row, score, mass in zip(rows, scores, masses):
        content.append(
            json.dumps(
                {
                    **{
                        key: row[key]
                        for key in (
                            "id",
                            "task",
                            "target",
                            "source",
                            "group_id",
                            "split",
                        )
                    },
                    "score": float(score),
                    "yes_no_token_mass": float(mass),
                },
                allow_nan=False,
            )
        )
    path.write_text("\n".join(content) + "\n")


def counts(rows):
    return dict(Counter(f"{r['split']}/{r['task']}/{r['target']}" for r in rows))
