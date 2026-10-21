"""Frozen probes and last-stage CLIP fine-tuning with masked target supervision.

Selection uses development AP only. Test/transfer labels never enter optimization,
early stopping, threshold selection, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import ColorJitter, RandomHorizontalFlip

from anomaly_eval.models import MODELS, load_image
from anomaly_eval.training.data import (
    TASKS,
    group_class_weights,
    manifest_hash,
    read_records,
)

ROOT = Path(__file__).resolve().parents[2]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pool_views(features, lengths):
    """Preserve endpoint, pooled context and temporal change, without future input."""
    groups = features.split([int(x) for x in lengths], dim=0)
    return F.normalize(
        torch.stack(
            [torch.cat([x[-1], x.mean(0), x.amax(0), x[-1] - x[0]]) for x in groups]
        ),
        dim=-1,
    )


def masked_binary_loss(values, tasks, targets):
    """Unknown objectives have zero loss/gradient for this training example."""
    return F.binary_cross_entropy_with_logits(
        values.gather(1, tasks[:, None])[:, 0].float(), targets
    )


def select_points(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(y) != len(score) or not np.isfinite(score).all() or set(y) != {0, 1}:
        raise ValueError("Calibration requires finite scores and both dev classes")
    choices = np.r_[np.nextafter(score.max(), np.inf), np.unique(score)]
    values = []
    for threshold in choices:
        pred = score >= threshold
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        fpr = fp / int((y == 0).sum())
        values.append((f1, -fpr, float(threshold)))
    best = max(values)
    budget = max(x for x in values if -x[1] <= 0.05)
    return {"dev_max_f1": best[2], "dev_fpr_budget": budget[2], "raw": 0.0}


def binary_metrics(y, score, threshold):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    tn = int(((~pred) & (y == 0)).sum())
    return {
        "n": len(y),
        "positive": int(y.sum()),
        "negative": int((1 - y).sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "fpr": fp / max(1, fp + tn),
        "ap": float(average_precision_score(y, score)) if len(set(y)) == 2 else None,
        "auroc": float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
    }


def evaluate(rows, scores, thresholds=None):
    output = {}
    for task in TASKS:
        indices = [i for i, r in enumerate(rows) if r["task"] == task]
        if not indices:
            continue
        y = np.array([rows[i]["target"] for i in indices])
        values = scores[indices, TASKS.index(task)]
        points = select_points(y, values) if thresholds is None else thresholds[task]
        output[task] = {
            "thresholds": points,
            "operating_points": {
                key: binary_metrics(y, values, value) for key, value in points.items()
            },
            "groups": len({rows[i]["group_id"] for i in indices}),
            "sources": {},
        }
        for source in sorted({rows[i]["source"] for i in indices}):
            selected = [i for i in indices if rows[i]["source"] == source]
            yy = np.array([rows[i]["target"] for i in selected])
            ss = scores[selected, TASKS.index(task)]
            output[task]["sources"][source] = {
                key: binary_metrics(yy, ss, value) for key, value in points.items()
            }
    return output


def mean_ap(evaluation):
    return float(
        np.mean(
            [
                np.mean(
                    [
                        source["raw"]["ap"]
                        for source in v["sources"].values()
                        if source["raw"]["ap"] is not None
                    ]
                )
                for v in evaluation.values()
                if v["operating_points"]["raw"]["ap"] is not None
            ]
        )
    )


class Encoder(nn.Module):
    def __init__(self, name, device):
        super().__init__()
        self.name = name
        self.device_name = device
        spec = MODELS[name]
        self.revision = spec.revision
        self.augment = ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)
        self.flip = RandomHorizontalFlip(0.5)
        definitions = json.loads(
            (ROOT / "anomaly_eval/config/anomalies.json").read_text()
        )
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
            from huggingface_hub import snapshot_download

            directory = snapshot_download(
                spec.checkpoint, revision=spec.revision, local_files_only=True
            )
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
                spec.checkpoint,
                revision=spec.revision,
                local_files_only=True,
                use_fast=False,
            )
            self.backbone = AutoModel.from_pretrained(
                spec.checkpoint,
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
            # Last-only text decision is the zero-shot baseline before fitting.
            self.head.weight[:, : self.dimension].copy_(
                self.text_direction * self.scale
            )

    def image_tensor(self, path, augment=False):
        image = load_image(path, 768, "letterbox")
        if augment:
            image = self.augment(self.flip(image))
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

    def unfreeze_last_stage(self):
        patterns = (
            ("visual.trunk.stages.3.", "visual.trunk.final_conv.", "visual.trunk.head.")
            if self.name == "mobileclip2-s2"
            else (
                "vision_model.encoder.layers.11.",
                "vision_model.post_layernorm.",
                "vision_model.head.",
            )
        )
        names = []
        for name, parameter in self.backbone.named_parameters():
            if name.startswith(patterns):
                parameter.requires_grad_(True)
                names.append(name)
        if not names:
            raise RuntimeError("No last-stage parameters matched this architecture")
        self.backbone.eval()  # Freeze all BatchNorm statistics/dropout even while weights adapt.
        return names


class Images(Dataset):
    def __init__(self, rows, encoder, augment=False):
        self.rows = rows
        self.encoder = encoder
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return (
            torch.stack(
                [
                    self.encoder.image_tensor(ROOT / p, self.augment)
                    for p in row["views"]
                ]
            ),
            TASKS.index(row["task"]),
            float(row["target"]),
            index,
        )


def collate(batch):
    images, tasks, targets, indices = zip(*batch)
    return (
        torch.cat(images),
        [len(x) for x in images],
        torch.tensor(tasks),
        torch.tensor(targets),
        torch.tensor(indices),
    )


def loader(rows, encoder, batch_size, workers=6, augment=False, sampler=None):
    return DataLoader(
        Images(rows, encoder, augment),
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=encoder.device_name.startswith("cuda"),
        collate_fn=collate,
        persistent_workers=False,
    )


def autocast(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )


def extract(rows, encoder, batch_size, workers):
    output = []
    encoder.eval()
    with torch.no_grad():
        for images, lengths, _, _, _ in loader(rows, encoder, batch_size, workers):
            with autocast(encoder.device_name):
                features = encoder.encode(
                    images.to(encoder.device_name, non_blocking=True), lengths
                )
            output.append(features.cpu().numpy())
    return np.concatenate(output)


def frozen_extract(rows, encoder, batch_size, workers):
    if any(p.requires_grad for p in encoder.backbone.parameters()):
        raise ValueError("Feature cache is only valid for the immutable frozen base")
    cache_path = ROOT / "data/models/clip_training_features" / (encoder.name + ".pt")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = (
        torch.load(cache_path, map_location="cpu", weights_only=True)
        if cache_path.exists()
        else {}
    )

    def key(row):
        identity = [
            encoder.revision,
            "letterbox768_native_v1",
            "temporal_last_mean_max_delta_v1",
            row["pixel_hashes"],
        ]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    keys = [key(row) for row in rows]
    missing = [i for i, k in enumerate(keys) if k not in cache]
    if missing:
        values = extract([rows[i] for i in missing], encoder, batch_size, workers)
        for i, value in zip(missing, values):
            cache[keys[i]] = torch.from_numpy(value.copy())
        temporary = cache_path.with_suffix(".tmp")
        torch.save(cache, temporary)
        temporary.replace(cache_path)
    print(
        encoder.name,
        "frozen cache",
        len(rows) - len(missing),
        "hits",
        len(missing),
        "new",
        flush=True,
    )
    return torch.stack([cache[k] for k in keys]).numpy()


def logits(features, head):
    return (
        features @ head.weight.detach().cpu().numpy().T
        + head.bias.detach().cpu().numpy()
    )


def zero_logits(features, encoder):
    last = features[:, : encoder.dimension]
    last = last / np.maximum(np.linalg.norm(last, axis=1, keepdims=True), 1e-12)
    return last @ encoder.text_direction.detach().cpu().numpy().T * encoder.scale


def score_predictions(path, rows, scores):
    with path.open("w") as stream:
        for row, values in zip(rows, scores):
            stream.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "task": row["task"],
                        "target": row["target"],
                        "source": row["source"],
                        "group_id": row["group_id"],
                        "scores": dict(zip(TASKS, map(float, values))),
                    }
                )
                + "\n"
            )


def checkpoint(path, encoder, metadata):
    # Save every trained tensor and head, plus the exact immutable base revision.
    parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in encoder.named_parameters()
        if parameter.requires_grad
    }
    torch.save(
        {
            "format_version": 1,
            "model": encoder.name,
            "base_revision": encoder.revision,
            "tasks": TASKS,
            "parameters": parameters,
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path, encoder):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["model"] != encoder.name or state["base_revision"] != encoder.revision:
        raise ValueError("Checkpoint/base revision mismatch")
    with torch.no_grad():
        parameters = dict(encoder.named_parameters())
        for name, value in state["parameters"].items():
            parameters[name].copy_(value.to(encoder.device_name))
    return state


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("This accuracy training run requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    rows = read_records(args.manifest, ROOT)
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_rows = [r for r in rows if r["split"] == "dev"]
    out = args.output / args.model
    out.mkdir(parents=True, exist_ok=False)
    identity = {
        "code_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "anomaly_eval/training/data.py",
                ROOT / "anomaly_eval/models.py",
            ]
        },
        "model": args.model,
        "manifest_sha256": manifest_hash(args.manifest),
        "seed": args.seed,
        "device": "cuda",
        "precision": "float32 weights; bfloat16 autocast",
        "batch_size_samples": args.batch_size,
        "cudnn_benchmark": args.cudnn_benchmark,
        "threads": args.threads,
        "epochs": args.epochs,
        "learning_rates": args.learning_rates,
        "selection": "development macro task AP, averaging dataset sources within each task",
        "image_policy": "letterbox 768 before native processor; no random crops",
        "temporal_pool": "normalized concatenation of last, mean, max, last-first frame embeddings",
        "masked_targets": True,
        "BN_statistics": "frozen",
        "task_order": list(TASKS),
    }
    save_json(out / "state.json", {"status": "extracting_train_dev", **identity})
    encoder = Encoder(args.model, "cuda")
    print(
        args.model,
        "extracting",
        len(train_rows),
        "train",
        len(dev_rows),
        "dev",
        flush=True,
    )
    train_features = frozen_extract(train_rows, encoder, args.batch_size, args.workers)
    dev_features = frozen_extract(dev_rows, encoder, args.batch_size, args.workers)
    np.savez_compressed(
        out / "frozen_features.npz", train=train_features, dev=dev_features
    )
    if args.prepare_features_only:
        save_json(out / "state.json", {"status": "features_prepared_only", **identity})
        return
    zero_dev_scores = zero_logits(dev_features, encoder)
    zero_dev = evaluate(dev_rows, zero_dev_scores)
    probe_candidates = []
    weights = np.array(group_class_weights(train_rows))
    weights *= len(weights) / weights.sum()
    for c in [1.0, 10.0, 100.0]:
        weight = np.zeros((len(TASKS), train_features.shape[-1]), np.float32)
        bias = np.zeros(len(TASKS), np.float32)
        supported = []
        for ti, task in enumerate(TASKS):
            indices = [i for i, r in enumerate(train_rows) if r["task"] == task]
            if not indices:
                continue
            labels = np.array([train_rows[i]["target"] for i in indices])
            model = LogisticRegression(C=c, max_iter=2000, random_state=args.seed).fit(
                train_features[indices], labels, sample_weight=weights[indices]
            )
            weight[ti] = model.coef_[0]
            bias[ti] = model.intercept_[0]
            supported.append(task)
        scores = dev_features @ weight.T + bias
        evaluation = evaluate(dev_rows, scores)
        candidate = {
            "C": c,
            "dev_macro_ap": mean_ap(evaluation),
            "dev": evaluation,
            "weight": weight,
            "bias": bias,
        }
        probe_candidates.append(candidate)
        print(args.model, "probe C", c, "dev AP", candidate["dev_macro_ap"], flush=True)
    probe = max(probe_candidates, key=lambda x: x["dev_macro_ap"])
    with torch.no_grad():
        encoder.head.weight.copy_(torch.from_numpy(probe["weight"]).to("cuda"))
        encoder.head.bias.copy_(torch.from_numpy(probe["bias"]).to("cuda"))
    checkpoint(
        out / "linear_probe.pt",
        encoder,
        {
            "C": probe["C"],
            "manifest_sha256": identity["manifest_sha256"],
            "supported_tasks": supported,
        },
    )
    save_json(
        out / "linear_probe_dev.json",
        {k: v for k, v in probe.items() if k not in {"weight", "bias"}},
    )
    selected_names = encoder.unfreeze_last_stage()
    trainable = [p for p in encoder.backbone.parameters() if p.requires_grad]
    anchor = {
        n: p.detach().clone()
        for n, p in encoder.backbone.named_parameters()
        if p.requires_grad
    }
    original = {
        n: p.detach().clone() for n, p in encoder.named_parameters() if p.requires_grad
    }
    print(
        args.model,
        "trainable vision parameters",
        sum(p.numel() for p in trainable),
        "names",
        len(selected_names),
        flush=True,
    )
    save_json(
        out / "training_parameters.json",
        {
            "vision_names": selected_names,
            "vision_parameters": sum(p.numel() for p in trainable),
            "head_parameters": sum(p.numel() for p in encoder.head.parameters()),
            "augmentation": "horizontal flip .5; brightness/contrast .15; saturation .1; no crop",
        },
    )
    teacher = torch.from_numpy(train_features).to("cuda")
    history = []
    best_global = None
    for learning_rate in args.learning_rates:
        with torch.no_grad():
            for name, parameter in encoder.named_parameters():
                if name in original:
                    parameter.copy_(original[name])
        torch.manual_seed(args.seed)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            len(train_rows),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = loader(
            train_rows, encoder, args.batch_size, args.workers, True, sampler
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": trainable, "lr": learning_rate},
                {"params": encoder.head.parameters(), "lr": 3e-4},
            ],
            weight_decay=0.01,
        )
        best_run = -1.0
        stale = 0
        for epoch in range(1, args.epochs + 1):
            encoder.eval()  # Deliberately freeze BN buffers and pretrained dropout.
            losses = []
            for images, lengths, tasks, targets, indices in train_loader:
                images = images.to("cuda", non_blocking=True)
                tasks = tasks.to("cuda")
                targets = targets.to("cuda")
                indices = indices.to("cuda")
                optimizer.zero_grad(set_to_none=True)
                with autocast("cuda"):
                    features = encoder.encode(images, lengths)
                    values = encoder.head(F.dropout(features, p=0.1, training=True))
                    # Only the annotated target contributes to the loss.
                    loss = masked_binary_loss(values, tasks, targets)
                    loss = loss + 0.1 * F.mse_loss(features.float(), teacher[indices])
                    loss = loss + 0.1 * sum(
                        (p - anchor[n]).square().mean()
                        for n, p in encoder.backbone.named_parameters()
                        if p.requires_grad
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*trainable, *encoder.head.parameters()], 1.0
                )
                optimizer.step()
                losses.append(float(loss.detach()))
            validation = extract(dev_rows, encoder, args.batch_size, args.workers)
            dev_scores = logits(validation, encoder.head)
            evaluation = evaluate(dev_rows, dev_scores)
            ap = mean_ap(evaluation)
            entry = {
                "learning_rate": learning_rate,
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "dev_macro_ap": ap,
                "dev": evaluation,
            }
            history.append(entry)
            print(
                args.model,
                "lr",
                learning_rate,
                "epoch",
                epoch,
                "loss",
                round(entry["loss"], 4),
                "dev AP",
                round(ap, 4),
                flush=True,
            )
            if ap > best_run + 1e-5:
                best_run = ap
                stale = 0
            else:
                stale += 1
            if best_global is None or ap > best_global["dev_macro_ap"]:
                best_global = deepcopy(entry)
                checkpoint(
                    out / "finetuned_best.pt",
                    encoder,
                    {
                        **entry,
                        "manifest_sha256": identity["manifest_sha256"],
                        "supported_tasks": supported,
                    },
                )
            save_json(out / "history.json", history)
            save_json(
                out / "state.json",
                {
                    "status": "training",
                    **identity,
                    "current": entry,
                    "best": best_global,
                },
            )
            if stale >= 3:
                break
        del optimizer, train_loader
    # Freeze selection before computing any held-out predictions or metrics.
    dev_candidates = {
        "zero_shot": zero_dev,
        "linear_probe": probe["dev"],
        "finetuned": best_global["dev"],
    }
    selected = max(dev_candidates, key=lambda name: mean_ap(dev_candidates[name]))
    task_routes = {
        task: max(
            dev_candidates, key=lambda name: mean_ap({task: dev_candidates[name][task]})
        )
        for task in supported
    }
    save_json(
        out / "selection.json",
        {
            "selected": selected,
            "zero_shot_dev_macro_ap": mean_ap(zero_dev),
            "task_routes": task_routes,
            "routing_selection": "Per-task mean development-source AP; ties prefer the simpler baseline",
            "linear_probe_dev_macro_ap": probe["dev_macro_ap"],
            "finetuned_dev_macro_ap": best_global["dev_macro_ap"],
            "finetuned_lr": best_global["learning_rate"],
            "finetuned_epoch": best_global["epoch"],
            "selection_uses_test": False,
        },
    )
    results = {
        "zero_shot": {"dev": zero_dev},
        "linear_probe": {"dev": probe["dev"]},
        "finetuned": {"dev": best_global["dev"]},
    }
    # All three protocols are fixed before looking at the held-out results.
    with torch.no_grad():
        for n, p in encoder.backbone.named_parameters():
            if n in anchor:
                p.copy_(anchor[n])
    for split in ["test", "external"]:
        subset = [r for r in rows if r["split"] == split]
        if not subset:
            continue
        frozen = extract(subset, encoder, args.batch_size, args.workers)
        zscores = zero_logits(frozen, encoder)
        pscores = frozen @ probe["weight"].T + probe["bias"]
        for key, scores in [("zero_shot", zscores), ("linear_probe", pscores)]:
            thresholds = {t: v["thresholds"] for t, v in results[key]["dev"].items()}
            results[key][split] = evaluate(subset, scores, thresholds)
            score_predictions(out / f"{key}_{split}_predictions.jsonl", subset, scores)
    load_checkpoint(out / "finetuned_best.pt", encoder)
    for split in ["test", "external"]:
        subset = [r for r in rows if r["split"] == split]
        if not subset:
            continue
        scores = logits(
            extract(subset, encoder, args.batch_size, args.workers), encoder.head
        )
        thresholds = {
            t: v["thresholds"] for t, v in results["finetuned"]["dev"].items()
        }
        results["finetuned"][split] = evaluate(subset, scores, thresholds)
        score_predictions(out / f"finetuned_{split}_predictions.jsonl", subset, scores)
    save_json(out / "results.json", results)
    save_json(
        out / "state.json",
        {
            "status": "complete",
            **identity,
            "supported_tasks": supported,
            "selected": selected,
            "gpu_name": torch.cuda.get_device_name(),
            "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        },
    )
    print(args.model, "COMPLETE selected", selected, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/clip_training_20261015/samples.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["mobileclip2-s2", "siglip2-b16-224"], required=True
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[1e-5, 3e-5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-features-only", action="store_true")
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Opt-in convolution autotuning; expensive with variable frame batches",
    )
    args = parser.parse_args(argv)
    if (
        min(args.batch_size, args.threads, args.epochs) < 1
        or args.workers < 0
        or any(x <= 0 for x in args.learning_rates)
    ):
        parser.error("Invalid training resource/hyperparameter values")
    train(args)


if __name__ == "__main__":
    main()
