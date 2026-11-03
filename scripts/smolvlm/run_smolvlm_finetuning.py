"""Prepare, train and evaluate source-held-out five-target SmolVLM LoRA."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from anomaly_eval.training.clip import save_json
from anomaly_eval.training.data import TASKS, read_records
from anomaly_eval.training.smolvlm import (
    ROOT,
    SPEC,
    TokenBatches,
    VisualImages,
    attach_lora,
    counts,
    digest,
    epoch_indices,
    evaluate_scores,
    local_checkpoint,
    macro_ap,
    predict,
    processor_and_model,
    response_ids,
    response_loss,
    seed_everything,
    task_ap,
    write_predictions,
)

DEFAULT = ROOT / "results/smolvlm_finetuning_20261019"


def prepare(out):
    source = ROOT / "data/strategy_study_20261016"
    inputs = [source / "samples.jsonl", source / "historical.jsonl"]
    rows = [json.loads(s) for p in inputs for s in p.read_text().splitlines()]
    destination = out / "samples.jsonl"
    content = "".join(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: r["id"]))
    if destination.exists() and destination.read_text() != content:
        raise ValueError("Refuse to change frozen SmolVLM source splits")
    destination.write_text(content)
    read_records(destination, ROOT, check_files=True)
    protocol = {
        "model": SPEC.checkpoint,
        "revision": SPEC.revision,
        "checkpoint_sha256": digest(Path(local_checkpoint()) / "model.safetensors"),
        "manifest_sha256": digest(destination),
        "inputs": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "counts": counts(rows),
        "examples": len(rows),
        "optimization_examples": sum(r["split"] == "train" for r in rows),
        "selection_examples": sum(r["split"] == "dev" for r in rows),
        "unique_images": len({p for r in rows for p in r["pixel_hashes"]}),
        "image_policy": "EXIF RGB, longest edge <=512, centered gray127 square letterbox; processor 512, no image splitting",
        "views": "All existing ordered views, one per static example, four per temporal window, eight per littering clip; no future inputs added",
        "training": "Language layers only, rank8 alpha16 dropout0.05 LoRA on q/k/v/o/gate/up/down; frozen vision, connector, embeddings and LM head",
        "loss": "Full-vocabulary first-answer-token cross entropy on exactly one annotated target; all prompt/image positions excluded, no unlabelled objective treated as negative",
        "sampling": "Every training example once per epoch plus an equally sized resampled pass balancing task/class/source/group; shuffle jointly",
        "selection": "Best development macro AP over five tasks, mean dataset-source AP within each task; primary two LRs, seed43 repeats primary winning LR",
        "task_routes": "Development-only choice between frozen original and two best-per-LR primary checkpoints; ties prefer original",
        "thresholds": "Raw Yes-minus-No margin0, development maximumF1, development pooled FPR<=5%; frozen before test inference",
        "score": "First-response token logit( Yes)-logit( No), conditional binary ranking, not calibrated probability; also save total vocabulary probability of the two tokens",
        "precision": "Frozen visual tokens float32, TF32 off; LoRA training bfloat16, development/checkpoint selection and final scoring float32",
        "qualification": "Retrospective source holdouts, previously inspected in earlier studies; provisional duration/ownership/sleep labels remain; no claim of fresh prospective deployment validation",
        "native_generation": "Binary forced-choice detector, no trained abstention or explanations, not comparable to old six-label free-generation accuracy",
        "no_latency_testing": True,
        "external_sources": [
            "https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct",
            "https://huggingface.co/docs/peft/v0.17.0/package_reference/lora",
            "https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/idefics3/modeling_idefics3.py",
        ],
    }
    path = out / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol already frozen")
    save_json(path, protocol)
    print(
        "Prepared",
        protocol["examples"],
        "examples",
        protocol["unique_images"],
        "images",
        flush=True,
    )


def features(out, batch_size, workers, *, audited_rows=None):
    rows = (
        read_records(out / "samples.jsonl", ROOT, check_files=False)
        if audited_rows is None
        else audited_rows
    )
    protocol = json.loads((out / "protocol.json").read_text())
    cache = out / "visual_cache"
    index_path = cache / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["manifest_sha256"] != protocol["manifest_sha256"]:
            raise ValueError("Visual cache source manifest changed")
        print("Reusing completed frozen visual cache", flush=True)
        return
    cache.mkdir(exist_ok=True)
    paths = {}
    for row in rows:
        for path, pixel in zip(row["views"], row["pixel_hashes"]):
            paths.setdefault(pixel, path)
    pixels = sorted(paths)
    processor, model = processor_and_model()
    loader = DataLoader(
        VisualImages([paths[p] for p in pixels], processor),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    tokens = np.lib.format.open_memmap(
        cache / "tokens.npy", mode="w+", dtype=np.float32, shape=(len(pixels), 64, 960)
    )
    cursor, started = 0, time.monotonic()
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            output = model.model.get_image_features(**batch).float().cpu().numpy()
            if output.shape[1:] != (64, 960) or not np.isfinite(output).all():
                raise ValueError("Unexpected visual token shape or nonfinite values")
            tokens[cursor : cursor + len(output)] = output
            cursor += len(output)
            if cursor % (10 * batch_size) == 0 or cursor == len(pixels):
                print(
                    "Visual encoding",
                    cursor,
                    "/",
                    len(pixels),
                    round(time.monotonic() - started, 1),
                    "s",
                    flush=True,
                )
    tokens.flush()
    if cursor != len(pixels):
        raise ValueError("Incomplete frozen visual token extraction")
    save_json(
        index_path,
        {
            "status": "complete",
            "manifest_sha256": protocol["manifest_sha256"],
            "checkpoint_sha256": protocol["checkpoint_sha256"],
            "pixel_to_index": {p: i for i, p in enumerate(pixels)},
            "pixel_to_path": paths,
            "shape": list(tokens.shape),
            "dtype": "float32",
            "TF32": False,
            "device": "cuda",
            "tokens_sha256": digest(cache / "tokens.npy"),
            "seconds": time.monotonic() - started,
        },
    )


def load_rows(out):
    protocol = json.loads((out / "protocol.json").read_text())
    if digest(out / "samples.jsonl") != protocol["manifest_sha256"]:
        raise ValueError("Frozen input manifest changed")
    rows = read_records(out / "samples.jsonl", ROOT, check_files=False)
    return (
        rows,
        [r for r in rows if r["split"] == "train"],
        [r for r in rows if r["split"] == "dev"],
    )


def baseline(out, batch_size):
    _, _, dev = load_rows(out)
    destination = out / "baseline"
    destination.mkdir(exist_ok=True)
    if (destination / "development_results.json").exists():
        return
    processor, model = processor_and_model()
    batches = TokenBatches(dev, processor, out / "visual_cache")
    scores, masses = predict(model, batches, dev, batch_size, progress="Original dev")
    write_predictions(destination / "dev_predictions.jsonl", dev, scores, masses)
    save_json(destination / "development_results.json", evaluate_scores(dev, scores))
    print("Original dev macroAP", macro_ap(dev, scores), flush=True)


def train_candidate(out, seed, lr, epochs, patience, batch_size, accumulation):
    _, train, dev = load_rows(out)
    directory = out / f"seed_{seed}" / f"lr_{lr:g}"
    if (directory / "state.json").exists():
        state = json.loads((directory / "state.json").read_text())
        if state["status"] == "complete":
            print("Reusing completed", directory, flush=True)
            return
        raise ValueError(
            f"Preserve incomplete training attempt before retrying: {directory}"
        )
    directory.mkdir(parents=True)
    seed_everything(seed)
    processor, model = processor_and_model(dtype=torch.bfloat16)
    model = attach_lora(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    train_batches = TokenBatches(train, processor, out / "visual_cache")
    dev_batches = TokenBatches(dev, processor, out / "visual_cache")
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(math.ceil(len(train) * 2 / batch_size) / accumulation)
    total_steps = epochs * steps_per_epoch
    warmup = max(1, int(0.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min((step + 1) / warmup, 1.0)
            * max(0.0, 1.0 - max(0, step - warmup) / max(1, total_steps - warmup))
        ),
    )
    state = {
        "status": "training",
        "seed": seed,
        "learning_rate": lr,
        "epochs_max": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "gradient_accumulation": accumulation,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "training_rows": len(train),
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(),
        "selection_precision": "float32",
        "manifest_sha256": digest(out / "samples.jsonl"),
    }
    save_json(directory / "state.json", state)
    best, stale, history, started = -1.0, 0, [], time.monotonic()
    ids = response_ids(processor)
    for epoch in range(1, epochs + 1):
        indices = epoch_indices(train, seed, epoch)
        model.to(dtype=torch.bfloat16)
        # PEFT keeps adapter optimizer state in float32; keep trainable weights there.
        for parameter in parameters:
            parameter.data = parameter.data.float()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, visits = 0.0, 0
        chunks = [
            indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
        ]
        for step, chunk in enumerate(chunks):
            inputs, targets = train_batches.batch(chunk, torch.bfloat16)
            group_start = (step // accumulation) * accumulation
            group_samples = sum(
                len(x) for x in chunks[group_start : group_start + accumulation]
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[
                    :, -1
                ]
                loss = response_loss(logits, targets, ids)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite response-token loss")
            (loss * (len(chunk) / group_samples)).backward()
            loss_sum += float(loss.detach()) * len(chunk)
            visits += len(chunk)
            if (step + 1) % accumulation == 0 or step + 1 == len(chunks):
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not torch.isfinite(norm):
                    raise ValueError("Nonfinite LoRA gradient")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(
                    "Train",
                    seed,
                    lr,
                    "epoch",
                    epoch,
                    visits,
                    "/",
                    len(indices),
                    "loss",
                    round(loss_sum / visits, 4),
                    flush=True,
                )
        model.float().eval()
        scores, masses = predict(
            model, dev_batches, dev, batch_size, progress=f"Dev {seed}/{lr}/{epoch}"
        )
        metric = macro_ap(dev, scores)
        entry = {
            "epoch": epoch,
            "loss": loss_sum / visits,
            "visits": visits,
            "unique_training_examples": len(set(indices)),
            "dev_macro_ap": metric,
            "dev_task_source_ap": {task: task_ap(dev, scores, task) for task in TASKS},
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(entry)
        save_json(directory / "history.json", history)
        print("Epoch result", json.dumps(entry), flush=True)
        if metric > best:
            best, stale = metric, 0
            model.save_pretrained(directory / "adapter", safe_serialization=True)
            processor.save_pretrained(directory / "processor")
            write_predictions(directory / "dev_predictions.jsonl", dev, scores, masses)
            save_json(
                directory / "development_results.json", evaluate_scores(dev, scores)
            )
            state.update(best_epoch=epoch, best_dev_macro_ap=best)
        else:
            stale += 1
        save_json(directory / "state.json", state)
        if stale >= patience:
            break
    state.update(
        status="complete",
        epochs_completed=len(history),
        seconds=time.monotonic() - started,
    )
    save_json(directory / "state.json", state)


def select(out, seed, lrs):
    _, _, dev = load_rows(out)
    choices = {"original": out / "baseline"}
    choices.update(
        {f"seed_{seed}/lr_{lr:g}": out / f"seed_{seed}" / f"lr_{lr:g}" for lr in lrs}
    )
    scores = {
        name: np.array(
            [
                json.loads(s)["score"]
                for s in (path / "dev_predictions.jsonl").read_text().splitlines()
            ]
        )
        for name, path in choices.items()
    }
    trained = [name for name in choices if name != "original"]
    best_joint = max(trained, key=lambda name: macro_ap(dev, scores[name]))
    routes = {
        task: max(choices, key=lambda name: task_ap(dev, scores[name], task))
        for task in TASKS
    }
    selection = {
        "seed": seed,
        "joint_adapter": best_joint,
        "task_routes": routes,
        "development_macro_ap": {
            name: macro_ap(dev, value) for name, value in scores.items()
        },
        "route_dev_source_ap": {
            task: task_ap(dev, scores[name], task) for task, name in routes.items()
        },
        "thresholds": {
            name: {
                task: value["thresholds"]
                for task, value in evaluate_scores(dev, score).items()
            }
            for name, score in scores.items()
        },
        "selection_used_test": False,
        "manifest_sha256": digest(out / "samples.jsonl"),
        "adapters_sha256": {
            name: digest(choices[name] / "adapter/adapter_model.safetensors")
            for name in trained
        },
    }
    frozen = out / f"seed_{seed}/selection.json"
    if frozen.exists():
        if json.loads(frozen.read_text()) != selection:
            raise ValueError("Refuse to change an already frozen model selection")
    else:
        save_json(frozen, selection)
    return selection


def heldout(out, seed, batch_size):
    from peft import PeftModel

    rows, _, _ = load_rows(out)
    selection = json.loads((out / f"seed_{seed}/selection.json").read_text())
    required = set(selection["task_routes"].values()) | {
        selection["joint_adapter"],
        "original",
    }
    evaluation = out / "evaluation"
    evaluation.mkdir(exist_ok=True)
    for name in sorted(required):
        destination = evaluation / name.replace("/", "__")
        destination.mkdir(exist_ok=True)
        if (destination / "results.json").exists():
            continue
        processor, model = processor_and_model()
        if name != "original":
            adapter = out / name / "adapter"
            if (
                digest(adapter / "adapter_model.safetensors")
                != selection["adapters_sha256"][name]
            ):
                raise ValueError("Selected adapter changed before evaluation")
            model = PeftModel.from_pretrained(model, adapter).float().eval()
        results = {}
        for split in ("test", "external"):
            selected = [r for r in rows if r["split"] == split]
            batches = TokenBatches(selected, processor, out / "visual_cache")
            scores, masses = predict(
                model, batches, selected, batch_size, progress=name + " " + split
            )
            write_predictions(
                destination / f"{split}_predictions.jsonl", selected, scores, masses
            )
            results[split] = evaluate_scores(
                selected, scores, selection["thresholds"][name]
            )
        save_json(destination / "results.json", results)
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["prepare", "features", "baseline", "train", "evaluate", "all"],
        default="all",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[3e-5, 1e-4])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("prepare", "all"):
        prepare(out)
    if args.stage in ("features", "all"):
        features(out, args.image_batch_size, args.workers)
    if args.stage in ("baseline", "all"):
        baseline(out, args.batch_size)
    if args.stage in ("train", "all"):
        if args.seed != 42:
            primary = json.loads((out / "seed_42/selection.json").read_text())
            selected_state = json.loads(
                (out / primary["joint_adapter"] / "state.json").read_text()
            )
            if args.learning_rates != [selected_state["learning_rate"]]:
                parser.error(
                    "Secondary seeds must repeat only the primary winning learning rate"
                )
        for lr in args.learning_rates:
            train_candidate(
                out,
                args.seed,
                lr,
                args.epochs,
                args.patience,
                args.batch_size,
                args.accumulation,
            )
        selection = select(out, args.seed, args.learning_rates)
        print("Selection frozen", json.dumps(selection), flush=True)
    if args.stage in ("evaluate", "all"):
        heldout(out, args.seed, args.batch_size)


if __name__ == "__main__":
    main()
