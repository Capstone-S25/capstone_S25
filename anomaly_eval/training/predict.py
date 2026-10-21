"""Run the development-selected five-target CLIP bundle on ordered images."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.runtime import resolve_device
from anomaly_eval.training.clip import (
    Encoder,
    autocast,
    load_checkpoint,
    logits,
    zero_logits,
)
from anomaly_eval.training.data import TASKS

ROOT = Path(__file__).resolve().parents[2]


def temporal_indices(timestamps, count):
    if timestamps is None:
        return np.rint(np.linspace(0, count - 1, 4)).astype(int).tolist()
    times = np.asarray(timestamps)
    targets = np.linspace(max(times[0], times[-1] - 20), times[-1], 4)
    return [int(np.argmin(abs(times - value))) for value in targets]


def temporal_support(task, timestamps, count):
    if task == "littering":
        return count >= 8
    if task not in {"loitering", "unattended_items"}:
        return True
    valid = bool(
        count >= 4
        and timestamps is not None
        and len(timestamps) == count
        and len(set(timestamps)) >= 4
        and timestamps[-1] - timestamps[0] >= 19.5
    )
    if not valid:
        return False
    indices = temporal_indices(timestamps, count)
    selected = [timestamps[i] for i in indices]
    return (
        len(set(indices)) == 4
        and selected[-1] - selected[0] >= 19.5
        and selected[0] >= timestamps[-1] - 20.5
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=ROOT / "results/clip_finetuning_20261015"
    )
    parser.add_argument(
        "--model", choices=["mobileclip2-s2", "siglip2-b16-224"], required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--images",
        type=Path,
        nargs="+",
        required=True,
        help="Chronological views of one scene; static objectives use the last view",
    )
    parser.add_argument(
        "--timestamps",
        type=float,
        nargs="+",
        help="Seconds for the corresponding images; duration objectives need ~20 seconds of history",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=["float32", "bfloat16"], default="float32"
    )
    parser.add_argument(
        "--operating-point",
        choices=["raw", "dev_max_f1", "dev_fpr_budget"],
        default="dev_max_f1",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.timestamps is not None and (
        len(args.timestamps) != len(args.images)
        or any(a > b for a, b in zip(args.timestamps, args.timestamps[1:]))
        or not np.isfinite(args.timestamps).all()
    ):
        parser.error(
            "Timestamps must be finite, chronological and match the image count"
        )
    if not all(p.is_file() for p in args.images):
        parser.error("Every input image must exist")
    directory = args.study / f"seed_{args.seed}" / args.model
    evaluation = (
        directory / "float32_evaluation" if args.precision == "float32" else directory
    )
    if not (evaluation / "results.json").exists():
        raise ValueError(
            "Requested evaluation precision is unavailable; complete the matching quality evaluation first"
        )
    selection = json.loads((evaluation / "selection.json").read_text())
    results = json.loads((evaluation / "results.json").read_text())
    state = json.loads((directory / "state.json").read_text())
    if state["status"] != "complete":
        raise ValueError("Training/evaluation bundle is incomplete")
    if tuple(state["task_order"]) != TASKS:
        raise ValueError("Classifier target order changed")
    provenance = json.loads((args.study / "provenance.json").read_text())
    if (
        hashlib.sha256((ROOT / provenance["definitions_path"]).read_bytes()).hexdigest()
        != provenance["definitions_sha256"]
    ):
        raise ValueError(
            "Target definitions changed since training; restore the recorded study definitions"
        )
    device = resolve_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = args.precision != "float32"
    torch.backends.cudnn.allow_tf32 = args.precision != "float32"
    torch.backends.cudnn.benchmark = False

    def precision_context():
        return nullcontext() if args.precision == "float32" else autocast(device)

    encoder = Encoder(args.model, device).eval()
    images = torch.stack([encoder.image_tensor(p) for p in args.images]).to(device)
    # Separate static endpoint and temporal features; do not silently change the static task protocol.
    temporal = temporal_indices(args.timestamps, len(args.images))
    litter = np.rint(np.linspace(0, len(args.images) - 1, 8)).astype(int).tolist()
    images = torch.cat([images[-1:], images[temporal], images[litter]])
    lengths = [1, 4, 8]
    needed = set(selection["task_routes"].values())
    scores = {}
    with torch.no_grad():
        with precision_context():
            features = encoder.encode(images, lengths).cpu().numpy()
        if "zero_shot" in needed:
            scores["zero_shot"] = zero_logits(features, encoder)
        if "linear_probe" in needed:
            load_checkpoint(directory / "linear_probe.pt", encoder)
            scores["linear_probe"] = logits(features, encoder.head)
        if "finetuned" in needed:
            load_checkpoint(directory / "finetuned_best.pt", encoder)
            with precision_context():
                features = encoder.encode(images, lengths).cpu().numpy()
            scores["finetuned"] = logits(features, encoder.head)
    predictions = {}
    for ti, task in enumerate(TASKS):
        method = selection["task_routes"].get(task)
        if method is None:
            predictions[task] = {
                "decision": "unknown",
                "reason": "no supported training target",
            }
            continue
        row = (
            0
            if task in {"smoking", "rough_sleeping"}
            else 2
            if task == "littering"
            else 1
        )
        value = float(scores[method][row, ti])
        threshold = results[method]["dev"][task]["thresholds"][args.operating_point]
        supported = temporal_support(task, args.timestamps, len(args.images))
        predictions[task] = {
            "score": value,
            "threshold": threshold,
            "decision": ("anomaly" if value >= threshold else "normal")
            if supported
            else "unknown",
            "method_selected_on_development": method,
            "evidence_status": "provisional_visual_labels"
            if task in {"rough_sleeping", "unattended_items"}
            else "dataset_annotations",
        }
        if not supported:
            predictions[task]["reason"] = (
                "Provide at least eight uniformly sampled views of the whole littering clip"
                if task == "littering"
                else "Provide four distinct chronological timestamps spanning approximately 20 seconds of past context"
            )
    output = {
        "model": args.model,
        "precision": args.precision,
        "base_revision": encoder.revision,
        "seed": args.seed,
        "operating_point": args.operating_point,
        "predictions": predictions,
        "input_images": [str(p) for p in args.images],
        "timestamps": args.timestamps,
        "scope": "Scene/clip classification; no object bounding boxes or housing-status inference",
    }
    value = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value)
    else:
        print(value, end="")


if __name__ == "__main__":
    main()
