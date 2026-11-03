"""Re-evaluate fixed trained models in float32; calibrate only on development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.training.clip import (
    Encoder,
    evaluate,
    load_checkpoint,
    loader,
    logits,
    save_json,
    score_predictions,
    zero_logits,
)
from anomaly_eval.training.data import read_records

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/clip_finetuning_20261015"


def extract(rows, encoder, batch_size=32):
    values = []
    encoder.eval()
    with torch.no_grad():
        for images, lengths, _, _, _ in loader(rows, encoder, batch_size, 6):
            values.append(
                encoder.encode(images.to("cuda", non_blocking=True), lengths)
                .cpu()
                .numpy()
            )
    return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["mobileclip2-s2", "siglip2-b16-224"], required=True
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    rows = read_records(ROOT / "data/clip_training_20261015/samples.jsonl", ROOT)
    by_split = {
        split: [r for r in rows if r["split"] == split]
        for split in ["dev", "test", "external"]
    }
    encoder = Encoder(args.model, "cuda").eval()
    frozen = {}
    print(args.model, "float32 base features", flush=True)
    for split, subset in by_split.items():
        frozen[split] = extract(subset, encoder)
        print(args.model, "base complete", split, len(subset), flush=True)
    for seed in [42, 43, 44]:
        directory = RUN / f"seed_{seed}" / args.model
        out = directory / "float32_evaluation"
        out.mkdir(exist_ok=False)
        original = json.loads((directory / "selection.json").read_text())
        selection = {
            **original,
            "methods_frozen_from_training": True,
            "precision": "float32 weights and operations; TF32 disabled",
            "calibration": "Refit operating thresholds on float32 development scores only; routes, weights and hyperparameters unchanged",
        }
        checkpoint_hashes = {
            name: hashlib.file_digest(
                (directory / name).open("rb"), "sha256"
            ).hexdigest()
            for name in ["linear_probe.pt", "finetuned_best.pt"]
        }
        save_json(
            out / "state.json",
            {
                "status": "evaluating",
                "precision": "float32",
                "TF32": False,
                "checkpoint_sha256": checkpoint_hashes,
            },
        )
        results = {}
        for method in ["zero_shot", "linear_probe", "finetuned"]:
            if method == "linear_probe":
                load_checkpoint(directory / "linear_probe.pt", encoder)
            if method == "finetuned":
                load_checkpoint(directory / "finetuned_best.pt", encoder)
            scores = {}
            for split, subset in by_split.items():
                if method == "zero_shot":
                    scores[split] = zero_logits(frozen[split], encoder)
                elif method == "linear_probe":
                    scores[split] = logits(frozen[split], encoder.head)
                else:
                    scores[split] = logits(extract(subset, encoder), encoder.head)
            dev = evaluate(by_split["dev"], scores["dev"])
            thresholds = {task: value["thresholds"] for task, value in dev.items()}
            results[method] = {"dev": dev}
            for split in ["test", "external"]:
                results[method][split] = evaluate(
                    by_split[split], scores[split], thresholds
                )
            for split, subset in by_split.items():
                score_predictions(
                    out / f"{method}_{split}_predictions.jsonl", subset, scores[split]
                )
            print(args.model, seed, method, "float32 metrics complete", flush=True)
        save_json(out / "results.json", results)
        save_json(out / "selection.json", selection)
        save_json(
            out / "state.json",
            {
                "status": "complete",
                "precision": "float32",
                "TF32": False,
                "routes_frozen_before_reevaluation": True,
                "threshold_fit_split": "dev",
                "checkpoint_sha256": checkpoint_hashes,
            },
        )
    print(args.model, "ALL FLOAT32 EVALUATIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
