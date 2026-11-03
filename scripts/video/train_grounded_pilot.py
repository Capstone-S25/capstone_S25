"""Isolate the effect of open-vocabulary track geometry in the temporal head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.training.clip import select_points
from anomaly_eval.training.strategies import source_ap
from scripts.video.train_video_pilot import batch_arrays, fit_temporal, save


def grounded_features(data, features):
    tracks = {
        r["path"]: r["geometry"]
        for r in map(
            json.loads, (data / "grounding_tracks_all.jsonl").read_text().splitlines()
        )
    }
    features = dict(features)
    features["geometry"] = features["geometry"].copy()
    for i, p in enumerate(features["paths"].tolist()):
        if p in tracks:
            features["geometry"][i] = tracks[p]
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["mobileclip2-s2", "siglip2-b16-224"], required=True
    )
    args = parser.parse_args()
    data = args.data.resolve()
    output = args.run.resolve() / args.model
    if (output / "grounding_selection.json").exists():
        raise ValueError("Grounding selection already complete")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(data / "features" / args.model / "features.npz") as archive:
        features = {k: archive[k] for k in archive.files}
    features = grounded_features(data, features)
    rows = [
        r
        for r in map(
            json.loads, (data / "modeling_windows.jsonl").read_text().splitlines()
        )
        if r["task"] == "unattended_items" and r["target"] >= 0
    ]
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    ta, da = batch_arrays(train, features), batch_arrays(dev, features)
    candidates = []
    predictions = {}
    for seed in [42, 43, 44]:
        checkpoint, scores, history = fit_temporal(train, dev, ta, da, True, seed)
        checkpoint["geometry_source"] = "grounding_dino"
        name = f"grounded_track_temporal_seed_{seed}"
        torch.save(checkpoint, output / "unattended_items" / (name + ".pt"))
        save(output / "unattended_items" / (name + "_history.json"), history)
        meta = {
            "name": name,
            "seed": seed,
            "epoch": checkpoint["epoch"],
            "selection": source_ap(dev, scores),
            "thresholds": select_points([r["target"] for r in dev], scores),
            "geometry_source": "grounding_dino",
        }
        candidates.append(meta)
        predictions[name] = {"ids": [r["id"] for r in dev], "scores": scores.tolist()}
        print("GROUNDED HEAD", args.model, name, meta, flush=True)
    save(
        output / "grounding_selection.json",
        {
            "candidates": candidates,
            "development_predictions": predictions,
            "geometry_sha256": hashlib.sha256(
                (data / "grounding_tracks_all.jsonl").read_bytes()
            ).hexdigest(),
        },
    )


if __name__ == "__main__":
    main()
