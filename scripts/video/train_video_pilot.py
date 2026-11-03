"""Controlled static/temporal/track ablations; all choices use development only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from anomaly_eval.training.clip import select_points
from anomaly_eval.training.data import group_class_weights
from anomaly_eval.training.strategies import represent, source_ap
from anomaly_eval.training.temporal import TemporalHead


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def batch_arrays(rows, features):
    index = {p: i for i, p in enumerate(features["paths"].tolist())}
    n = len(rows)
    steps = max(len(r["views"]) for r in rows)
    d = features["full"].shape[1]
    visual = np.zeros((n, steps, d * 3), np.float32)
    geometry = np.zeros((n, steps, features["geometry"].shape[1]), np.float32)
    timing = np.zeros((n, steps, 2), np.float32)
    lengths = []
    for i, r in enumerate(rows):
        ids = [index[p] for p in r["views"]]
        lengths.append(len(ids))
        visual[i, : len(ids)] = np.concatenate(
            [features[k][ids] for k in ["full", "crop_mean", "crop_max"]], axis=-1
        )
        geometry[i, : len(ids)] = features["geometry"][ids]
        times = np.asarray(r["timestamps"], dtype=np.float32)
        timing[i, : len(ids), 0] = np.minimum((times - times[0]) / 20, 6)
        timing[i, : len(ids), 1] = np.minimum(np.diff(times, prepend=times[0]) / 5, 6)
    return visual, geometry, timing, np.asarray(lengths, dtype=np.int64)


def predict_temporal(model, arrays, device="cuda"):
    model.eval()
    out = []
    with torch.inference_mode():
        for start in range(0, len(arrays[0]), 128):
            inputs = [
                torch.from_numpy(v[start : start + 128]).to(device) for v in arrays
            ]
            out.append(model(*inputs).cpu().numpy())
    return np.concatenate(out)


def fit_temporal(train, dev, train_arrays, dev_arrays, geometry, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TemporalHead(
        train_arrays[0].shape[-1], train_arrays[1].shape[-1] if geometry else 0
    ).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    weights = np.asarray(group_class_weights(train))
    weights /= weights.sum()
    rng = np.random.default_rng(seed)
    y = torch.tensor([r["target"] for r in train], dtype=torch.float32, device="cuda")
    tensors = [torch.from_numpy(x).cuda() for x in train_arrays]
    best = (-1.0, -1.0)
    state = None
    stale = 0
    history = []
    best_epoch = 0
    for epoch in range(1, 61):
        model.train()
        sampled = rng.choice(len(train), len(train), replace=True, p=weights)
        for start in range(0, len(sampled), 128):
            ids = torch.as_tensor(sampled[start : start + 128], device="cuda")
            scores = model(*[x[ids] for x in tensors])
            loss = nn.functional.binary_cross_entropy_with_logits(scores, y[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scores = predict_temporal(model, dev_arrays)
        metrics = source_ap(dev, scores)
        key = (metrics["worst_source_ap"], metrics["mean_source_ap"])
        history.append({"epoch": epoch, **metrics})
        if key > best:
            best = key
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            best_epoch = epoch
        else:
            stale += 1
        if stale >= 8:
            break
    model.load_state_dict(state)
    scores = predict_temporal(model, dev_arrays)
    return (
        {
            "kind": "temporal",
            "geometry": geometry,
            "seed": seed,
            "epoch": best_epoch,
            "visual_dimension": train_arrays[0].shape[-1],
            "geometry_dimension": train_arrays[1].shape[-1] if geometry else 0,
            "state_dict": state,
        },
        scores,
        history,
    )


def static_matrix(arrays, scene_only=False):
    visual, _, _, lengths = arrays
    d = visual.shape[-1] // 3 if scene_only else visual.shape[-1]
    return np.stack([represent(x[:length, :d]) for x, length in zip(visual, lengths)])


def score_checkpoint(checkpoint, arrays, device="cuda"):
    if checkpoint["kind"] == "linear":
        x = static_matrix(arrays, checkpoint["scene_only"])
        return (x @ checkpoint["weight"].numpy().T + checkpoint["bias"].numpy())[:, 0]
    model = TemporalHead(
        checkpoint["visual_dimension"], checkpoint["geometry_dimension"]
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return predict_temporal(model, arrays, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["mobileclip2-s2", "siglip2-b16-224"], required=True
    )
    args = parser.parse_args()
    data = args.data.resolve()
    output = args.output.resolve() / args.model
    output.mkdir(parents=True, exist_ok=True)
    if (output / "selection.json").exists():
        raise ValueError("Completed model selection is immutable")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = [
        r
        for r in map(
            json.loads, (data / "modeling_windows.jsonl").read_text().splitlines()
        )
        if r["target"] >= 0
    ]
    # Training does not open the fresh cohort. Feature extraction has no fitting.
    with np.load(data / "features" / args.model / "features.npz") as archive:
        f = {k: archive[k] for k in archive.files}
    selections = {}
    dev_predictions = {}
    for task in ["smoking", "unattended_items"]:
        train = [r for r in rows if r["task"] == task and r["split"] == "train"]
        dev = [r for r in rows if r["task"] == task and r["split"] == "dev"]
        ta, da = batch_arrays(train, f), batch_arrays(dev, f)
        candidates = []
        directory = output / task
        directory.mkdir(exist_ok=True)
        for name, scene_only in [("scene_linear", True), ("region_linear", False)]:
            x, _dx = static_matrix(ta, scene_only), static_matrix(da, scene_only)
            weights = np.asarray(group_class_weights(train))
            weights *= len(train) / weights.sum()
            model = LogisticRegression(C=1.0, max_iter=600, tol=1e-4).fit(
                x, [r["target"] for r in train], sample_weight=weights
            )
            checkpoint = {
                "kind": "linear",
                "scene_only": scene_only,
                "weight": torch.from_numpy(model.coef_.astype(np.float32)),
                "bias": torch.from_numpy(model.intercept_.astype(np.float32)),
            }
            scores = score_checkpoint(checkpoint, da)
            meta = {
                "name": name,
                "seed": None,
                "selection": source_ap(dev, scores),
                "thresholds": select_points([r["target"] for r in dev], scores),
            }
            torch.save(checkpoint, directory / (name + ".pt"))
            candidates.append(meta)
            dev_predictions[task + "/" + name] = {
                "ids": [r["id"] for r in dev],
                "scores": scores.tolist(),
            }
            print(args.model, task, name, meta["selection"], flush=True)
        for name, geom in [("region_temporal", False), ("region_track_temporal", True)]:
            for seed in [42, 43, 44]:
                checkpoint, scores, history = fit_temporal(
                    train, dev, ta, da, geom, seed
                )
                key = f"{name}_seed_{seed}"
                meta = {
                    "name": key,
                    "seed": seed,
                    "epoch": checkpoint["epoch"],
                    "selection": source_ap(dev, scores),
                    "thresholds": select_points([r["target"] for r in dev], scores),
                }
                torch.save(checkpoint, directory / (key + ".pt"))
                save(directory / (key + "_history.json"), history)
                candidates.append(meta)
                dev_predictions[task + "/" + key] = {
                    "ids": [r["id"] for r in dev],
                    "scores": scores.tolist(),
                }
                print(
                    args.model,
                    task,
                    key,
                    "epoch",
                    checkpoint["epoch"],
                    meta["selection"],
                    flush=True,
                )
        # Common preregistered selection: prioritize the weakest development source.
        best = max(
            candidates,
            key=lambda c: (
                c["selection"]["worst_source_ap"],
                c["selection"]["mean_source_ap"],
            ),
        )
        selections[task] = {
            "primary": best,
            "candidates": candidates,
            "train_n": len(train),
            "dev_n": len(dev),
        }
    save(output / "development_predictions.json", dev_predictions)
    save(
        output / "selection.json",
        {
            "model": args.model,
            "tasks": selections,
            "selection_rule": "maximum worst development-source AP; mean source AP tie-break",
            "manifest_sha256": hashlib.sha256(
                (data / "modeling_windows.jsonl").read_bytes()
            ).hexdigest(),
            "prepared_manifest_sha256": hashlib.sha256(
                (data / "windows.jsonl").read_bytes()
            ).hexdigest(),
            "feature_sha256": hashlib.sha256(
                (data / "features" / args.model / "features.npz").read_bytes()
            ).hexdigest(),
            "precision": "float32; TF32 disabled",
            "fresh_test_labels_used": False,
        },
    )


if __name__ == "__main__":
    main()
