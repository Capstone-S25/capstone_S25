"""Smoking multiple-instance head with additional training-person supervision."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from anomaly_eval.training.clip import select_points
from anomaly_eval.training.data import group_class_weights
from anomaly_eval.training.strategies import overlap, source_ap
from scripts.train_video_pilot import save

ROOT = Path(__file__).resolve().parents[1]


def instance_arrays(rows, features):
    paths = features["paths"].tolist()
    index = {p: i for i, p in enumerate(paths)}
    by_image = defaultdict(list)
    for i, (image, person) in enumerate(
        zip(features["roi_image_indices"], features["roi_is_person"])
    ):
        if person:
            by_image[int(image)].append(i)
    arrays = []
    for row in rows:
        images = [index[p] for p in row["views"]]
        values = []
        for image in images:
            ids = by_image[image]
            values.extend(
                features["roi_features"][ids]
                if ids
                else features["full"][image : image + 1]
            )
        arrays.append(np.asarray(values, dtype=np.float32))
    return arrays


def score_person(checkpoint, instances):
    weight = checkpoint["weight"].numpy().reshape(-1)
    bias = float(checkpoint["bias"].item())
    return np.asarray(
        [float((x @ weight + bias).max()) for x in instances], dtype=np.float32
    )


def annotated_training_regions(rows, features):
    allowed = {p for r in rows if r["source"] == "smoking_cctv" for p in r["views"]}
    annotations = {}
    base = ROOT / "data/raw/datasets/smoking/smoking-video.v1i.coco"
    for p in base.glob("*/_annotations.coco.json"):
        coco = json.loads(p.read_text())
        images = {r["id"]: r for r in coco["images"]}
        for a in coco["annotations"]:
            if a["category_id"] not in {1, 2}:
                continue
            image = images[a["image_id"]]
            path = str((p.parent / image["file_name"]).relative_to(ROOT))
            if path not in allowed:
                continue
            x, y, w, h = a["bbox"]
            iw, ih = image["width"], image["height"]
            annotations.setdefault(path, []).append(
                (
                    [x / iw, y / ih, (x + w) / iw, (y + h) / ih],
                    int(a["category_id"] == 2),
                )
            )
    values = []
    targets = []
    paths = features["paths"].tolist()
    # Match automatically selected contextual person crops to training labels.
    for i, (image_index, box, person) in enumerate(
        zip(
            features["roi_image_indices"],
            features["roi_boxes"],
            features["roi_is_person"],
        )
    ):
        path = paths[int(image_index)]
        if not person or path not in annotations:
            continue
        # Padding may be clipped at image borders; matching the contextual box
        # directly is conservative and needs only 0.3 IoU.
        pairs = sorted(
            [(overlap(box, a), y) for a, y in annotations[path]], reverse=True
        )
        if not pairs or pairs[0][0] < 0.3:
            continue
        if any(v >= pairs[0][0] - 0.05 and y != pairs[0][1] for v, y in pairs):
            continue
        values.append(features["roi_features"][i])
        targets.append(pairs[0][1])
    if len(set(targets)) != 2:
        raise ValueError("Both training-region classes are required")
    return np.stack(values), np.asarray(targets, dtype=np.float32)


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
    if not (output / "selection.json").exists():
        raise ValueError("Run main pilot selection first")
    if (output / "person_selection.json").exists():
        raise ValueError("Person selection already complete")
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = [
        r
        for r in map(
            json.loads, (data / "modeling_windows.jsonl").read_text().splitlines()
        )
        if r["task"] == "smoking"
    ]
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "dev"]
    with np.load(data / "features" / args.model / "features.npz") as archive:
        features = {k: archive[k] for k in archive.files}
    instances = instance_arrays(train, features)
    dev_instances = instance_arrays(dev, features)
    rx, ry = annotated_training_regions(train, features)
    region_x = torch.from_numpy(rx).cuda()
    region_y = torch.from_numpy(ry).cuda()
    region_weights = np.where(ry > 0, 1 / max(1, ry.sum()), 1 / max(1, (1 - ry).sum()))
    region_weights /= region_weights.sum()
    weights = np.asarray(group_class_weights(train))
    weights /= weights.sum()
    max_length = max(map(len, instances))
    d = rx.shape[1]
    x = np.zeros((len(train), max_length, d), dtype=np.float32)
    mask = np.zeros((len(train), max_length), dtype=bool)
    for i, v in enumerate(instances):
        x[i, : len(v)] = v
        mask[i, : len(v)] = True
    x = torch.from_numpy(x).cuda()
    mask = torch.from_numpy(mask).cuda()
    y = torch.tensor([r["target"] for r in train], device="cuda", dtype=torch.float32)
    candidates = []
    predictions = {}
    for hard_negatives, seed in [(h, s) for h in [False, True] for s in [42, 43, 44]]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        head = nn.Linear(d, 1).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.1)
        best = (-1.0, -1.0)
        saved = None
        stale = 0
        best_epoch = 0
        history = []
        sampling_weights = weights.copy()
        for epoch in range(1, 61):
            if hard_negatives and epoch >= 4:
                with torch.no_grad():
                    difficulty = (
                        torch.sigmoid(
                            head(x)[:, :, 0].masked_fill(~mask, -1e4).max(1).values
                        )
                        .cpu()
                        .numpy()
                    )
                sampling_weights = weights.copy()
                for source in sorted({r["source"] for r in train}):
                    normal = np.asarray(
                        [
                            i
                            for i, r in enumerate(train)
                            if r["source"] == source and r["target"] == 0
                        ]
                    )
                    if not len(normal):
                        continue
                    sampling_weights[normal] *= 1 + 2 * difficulty[normal]
                    sampling_weights[normal] *= (
                        weights[normal].sum() / sampling_weights[normal].sum()
                    )
                sampling_weights /= sampling_weights.sum()
            ids = rng.choice(len(train), len(train), replace=True, p=sampling_weights)
            for start in range(0, len(ids), 128):
                idx = torch.as_tensor(ids[start : start + 128], device="cuda")
                logits = (
                    head(x[idx])[:, :, 0].masked_fill(~mask[idx], -1e4).max(1).values
                )
                weak = nn.functional.binary_cross_entropy_with_logits(logits, y[idx])
                ri = torch.as_tensor(
                    rng.choice(len(ry), 256, replace=True, p=region_weights),
                    device="cuda",
                )
                strong = nn.functional.binary_cross_entropy_with_logits(
                    head(region_x[ri])[:, 0], region_y[ri]
                )
                loss = weak + 0.5 * strong
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            checkpoint = {
                "kind": "person_mil",
                "weight": head.weight.detach().cpu().clone(),
                "bias": head.bias.detach().cpu().clone(),
                "seed": seed,
                "epoch": epoch,
                "additional_supervision": "training camera matched Smoking/Not Smoking COCO regions",
                "region_samples": len(ry),
                "region_positive": int(ry.sum()),
                "hard_negative_mining": hard_negatives,
            }
            scores = score_person(checkpoint, dev_instances)
            metrics = source_ap(dev, scores)
            history.append({"epoch": epoch, **metrics})
            key = (metrics["worst_source_ap"], metrics["mean_source_ap"])
            if key > best:
                best = key
                saved = checkpoint
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= 8:
                break
        name = f"person_supervised{'_hardneg' if hard_negatives else ''}_seed_{seed}"
        torch.save(saved, output / "smoking" / (name + ".pt"))
        save(output / "smoking" / (name + "_history.json"), history)
        scores = score_person(saved, dev_instances)
        candidates.append(
            {
                "name": name,
                "seed": seed,
                "epoch": best_epoch,
                "selection": source_ap(dev, scores),
                "thresholds": select_points([r["target"] for r in dev], scores),
                "region_samples": len(ry),
                "region_positive": int(ry.sum()),
                "hard_negative_mining": hard_negatives,
            }
        )
        predictions[name] = {"ids": [r["id"] for r in dev], "scores": scores.tolist()}
        print("PERSON HEAD", args.model, name, candidates[-1], flush=True)
    save(
        output / "person_selection.json",
        {
            "candidates": candidates,
            "development_predictions": predictions,
            "selection_rule": "same worst development-source AP; auxiliary experiment; main selection remains immutable",
        },
    )


if __name__ == "__main__":
    main()
