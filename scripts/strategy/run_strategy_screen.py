"""Select representation strategies on development; open fresh test scores last."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from anomaly_eval.training.clip import Encoder, binary_metrics, select_points
from anomaly_eval.training.data import TASKS, group_class_weights
from anomaly_eval.training.strategies import geometry, normalize, represent, source_ap

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/strategy_study_20261016"
RUN = ROOT / "results/strategy_study_20261016"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def head_fit(x, y, weights, dev_x, dev_rows, c):
    model = LogisticRegression(C=c, max_iter=400, tol=1e-4, solver="lbfgs")
    model.fit(x, y, sample_weight=weights)
    return {
        "kind": "linear",
        "weight": torch.from_numpy(model.coef_.astype(np.float32)),
        "bias": torch.from_numpy(model.intercept_.astype(np.float32)),
    }, model.decision_function(dev_x)


def mlp_fit(x, y, weights, dev_x, dev_rows):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
    ).cuda()
    tx = torch.from_numpy(x).cuda()
    ty = torch.tensor(y, dtype=torch.float32, device="cuda")
    dx = torch.from_numpy(dev_x).cuda()
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.tensor(weights),
        len(y),
        replacement=True,
        generator=torch.Generator().manual_seed(42),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best, saved, stale, best_epoch = -1, None, 0, 0
    for epoch in range(1, 81):
        model.train()
        indices = list(sampler)
        for start in range(0, len(indices), 128):
            idx = torch.tensor(indices[start : start + 128], device="cuda")
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(tx[idx])[:, 0], ty[idx]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(dx)[:, 0].cpu().numpy()
        score = source_ap(dev_rows, prediction)["mean_source_ap"]
        if score > best + 1e-8:
            best, saved, stale, best_epoch = (
                score,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                0,
                epoch,
            )
        else:
            stale += 1
        if stale >= 8:
            break
    model.load_state_dict(saved)
    with torch.no_grad():
        prediction = model(dx)[:, 0].cpu().numpy()
    return {
        "kind": "mlp",
        "state_dict": saved,
        "input_dimension": x.shape[1],
        "epoch": best_epoch,
    }, prediction


def predict(head, values):
    if head["kind"] == "linear":
        return (values @ head["weight"].numpy().T + head["bias"].numpy())[:, 0]
    model = nn.Sequential(
        nn.Linear(head["input_dimension"], 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )
    model.load_state_dict(head["state_dict"])
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(values))[:, 0].numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=["mobileclip2-s2", "siglip2-b16-224"]
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = DATA / "samples.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    output = RUN / args.model
    output.mkdir(parents=True, exist_ok=False)
    detections = {
        r["path"]: r["detections"]
        for r in map(json.loads, (DATA / "detections.jsonl").read_text().splitlines())
    }
    features = {}
    for stage, alphas in [("full", [0, 0.25, 0.5, 1]), ("crop", [0])]:
        for alpha in alphas:
            path = DATA / "features" / args.model / f"main_{stage}_alpha_{alpha:g}.npz"
            value = np.load(path)
            features[stage, alpha] = dict(
                zip(value["paths"].tolist(), value["features"])
            )
    encoder = Encoder(args.model, "cuda").eval()
    text_weights = (encoder.text_direction * encoder.scale).detach().cpu().clone()
    del encoder
    torch.cuda.empty_cache()
    recipes = [f"full_pool_a{a:g}" for a in [0, 0.25, 0.5, 1]] + [
        "endpoint_a0",
        "full_ordered_a0",
        "full_ordered_a1",
        "full_crop_pool_a0",
        "full_crop_ordered_a0",
        "full_geometry_a0",
    ]
    matrices = {}
    for recipe in recipes:
        alpha = float(recipe.rsplit("a", 1)[1])
        values = []
        for row in rows:
            full = represent(
                [features["full", alpha][p] for p in row["views"]],
                ordered="ordered" in recipe,
            )
            if recipe.startswith("endpoint"):
                full = features["full", alpha][row["views"][-1]]
            if "crop" in recipe:
                crop = represent(
                    [features["crop", 0][p] for p in row["views"]],
                    ordered="ordered" in recipe,
                )
                full = normalize(np.concatenate([full, crop]))
            if "geometry" in recipe:
                full = normalize(
                    np.concatenate(
                        [full, geometry([detections[p] for p in row["views"]])]
                    )
                )
            values.append(full)
        matrices[recipe] = np.stack(values).astype(np.float32)
    summaries, selections, chosen_heads = {}, {}, {}
    for task in TASKS:
        train = [
            i for i, r in enumerate(rows) if r["task"] == task and r["split"] == "train"
        ]
        dev = [
            i for i, r in enumerate(rows) if r["task"] == task and r["split"] == "dev"
        ]
        train_rows, dev_rows = [rows[i] for i in train], [rows[i] for i in dev]
        y = np.array([r["target"] for r in train_rows])
        weights = np.array(group_class_weights(train_rows))
        weights *= len(weights) / weights.sum()
        zero_head = {
            "kind": "linear",
            "weight": text_weights[TASKS.index(task) : TASKS.index(task) + 1],
            "bias": torch.zeros(1),
        }
        zero_scores = predict(zero_head, matrices["endpoint_a0"][dev])
        candidates = [
            (
                {
                    "strategy": "endpoint_a0",
                    "classifier": "zero_shot",
                    **source_ap(dev_rows, zero_scores),
                },
                zero_head,
                zero_scores,
            )
        ]
        for recipe in recipes:
            values = matrices[recipe]
            for c in [0.1, 1, 10]:
                head, prediction = head_fit(
                    values[train], y, weights, values[dev], dev_rows, c
                )
                metrics = source_ap(dev_rows, prediction)
                candidate = {
                    "strategy": recipe,
                    "classifier": "linear",
                    "C": c,
                    **metrics,
                }
                candidates.append((candidate, head, prediction))
            print(args.model, task, recipe, "development fits complete", flush=True)
        for recipe in ["full_pool_a0", "full_crop_pool_a0"]:
            values = matrices[recipe]
            head, prediction = mlp_fit(values[train], y, weights, values[dev], dev_rows)
            candidate = {
                "strategy": recipe,
                "classifier": "mlp",
                "epoch": head["epoch"],
                **source_ap(dev_rows, prediction),
            }
            candidates.append((candidate, head, prediction))
        selected, head, dev_scores = max(
            candidates, key=lambda x: x[0]["mean_source_ap"]
        )
        thresholds = select_points([r["target"] for r in dev_rows], dev_scores)
        chosen_heads[task] = head
        selections[task] = {
            **selected,
            "thresholds": thresholds,
            "selection_uses_test": False,
        }
        summaries[task] = [candidate for candidate, _, _ in candidates]
        # Save every candidate, including unsuccessful strategies.
        torch.save(
            {"candidates": [{"selection": c, "head": h} for c, h, _ in candidates]},
            output / f"{task}_candidates.pt",
        )
        save(
            output / f"{task}_development_predictions.json",
            {
                "ids": [r["id"] for r in dev_rows],
                "selected_scores": dev_scores.tolist(),
            },
        )
    # This record is written before computing or reporting any new holdout scores.
    save(
        output / "selection.json",
        {
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "tasks": selections,
            "selection_split": "dev",
            "primary_seed": 42,
        },
    )
    save(output / "development_results.json", summaries)
    test_rows = []
    test_metrics = {}
    for task, selection in selections.items():
        indices = [
            i for i, r in enumerate(rows) if r["task"] == task and r["split"] == "test"
        ]
        if not indices:
            continue
        values = predict(chosen_heads[task], matrices[selection["strategy"]][indices])
        target = [rows[i]["target"] for i in indices]
        test_metrics[task] = {
            op: binary_metrics(target, values, threshold)
            for op, threshold in selection["thresholds"].items()
        }
        test_rows.extend(
            {
                "id": rows[i]["id"],
                "source": rows[i]["source"],
                "group_id": rows[i]["group_id"],
                "task": task,
                "target": rows[i]["target"],
                "score": float(score),
            }
            for i, score in zip(indices, values)
        )
    save(output / "fresh_test_results.json", test_metrics)
    (output / "fresh_test_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in test_rows)
    )
    torch.save(
        {"model": args.model, "heads": chosen_heads, "selections": selections},
        output / "selected_heads.pt",
    )
    save(
        output / "state.json",
        {
            "status": "complete",
            "task_count": len(selections),
            "candidate_count_per_task": len(candidates),
            "fresh_test_tasks": list(test_metrics),
            "precision": "float32; TF32 disabled",
            "no_latency_testing": True,
        },
    )
    print("STRATEGY SCREEN COMPLETE", args.model, flush=True)


if __name__ == "__main__":
    main()
