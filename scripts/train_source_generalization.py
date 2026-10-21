"""Controlled old/new/mixed-source smoking heads on fixed CLIP features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/generalization_pilot_20261020/xrf55"
OUT = ROOT / "results/generalization_pilot_20261020/smoking"
OLD = ROOT / "data/strategy_study_20261016"
MODELS = ("mobileclip2-s2", "siglip2-b16-224")
RECIPES = ("full_pool_a0", "full_crop_pool_a0")
FAMILIES = ("old_only", "new_only", "mixed")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def freeze():
    if (OUT / "protocol.json").exists():
        raise ValueError("Experiment protocol is already frozen")
    acquisition = DATA / "acquisition_protocol.json"
    sources = [
        OLD / "samples.jsonl",
        OLD / "historical.jsonl",
        acquisition,
        Path(__file__),
        ROOT / "scripts/detect_strategy_regions.py",
        ROOT / "scripts/extract_strategy_features.py",
        ROOT / "scripts/run_strategy_screen.py",
        ROOT / "scripts/evaluate_strategy_transfer.py",
    ]
    protocol = {
        "date": "2026-10-20",
        "status": "frozen_before_new_source_model_predictions",
        "models": MODELS,
        "families": FAMILIES,
        "representations": RECIPES,
        "linear_C": [0.1, 1, 10],
        "classifier_fits": 36,
        "new_backbone_finetunes": 0,
        "encoder": "Pinned pretrained encoders, immutable; normalized float32 features; TF32 off",
        "region_policy": "Identical existing automatic COCO contextual crops; no source boxes",
        "views": "All16 uniformly spaced XRF55 views; existing ordered old views unchanged. Native historical strategy reference uses only last frame for smoking.",
        "training": "Logistic regression, balanced task/class then source then recording/author group; identical optimization in each arm",
        "selection": "Within each old/new/mixed family maximize worst dev-source recall at a single every-source <=5% FPR cutoff; tie-break mean recall then worst/mean AP. Same combined dev sources for all families.",
        "thresholds": "Raw0; dev maximumF1; a global threshold satisfying <=5% empirical FPR in every development source; no test fitting",
        "fresh_test": "XRF55 scene4; all20 repetitions; three numeric subject groups; no train/development subject aliases",
        "qualification": "New scene/subject-separated scripted laboratory source, not fresh CCTV validation. Historical tests are retrospective; primary-action controls non-exhaustive; no continuous smoking event/FAR claims.",
        "source_files_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "no_latency_testing": True,
        "no_test_seed_or_route_substitution": True,
    }
    save(OUT / "protocol.json", protocol)
    print(
        "SMOKING SOURCE PROTOCOL FROZEN;36 classifier fits;0 new backbone fine-tunes",
        flush=True,
    )


def check():
    p = json.loads((OUT / "protocol.json").read_text())
    for name, sha in p["source_files_sha256"].items():
        assert digest(ROOT / name) == sha, name
    return p


def feature_map(model, tag, new=False):
    output = {}
    for stage in ["full", "crop"]:
        base = DATA if new else OLD
        path = base / "features" / model / f"{tag}_{stage}_alpha_0.npz"
        if path.exists():
            with np.load(path) as f:
                output[stage, 0] = dict(zip(f["paths"].tolist(), f["features"]))
    return output


def context(model, include_historical=False):
    old = [r for r in rows(OLD / "samples.jsonl") if r["task"] == "smoking"]
    if include_historical:
        old += [r for r in rows(OLD / "historical.jsonl") if r["task"] == "smoking"]
    fresh = rows(DATA / "samples.jsonl")
    # Every source identity is distinct; subjects remain disjoint across splits.
    assert len({r["id"] for r in old + fresh}) == len(old + fresh)
    expected = {p: h for r in old for p, h in zip(r["views"], r["pixel_hashes"])}
    all_old_pixels = set(expected.values())
    new_pixels = {h for r in fresh for h in r["pixel_hashes"]}
    assert not (all_old_pixels & new_pixels), (
        "New source shares model-visible decoded pixels with old inputs"
    )
    features = feature_map(model, "main")
    if include_historical:
        historical = feature_map(model, "historical")
        for key, value in historical.items():
            features[key].update(value)
    for key, value in feature_map(model, "xrf55", new=True).items():
        features[key].update(value)
    return old, fresh, features


def metrics(dev, scores):
    from anomaly_eval.training.clip import binary_metrics
    from scripts.run_event_improvements import source_budget_threshold

    threshold = source_budget_threshold(dev, scores, 0.05)
    cells = {}
    for source in sorted({r["source"] for r in dev}):
        ids = [i for i, r in enumerate(dev) if r["source"] == source]
        cells[source] = binary_metrics(
            [dev[i]["target"] for i in ids], scores[ids], threshold
        )
    ap = [r["ap"] for r in cells.values()]
    recall = [r["recall"] for r in cells.values()]
    key = (min(recall), float(np.mean(recall)), min(ap), float(np.mean(ap)))
    return threshold, cells, key


def fit(model):
    import torch
    from sklearn.linear_model import LogisticRegression

    from anomaly_eval.training.clip import select_points
    from anomaly_eval.training.data import group_class_weights
    from scripts.evaluate_strategy_transfer import matrix
    from scripts.run_strategy_screen import predict

    check()
    torch.set_num_threads(8)
    directory = OUT / model
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "selection.json").exists():
        raise ValueError("Completed selection is immutable")
    old, fresh, features = context(model)
    # No test data is passed to fitting or candidate evaluation.
    train = [r for r in old + fresh if r["split"] == "train"]
    dev = [r for r in old + fresh if r["split"] == "dev"]
    candidates, heads, development = {}, {}, {}
    for recipe in RECIPES:
        tx = matrix(train, recipe, features, {})
        dx = matrix(dev, recipe, features, {})
        for family in FAMILIES:
            ids = [
                i
                for i, r in enumerate(train)
                if family == "mixed"
                or (r["source"] == "xrf55_rgb_public") == (family == "new_only")
            ]
            subset = [train[i] for i in ids]
            weights = np.asarray(group_class_weights(subset))
            weights *= len(weights) / weights.sum()
            y = np.asarray([r["target"] for r in subset])
            for c in [0.1, 1, 10]:
                estimator = LogisticRegression(
                    C=c, max_iter=1000, tol=1e-6, solver="lbfgs"
                )
                estimator.fit(tx[ids], y, sample_weight=weights)
                head = {
                    "kind": "linear",
                    "weight": torch.from_numpy(estimator.coef_.astype(np.float32)),
                    "bias": torch.from_numpy(estimator.intercept_.astype(np.float32)),
                }
                scores = predict(head, dx)
                threshold, cells, key = metrics(dev, scores)
                name = f"{family}__{recipe}__C{c:g}"
                points = select_points([r["target"] for r in dev], scores)
                # Replace pooled budget point with the predeclared every-source one.
                points["dev_fpr_budget"] = threshold
                candidates[name] = {
                    "family": family,
                    "recipe": recipe,
                    "C": c,
                    "training_examples": len(ids),
                    "thresholds": points,
                    "development": cells,
                    "selection_key": key,
                    "converged": bool(estimator.n_iter_[0] < estimator.max_iter),
                }
                heads[name] = head
                development[name] = {
                    "ids": [r["id"] for r in dev],
                    "scores": scores.tolist(),
                }
                print(model, name, "dev_key", key, flush=True)
    selected = {
        family: max(
            (k for k, v in candidates.items() if v["family"] == family),
            key=lambda k: candidates[k]["selection_key"],
        )
        for family in FAMILIES
    }
    torch.save({"model": model, "heads": heads}, directory / "heads.pt")
    save(directory / "development_predictions.json", development)
    save(
        directory / "selection.json",
        {
            "selected": selected,
            "candidates": candidates,
            "selection_uses_test": False,
            "protocol_sha256": digest(OUT / "protocol.json"),
            "new_manifest_sha256": digest(DATA / "samples.jsonl"),
            "heads_sha256": digest(directory / "heads.pt"),
        },
    )


def evaluate(model):
    import torch

    from anomaly_eval.training.clip import binary_metrics
    from scripts.evaluate_strategy_transfer import matrix
    from scripts.run_strategy_screen import predict

    check()
    directory = OUT / model
    if (directory / "results.json").exists():
        raise ValueError("Scored output is immutable")
    selection = json.loads((directory / "selection.json").read_text())
    assert digest(directory / "heads.pt") == selection["heads_sha256"]
    old, fresh, features = context(model, include_historical=True)
    test = [r for r in old + fresh if r["split"] in {"test", "external"}]
    heads = torch.load(directory / "heads.pt", weights_only=True, map_location="cpu")[
        "heads"
    ]
    results = {}
    predictions = []
    for family, name in selection["selected"].items():
        info = selection["candidates"][name]
        scores = predict(heads[name], matrix(test, info["recipe"], features, {}))
        for source in sorted({r["source"] for r in test}):
            ids = [i for i, r in enumerate(test) if r["source"] == source]
            y = [test[i]["target"] for i in ids]
            results[family + "/" + source] = {
                "candidate": name,
                "operating_points": {
                    point: binary_metrics(y, scores[ids], threshold)
                    for point, threshold in info["thresholds"].items()
                },
                "previously_inspected": source != "xrf55_rgb_public",
            }
        predictions += [
            {
                **{k: r[k] for k in ["id", "source", "split", "target", "group_id"]},
                "family": family,
                "candidate": name,
                "score": float(s),
                "action": r.get("action"),
                "scene": r.get("scene"),
                "subject": r.get("subject"),
            }
            for r, s in zip(test, scores)
        ]
    # Genuine original contrastive-score control, same full-image clip pooling.
    original_candidates = torch.load(
        ROOT / "results/strategy_study_20261016" / model / "smoking_candidates.pt",
        weights_only=True,
        map_location="cpu",
    )["candidates"]
    original_head = next(
        c["head"]
        for c in original_candidates
        if c["selection"]["classifier"] == "zero_shot"
    )
    direction = original_head["weight"].numpy().reshape(-1)
    scores = np.asarray(
        [
            np.mean([features["full", 0][p] @ direction for p in r["views"]])
            for r in test
        ]
    )
    for source in sorted({r["source"] for r in test}):
        ids = [i for i, r in enumerate(test) if r["source"] == source]
        results["pretrained_text/" + source] = {
            "operating_points": {
                "raw": binary_metrics([test[i]["target"] for i in ids], scores[ids], 0)
            }
        }
    predictions += [
        {
            **{k: r[k] for k in ["id", "source", "split", "target", "group_id"]},
            "family": "pretrained_text",
            "candidate": "mean_full_image_text_margin",
            "score": float(s),
            "action": r.get("action"),
            "scene": r.get("scene"),
            "subject": r.get("subject"),
        }
        for r, s in zip(test, scores)
    ]
    legacy_dir = ROOT / "results/strategy_study_20261016" / model
    legacy_policy = json.loads((legacy_dir / "selection.json").read_text())["tasks"][
        "smoking"
    ]
    legacy_head = torch.load(
        legacy_dir / "selected_heads.pt", weights_only=True, map_location="cpu"
    )["heads"]["smoking"]
    native_rows = [{**r, "views": r["views"][-1:]} for r in test]
    legacy_scores = predict(
        legacy_head, matrix(native_rows, legacy_policy["strategy"], features, {})
    )
    for source in sorted({r["source"] for r in test}):
        ids = [i for i, r in enumerate(test) if r["source"] == source]
        results["legacy_native/" + source] = {
            "operating_points": {
                point: binary_metrics(
                    [test[i]["target"] for i in ids], legacy_scores[ids], cutoff
                )
                for point, cutoff in legacy_policy["thresholds"].items()
            },
            "qualification": "Frozen16October strategy; last-frame native smoking inputs and original development cutoffs; controlled source-addition comparison is old_only versus mixed.",
        }
    predictions += [
        {
            **{k: r[k] for k in ["id", "source", "split", "target", "group_id"]},
            "family": "legacy_native",
            "candidate": legacy_policy["strategy"],
            "score": float(s),
            "action": r.get("action"),
            "scene": r.get("scene"),
            "subject": r.get("subject"),
        }
        for r, s in zip(test, legacy_scores)
    ]
    # Preserve person/negative-action breakdown on the new source at frozen cutoffs.
    breakdown = []
    for family, name in selection["selected"].items():
        p = [
            r
            for r in predictions
            if r["family"] == family and r["source"] == "xrf55_rgb_public"
        ]
        cutoff = selection["candidates"][name]["thresholds"]["dev_fpr_budget"]
        for subject in sorted({r["subject"] for r in p}):
            group = [r for r in p if r["subject"] == subject]
            breakdown.append(
                {
                    "family": family,
                    "subject": subject,
                    "metrics": binary_metrics(
                        [r["target"] for r in group],
                        [r["score"] for r in group],
                        cutoff,
                    ),
                }
            )
        for action in sorted({r["action"] for r in p}):
            group = [r for r in p if r["action"] == action]
            breakdown.append(
                {
                    "family": family,
                    "action": action,
                    "examples": len(group),
                    "positive_prediction_rate": sum(r["score"] >= cutoff for r in group)
                    / len(group),
                }
            )
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in predictions)
    )
    save(
        directory / "results.json",
        {
            "status": "complete",
            "results": results,
            "new_source_breakdown": breakdown,
            "qualification": json.loads((OUT / "protocol.json").read_text())[
                "qualification"
            ],
            "selection_sha256": digest(directory / "selection.json"),
            "examples": len(test),
        },
    )
    print("SMOKING EVALUATED", model, len(test), "held-out cases", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "fit", "evaluate"])
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif not args.model:
        parser.error("--model required")
    elif args.stage == "fit":
        fit(args.model)
    else:
        evaluate(args.model)


if __name__ == "__main__":
    main()
