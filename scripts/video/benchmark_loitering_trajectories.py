"""Day-held-out trajectory-component benchmark on published thermal annotations."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from anomaly_eval.training.clip import binary_metrics, select_points
from anomaly_eval.training.strategies import source_ap

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/strategy_sources_20261016/thermal/annotations_loitering.csv"
OUT = ROOT / "results/strategy_study_20261016/thermal_trajectories"


def describe(points):
    values = np.asarray(sorted(set(points)), dtype=np.float64)
    frames, position = values[:, 0], values[:, 1:]
    delta = np.diff(position, axis=0)
    gaps = np.maximum(np.diff(frames), 1)
    step = np.linalg.norm(delta, axis=1)
    path = float(step.sum())
    net = float(np.linalg.norm(position[-1] - position[0]))
    speed = step / gaps
    cosine = (delta[1:] * delta[:-1]).sum(1) / np.maximum(step[1:] * step[:-1], 1e-9)
    span = float(frames[-1] - frames[0] + 1)
    extent = np.ptp(position, axis=0)
    return [
        np.log1p(span),
        np.log1p(len(values)),
        min(len(values) / span, 1),
        np.log1p(path),
        np.log1p(net),
        net / max(path, 1e-9),
        np.log1p(extent[0]),
        np.log1p(extent[1]),
        np.log1p(np.linalg.norm(position - position.mean(0), axis=1).mean()),
        float(np.median(speed)),
        float(np.quantile(speed, 0.9)),
        float(np.mean(speed < 0.25)),
        float(np.mean(cosine)) if len(cosine) else 1,
        float(np.mean(cosine < 0)) if len(cosine) else 0,
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    tracks = {}
    with SOURCE.open() as stream:
        for row in csv.DictReader(stream):
            if row["class"] != "human":
                continue
            key = row["id"]
            track = tracks.setdefault(
                key, {"day": row["folder"], "labels": set(), "points": []}
            )
            if track["day"] != row["folder"]:
                raise ValueError("Track identity crosses dates")
            track["labels"].add(int(row["loitering"]))
            track["points"].append(
                (int(row["frame"]), float(row["x"]), float(row["y"]))
            )
    rows, rejected = [], Counter()
    for key, track in tracks.items():
        if len(track["labels"]) != 1:
            rejected["conflicting_track_labels"] += 1
            continue
        if len({v[0] for v in track["points"]}) < 8:
            rejected["fewer_than_8_observations"] += 1
            continue
        rows.append(
            {
                "id": key,
                "day": track["day"],
                "source": track["day"],
                "target": next(iter(track["labels"])),
                "features": describe(track["points"]),
            }
        )
    days = sorted({r["day"] for r in rows})
    n = max(1, round(len(days) * 0.2))
    assignment = {
        day: "train"
        if i < len(days) - 2 * n
        else "dev"
        if i < len(days) - n
        else "test"
        for i, day in enumerate(days)
    }
    for row in rows:
        row["split"] = assignment[row["day"]]
    for split in ["train", "dev", "test"]:
        assert {r["target"] for r in rows if r["split"] == split} == {0, 1}
    (OUT / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    subsets = {
        s: [r for r in rows if r["split"] == s] for s in ["train", "dev", "test"]
    }
    x = {s: np.array([r["features"] for r in rr]) for s, rr in subsets.items()}
    y = {s: np.array([r["target"] for r in rr]) for s, rr in subsets.items()}
    candidates = [
        (
            "duration_only",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", max_iter=400),
            ),
            [0],
        ),
        (
            "trajectory_linear",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(class_weight="balanced", C=1, max_iter=400),
            ),
            list(range(14)),
        ),
        (
            "trajectory_forest",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=16,
                random_state=42,
            ),
            list(range(14)),
        ),
        (
            "trajectory_boosting",
            HistGradientBoostingClassifier(
                max_iter=200,
                max_leaf_nodes=15,
                l2_regularization=1,
                class_weight="balanced",
                random_state=42,
            ),
            list(range(14)),
        ),
    ]
    results, fitted = {}, {}
    for name, model, columns in candidates:
        model.fit(x["train"][:, columns], y["train"])
        score = model.predict_proba(x["dev"][:, columns])[:, 1]
        # Probability scores are converted to logits for consistent zero thresholds.
        score = np.log(np.maximum(score, 1e-8) / np.maximum(1 - score, 1e-8))
        thresholds = select_points(y["dev"], score)
        results[name] = {
            "dev": source_ap(subsets["dev"], score),
            "thresholds": thresholds,
        }
        fitted[name] = (model, columns)
        print(name, results[name]["dev"], flush=True)
    selected = max(results, key=lambda k: results[k]["dev"]["mean_source_ap"])
    selection = {
        "selected": selected,
        "selection_uses_test": False,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "days": assignment,
        "counts": dict(Counter(f"{r['split']}/{r['target']}" for r in rows)),
        "rejected": dict(rejected),
        "input": "author-provided complete human trajectories, pixel x/y; frame units, not assumed seconds",
        "qualification": "Offline annotated-track thermal component benchmark; not automatic tracking or RGB CLIP detection; one physical camera with dates/seasons held out",
    }
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    for name, (model, columns) in fitted.items():
        probability = model.predict_proba(x["test"][:, columns])[:, 1]
        score = np.log(
            np.maximum(probability, 1e-8) / np.maximum(1 - probability, 1e-8)
        )
        results[name]["test"] = {
            op: binary_metrics(y["test"], score, threshold)
            for op, threshold in results[name]["thresholds"].items()
        }
        (OUT / (name + "_test_predictions.jsonl")).write_text(
            "".join(
                json.dumps(
                    {
                        "id": row["id"],
                        "day": row["day"],
                        "target": row["target"],
                        "score": float(value),
                    }
                )
                + "\n"
                for row, value in zip(subsets["test"], score)
            )
        )
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (OUT / "state.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "tracks": len(rows),
                "days": len(days),
                "selected": selected,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        "THERMAL TRAJECTORY STUDY COMPLETE", len(rows), len(days), selected, flush=True
    )


if __name__ == "__main__":
    main()
