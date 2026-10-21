"""Persist the already selected trajectory estimator and verify score replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/strategy_study_20261016/thermal_trajectories"


def main():
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["selected"] == "trajectory_forest"
    target = OUT / "selected_estimator.joblib"
    if target.exists():
        raise ValueError("Selected estimator already exported")
    rows = [
        json.loads(line) for line in (OUT / "manifest.jsonl").read_text().splitlines()
    ]
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=16,
        random_state=42,
    )
    model.fit(np.array([r["features"] for r in train]), [r["target"] for r in train])
    probability = model.predict_proba(np.array([r["features"] for r in test]))[:, 1]
    score = np.log(np.maximum(probability, 1e-8) / np.maximum(1 - probability, 1e-8))
    saved = [
        json.loads(line)
        for line in (OUT / "trajectory_forest_test_predictions.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [r["id"] for r in saved] == [r["id"] for r in test]
    assert np.allclose(score, [r["score"] for r in saved], rtol=0, atol=1e-10)
    joblib.dump(
        {
            "estimator": model,
            "descriptor": "scripts.benchmark_loitering_trajectories.describe; 14 features in recorded order",
            "thresholds": json.loads((OUT / "results.json").read_text())[
                "trajectory_forest"
            ]["thresholds"],
            "qualification": selection["qualification"],
        },
        target,
        compress=3,
    )
    (OUT / "estimator_export.json").write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "prediction_replay_matches": True,
                "selection_or_threshold_changed": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("Selected trajectory estimator exported; held-out scores match.", flush=True)


if __name__ == "__main__":
    main()
