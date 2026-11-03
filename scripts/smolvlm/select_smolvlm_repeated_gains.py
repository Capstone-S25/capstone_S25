"""Retrospective optional routing: require development gains in both seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anomaly_eval.training.clip import save_json
from anomaly_eval.training.data import TASKS
from anomaly_eval.training.smolvlm import ROOT, digest, task_ap

RULE = "Keep the primary target route only when its source-averaged development AP and the independent-seed joint adapter's development AP both strictly exceed the original; otherwise use the original. No held-out score enters this function."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study", type=Path, default=ROOT / "results/smolvlm_finetuning_20261019"
    )
    parser.add_argument("--freeze-rule-only", action="store_true")
    args = parser.parse_args()
    study = args.study
    policy = {
        "rule": RULE,
        "qualification": "Exploratory rule specified after primary held-out results were inspected and after the first independent-seed development epoch. Retrospective safeguard, not fresh prospective evidence. Independent optimization seeds do not provide independent cameras or annotation review.",
    }
    path = study / "repeated_gain_protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == policy
    else:
        save_json(path, policy)
    if args.freeze_rule_only:
        print(
            "Optional repeated-development-gain rule frozen; primary selection unchanged"
        )
        return
    primary = json.loads((study / "seed_42/selection.json").read_text())
    secondary = json.loads((study / "seed_43/selection.json").read_text())
    rows = [json.loads(s) for s in (study / "samples.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["split"] == "dev"]
    routes = set(primary["task_routes"].values()) | {
        "original",
        secondary["joint_adapter"],
    }
    scores = {}
    for route in routes:
        directory = study / "baseline" if route == "original" else study / route
        predictions = [
            json.loads(s)
            for s in (directory / "dev_predictions.jsonl").read_text().splitlines()
        ]
        assert [r["id"] for r in predictions] == [r["id"] for r in rows]
        scores[route] = np.array([r["score"] for r in predictions])
    choices, gains = {}, {}
    for task in TASKS:
        original = task_ap(rows, scores["original"], task)
        route = primary["task_routes"][task]
        first = task_ap(rows, scores[route], task) - original
        second = task_ap(rows, scores[secondary["joint_adapter"]], task) - original
        choices[task] = route if first > 0 and second > 0 else "original"
        gains[task] = {
            "primary_development_ap_gain": first,
            "secondary_development_ap_gain": second,
        }
    result = {
        **policy,
        "task_routes": choices,
        "development_gains": gains,
        "primary_selection_sha256": digest(study / "seed_42/selection.json"),
        "secondary_selection_sha256": digest(study / "seed_43/selection.json"),
        "used_heldout_predictions": False,
    }
    save_json(study / "repeated_gain_routes.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
