"""Replay a frozen pilot head on verified cached video/image representations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from anomaly_eval.training.video import make_alerts, unattended_state
from scripts.train_grounded_pilot import grounded_features
from scripts.train_person_pilot import instance_arrays, score_person
from scripts.train_video_pilot import batch_arrays, save, score_checkpoint


def predict(
    data, run, model, identifier, kind="recording", device="cuda", source_gate=False
):
    rows = [
        r
        for name in ["verified_windows.jsonl", "verified_fresh_avss.jsonl"]
        for r in map(json.loads, (data / name).read_text().splitlines())
        if r["recording_id" if kind == "recording" else "id"] == identifier
    ]
    if not rows:
        raise ValueError("Unknown pilot recording/window identifier")
    tasks = {r["task"] for r in rows}
    if len(tasks) != 1:
        raise ValueError("Recording contains multiple tasks")
    task = next(iter(tasks))
    rows.sort(key=lambda r: r["timestamps"][-1])
    selection = json.loads((run / model / "joint_selection.json").read_text())
    entry = selection["tasks"][task]["primary"]
    path = run / model / task / (entry["name"] + ".pt")
    head = torch.load(path, weights_only=True, map_location="cpu")
    with np.load(data / "features" / model / "features.npz") as archive:
        features = {k: archive[k] for k in archive.files}
    if head.get("geometry_source") == "grounding_dino":
        features = grounded_features(data, features)
    scores = (
        score_person(head, instance_arrays(rows, features))
        if head["kind"] == "person_mil"
        else score_checkpoint(head, batch_arrays(rows, features), device)
    )
    if task == "unattended_items":
        policies = json.loads((run / model / "event_policies.json").read_text())
        threshold = policies[task]["event_policies"][entry["name"]]["threshold"]
    else:
        threshold = entry["thresholds"]["dev_fpr_budget"]
    gate = np.ones(len(rows), dtype=bool)
    if source_gate:
        if task != "unattended_items" or any(
            not r["recording_id"].startswith("avss:") for r in rows
        ):
            raise ValueError("Known AVSS source rule is defined only for that cohort")
        tracks = {
            r["path"]: r["tracks"]
            for r in map(
                json.loads,
                (data / "grounding_tracks_all.jsonl").read_text().splitlines(),
            )
        }
        gate = np.asarray([unattended_state(tracks[r["views"][-1]]) for r in rows])
    times = [r["timestamps"][-1] for r in rows]
    return {
        "model": model,
        "task": task,
        "candidate": entry["name"],
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "scope": "cached-feature research replay; does not decode a new video or benchmark latency",
        "identifier": identifier,
        "score_type": "uncalibrated classifier logits",
        "threshold": threshold,
        "source_criteria_gate": source_gate,
        "alerts": make_alerts(times, np.where(gate, scores, -1e30), threshold)
        if task == "unattended_items" and kind == "recording"
        else None,
        "decisions": [
            {
                "id": r["id"],
                "timestamp": t,
                "score": float(s),
                "state_predicted": bool(s >= threshold and g),
            }
            for r, t, s, g in zip(rows, times, scores, gate)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["mobileclip2-s2", "siglip2-b16-224"], required=True
    )
    identifiers = parser.add_mutually_exclusive_group(required=True)
    identifiers.add_argument("--recording-id")
    identifiers.add_argument("--window-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--known-avss-rule", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = predict(
        args.data.resolve(),
        args.run.resolve(),
        args.model,
        args.recording_id or args.window_id,
        "recording" if args.recording_id else "window",
        args.device,
        args.known_avss_rule,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save(args.output, result)
    print("PILOT PREDICTION WRITTEN", args.output, flush=True)


if __name__ == "__main__":
    main()
