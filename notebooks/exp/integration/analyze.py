"""Summarise an integration run (run_integration.py OUT): gate time per frame and lateness against the real-time
schedule, and per-candidate latency from the alert firing to the VLM decision (crop-check hold + queue + VLM).
Usage (from notebooks/): ../.venv/bin/python exp/integration/analyze.py OUT [OUT ...]
"""
import json
import sys
from pathlib import Path

import numpy as np


def pct(v, q):
    return round(float(np.percentile(v, q)), 3) if len(v) else None


def summary(out):
    out = Path(out)
    run = json.loads((out / "run.json").read_text())
    frames = [json.loads(l) for f in out.glob("*_frames.jsonl") for l in f.read_text().splitlines()]
    gate = np.array([f["gate_s"] for f in frames]) * 1000
    late = np.array([f["late_s"] for f in frames])
    cands = [json.loads(l) for f in out.glob("*/candidates.jsonl") for l in f.read_text().splitlines()]
    done = [c for c in cands if c["status"] == "completed"]
    hold = np.array([c["integration"]["hold_s"] for c in done])
    queue = np.array([c["queue_seconds"] for c in done])
    service = np.array([c["service_seconds"] for c in done])
    e2e = hold + np.array([max(0.0, c["integration"]["late_s"]) for c in done]) + queue + service
    minutes = run["stream_s"] / 60
    return {
        "run": out.name, "cameras": len(run["plans"]), "video_minutes_per_camera": round(run["plans"][0]["seconds"] / 60, 1),
        "frames": len(frames),
        "gate ms per frame p50 / p95 / max": [pct(gate, 50), pct(gate, 95), pct(gate, 100)],
        "frames started > 0.2 s late (share)": round(float(np.mean(late > 0.2)), 4) if len(late) else None,
        "lateness s p95 / max": [pct(late, 95), pct(late, 100)],
        "candidates": len(cands), "per camera per minute": round(len(cands) / max(minutes, 1e-9) / len(run["plans"]), 2),
        "rejected (queue full)": sum(c["status"] == "rejected" for c in cands),
        "VLM service s p50 / p95 / max": [pct(service, 50), pct(service, 95), pct(service, 100)],
        "queue wait s p50 / p95 / max": [pct(queue, 50), pct(queue, 95), pct(queue, 100)],
        "alert -> VLM decision s p50 / p95 / max": [pct(e2e, 50), pct(e2e, 95), pct(e2e, 100)],
        "drain after stream end s": round(run["drain_after_stream_s"], 1),
    }


if __name__ == "__main__":
    for o in sys.argv[1:]:
        print(json.dumps(summary(o), indent=1))
