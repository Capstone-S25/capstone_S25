"""Experiment harness for raising Stage 1 recall without re-running YOLO.

A *method* is a function `method(result, dataset_name) -> list[event dict]` that reads the cached
per-frame observations of one sequence (persons with boxes / track ids / pose keypoints, items) and
returns candidates. Each event needs: timestamp_s, clip_window_s [t0, t1], roi_box [x1, y1, x2, y2],
event_type. Methods are scored together with (or instead of) the current rules.

Splits (the held-out half is only loaded when STAGE1_HELDOUT=1, so tuning never touches it):
    littering   tune: 230 train clips             held-out: 100 valid+test clips
    avenue      tune: odd-numbered test videos    held-out: even-numbered test videos
    normal      tune: odd Avenue training videos  held-out: even ones (no anomalies; false alarms)
    smoking     tune: even camera-days            held-out: odd camera-days
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from stage1_bench import evaluate_run, summarize
from stage1_gate import DEFAULT_RULES, GateConfig, replay_rules

NB = Path(__file__).resolve().parent
HELDOUT = os.environ.get("STAGE1_HELDOUT") == "1"

Method = Callable[[object, str], list]


def _load(path):
    return pickle.loads(Path(path).read_bytes())


def _num(seq_id: str) -> int:
    return int(seq_id.split("/")[-1])


def load_splits(heldout: bool = HELDOUT) -> dict[str, list]:
    avenue = _load(NB / "bench_cache" / "avenue.pkl")
    smoke = _load(NB / "bench_cache" / "smoke.pkl")
    days = sorted({g.seq.seq_id.split("#")[0] for g, _ in smoke})
    day = lambda g: days.index(g.seq.seq_id.split("#")[0]) % 2
    want = 0 if heldout else 1          # odd numbers / odd day index are the tuning half
    splits = {
        "avenue": [x for x in avenue if x[0].seq.seq_id.startswith("test/") and _num(x[0].seq.seq_id) % 2 == want],
        "normal": [x for x in avenue if x[0].seq.seq_id.startswith("train/") and _num(x[0].seq.seq_id) % 2 == want],
        "smoking": [x for x in smoke if day(x[0]) == (1 if heldout else 0)],
    }
    splits["littering"] = _load(NB / "bench_cache" / ("litter_clips.pkl" if heldout else "litter_clips_train.pkl"))
    return splits


BASE_CFG = GateConfig(rules=DEFAULT_RULES)


def baseline(result, dataset: str, cfg: GateConfig = BASE_CFG) -> list:
    return replay_rules(result, cfg)[0]


def combine(*methods: Method) -> Method:
    def run(result, dataset):
        out = []
        for m in methods:
            out += m(result, dataset)
        return out
    return run


def score(method: Method, splits: dict | None = None, name: str = "") -> dict:
    """One row: recall on each anomaly set, false alarms per minute on normal footage."""
    splits = splits or SPLITS
    row = {"method": name}
    for ds in ("littering", "avenue", "smoking"):
        rows = [evaluate_run(g, r, method(r, ds)) for g, r in splits[ds]]
        s = summarize(rows, ds)
        row[f"{ds}_event_%"] = s["event_recall_%"]
        row[f"{ds}_crop_%"] = s["crop_recall_%"]
        if ds != "littering":
            row[f"{ds}_reduction_%"] = s["review_reduction_%"]
    n = minutes = 0.0
    for g, r in splits["normal"]:
        n += len(method(r, "normal"))
        minutes += r.duration_s / 60.0
    row["normal_FA_per_min"] = round(n / minutes, 2)
    return row


def per_action(method: Method, splits: dict | None = None) -> pd.DataFrame:
    splits = splits or SPLITS
    rows = [evaluate_run(g, r, method(r, "littering")) for g, r in splits["littering"]]
    out = []
    for tag in sorted({r["tag"] for r in rows}):
        s = summarize([r for r in rows if r["tag"] == tag], tag)
        out.append({"action": tag, "event_%": s["event_recall_%"], "crop_%": s["crop_recall_%"]})
    return pd.DataFrame(out).set_index("action")


def missed(method: Method, dataset: str, splits: dict | None = None) -> pd.DataFrame:
    splits = splits or SPLITS
    out = []
    for g, r in splits[dataset]:
        row = evaluate_run(g, r, method(r, dataset))
        for m in row["missed"]:
            out.append({**m, "tag": g.tag})
    return pd.DataFrame(out)


def make_event(event_type: str, t: float, roi, pad: float = 2.5, **extra) -> dict:
    return {"event_type": event_type, "timestamp_s": round(float(t), 2),
            "clip_window_s": [round(max(0.0, t - pad), 2), round(t + pad, 2)],
            "roi_box": [int(v) for v in roi], **extra}


SPLITS = load_splits()

if __name__ == "__main__":
    print({k: len(v) for k, v in SPLITS.items()}, "heldout" if HELDOUT else "tune")
    print(pd.DataFrame([score(baseline, name="current rules")]).T)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def dedup(method: Method, window_s: float = 2.5, iou_thr: float = 0.5) -> Method:
    """Merge candidates that point at the same region within window_s: the VLM only needs one crop.
    Causal: an event is dropped only if an earlier-or-equal kept event covers it."""
    def run(result, dataset):
        kept = []
        for e in sorted(method(result, dataset), key=lambda e: e["timestamp_s"]):
            if not any(e["timestamp_s"] - k["timestamp_s"] <= window_s and _iou(e["roi_box"], k["roi_box"]) >= iou_thr
                       for k in kept[-50:]):
                kept.append(e)
        return kept
    return run
