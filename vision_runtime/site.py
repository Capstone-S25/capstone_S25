"""Per-camera thresholds from a site's own ordinary footage.

The classifier's thresholds were set on other people's cameras. A camera can
have something in view that scores high all day (a doorway that looks like a
bag, a paved corner that looks like the littering clips). The remedy needs no
labels: take a stretch of that camera's footage in which the behaviour did
not happen, and raise the camera's threshold for a behaviour until at most a
small share of those windows would be flagged.

Thresholds are only ever raised, per camera and behaviour, and only for the
model and operating point they were measured with. `tools/calibrate_site.py`
writes the file; `EnsembleStrategy` reads it when it starts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import runtime_root

NAME = "site_calibration.json"


def location():
    custom = os.environ.get("VISION_SITE_CALIBRATION")
    return Path(custom).expanduser() if custom else runtime_root() / NAME


def read(path=None):
    path = Path(path or location())
    if not path.is_file():
        return {"cameras": {}}
    return json.loads(path.read_text())


def load(model, operating_point, path=None):
    """{camera: {task: threshold}} measured with this model and operating point."""
    out = {}
    for camera, entry in read(path)["cameras"].items():
        if (
            entry.get("model") == model
            and entry.get("operating_point") == operating_point
        ):
            out[camera] = {t: float(v["threshold"]) for t, v in entry["tasks"].items()}
    return out


def journal_scores(files, model):
    """Screening scores per task from event journals (one per window)."""
    scores = {}
    for file in files:
        seen = set()
        for line in Path(file).read_text().splitlines():
            row = json.loads(line)
            if row.get("status") != "screened" or row["event_id"] in seen:
                continue
            seen.add(row["event_id"])
            for task, value in (row.get("predictions") or {}).items():
                if "score" in value and value.get("strategy") == model:
                    scores.setdefault(task, []).append(float(value["score"]))
    return scores


def calibrate(scores, thresholds, share=0.02, minimum=30):
    """Raise each task's threshold until at most `share` of the ordinary
    windows are flagged. Tasks with fewer than `minimum` windows are skipped."""
    if not 0 <= share < 1:
        raise ValueError("share must be at least 0 and below 1")
    out = {}
    for task, base in thresholds.items():
        values = sorted(scores.get(task, []))
        if len(values) < minimum:
            continue
        # Smallest threshold that leaves at most share * n windows at or above it.
        allowed = int(share * len(values))
        cut = values[len(values) - allowed - 1]
        raised = max(float(base), cut + 1e-6)
        out[task] = {
            "threshold": raised,
            "model_threshold": float(base),
            "windows": len(values),
            "flagged_before": sum(v >= base for v in values) / len(values),
            "flagged_after": sum(v >= raised for v in values) / len(values),
        }
    return out


def save(camera, model, operating_point, tasks, share, path=None):
    path = Path(path or location())
    table = read(path)
    table["cameras"][camera] = {
        "model": model,
        "operating_point": operating_point,
        "share": share,
        "created_unix_s": time.time(),
        "tasks": tasks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_text(json.dumps(table, indent=1) + "\n")
    temporary.replace(path)
    return path


def clear(camera, path=None):
    path = Path(path or location())
    table = read(path)
    removed = table["cameras"].pop(camera, None) is not None
    if removed:
        path.write_text(json.dumps(table, indent=1) + "\n")
    return removed
