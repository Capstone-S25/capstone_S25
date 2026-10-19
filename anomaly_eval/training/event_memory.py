"""Causal item-state smoothing and per-item alert memory for research replays.

These policies use automatic boxes only. They do not supply ownership truth or
labels for unreviewed footage. The original video-pilot tracker stays unchanged.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from anomaly_eval.training.strategies import overlap
from anomaly_eval.training.video import CausalTracker, center


def poisson_upper_rate(count, seconds, confidence=0.95):
    """One-sided exact Poisson rate bound per hour, conditional on exposure."""
    if count < 0 or seconds <= 0 or not 0 < confidence < 1:
        raise ValueError("Nonnegative count and positive reviewed exposure required")
    return float(chi2.ppf(confidence, 2 * (count + 1)) / 2 * 3600 / seconds)


class RobustStationarityTracker(CausalTracker):
    """Reject isolated box jitter; sustained displacement still resets evidence.

    Three observed centers form a trailing median. Displacement beyond the
    original .015 frame threshold must persist for one second. Pending movement
    blocks alerts immediately. Gaps longer than two seconds lose the evidence.
    Owner association is still the original proximity heuristic.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = {}

    def update(self, time, detections):
        live = super().update(time, detections)
        for t in live:
            key = t["id"]
            if key not in self.history:
                self.history[key] = {
                    "centers": deque(maxlen=3),
                    "anchor": center(t["box"]),
                    "since": time,
                    "last": time,
                    "moving_since": None,
                }
            h = self.history[key]
            if time - h["last"] > self.max_gap:
                h["centers"].clear()
                h.update(anchor=center(t["box"]), since=time, moving_since=None)
            h["centers"].append(center(t["box"]))
            smoothed = np.median(h["centers"], axis=0)
            displaced = bool(np.linalg.norm(smoothed - h["anchor"]) > 0.015)
            if displaced:
                if h["moving_since"] is None:
                    h["moving_since"] = time
                if time - h["moving_since"] >= 1.0:
                    h["since"] = h["moving_since"]
                    h["anchor"] = smoothed
                    h["moving_since"] = None
            else:
                h["moving_since"] = None
            t["stationary"] = float(time - h["since"])
            t["motion_pending"] = displaced
            h["last"] = time
        self.history = {k: v for k, v in self.history.items() if k in self.tracks}
        return live


class ItemEventMemory:
    """Emit once per item until observed owner return, motion, or long absence.

    Spatial association preserves alarm memory across short track-ID changes.
    It never extends stationarity or invents owner history across such changes.
    Simultaneous items have one-to-one memory assignments. A gap clears pending
    persistence, while an emitted alarm remains latched for at most 30 seconds
    without observations. Motion/owner return must persist for 3/5 seconds to
    rearm. Those fixed engineering settings were not fitted to test labels.
    """

    def __init__(self, stationary_seconds=30.0, persistence=2.0):
        self.stationary_seconds, self.persistence = stationary_seconds, persistence
        self.memories, self.next_id, self.last = {}, 0, None

    def update(self, time, tracks):
        if not np.isfinite(time) or (self.last is not None and time <= self.last):
            raise ValueError("Event memory needs strictly increasing finite times")
        self.last = time
        self.memories = {
            k: v for k, v in self.memories.items() if time - v["last"] <= 30.0
        }
        items = [t for t in tracks if t["kind"] == "item"]
        keys = list(self.memories)
        costs = np.full((len(keys), len(items)), 1e6)
        for i, key in enumerate(keys):
            m = self.memories[key]
            for j, item in enumerate(items):
                iou = overlap(m["box"], item["box"])
                distance = np.linalg.norm(center(m["box"]) - center(item["box"]))
                if iou >= 0.2 or distance <= 0.025:
                    costs[i, j] = 1 - iou + distance
        assignments = {}
        if costs.size:
            a, b = linear_sum_assignment(costs)
            assignments = {j: keys[i] for i, j in zip(a, b) if costs[i, j] < 1e5}
        alerts = []
        for j, item in enumerate(items):
            key = assignments.get(j)
            if key is None:
                key = self.next_id
                self.next_id += 1
                self.memories[key] = {
                    "box": item["box"],
                    "last": time,
                    "latched": False,
                    "above": None,
                    "return_since": None,
                    "motion_since": None,
                }
            m = self.memories[key]
            if time - m["last"] > 1.1:
                m.update(above=None, return_since=None, motion_since=None)
            returned = item["owner"] is not None and item["owner_absence"] == 0
            moving = item.get("motion_pending", False) or item["stationary"] < 1.0
            for condition, field, duration in [
                (returned, "return_since", 5.0),
                (moving, "motion_since", 3.0),
            ]:
                if not condition:
                    m[field] = None
                elif m[field] is None:
                    m[field] = time
                elif time - m[field] >= duration:
                    m["latched"] = False
            eligible = (
                item["owner"] is not None
                and item["stationary"] >= self.stationary_seconds
                and item["owner_absence"] >= 0.5
                and not item.get("motion_pending", False)
            )
            if eligible:
                if m["above"] is None:
                    m["above"] = time
                if not m["latched"] and time - m["above"] >= self.persistence:
                    alerts.append(
                        {"time": float(time), "memory_id": key, "track_id": item["id"]}
                    )
                    m["latched"] = True
            else:
                m["above"] = None
            m.update(box=item["box"], last=time)
        return alerts
