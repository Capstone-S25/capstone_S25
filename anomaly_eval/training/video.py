"""Causal association tracks and explicit event accounting for video pilots.

This small Hungarian/velocity tracker is a baseline, not ByteTrack. Ownership
is a proximity hypothesis requiring observed history, never ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from anomaly_eval.training.strategies import overlap

TRACK_FEATURES = (
    "person_count",
    "item_count",
    "person_age",
    "item_age",
    "person_stationary",
    "item_stationary",
    "owner_absence",
    "owned_item_fraction",
    "person_coverage",
    "item_coverage",
    "person_speed",
    "item_speed",
    "nearest_person_item",
    "person_area",
    "item_area",
    "person_confidence",
    "item_confidence",
    "owner_observed_fraction",
)


def center(box):
    return (np.asarray(box[:2]) + np.asarray(box[2:])) / 2


def proximity(person, item):
    """Distance from item center to person rectangle, in person heights."""
    p, b = np.asarray(person), center(item)
    distance = np.maximum(np.maximum(p[:2] - b, b - p[2:]), 0)
    return float(np.linalg.norm(distance) / max(p[3] - p[1], 0.02))


@dataclass
class Track:
    id: int
    kind: str
    box: np.ndarray
    born: float
    last: float
    confidence: float
    stationary_since: float
    anchor: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    hits: int = 1
    owner: int | None = None
    candidate: int | None = None
    candidate_since: float | None = None
    absent_since: float | None = None


class CausalTracker:
    def __init__(
        self, max_gap=2.0, owner_seconds=2.0, new_threshold=0.5, low_threshold=0.2
    ):
        self.max_gap, self.owner_seconds = max_gap, owner_seconds
        self.new_threshold, self.low_threshold = new_threshold, low_threshold
        self.tracks, self.next_id, self.time = {}, 0, None

    def update(self, time, detections):
        if not np.isfinite(time) or (self.time is not None and time <= self.time):
            raise ValueError("Tracking requires finite, strictly increasing timestamps")
        self.time = float(time)
        self.tracks = {
            k: t for k, t in self.tracks.items() if time - t.last <= self.max_gap
        }
        found = []
        for d in detections:
            box = np.asarray(d["box"], dtype=float)
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or np.any(box[2:] <= box[:2])
            ):
                raise ValueError("Invalid detection box")
            if (
                d["label"] not in {1, 27, 31, 33, 1001}
                or d["confidence"] < self.low_threshold
            ):
                continue
            found.append(
                {**d, "box": box, "kind": "person" if d["label"] == 1 else "item"}
            )
        live, used = [], set()
        for kind in ["person", "item"]:
            tracks = [t for t in self.tracks.values() if t.kind == kind]
            indices = [i for i, d in enumerate(found) if d["kind"] == kind]
            costs = np.full((len(tracks), len(indices)), 1e6)
            for i, t in enumerate(tracks):
                dt = time - t.last
                shift = t.velocity * dt
                predicted = t.box + np.tile(shift, 2)
                for j, index in enumerate(indices):
                    box = found[index]["box"]
                    distance = float(np.linalg.norm(center(predicted) - center(box)))
                    iou = overlap(predicted, box)
                    if iou >= 0.05 or distance < (0.12 if kind == "person" else 0.06):
                        costs[i, j] = 1 - iou + distance * 2
            if costs.size:
                ti, di = linear_sum_assignment(costs)
                for i, j in zip(ti, di):
                    if costs[i, j] >= 1e5:
                        continue
                    t, d = tracks[i], found[indices[j]]
                    dt = time - t.last
                    t.velocity = (
                        0.5 * t.velocity + 0.5 * (center(d["box"]) - center(t.box)) / dt
                    )
                    if np.linalg.norm(center(d["box"]) - t.anchor) > 0.015:
                        t.stationary_since = time
                        t.anchor = center(d["box"])
                    t.box = d["box"]
                    t.last = time
                    t.confidence = d["confidence"]
                    t.hits += 1
                    live.append(t)
                    used.add(indices[j])
        for i, d in enumerate(found):
            if i in used or d["confidence"] < self.new_threshold:
                continue
            t = Track(
                self.next_id,
                d["kind"],
                d["box"],
                time,
                time,
                d["confidence"],
                time,
                center(d["box"]),
            )
            self.tracks[t.id] = t
            self.next_id += 1
            live.append(t)
        persons = [t for t in live if t.kind == "person"]
        for item in [t for t in live if t.kind == "item"]:
            if item.owner is None:
                near = sorted((proximity(p.box, item.box), p.id) for p in persons)
                candidate = near[0][1] if near and near[0][0] <= 0.75 else None
                if candidate is None or candidate != item.candidate:
                    item.candidate = candidate
                    item.candidate_since = time if candidate is not None else None
                if (
                    candidate is not None
                    and time - item.candidate_since >= self.owner_seconds
                ):
                    item.owner = candidate
            if item.owner is not None:
                owner = next((p for p in persons if p.id == item.owner), None)
                if owner is not None and proximity(owner.box, item.box) <= 1.0:
                    item.absent_since = None
                elif item.absent_since is None:
                    item.absent_since = time
        return [self._snapshot(t, time) for t in sorted(live, key=lambda t: t.id)]

    @staticmethod
    def _snapshot(t, time):
        return {
            "id": t.id,
            "kind": t.kind,
            "box": t.box.tolist(),
            "confidence": float(t.confidence),
            "age": float(time - t.born),
            "stationary": float(time - t.stationary_since),
            "speed": float(np.linalg.norm(t.velocity)),
            "hits": t.hits,
            "owner": t.owner,
            "owner_absence": 0.0
            if t.absent_since is None
            else float(time - t.absent_since),
        }


def track_features(tracks):
    persons = [t for t in tracks if t["kind"] == "person"]
    items = [t for t in tracks if t["kind"] == "item"]
    maximum = lambda rows, k: max((r[k] for r in rows), default=0.0)
    distance = min(
        (proximity(p["box"], b["box"]) for p in persons for b in items), default=3.0
    )
    area = lambda rows: max(
        ((t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1]) for t in rows),
        default=0.0,
    )
    owned = [t for t in items if t["owner"] is not None]
    visible_ids = {p["id"] for p in persons}
    return np.asarray(
        [
            min(len(persons), 10) / 10,
            min(len(items), 5) / 5,
            min(maximum(persons, "age"), 120) / 120,
            min(maximum(items, "age"), 120) / 120,
            min(maximum(persons, "stationary"), 120) / 120,
            min(maximum(items, "stationary"), 120) / 120,
            min(maximum(items, "owner_absence"), 120) / 120,
            len(owned) / max(1, len(items)),
            float(bool(persons)),
            float(bool(items)),
            min(maximum(persons, "speed"), 1),
            min(maximum(items, "speed"), 1),
            min(distance, 3) / 3,
            area(persons),
            area(items),
            maximum(persons, "confidence"),
            maximum(items, "confidence"),
            sum(t["owner"] in visible_ids for t in owned) / max(1, len(owned)),
        ],
        dtype=np.float32,
    )


def make_alerts(
    times, scores, threshold, persistence=2.0, release=1.0, cooldown=10.0, max_gap=1.1
):
    """Emit actual decision times, without backdating to the first high score."""
    if len(times) != len(scores) or persistence < 0 or release < 0 or cooldown < 0:
        raise ValueError("Invalid alert inputs")
    above = below = last = None
    last_alert = -np.inf
    active = False
    alerts = []
    for time, score in zip(times, scores):
        score = float(score)
        if not np.isfinite(time) or (last is not None and time <= last):
            raise ValueError("Alert timestamps must be finite and strictly increasing")
        if last is not None and time - last > max_gap:
            above = below = None
            active = False
        last = time
        if not np.isfinite(score):
            above = below = None
            active = False
            continue
        if score >= threshold:
            below = None
            if above is None:
                above = time
            if (
                not active
                and time - above >= persistence
                and time - last_alert >= cooldown
            ):
                alerts.append(float(time))
                active = True
                last_alert = time
        else:
            above = None
            if below is None:
                below = time
            if time - below >= release:
                active = False
    return alerts


def interval_seconds(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b < a:
            raise ValueError("Reversed interval")
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return float(sum(b - a for a, b in merged))


def unattended_state(tracks, stationary_seconds=30.0, absence_seconds=0.5):
    """Require stationary and absent-owner evidence on the same observed item.

    Missing ownership/detections produce a negative inference decision, not a
    newly asserted normal annotation.
    """
    return any(
        t["kind"] == "item"
        and t["owner"] is not None
        and t["stationary"] >= stationary_seconds
        and t["owner_absence"] >= absence_seconds
        for t in tracks
    )


def event_metrics(alerts, events, normal_intervals):
    """One alert per event; early/normal alerts and duplicate alarms stay visible.

    Onset-only matching is deliberately strict: an alert before the labelled
    onset cannot later become a true positive merely by remaining active.
    Unreviewed time is excluded from false-alert exposure.
    """
    matched = set()
    delays = []
    normal = duplicate = unknown = 0
    for time in sorted(alerts):
        eligible = [i for i, (a, b) in enumerate(events) if a <= time <= b]
        available = [i for i in eligible if i not in matched]
        if available:
            i = min(available, key=lambda i: events[i][1])
            matched.add(i)
            delays.append(time - events[i][0])
        elif eligible:
            duplicate += 1
        elif any(a <= time <= b for a, b in normal_intervals):
            normal += 1
        else:
            unknown += 1
    normal_seconds = interval_seconds(normal_intervals)
    reviewed_seconds = interval_seconds(list(normal_intervals) + list(events))
    return {
        "events": len(events),
        "detected_events": len(matched),
        "event_recall": len(matched) / len(events) if events else None,
        "normal_false_alerts": normal,
        "duplicate_alerts": duplicate,
        "unknown_time_alerts": unknown,
        "normal_seconds": normal_seconds,
        "reviewed_seconds": reviewed_seconds,
        "false_alerts_per_normal_camera_hour": normal * 3600 / normal_seconds
        if normal_seconds
        else None,
        "unmatched_alerts_per_reviewed_camera_hour": (normal + duplicate)
        * 3600
        / reviewed_seconds
        if reviewed_seconds
        else None,
        "delays_seconds": delays,
        "median_delay_seconds": float(np.median(delays)) if delays else None,
    }
