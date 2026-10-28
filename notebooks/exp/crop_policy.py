"""Crop policy: better *packaging* of the existing Stage 1 candidates (no new detectors).

Deployed as combine(baseline, crop_policy_method). Whenever a baseline person-triggered rule fires
on person P at time t, this method may emit extra candidates at the same t that cover people near P,
because the rule often fires on a bystander (or on the wrong one of two people standing together)
while the real actor is right next to them.

Policies (`mode`):
  "each"    one extra candidate per selected neighbour (its own box -> small, detailed crops)
  "union"   one extra candidate whose roi is the union of P and the selected neighbours
  "expand"  one extra candidate: P's box grown by `expand` (context), optionally extended down
            to the ground (`foot_pad`), no use of other people
Neighbours are people whose box lies within `radius_h` trigger-person heights of P (box-to-box gap),
ranked by distance or by recent "activity" (causal motion + pose change over the last `act_window_s`),
optionally required to have activity >= `min_activity`, capped at `k`.

Optional extra trigger (`trigger_wrist`): fire on any person whose wrist moved >= trigger_wrist body
heights relative to their own box within act_window_s (arm swing / throw). This is a new cue, not
packaging, and is the only way here to buy *event* recall; set it to None for packaging only.

Everything is causal: the baseline engine is replayed frame by frame and the extra candidates at
frame t only use observations up to t. Nothing is fitted on data. Per frame cost is a few numpy ops
on boxes/keypoints of the people visible.
"""
from __future__ import annotations

import math
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks")

from exp_harness import BASE_CFG, make_event  # noqa: E402
from stage1_gate import KP_CONF, RuleEngine  # noqa: E402

PERSON_TYPES = ("reach_down", "fast_motion", "sustained_presence", "hand_to_face")

DEFAULTS = dict(
    # recommended operating point "pack_k1_wrist0.36" (tune half, combined with baseline):
    # littering crop 52.2 -> 73.9, avenue event 76 -> 92, normal FA 7.05 -> 14.5 / min
    mode="each",            # "each" | "union" | "expand"
    types=("reach_down",),  # baseline rules whose candidates get re-packaged
    radius_h=99.0,          # neighbour search radius (trigger-person heights); 99 = whole frame
    k=1,                    # extra neighbour crops per trigger
    rank="activity",        # "activity" | "dist"
    min_activity=0.0,
    act_window_s=0.5,
    cooldown_s=5.0,         # one extra crop per person per cooldown
    expand=0.0,
    foot_pad=0.0,
    max_area_frac=1.0,
    trigger_wrist=0.36,     # optional extra trigger: wrist moved >= this many body heights (rel. to box) in act_window_s; None = off
    trigger_pose=None,      # optional extra trigger: mean keypoint change >= this
    trigger_cooldown_s=30.0,
    trigger_min_h=0.0,      # ignore people shorter than this fraction of frame height
    datasets=None,          # None = every dataset; or a tuple to restrict (only for analysis)
)


def _gap_h(a, b, h):
    """Box-to-box gap (0 when overlapping) in units of h."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy) / h


def _union(boxes):
    b = np.asarray(boxes, dtype=float)
    return [b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()]


def _clip(b, hw):
    h, w = hw
    return [max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])]


def crop_area_frac(roi, hw, pad=0.4):
    """Area of the crop the VLM receives (roi padded 40% like HandoffWriter, clipped) / frame area."""
    x1, y1, x2, y2 = roi
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    c = _clip([x1 - px, y1 - py, x2 + px, y2 + py], hw)
    return max(0.0, c[2] - c[0]) * max(0.0, c[3] - c[1]) / float(hw[0] * hw[1])


class _Activity:
    """Causal per-track motion/pose-change score, keyed by tracker raw_id."""

    def __init__(self, window_s):
        self.window_s = window_s
        self.hist: dict[int, deque] = {}

    def update(self, obs):
        t = obs.t
        for p in obs.persons:
            d = self.hist.setdefault(p["raw_id"], deque())
            d.append((t, p["box"], p.get("kpts")))
            while d and d[0][0] < t - 3 * self.window_s - 1.0:
                d.popleft()
        for rid in [r for r, d in self.hist.items() if t - d[-1][0] > 5.0]:
            del self.hist[rid]

    def score(self, p, t, parts=False):
        d = self.hist.get(p["raw_id"])
        if not d or len(d) < 2:
            return (0.0, 0.0, 0.0) if parts else 0.0
        box, kp = p["box"], p.get("kpts")
        h = max(1.0, box[3] - box[1])
        # reference = oldest entry inside the window, else the previous one (sparse 1 fps sources)
        ref = next((e for e in d if e[0] >= t - self.window_s and e[0] < t), d[-2])
        if ref[0] >= t:
            return (0.0, 0.0, 0.0) if parts else 0.0
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        rx, ry = (ref[1][0] + ref[1][2]) / 2, (ref[1][1] + ref[1][3]) / 2
        motion = math.hypot(cx - rx, cy - ry) / h
        pose = wrist = 0.0
        if kp is not None and ref[2] is not None:
            ok = (kp[:, 2] >= KP_CONF) & (ref[2][:, 2] >= KP_CONF)
            if ok.sum() >= 4:
                # pose change relative to the box (removes translation), body-height units
                a = (kp[:, :2] - [box[0], box[1]]) / h
                b = (ref[2][:, :2] - [ref[1][0], ref[1][1]]) / max(1.0, ref[1][3] - ref[1][1])
                dist = np.linalg.norm(a - b, axis=1)
                pose = float(np.mean(dist[ok]))
                w = [dist[i] for i in (9, 10) if ok[i]]
                wrist = float(max(w)) if w else 0.0
        if parts:
            return motion, pose, wrist
        return motion + 2.0 * pose


def make_method(**params):
    P = {**DEFAULTS, **params}
    types = tuple(P["types"])

    def method(result, dataset):
        if P["datasets"] is not None and dataset not in P["datasets"]:
            return []
        engine = RuleEngine(BASE_CFG, result.fps)
        act = _Activity(P["act_window_s"])
        last_emit: dict[str, float] = {}      # dedup key -> last time an extra crop covered it
        out = []
        for obs in result.observations:
            act.update(obs)
            evs = engine.step(obs)
            cids = engine.last_cids
            if P["trigger_wrist"] is not None or P["trigger_pose"] is not None:
                for c, p in zip(cids, obs.persons):
                    if (p["box"][3] - p["box"][1]) < P["trigger_min_h"] * obs.frame_hw[0]:
                        continue
                    m, po, w = act.score(p, obs.t, parts=True)
                    hit = (P["trigger_wrist"] is not None and w >= P["trigger_wrist"]) or \
                          (P["trigger_pose"] is not None and po >= P["trigger_pose"])
                    key = f"a{c}"
                    if hit and obs.t - last_emit.get(key, -1e9) >= P["trigger_cooldown_s"]:
                        last_emit[key] = obs.t
                        out.append(make_event("pose_activity", obs.t, p["box"], policy="activity_trigger",
                                              wrist=round(w, 3), pose=round(po, 3)))
            if not evs or not obs.persons:
                continue
            by_cid = {f"person{c}": (c, p) for c, p in zip(cids, obs.persons)}
            for e in evs:
                if e["event_type"] not in types or e["entity_id"] not in by_cid:
                    continue
                tc, tp = by_cid[e["entity_id"]]
                h = max(1.0, tp["box"][3] - tp["box"][1])
                t = obs.t
                if P["mode"] == "expand":
                    key = f"x{tc}"
                    if t - last_emit.get(key, -1e9) < P["cooldown_s"]:
                        continue
                    b = list(tp["box"])
                    bw, bh = b[2] - b[0], b[3] - b[1]
                    roi = [b[0] - P["expand"] * bw, b[1] - P["expand"] * bh,
                           b[2] + P["expand"] * bw, b[3] + P["expand"] * bh + P["foot_pad"] * bh]
                    last_emit[key] = t
                    out.append(make_event("context_" + e["event_type"], t, _clip(roi, obs.frame_hw),
                                          policy="expand", trigger=e["entity_id"]))
                    continue
                cands = []
                for c, p in zip(cids, obs.persons):
                    if c == tc:
                        continue
                    g = _gap_h(tp["box"], p["box"], h)
                    if g > P["radius_h"]:
                        continue
                    a = act.score(p, t)
                    if a < P["min_activity"]:
                        continue
                    cands.append((g, a, c, p))
                if not cands:
                    continue
                cands.sort(key=(lambda x: -x[1]) if P["rank"] == "activity" else (lambda x: x[0]))
                cands = cands[: P["k"]]
                if P["mode"] == "each":
                    for g, a, c, p in cands:
                        key = f"p{c}"
                        if t - last_emit.get(key, -1e9) < P["cooldown_s"]:
                            continue
                        last_emit[key] = t
                        out.append(make_event("neighbour_" + e["event_type"], t, p["box"], policy="each",
                                              trigger=e["entity_id"], activity=round(a, 3)))
                else:  # union
                    members = sorted([tc] + [c for _, _, c, _ in cands])
                    key = "u" + ",".join(map(str, members))
                    # skip if every member was already covered recently
                    if all(t - last_emit.get(f"p{c}", -1e9) < P["cooldown_s"] for c in members):
                        continue
                    roi = _union([tp["box"]] + [p["box"] for _, _, _, p in cands])
                    # shrink membership until the crop fits the area budget
                    while crop_area_frac(roi, obs.frame_hw) > P["max_area_frac"] and len(cands) > 1:
                        cands = cands[:-1]
                        roi = _union([tp["box"]] + [p["box"] for _, _, _, p in cands])
                    if P["foot_pad"]:
                        roi[3] += P["foot_pad"] * h
                    members = [tc] + [c for _, _, c, _ in cands]
                    for c in members:
                        last_emit[f"p{c}"] = t
                    out.append(make_event("group_" + e["event_type"], t, _clip(roi, obs.frame_hw), policy="union",
                                          trigger=e["entity_id"], members=len(cands) + 1))
        return out

    return method


METHOD = make_method()
