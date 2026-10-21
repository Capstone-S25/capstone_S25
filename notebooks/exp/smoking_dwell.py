"""Pose-free "person stays in an area" candidates, aimed at smoking recall on ~1 fps CCTV.

At 1 fps with gaps the tracker id of a standing person changes almost every frame, so dwell timed on
track ids (the baseline `loiter` rule, 20 s) almost never completes. Here dwell is timed on *position
spots*: a spot is a foot position that keeps being occupied by *some* person box, whatever its tracker
id, with gaps of up to `spot_keep_s` allowed. Rules (each fires at most once per spot per cooldown):
  * dwell        - the spot has been occupied >= dwell_s and the person moved < still_h body heights
                   since then: re-sample it every cooldown_s seconds of wall time.
  * first_seen   - (first_snapshot) a brand-new spot away from the frame border: one snapshot, so a
                   person seen in only one or two frames still reaches the VLM.
  * reseen       - (reseen_after_s) a stationary spot that went unobserved for a while is re-sampled
                   early (floor min_gap_s). Helps on the gappy smoking set; see risks in the report.
  * reacquired   - (reacquired) tracker id changed at a stationary spot. Tested, not recommended.
  * hand_near_head - (pose_cue) SEPARATE, NOT pose-free variant: lenient wrist-above-shoulder /
                   wrist-near-head lets a non-stationary person fire too. Added nothing on tune data.
Cost control: location dedupe (no new candidate within dedupe_h body heights of one sent in the
last cooldown), grouping (one candidate's ROI is the union of nearby stationary people, so a group
of workers is one VLM call, capped at group_max_frac of the frame) and an optional per-minute budget.
Spot association can be swapped for TrackStitcher canonical ids (assoc="cid") or raw ids ("raw").

Causal: a candidate at time t uses only observations with time <= t and is stamped t. Nothing is fitted.
Deploy as combine(baseline, make_method(**DEFAULTS)); OPERATING_POINTS holds the reported Pareto set.
"""
from __future__ import annotations

import math
import sys
from collections import deque

sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks")

import numpy as np  # noqa: E402

from exp_harness import make_event  # noqa: E402
from stage1_gate import GateConfig, TrackStitcher  # noqa: E402

KP_CONF = 0.3
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6
L_WRIST, R_WRIST = 9, 10

DEFAULTS = dict(
    assoc="spot",          # "spot" (position clusters across broken tracks) | "cid" | "raw"
    assoc_h=0.5,           # a person joins a spot if its foot is within this many body heights
    spot_keep_s=30.0,      # a spot survives this long without being seen (gaps allowed)
    dwell_s=1.0,           # spot must have been occupied at least this long before it can fire
    still_h=0.5,           # max foot displacement (body heights) since the dwell anchor
    cooldown_s=10.0,       # re-sample each stationary spot at most once per this many seconds
    first_snapshot=False,  # candidate on a brand-new, interior, not-yet-moving spot
    edge_frac=0.05,        # "interior" = foot box this far from the frame border (fraction of size)
    budget_per_min=None,   # optional cap on dwell candidates per rolling minute
    pose_cue=False,        # separate variant: lenient hand-near-head lets moving people fire too
    pose_head_h=0.25,
    reseen_after_s=None,   # also re-sample a stationary spot that went unobserved for >= this long
    reacquired=False,      # also re-sample when the tracker id at a stationary spot changes
    min_gap_s=3.0,         # floor between two candidates of one spot for the reseen/reacquired triggers
    group_h=0.0,           # >0: one candidate also covers stationary people within this many heights
    group_max_frac=0.6,    # ... as long as the union crop stays below this fraction of the frame
    dedupe_h=0.0,         # >0: skip if a candidate was sent within this many body heights ...
    dedupe_s=None,         # ... in the last dedupe_s seconds (None = cooldown_s)
    pad_s=2.5,
)
_ALL = dict(DEFAULTS)

# Reported operating points (tune half, combined with the baseline rules).
_G = dict(dwell_s=1.0, still_h=0.5, first_snapshot=True, group_h=8.0, group_max_frac=0.6, assoc_h=0.5)
OPERATING_POINTS = {
    "A_lean":   dict(_G, cooldown_s=60.0, dedupe_h=2.0, spot_keep_s=60.0),
    "A2":       dict(_G, cooldown_s=60.0, dedupe_h=1.0, spot_keep_s=15.0),
    "B_mid":    dict(_G, cooldown_s=30.0, dedupe_h=1.0, spot_keep_s=30.0, group_h=4.0),
    "C_recall": dict(_G, cooldown_s=10.0, dedupe_h=2.0, spot_keep_s=15.0, reseen_after_s=2.0, budget_per_min=6),
    "C2":       dict(_G, cooldown_s=10.0, dedupe_h=2.0, spot_keep_s=15.0, reseen_after_s=2.0, budget_per_min=8),
}
DEFAULTS = {**_ALL, **OPERATING_POINTS["B_mid"]}   # recommended


def _foot(b):
    return ((b[0] + b[2]) / 2.0, float(b[3]))


def _kp(k, i):
    return (float(k[i, 0]), float(k[i, 1])) if k is not None and k[i, 2] >= KP_CONF else None


def hand_near_head(p, head_h: float) -> bool:
    """Lenient pose cue: either wrist above its shoulder line, or within head_h body heights of the head."""
    k = p.get("kpts")
    if k is None:
        return False
    h = max(1.0, p["box"][3] - p["box"][1])
    face = [q for q in (_kp(k, i) for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)) if q]
    shoulders = [q for q in (_kp(k, L_SH), _kp(k, R_SH)) if q]
    for w in (_kp(k, L_WRIST), _kp(k, R_WRIST)):
        if w is None:
            continue
        if shoulders and w[1] < min(s[1] for s in shoulders):
            return True
        if face:
            fx, fy = np.mean(face, axis=0)
            if math.hypot(w[0] - fx, w[1] - fy) / h <= head_h:
                return True
    return False


class DwellEngine:
    """Causal, per-camera state. Call step(obs) once per frame in time order."""

    def __init__(self, fps: float, **params):
        self.p = {**DEFAULTS, **params}
        self.fps = fps
        self.spots: list[dict] = []
        self.by_id: dict = {}
        self.stitcher = TrackStitcher(GateConfig()) if self.p["assoc"] == "cid" else None
        self.fired: deque = deque()      # times of dwell candidates, for the budget
        self.next_spot = 0
        self.fired_at: deque = deque()   # (t, foot_x, foot_y, h) of sent candidates, for location dedupe

    # -- association --------------------------------------------------------------------------
    def _new_spot(self, f, h, t):
        s = {"id": self.next_spot, "foot": f, "h": h, "first_t": t, "last_t": t,
             "hist": deque(), "last_fire": -1e9, "snapshotted": False, "new": True}
        self.next_spot += 1
        return s

    def _assign_spots(self, persons, t):
        p = self.p
        self.spots = [s for s in self.spots if t - s["last_t"] <= p["spot_keep_s"]]
        pairs = []
        for i, q in enumerate(persons):
            f, h = _foot(q["box"]), max(1.0, q["box"][3] - q["box"][1])
            for j, s in enumerate(self.spots):
                ratio = h / s["h"]
                if not 0.6 <= ratio <= 1.67:
                    continue
                d = math.hypot(f[0] - s["foot"][0], f[1] - s["foot"][1]) / max(h, s["h"])
                if d <= p["assoc_h"]:
                    pairs.append((d, i, j))
        pairs.sort()
        out = [None] * len(persons)
        used = set()
        for d, i, j in pairs:
            if out[i] is None and j not in used:
                out[i] = self.spots[j]
                out[i]["new"] = False
                used.add(j)
        for i, q in enumerate(persons):
            if out[i] is None:
                s = self._new_spot(_foot(q["box"]), max(1.0, q["box"][3] - q["box"][1]), t)
                self.spots.append(s)
                out[i] = s
        return out

    def _assign_ids(self, persons, t):
        keys = self.stitcher.assign(persons, t) if self.stitcher else [q["raw_id"] for q in persons]
        out = []
        for k, q in zip(keys, persons):
            s = self.by_id.get(k)
            if s is None or t - s["last_t"] > self.p["spot_keep_s"]:
                s = self.by_id[k] = self._new_spot(_foot(q["box"]), max(1.0, q["box"][3] - q["box"][1]), t)
            else:
                s["new"] = False
            out.append(s)
        horizon = self.p["spot_keep_s"]
        for k in [k for k, s in self.by_id.items() if t - s["last_t"] > horizon]:
            del self.by_id[k]
        return out

    # -- rules --------------------------------------------------------------------------------
    def step(self, obs) -> list[dict]:
        p, t = self.p, obs.t
        persons = obs.persons
        if not persons:
            return []
        spots = self._assign_spots(persons, t) if p["assoc"] == "spot" else self._assign_ids(persons, t)
        while self.fired and self.fired[0] < t - 60.0:
            self.fired.popleft()
        fh, fw = obs.frame_hw
        cands, stills = [], []
        for i, (q, s) in enumerate(zip(persons, spots)):
            f, h = _foot(q["box"]), max(1.0, q["box"][3] - q["box"][1])
            s["hist"].append((t, f[0], f[1], h))
            while len(s["hist"]) > 1 and s["hist"][0][0] < t - max(p["dwell_s"], 1.0) - p["spot_keep_s"]:
                s["hist"].popleft()
            # spot centre follows the person (EMA), height too
            s["foot"] = (0.5 * s["foot"][0] + 0.5 * f[0], 0.5 * s["foot"][1] + 0.5 * f[1])
            s["h"] = 0.7 * s["h"] + 0.3 * h
            unseen = t - s["last_t"] if not s["new"] else 0.0
            reacq = (not s["new"]) and s.get("raw") is not None and s["raw"] != q["raw_id"]
            s["last_t"] = t
            s["raw"] = q["raw_id"]

            still = self._still(s, t, h)
            stills.append(still)
            reason = None
            since = t - s["last_fire"]
            if still and since >= p["cooldown_s"]:
                reason = "dwell"
            elif still and since >= p["min_gap_s"] and p["reseen_after_s"] is not None and unseen >= p["reseen_after_s"]:
                reason = "reseen"          # spot was not observed for a while: what we knew is stale
            elif still and since >= p["min_gap_s"] and p["reacquired"] and reacq:
                reason = "reacquired"      # tracker lost and re-found the person at the same place
            elif p["first_snapshot"] and s["new"] and not s["snapshotted"]:
                x1, y1, x2, y2 = q["box"]
                m = p["edge_frac"]
                if x1 > m * fw and x2 < (1 - m) * fw and y2 < (1 - m) * fh:
                    reason = "first_seen"
            elif p["pose_cue"] and since >= p["cooldown_s"] and hand_near_head(q, p["pose_head_h"]):
                reason = "hand_near_head"
            if reason is not None:
                cands.append((since, reason, i))

        dd_s = p["cooldown_s"] if p["dedupe_s"] is None else p["dedupe_s"]
        while self.fired_at and self.fired_at[0][0] < t - max(dd_s, p["min_gap_s"]):
            self.fired_at.popleft()
        events, done = [], set()
        # oldest-sampled spot first, so a budget is shared fairly
        for since, reason, i in sorted(cands, key=lambda c: -c[0]):
            if i in done:
                continue
            if p["budget_per_min"] is not None and len(self.fired) >= p["budget_per_min"]:
                break
            q, s = persons[i], spots[i]
            if reason == "first_seen":
                s["snapshotted"] = True
            f, h = _foot(q["box"]), max(1.0, q["box"][3] - q["box"][1])
            # location dedupe: spots fragment (occlusion, crowd swaps), places do not
            win = p["min_gap_s"] if reason in ("reseen", "reacquired") else dd_s
            if p["dedupe_h"] and any(t - ft < win and math.hypot(f[0] - fx, f[1] - fy) / max(h, fh_) < p["dedupe_h"]
                                     for ft, fx, fy, fh_ in self.fired_at):
                s["last_fire"] = t
                continue
            # group: one candidate covers the nearby stationary people too (one crop, one VLM call)
            members = [i]
            if p["group_h"]:
                near = sorted((math.hypot(_foot(persons[j]["box"])[0] - f[0], _foot(persons[j]["box"])[1] - f[1]) / h, j)
                              for j in range(len(persons)) if j != i and j not in done
                              and (stills[j] or any(c[2] == j for c in cands)))
                roi = list(q["box"])
                for d, j in near:
                    if d > p["group_h"]:
                        break
                    b = persons[j]["box"]
                    u = [min(roi[0], b[0]), min(roi[1], b[1]), max(roi[2], b[2]), max(roi[3], b[3])]
                    if u[2] - u[0] > p["group_max_frac"] * fw or u[3] - u[1] > p["group_max_frac"] * fh:
                        continue
                    roi = u
                    members.append(j)
            else:
                roi = q["box"]
            for j in members:
                done.add(j)
                spots[j]["last_fire"] = t
                fj = _foot(persons[j]["box"])
                self.fired_at.append((t, fj[0], fj[1], max(1.0, persons[j]["box"][3] - persons[j]["box"][1])))
            self.fired.append(t)
            events.append(make_event("sustained_presence", t, roi, pad=p["pad_s"], cue=reason,
                                     spot=int(s["id"]), n_people=len(members), dwell_s=round(t - s["first_t"], 2)))
        return events

    def _still(self, s, t, h) -> bool:
        p = self.p
        if t - s["first_t"] < p["dwell_s"] - 1e-6 or len(s["hist"]) < 2:
            return False
        cur = s["hist"][-1]
        # anchor = most recent point at least dwell_s old (allows gaps); all points since then must
        # stay within still_h of the current foot position
        anchor_i = None
        for i in range(len(s["hist"]) - 2, -1, -1):
            if s["hist"][i][0] <= t - p["dwell_s"] + 1e-6:
                anchor_i = i
                break
        if anchor_i is None:
            return False
        for (_, x, y, _) in list(s["hist"])[anchor_i:-1]:
            if math.hypot(x - cur[1], y - cur[2]) / h > p["still_h"]:
                return False
        return True


def make_method(**params):
    def method(result, dataset_name):
        eng = DwellEngine(result.fps, **params)
        out = []
        for obs in result.observations:
            out += eng.step(obs)
        return out
    method.params = {**DEFAULTS, **params}
    return method
