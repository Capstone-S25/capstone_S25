"""Pose cues for the littering ACT (Stage 1 add-on to the baseline rules).

The baseline only has "reach_down" (a wrist at knee level). Littering is often a throw, a toss or a
drop with the hand at waist height, which reach_down cannot see. This module adds cheap, causal
per-person cues computed from the pose keypoints and the person box of the current frame plus a
short history of the same track:

  arm_swing   fast wrist motion RELATIVE to the shoulder/hip centre, normalised by the track's body
              height (so walking, which moves the whole body, does not count). Uses a short lag
              (~0.2-0.4 s) so single-frame keypoint jitter is averaged out.
  arm_extend  wrist far from the body: horizontal distance from the shoulder (sideways / forward
              throw), or raised above the shoulder (overhand throw).
  low_hand    wrist clearly below the hip (dropping / setting something down), weaker than the
              baseline knee-level reach.
  bend        torso shortened vs the track's own median (shoulders dropping toward the hips) or
              box height shrinking vs the track's median (crouch).

Each cue produces a score; a person fires a "littering_act" candidate when a cue exceeds its
threshold, at most once per `cooldown_s` per (stitched) track. Small or low-confidence skeletons
are ignored because keypoint jitter on them looks like fast motion.

Everything is causal (only observations with time <= t are used) and pure numpy on boxes and
keypoints. Nothing is fitted on data; thresholds were swept on the tune halves of the littering
clips (recall) and the Avenue training videos (false alarms).
"""
from __future__ import annotations

import math
import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks")

from exp_harness import BASE_CFG, make_event  # noqa: E402
from stage1_gate import TrackStitcher, replay_rules  # noqa: E402

L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP, L_KNE, R_KNE = 5, 6, 7, 8, 9, 10, 11, 12, 13, 14

DEFAULTS = dict(
    # recommended operating point "C" (tune half: about +7.6 candidates/min over the baseline on normal footage)
    kp_conf=0.35,          # keypoint confidence needed for a joint to be used
    min_h_px=60.0,         # skip people shorter than this (keypoint noise dominates)
    extend_h=0.16,         # wrist horizontally >= this many body heights from its shoulder ...
    need_extend=3,         # ... in >= 3 frames of the persistence window (kills single-frame jitter)
    low_hand_h=0.03,       # wrist below the hip line by >= this (drop / set down at waist level) ...
    need_low=4,            # ... in >= 4 frames
    raise_h=None,          # wrist above shoulder (overhand throw); 0.0 with need_raise=3 is point "D"
    need_raise=3,
    outside_h=None,        # wrist outside the torso span; tried, no better than extend_h
    swing_h_per_s=99.0,    # wrist speed relative to torso (body heights/s); off: helps Avenue, not littering
    swing_min_disp_h=0.3,
    lag_s=0.3,
    max_dt_s=0.45,         # no speed cue if the previous pose sample is older than this (1 fps CCTV)
    bend_ratio=None,       # torso shortening vs the track median: no gain, off
    crouch_ratio=None,     # box shrinking vs the track median: no gain, off
    need_hits=1,           # default persistence for cues without their own need_<cue>
    hit_window_s=0.6,      # persistence window (stretched at low fps so it can hold need_<cue> frames)
    cooldown_s=5.0,        # one candidate per person per cooldown
    skip_if_baseline=True,  # do not duplicate a baseline candidate for the same person in the last baseline_gap_s
    baseline_gap_s=2.5,
    roi="person",          # "person" | "person+ground" | "group" (group/ground gave no crop-recall gain)
    hint_types=False,      # event_type per cue (True) or a single "littering_act" type
)


def _pt(k, i, c):
    return (float(k[i, 0]), float(k[i, 1])) if k[i, 2] >= c else None


def _mid(a, b):
    if a and b:
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return a or b


class _Track:
    __slots__ = ("hs", "torsos", "poses", "hits", "last_fire")

    def __init__(self):
        self.hs = deque(maxlen=60)       # (t, box h)
        self.torsos = deque(maxlen=60)   # torso length samples
        self.poses = deque(maxlen=20)    # (t, raw_id, {wrist side: rel (x, y)})
        self.hits = deque()
        self.last_fire = -1e9


def person_cues(p, t, tr: _Track, P) -> dict:
    """Cue scores for one person at time t, given its track history (updates the history)."""
    x1, y1, x2, y2 = p["box"]
    h_now = max(1.0, y2 - y1)
    tr.hs.append(h_now)
    H = float(np.median(tr.hs))
    k = p.get("kpts")
    out = {}
    if P["crouch_ratio"] is not None and len(tr.hs) >= 5:
        out["crouch"] = H / h_now          # >1 when the box is shorter than usual
    if k is None or H < P["min_h_px"]:
        return out
    c = P["kp_conf"]
    ls, rs, lh, rh = (_pt(k, i, c) for i in (L_SHO, R_SHO, L_HIP, R_HIP))
    sho, hip = _mid(ls, rs), _mid(lh, rh)
    anchor = sho or hip
    if anchor is None:
        return out
    if sho and hip:
        torso = abs(hip[1] - sho[1])
        if P["bend_ratio"] is not None and len(tr.torsos) >= 5:
            out["bend"] = float(np.median(tr.torsos)) / max(1.0, torso)
        tr.torsos.append(torso)
    rel = {}
    centre = _mid(sho, hip)
    body_x = [q[0] for q in (ls, rs, lh, rh) if q]
    lo_x, hi_x = min(body_x), max(body_x)
    for side, wi, ei, si, hi in (("l", L_WRI, L_ELB, ls, lh), ("r", R_WRI, R_ELB, rs, rh)):
        w = _pt(k, wi, c)
        if w is None:
            continue
        e = _pt(k, ei, c)
        if e is not None:
            fx, fy = w[0] - e[0], w[1] - e[1]
            fl = math.hypot(fx, fy)
            if fl > 0.04 * H:
                # forearm angle from hanging straight down (0) to horizontal (90) to up (180)
                ang = math.degrees(math.atan2(abs(fx), fy))
                out["forearm"] = max(out.get("forearm", 0.0), ang)
                s0 = si or sho
                if s0 is not None:
                    ux, uy = e[0] - s0[0], e[1] - s0[1]
                    ul = math.hypot(ux, uy)
                    if ul > 0.04 * H:
                        cosv = (ux * fx + uy * fy) / (ul * fl)   # 1 = straight arm
                        out["straight"] = max(out.get("straight", -1.0), cosv)
        # horizontal distance of the wrist OUTSIDE the torso's span (shoulders + hips): immune to
        # left/right keypoint swaps and to arms hanging in front of a frontal body
        out["outside"] = max(out.get("outside", 0.0), max(0.0, w[0] - hi_x, lo_x - w[0]) / H)
        if centre is not None:
            out["reach"] = max(out.get("reach", 0.0), math.hypot(w[0] - centre[0], w[1] - centre[1]) / H)
        rel[side] = ((w[0] - anchor[0]) / H, (w[1] - anchor[1]) / H)
        s = si or sho
        if s is not None:
            out["extend"] = max(out.get("extend", 0.0), abs(w[0] - s[0]) / H)
            if P["raise_h"] is not None:
                out["raise"] = max(out.get("raise", -9.0), (s[1] - w[1]) / H)
        hp = hi or hip
        if hp is not None and P["low_hand_h"] is not None:
            out["low"] = max(out.get("low", -9.0), (w[1] - hp[1]) / H)
    # wrist speed relative to the torso, against the sample closest to lag_s ago (same raw track id)
    if rel:
        best = None
        for (tp, rid, rp) in reversed(tr.poses):
            dt = t - tp
            if rid != p["raw_id"] or dt > P["max_dt_s"]:
                break
            if dt <= 0:
                continue
            if best is None or abs(dt - P["lag_s"]) < abs(best[0] - P["lag_s"]):
                best = (dt, rp)
        if best is not None:
            dt, rp = best
            sp = disp = 0.0
            for side, (wx, wy) in rel.items():
                if side in rp:
                    d = math.hypot(wx - rp[side][0], wy - rp[side][1])
                    if d / dt > sp:
                        sp, disp = d / dt, d
            out["swing"] = sp
            out["swing_disp"] = disp
        tr.poses.append((t, p["raw_id"], rel))
    return out


def fired(cues: dict, P) -> list[str]:
    f = []
    if cues.get("swing", 0) >= P["swing_h_per_s"] and cues.get("swing_disp", 0) >= P["swing_min_disp_h"]:
        f.append("swing")
    if P["extend_h"] is not None and cues.get("extend", 0) >= P["extend_h"]:
        f.append("extend")
    if P.get("outside_h") is not None and cues.get("outside", 0) >= P["outside_h"]:
        f.append("outside")
    if P["raise_h"] is not None and cues.get("raise", -9) >= P["raise_h"]:
        f.append("raise")
    if P["low_hand_h"] is not None and cues.get("low", -9) >= P["low_hand_h"]:
        f.append("low")
    if P["bend_ratio"] is not None and cues.get("bend", 0) >= 1.0 / P["bend_ratio"]:
        f.append("bend")
    if P["crouch_ratio"] is not None and cues.get("crouch", 0) >= 1.0 / P["crouch_ratio"]:
        f.append("crouch")
    return f


def _roi(box, mode, hw, others=()):
    x1, y1, x2, y2 = box
    if mode == "group":
        # union with people standing within group_gap_h body heights (the actor may be the neighbour
        # of whoever moved their arm); capped so the crop stays usable
        h = y2 - y1
        for o in others:
            gap = max(o[0] - x2, x1 - o[2], 0.0)
            if gap <= 0.5 * h and abs((o[3] - o[1]) / max(1.0, h) - 1) < 0.5:
                x1, y1, x2, y2 = min(x1, o[0]), min(y1, o[1]), max(x2, o[2]), max(y2, o[3])
        return [x1, y1, x2, y2]
    if mode == "person+ground":
        w, h = x2 - x1, y2 - y1
        H, W = hw
        return [max(0, x1 - 0.25 * w), y1, min(W, x2 + 0.25 * w), min(H, y2 + 0.15 * h)]
    return box


# Baseline candidates per cached sequence (the baseline is deterministic; this only saves replay time in sweeps).
# In a live system the same check is a lookup into the baseline rule engine's recent candidates.
_BASE_CACHE: dict[int, list] = {}


def make_method(**params):
    P = {**DEFAULTS, **params}

    def method(result, dataset):
        base = []
        if P["skip_if_baseline"]:
            key = id(result)
            if key not in _BASE_CACHE:
                _BASE_CACHE[key] = [(e["timestamp_s"], e["entity_id"]) for e in replay_rules(result, BASE_CFG)[0]]
            base = _BASE_CACHE[key]
        stitch = TrackStitcher(BASE_CFG)
        tracks: dict[int, _Track] = {}
        events = []
        # persistence window: at least hit_window_s, and long enough to hold need_hits frames at this fps
        nmax = max([P["need_hits"]] + [v for k, v in P.items() if k.startswith("need_")])
        window = max(P["hit_window_s"], (nmax - 0.5) / max(result.fps, 1e-6))
        for obs in result.observations:
            t = obs.t
            cids = stitch.assign(obs.persons, t)
            for cid, p in zip(cids, obs.persons):
                tr = tracks.get(cid)
                if tr is None:
                    tr = tracks[cid] = _Track()
                cues = person_cues(p, t, tr, P)
                now = fired(cues, P)
                for c in now:
                    tr.hits.append((t, c))
                while tr.hits and tr.hits[0][0] < t - window:
                    tr.hits.popleft()
                # a cue counts once it was true in need_<cue> (default need_hits) frames inside the window
                f = [c for c in now if sum(1 for _, x in tr.hits if x == c) >= P.get("need_" + c, P["need_hits"])]
                if not f or t - tr.last_fire < P["cooldown_s"]:
                    continue
                ent = f"person{cid}"
                if any(e == ent and 0 <= t - bt <= P["baseline_gap_s"] for bt, e in base):
                    continue    # baseline already sent this person moments ago (base events at bt <= t only)
                tr.last_fire = t
                etype = ("act_" + f[0]) if P["hint_types"] else "littering_act"
                events.append(make_event(etype, t, _roi(p["box"], P["roi"], obs.frame_hw, [q["box"] for q in obs.persons if q is not p]), pad=BASE_CFG.clip_pad_s,
                                         entity_id=ent, cues=f))
        return events

    return method
