"""Trajectory / motion anomaly cues for Stage 1 (aimed at Avenue-style scenes).

All cues run on an *id-robust* person track: ByteTrack at 12.5 fps often gives a running person a
new raw id every frame, so the baseline speed rule (which needs >= 4 consecutive points of one raw
id) never sees them. `Linker` re-attaches a new raw id to a track lost < link_gap_s ago when its
constant-velocity prediction (or a plausible running jump) lands near it with a similar height.

Cues (pure numpy on boxes; microseconds per person per frame):
  * fast      - robust speed (median step, capped by net displacement, normalised by the track's
                robust height, ignoring boxes clipped by the left/right frame edge).
  * turn      - direction reversal while moving (erratic / zig-zag path).
  * density   - the person's (foot x, foot y, box height, speed class) is rare under a smoothed
                4-D histogram of normal footage: catches people unusually close to the camera, in
                places people never are, or standing still where people only pass by. Must hold
                for `dens_hold` consecutive observations of the track; re-fires every repeat_s.
  * wrongdir  - sustained motion whose direction is rare for that grid cell in normal footage.
  * item      - COCO item (bag, bottle...) moving fast or upwards (thrown object).

Scene models ("density", "wrongdir") are per camera. They are fitted ONLY on SPLITS["normal"]
(tune half of Avenue training videos, no anomalies) and applied to the Avenue camera ("avenue" and
"normal" datasets). When scoring "normal" the model is fitted leave-one-video-out, so the reported
false-alarm rate is out-of-sample. Littering and smoking have no normal footage of their cameras,
so only the scene-free cues (fast, turn, item) can run there.

Everything is causal: a candidate at time t uses observations with time <= t.
"""
from __future__ import annotations

import math
import sys
from collections import deque

sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks")

import numpy as np

from exp_harness import SPLITS, make_event

EVENT_TYPES = {"fast": "fast_motion", "turn": "erratic_motion", "density": "unusual_presence",
               "wrongdir": "wrong_direction", "item": "thrown_object"}

DEFAULTS = dict(
    cues=("fast", "density"),
    # linking
    link_gap_s=0.6, link_dist_h=0.6, link_speed_h=3.5,
    # fast
    speed_h_per_s=3.0, speed_window_s=0.5, speed_min_pts=3,
    # turn
    turn_min_speed=0.8, turn_cos=-0.3, turn_window_s=0.6,
    # density model
    # dens_q: fire below this quantile of normal person-observation densities (overrides dens_thr)
    dens_bins=(16, 16, 10, 3), dens_smooth=1, dens_thr=1e-3, dens_q=0.005, dens_hold=5,
    # wrong direction
    grid=(8, 6), wrong_min_speed=0.35, wrong_p=0.03, wrong_hold_s=1.0,
    # item
    item_speed_frame=0.6, item_up_frame=0.35,
    # dedup: one candidate per track per cooldown_s (any cue); persistent cues re-fire every repeat_s
    cooldown_s=2.0, repeat_s=3.0,
    # persistent cues at the same place (a person standing near the camera, a static false detection):
    # each re-fire at a location (IoU >= loc_iou with an earlier fire) doubles the wait, up to backoff_max_s
    backoff=True, loc_iou=0.4, backoff_max_s=48.0,
    datasets=None,
)

SCENE_DATASETS = {"avenue", "normal"}
H_LO, H_HI = math.log(0.08), math.log(1.0)
SPEED_EDGES = (0.25, 1.0)


# --------------------------------------------------------------------------- id-robust tracks

class Linker:
    def __init__(self, P):
        self.P = P
        self.alias: dict[int, int] = {}
        self.tracks: dict[int, dict] = {}
        self.next = 0

    def assign(self, persons, t, fw):
        P = self.P
        out = [None] * len(persons)
        used = set()
        for i, p in enumerate(persons):
            tid = self.alias.get(p["raw_id"])
            if tid is not None and tid in self.tracks and tid not in used:
                out[i] = tid
                used.add(tid)
        for i, p in enumerate(persons):
            if out[i] is not None:
                continue
            b = p["box"]
            fx, fy, h = (b[0] + b[2]) / 2, b[3], max(1.0, b[3] - b[1])
            best, bd = None, 1e9
            for tid, tr in self.tracks.items():
                gap = t - tr["last_t"]
                if tid in used or gap <= 0 or gap > P["link_gap_s"] or not 0.7 <= h / tr["h"] <= 1.43:
                    continue
                d = math.hypot(fx - tr["fx"] - tr["vx"] * gap, fy - tr["fy"] - tr["vy"] * gap) / max(h, tr["h"])
                d_raw = math.hypot(fx - tr["fx"], fy - tr["fy"]) / max(h, tr["h"])
                if (d <= P["link_dist_h"] or d_raw <= P["link_speed_h"] * gap) and min(d, d_raw) < bd:
                    best, bd = tid, min(d, d_raw)
            if best is None:
                best = self.next
                self.next += 1
                self.tracks[best] = {"hist": deque(), "first_t": t, "last_t": t, "fx": fx, "fy": fy, "h": h,
                                     "vx": 0.0, "vy": 0.0, "last_fire": -1e9, "last_cue": {}, "fast_armed": True,
                                     "wrong_since": None, "rare_run": 0}
            self.alias[p["raw_id"]] = best
            out[i] = best
            used.add(best)
        for tid, p in zip(out, persons):
            tr = self.tracks[tid]
            b = p["box"]
            fx, fy, h = (b[0] + b[2]) / 2, b[3], max(1.0, b[3] - b[1])
            gap = t - tr["last_t"]
            if gap > 0:
                tr["vx"] = 0.5 * tr["vx"] + 0.5 * (fx - tr["fx"]) / gap
                tr["vy"] = 0.5 * tr["vy"] + 0.5 * (fy - tr["fy"]) / gap
            tr.update(last_t=t, fx=fx, fy=fy, h=0.7 * tr["h"] + 0.3 * h if tr["hist"] else h)
            edge = b[0] <= 2 or b[2] >= fw - 2
            tr["hist"].append((t, fx, fy, h, edge))
            while tr["hist"] and tr["hist"][0][0] < t - 3.0:
                tr["hist"].popleft()
        for tid in [k for k, tr in self.tracks.items() if t - tr["last_t"] > 5.0]:
            del self.tracks[tid]
        return out


def robust_h(hist):
    return float(np.percentile([h[3] for h in hist], 75))


def track_speed(hist, t, window, min_pts, skip_edge=True):
    pts = [h for h in hist if h[0] >= t - window and not (skip_edge and h[4])]
    if len(pts) < min_pts or pts[-1][0] - pts[0][0] < 0.4 * window:
        return None
    H = robust_h(hist)
    steps = [math.hypot(b[1] - a[1], b[2] - a[2]) / max(1e-6, b[0] - a[0]) / H for a, b in zip(pts, pts[1:])]
    net = math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]) / (pts[-1][0] - pts[0][0]) / H
    return min(float(np.median(steps)), 1.25 * net)


def track_vel(hist, t0, t1):
    pts = [h for h in hist if t0 <= h[0] <= t1]
    if len(pts) < 2 or pts[-1][0] - pts[0][0] < 0.15:
        return None
    dt = pts[-1][0] - pts[0][0]
    H = robust_h(hist)
    return (pts[-1][1] - pts[0][1]) / dt / H, (pts[-1][2] - pts[0][2]) / dt / H


def _dir_bin(dx, dy):
    return int(((math.atan2(dy, dx) + math.pi) / (2 * math.pi) * 8)) % 8


# --------------------------------------------------------------------------- scene models

def person_features(result, P):
    """Causal per-person features on the id-robust track: yields (obs, person, tid, track, feat)."""
    link = Linker(P)
    for o in result.observations:
        fh, fw = o.frame_hw
        tids = link.assign(o.persons, o.t, fw)
        for tid, p in zip(tids, o.persons):
            tr = link.tracks[tid]
            b = p["box"]
            h = max(1.0, b[3] - b[1])
            v = track_speed(tr["hist"], o.t, max(P["speed_window_s"], 3.0 / result.fps), P["speed_min_pts"], skip_edge=False)
            feat = {"x": (b[0] + b[2]) / 2 / fw, "y": b[3] / fh, "lh": math.log(max(1e-3, h / fh)), "v": v}
            yield o, p, tid, tr, feat


def _dens_index(f, bins):
    bx, by, bh, bv = bins
    ix = min(bx - 1, max(0, int(f["x"] * bx)))
    iy = min(by - 1, max(0, int(f["y"] * by)))
    ih = min(bh - 1, max(0, int((f["lh"] - H_LO) / (H_HI - H_LO) * bh)))
    if bv == 1:
        return ix, iy, ih, 0
    iv = None if f["v"] is None else (0 if f["v"] < SPEED_EDGES[0] else 1 if f["v"] < SPEED_EDGES[1] else 2)
    return ix, iy, ih, iv


def _box_smooth(a, r):
    """Sum over a (2r+1)^3 neighbourhood in the first three axes (zero padded)."""
    if r <= 0:
        return a
    out = a.copy()
    for ax in range(3):
        cur = out
        acc = np.zeros_like(cur)
        pad = [(0, 0)] * cur.ndim
        pad[ax] = (r, r)
        padded = np.pad(cur, pad)
        n = cur.shape[ax]
        for s in range(2 * r + 1):
            acc += np.take(padded, range(s, s + n), axis=ax)
        out = acc
    return out


class SceneModel:
    def __init__(self, P):
        self.P = P

    def fit(self, results):
        P = self.P
        bins = P["dens_bins"]
        cnt = np.zeros(bins)
        gx, gy = P["grid"]
        dirs = np.zeros((gx, gy, 8))
        for r in results:
            for o, p, tid, tr, f in person_features(r, P):
                ix, iy, ih, iv = _dens_index(f, bins)
                if iv is None:
                    cnt[ix, iy, ih, :] += 1.0 / bins[3]
                else:
                    cnt[ix, iy, ih, iv] += 1
                vv = track_vel(tr["hist"], o.t - 0.6, o.t)
                if vv and math.hypot(*vv) >= P["wrong_min_speed"]:
                    cx, cy = min(gx - 1, int(f["x"] * gx)), min(gy - 1, max(0, int(f["y"] * gy)))
                    dirs[cx, cy, _dir_bin(*vv)] += 1
        sm = _box_smooth(cnt, P["dens_smooth"])
        self.dens = sm / max(1.0, cnt.sum())
        self.dens_any = self.dens.sum(axis=3)
        self.dirs = (dirs + 0.5) / (dirs.sum(axis=2, keepdims=True) + 4.0)
        self.dir_n = dirs.sum(axis=2)
        # density of every normal person-observation, so a threshold can be given as a quantile
        self.train_dens = np.sort([self.density(f) for r in results for *_, f in person_features(r, P)])
        return self

    def threshold(self, P):
        # P is the caller's params (the cached model may have been built by another param set)
        if P.get("dens_q") is None:
            return P["dens_thr"]
        return float(np.quantile(self.train_dens, P["dens_q"]))

    def density(self, f):
        ix, iy, ih, iv = _dens_index(f, self.P["dens_bins"])
        return float(self.dens_any[ix, iy, ih] / self.P["dens_bins"][3] if iv is None else self.dens[ix, iy, ih, iv])


_SCENE_CACHE: dict = {}
_SCENE_KEYS = ("link_gap_s", "link_dist_h", "link_speed_h", "speed_window_s", "speed_min_pts", "dens_bins",
               "dens_smooth", "grid", "wrong_min_speed")


_FIT_NORMAL = None


def _fit_normal():
    """The scene model is always fitted on the TUNE half of the normal videos, also when the harness runs held-out."""
    global _FIT_NORMAL
    if _FIT_NORMAL is None:
        from exp_harness import load_splits
        _FIT_NORMAL = load_splits(heldout=False)["normal"]
    return _FIT_NORMAL


def scene_model(P, exclude=None):
    key = (tuple((k, str(P[k])) for k in _SCENE_KEYS), exclude)
    if key not in _SCENE_CACHE:
        _SCENE_CACHE[key] = SceneModel(P).fit([r for g, r in _fit_normal() if r.seq_id != exclude])
    return _SCENE_CACHE[key]


# --------------------------------------------------------------------------- method

def make_method(**params):
    P = {**DEFAULTS, **params}
    cues = set(P["cues"])

    def method(result, dataset):
        if P["datasets"] is not None and dataset not in P["datasets"]:
            return []
        scene = None
        if dataset in SCENE_DATASETS and cues & {"density", "wrongdir"}:
            scene = scene_model(P, exclude=result.seq_id if dataset == "normal" else None)
            dens_thr = scene.threshold(P)
        events = []
        items: dict[int, deque] = {}
        item_fire: dict[int, float] = {}
        last_obs = None
        places: list[list] = []          # [box, last_fire_t, interval]
        for o, p, tid, tr, f in person_features(result, P):
            t = o.t
            b = p["box"]
            fired = []
            if "fast" in cues:
                v = track_speed(tr["hist"], t, max(P["speed_window_s"], 3.0 / result.fps), P["speed_min_pts"])
                if v is not None:
                    if v >= P["speed_h_per_s"] and tr["fast_armed"]:
                        fired.append(("fast", v))
                    elif v < 0.6 * P["speed_h_per_s"]:
                        tr["fast_armed"] = True
            if "turn" in cues:
                w = P["turn_window_s"]
                a, c = track_vel(tr["hist"], t - 2 * w, t - w), track_vel(tr["hist"], t - w, t)
                if a and c:
                    na, nc = math.hypot(*a), math.hypot(*c)
                    if na > P["turn_min_speed"] and nc > P["turn_min_speed"]:
                        cos = (a[0] * c[0] + a[1] * c[1]) / (na * nc)
                        if cos < P["turn_cos"] and t - tr["last_cue"].get("turn", -1e9) >= P["repeat_s"]:
                            fired.append(("turn", cos))
            if scene is not None and "density" in cues:
                d = scene.density(f)
                tr["rare_run"] = tr["rare_run"] + 1 if d < dens_thr else 0
                if tr["rare_run"] >= P["dens_hold"] and t - tr["last_cue"].get("density", -1e9) >= P["repeat_s"] \
                        and (not P["backoff"] or _place_ok(places, b, t, P)):
                    fired.append(("density", d * 1e6))
            if scene is not None and "wrongdir" in cues:
                gx, gy = P["grid"]
                cx, cy = min(gx - 1, int(f["x"] * gx)), min(gy - 1, max(0, int(f["y"] * gy)))
                vv = track_vel(tr["hist"], t - 0.6, t)
                bad = bool(vv and math.hypot(*vv) >= P["wrong_min_speed"] and scene.dir_n[cx, cy] >= 20
                           and scene.dirs[cx, cy, _dir_bin(*vv)] < P["wrong_p"])
                if bad:
                    tr["wrong_since"] = t if tr["wrong_since"] is None else tr["wrong_since"]
                    if t - tr["wrong_since"] >= P["wrong_hold_s"] and t - tr["last_cue"].get("wrongdir", -1e9) >= P["repeat_s"]:
                        fired.append(("wrongdir", 0.0))
                else:
                    tr["wrong_since"] = None
            if fired and t - tr["last_fire"] >= P["cooldown_s"]:
                cue, val = fired[0]
                tr["last_fire"] = t
                tr["last_cue"][cue] = t
                if cue == "fast":
                    tr["fast_armed"] = False
                if cue == "density" and P["backoff"]:
                    _place_fire(places, b, t, P)
                events.append(make_event(EVENT_TYPES[cue], t, b, metric=round(float(val), 3), track=tid))
            if "item" in cues and o is not last_obs:
                last_obs = o
                events += _items(o, items, item_fire, P)
        return events

    return method


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def _place_ok(places, box, t, P):
    for pl in places:
        if _iou(pl[0], box) >= P["loc_iou"]:
            return t - pl[1] >= pl[2]
    return True


def _place_fire(places, box, t, P):
    for pl in places:
        if _iou(pl[0], box) >= P["loc_iou"]:
            pl[:] = [box, t, min(P["backoff_max_s"], 2 * pl[2])]
            return
    places.append([box, t, P["repeat_s"]])


def _items(o, items, item_fire, P):
    fh = o.frame_hw[0]
    out = []
    for it in o.items:
        hq = items.setdefault(it["raw_id"], deque())
        c = ((it["box"][0] + it["box"][2]) / 2, (it["box"][1] + it["box"][3]) / 2)
        hq.append((o.t, *c))
        while hq and hq[0][0] < o.t - 0.5:
            hq.popleft()
        if len(hq) >= 3 and hq[-1][0] - hq[0][0] >= 0.15:
            dt = hq[-1][0] - hq[0][0]
            vx, vy = (hq[-1][1] - hq[0][1]) / dt / fh, (hq[-1][2] - hq[0][2]) / dt / fh
            if (math.hypot(vx, vy) >= P["item_speed_frame"] or -vy >= P["item_up_frame"]) and \
                    o.t - item_fire.get(it["raw_id"], -1e9) >= P["cooldown_s"]:
                item_fire[it["raw_id"]] = o.t
                roi = list(it["box"])
                best = min(o.persons, key=lambda p: math.hypot((p["box"][0] + p["box"][2]) / 2 - c[0],
                                                               (p["box"][1] + p["box"][3]) / 2 - c[1]), default=None)
                if best is not None:
                    bb = best["box"]
                    roi = [min(roi[0], bb[0]), min(roi[1], bb[1]), max(roi[2], bb[2]), max(roi[3], bb[3])]
                out.append(make_event(EVENT_TYPES["item"], o.t, roi, metric=round(math.hypot(vx, vy), 3)))
    return out


if __name__ == "__main__":
    import pandas as pd
    from exp_harness import baseline, combine, score
    pd.set_option("display.width", 250)
    rows = [score(baseline, name="baseline"), score(combine(baseline, make_method(**DEFAULTS)), name="base+motion")]
    print(pd.DataFrame(rows).set_index("method").T)
