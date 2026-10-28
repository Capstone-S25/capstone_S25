"""One-class "unusual pose or motion" candidate rule for Stage 1.

Fitted ONLY on SPLITS["normal"] (tune half = odd-numbered Avenue training videos, no anomalies).

Features per person detection with pose keypoints (pure numpy, causal):
    pose : 13 joints (nose, shoulders, elbows, wrists, hips, knees, ankles) relative to the
           shoulder/hip midpoint, divided by torso length (floored at 0.15 x box height; box-based
           fallback when only shoulders or only hips are visible). Resolution-, position- and
           size-invariant, so it can transfer from 640x360 Avenue to 1080p littering clips.
    vel  : change of those normalised joint positions per second against the same tracker id's
           observation closest to `max_dt` seconds ago (only past frames; not available when the
           frame gap is long, e.g. 1 fps smoking CCTV, so smoking uses pose only).
Missing joints (conf < 0.3) are masked: distances use only the query's visible dims, and the bank's
missing dims are mean-filled.

Score: per "channel" (a subset of dims; the recommended set is "body_y" = vertical coordinates of
all 13 joints, which are unchanged by the person's yaw, and "legs" = x/y of hips, knees, ankles) the
RMS distance to the k nearest normal vectors (bank of normal people plus their mirror images),
divided by that channel's `cal_q` quantile on leave-one-video-out normal scores. The person's score
is the max over enabled channels, so 1.0 = "as unusual as the top (1-cal_q) of normal people".
A candidate fires when the score >= thr on >= min_hits observations of that person within
window_s (a single hit is enough when the previous observation of that person is > sparse_dt
old, i.e. low-fps sources), then that stitched person id is quiet for cooldown_s.

Honest normal FA: for dataset "normal" every video is scored with a bank (and calibration) fitted
WITHOUT that video (leave-one-video-out). Other datasets use the bank fitted on all 8 normal
tune videos. Calibration quantiles for the full bank are the pooled LOO quantiles.

Per-frame cost: per channel one (n_persons x 52) @ (52 x bank) product, bank = 4000 -> ~1 MFLOP per
person per channel; < 1 ms per frame on CPU for typical crowds.
"""
from __future__ import annotations

import sys
from collections import deque

import numpy as np

NB = "/Users/ianvarella/capstone_S25/notebooks"
if NB not in sys.path:
    sys.path.insert(0, NB)

from exp_harness import SPLITS, make_event  # noqa: E402
from stage1_gate import GateConfig, TrackStitcher  # noqa: E402

KP_CONF = 0.3
JOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]          # nose + body (COCO order)
_FLIP = {0: 0, 5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13, 15: 16, 16: 15}
FLIP_IDX = [JOINTS.index(_FLIP[j]) for j in JOINTS]
NJ = len(JOINTS)
D = 4 * NJ


def _dims(joints, vel=False, axes=(0, 1)):
    out = []
    for j in joints:
        i = JOINTS.index(j)
        base = 2 * NJ if vel else 0
        out += [base + 2 * i + a for a in axes]
    return np.array(out)


CHANNELS = {
    "body": _dims(JOINTS),
    "arms": _dims([5, 6, 7, 8, 9, 10]),
    "legs": _dims([11, 12, 13, 14, 15, 16]),
    "armvel": _dims([7, 8, 9, 10], vel=True),
    "bodyvel": _dims(JOINTS, vel=True),
    # vertical-only channels: invariant to the person's yaw (walking across vs facing the camera)
    "body_y": _dims(JOINTS, axes=(1,)),
    "arms_y": _dims([5, 6, 7, 8, 9, 10], axes=(1,)),
    "legs_y": _dims([11, 12, 13, 14, 15, 16], axes=(1,)),
    "armvel_y": _dims([7, 8, 9, 10], vel=True, axes=(1,)),
}


# --------------------------------------------------------------------------- features

def pose_vec(k, box):
    vis = k[:, 2] >= KP_CONF
    sh = [k[i, :2] for i in (5, 6) if vis[i]]
    hp = [k[i, :2] for i in (11, 12) if vis[i]]
    bh = max(1.0, box[3] - box[1])
    if sh and hp:
        s, h = np.mean(sh, 0), np.mean(hp, 0)
        scale = max(float(np.hypot(*(s - h))), 0.15 * bh)
        origin = 0.5 * (s + h)
    elif sh or hp:
        origin = np.mean(sh or hp, 0)
        scale = 0.3 * bh
    else:
        return None
    sel = k[JOINTS]
    return (sel[:, :2] - origin) / scale, sel[:, 2] >= KP_CONF


def flip(xy, vis):
    xy2 = xy[FLIP_IDX].copy()
    xy2[:, 0] *= -1
    return xy2, vis[FLIP_IDX]


class FeatureTracker:
    """Causal per-raw-tracker-id history for velocities."""

    def __init__(self, max_dt: float = 0.3, vel_clip: float = 8.0):
        self.prev = {}
        self.max_dt = max_dt
        self.vel_clip = vel_clip

    def feats(self, p, t):
        k = p.get("kpts")
        if k is None:
            return None
        pv = pose_vec(k, p["box"])
        if pv is None:
            return None
        xy, vis = pv
        vel = np.zeros_like(xy)
        vvis = np.zeros(NJ, bool)
        hist = self.prev.setdefault(p["raw_id"], deque())
        while hist and t - hist[0][0] > 1.5 * self.max_dt:
            hist.popleft()
        cands = [h for h in hist if 0.4 * self.max_dt <= t - h[0] <= 1.5 * self.max_dt]
        if cands:
            h = min(cands, key=lambda h: abs(t - h[0] - self.max_dt))
            vel = np.clip((xy - h[1]) / (t - h[0]), -self.vel_clip, self.vel_clip)
            vvis = vis & h[2]
        hist.append((t, xy.copy(), vis.copy()))
        if len(self.prev) > 256:                      # forget long-gone ids
            for rid in [r for r, h in self.prev.items() if not h or t - h[-1][0] > 5.0]:
                del self.prev[rid]
        return xy, vis, vel, vvis


def pack(xy, vis, vel, vvis, w_vel=1.0):
    v = np.concatenate([xy.ravel(), w_vel * vel.ravel()]).astype(np.float32)
    m = np.concatenate([np.repeat(vis, 2), np.repeat(vvis, 2)]).astype(np.float32)
    return v, m


# --------------------------------------------------------------------------- normal model

def collect(result, max_dt):
    ft = FeatureTracker(max_dt)
    rows = []
    for o in result.observations:
        for p in o.persons:
            f = ft.feats(p, o.t)
            if f is not None:
                rows.append(f)
    return rows


class NormalBank:
    def __init__(self, rows, bank_size=4000, min_vis=8, seed=0, aug_x=(1.0,)):
        rows = [f for f in rows if f[1].sum() >= min_vis]
        rng = np.random.default_rng(seed)
        if len(rows) > bank_size // 2:
            rows = [rows[i] for i in rng.choice(len(rows), bank_size // 2, replace=False)]
        V, M = [], []
        for xy, vis, vel, vvis in rows:
            for a in aug_x:
                # aug_x widens the pose horizontally (crude stand-in for other viewpoints)
                sc = np.array([a, 1.0])
                for f in ((xy * sc, vis, vel * sc, vvis), (*flip(xy * sc, vis), *flip(vel * sc, vvis))):
                    v, m = pack(*f)
                    V.append(v)
                    M.append(m)
        V, M = np.array(V), np.array(M)
        mean = (V * M).sum(0) / np.maximum(M.sum(0), 1)
        self.B = np.where(M > 0, V, mean).astype(np.float32)
        self.cal = {}

    def raw_scores(self, Q, Mq, k, channels):
        """dict channel -> (n,) RMS kNN distance over that channel's visible dims (nan if < 4 dims)."""
        out = {}
        for c in channels:
            d = CHANNELS[c]
            q, mq, b = Q[:, d], Mq[:, d], self.B[:, d]
            nvis = mq.sum(1)
            d2 = (mq * q ** 2).sum(1)[:, None] - 2 * (mq * q) @ b.T + mq @ (b ** 2).T
            d2 = np.maximum(d2, 0) / np.maximum(nvis, 1)[:, None]
            kk = min(k, d2.shape[1])
            s = np.sqrt(np.partition(d2, kk - 1, axis=1)[:, :kk].mean(1))
            s[nvis < 4] = np.nan
            out[c] = s
        return out


_MODELS = {}
_SCORES = {}
_CIDS = {}


def _vectors(rows, min_vis):
    rows = [f for f in rows if f[1].sum() >= min_vis]
    if not rows:
        return np.zeros((0, D), np.float32), np.zeros((0, D), np.float32)
    V, M = zip(*(pack(*f) for f in rows))
    return np.array(V), np.array(M)


def get_models(bank_size, min_vis, max_dt, k, cal_q, aug_x=(1.0,)):
    """Fit on SPLITS['normal'] (tune half) only: full bank + leave-one-video-out banks, each with
    per-channel calibration quantiles from LOO scores."""
    key = (bank_size, min_vis, max_dt, k, cal_q, tuple(aug_x))
    if key in _MODELS:
        return _MODELS[key]
    per_video = {r.seq_id: collect(r, max_dt) for g, r in SPLITS["normal"]}
    loo, pooled = {}, {c: [] for c in CHANNELS}
    for vid in per_video:
        bank = NormalBank([f for v, rows in per_video.items() if v != vid for f in rows], bank_size // len(aug_x), min_vis, aug_x=aug_x)
        loo[vid] = bank
        Q, Mq = _vectors(per_video[vid], min_vis)
        for c, s in bank.raw_scores(Q, Mq, k, CHANNELS).items():
            pooled[c].append(s[~np.isnan(s)])
    # LOO calibration of each LOO bank uses the other videos' held-out scores (no self-scoring)
    for vid, bank in loo.items():
        bank.cal = {c: float(np.quantile(np.concatenate([s for v, s in zip(per_video, pooled[c]) if v != vid]), cal_q))
                    for c in CHANNELS}
    full = NormalBank([f for rows in per_video.values() for f in rows], bank_size // len(aug_x), min_vis, aug_x=aug_x)
    full.cal = {c: float(np.quantile(np.concatenate(pooled[c]), cal_q)) for c in CHANNELS}
    _MODELS[key] = (full, loo)
    return full, loo


def person_scores(result, bank, p):
    """Causal per-observation channel scores (normalised): list of [dict or None] per observation."""
    ft = FeatureTracker(p["max_dt"])
    out = []
    for o in result.observations:
        vecs, masks, idx = [], [], []
        for i, per in enumerate(o.persons):
            f = ft.feats(per, o.t)
            if f is None or f[1].sum() < p["min_vis"]:
                continue
            v, m = pack(*f)
            vecs.append(v)
            masks.append(m)
            idx.append(i)
        sc = [None] * len(o.persons)
        if vecs:
            raw = bank.raw_scores(np.array(vecs), np.array(masks), p["k"], p["channels"])
            for j, i in enumerate(idx):
                sc[i] = {c: float(raw[c][j] / bank.cal[c]) for c in p["channels"]}
        out.append(sc)
    return out


# --------------------------------------------------------------------------- method

# Recommended operating point "B" (about +5.5 normal false alarms/min over the current rules when
# combined with them). Lighter: OPERATING_POINTS["A"]; heavier: OPERATING_POINTS["C"].
DEFAULTS = dict(channels=("body_y", "legs"), thr=1.0, ch_thr={"body_y": 0.8, "legs": 1.0}, k=5, bank_size=4000,
                min_vis=8, max_dt=0.3, cal_q=0.99, aug_x=(1.0,), window_s=1.0, min_hits=5, sparse_dt=0.5,
                cooldown_s=15.0, roi_grow=0.25)

OPERATING_POINTS = {
    "A": {**DEFAULTS, "ch_thr": {"body_y": 1.0, "legs": 1.5}, "min_hits": 3},
    "B": dict(DEFAULTS),
    "C": {**DEFAULTS, "ch_thr": {"body_y": 0.6, "legs": 1.25}, "min_hits": 4},
}


def make_method(**params):
    p = {**DEFAULTS, **params}
    fkey = (p["bank_size"], p["min_vis"], p["max_dt"], p["k"], p["cal_q"], tuple(p["aug_x"]))
    full, loo = get_models(*fkey)
    ch_thr = {c: (p["ch_thr"] or {}).get(c, p["thr"]) for c in p["channels"]}

    def scores_for(result, dataset):
        # memo for offline sweeps; live use computes the same scores frame by frame
        bank = loo.get(result.seq_id, full) if dataset == "normal" else full
        key = (fkey, tuple(p["channels"]), id(result), dataset)
        if key not in _SCORES:
            _SCORES[key] = person_scores(result, bank, p)
        return _SCORES[key]

    def cids_for(result):
        if id(result) not in _CIDS:
            st = TrackStitcher(GateConfig())
            _CIDS[id(result)] = [st.assign(o.persons, o.t) for o in result.observations]
        return _CIDS[id(result)]

    def person_score(sc):
        if sc is None:
            return None, None
        best, bc = -1.0, None
        for c in p["channels"]:
            v = sc[c]
            if v == v and v / ch_thr[c] > best:
                best, bc = v / ch_thr[c], c
        return (best, bc) if bc else (None, None)

    def method(result, dataset):
        hits, last, seen = {}, {}, {}
        events = []
        for o, sc_list, cids in zip(result.observations, scores_for(result, dataset), cids_for(result)):
            for cid, per, sc in zip(cids, o.persons, sc_list):
                s, ch = person_score(sc)
                prev_t = seen.get(cid, -1e9)
                seen[cid] = o.t
                dq = hits.setdefault(cid, deque())
                hot = s is not None and s >= 1.0
                if hot:
                    dq.append(o.t)
                while dq and dq[0] < o.t - p["window_s"]:
                    dq.popleft()
                need = 1 if o.t - prev_t > p["sparse_dt"] else p["min_hits"]
                if hot and len(dq) >= need and o.t - last.get(cid, -1e9) >= p["cooldown_s"]:
                    last[cid] = o.t
                    b = per["box"]
                    g = p["roi_grow"] * (b[3] - b[1])
                    roi = [b[0] - g, b[1] - g, b[2] + g, b[3] + g]
                    events.append(make_event("unusual_pose", o.t, roi, score=round(s, 3), channel=ch))
                    dq.clear()
        return events

    method.params = p
    method.scores_for = scores_for
    return method
