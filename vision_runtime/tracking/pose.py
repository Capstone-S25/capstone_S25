"""Inference-only Stage 1 helper. See docs/integration.md for source provenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "pose_action_mlp.npz"
KP_CONF = 0.3
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = (
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
)
ARMS = (L_ELB, R_ELB, L_WRI, R_WRI)
WIN = dict(period_s=1.0, win_s=1.0, min_hist_s=0.8, K=8)


class TrackWindows:
    """Online, causal window extractor: feed observations in time order; returns windows due at this frame."""

    def __init__(
        self,
        period_s=WIN["period_s"],
        win_s=WIN["win_s"],
        min_hist_s=WIN["min_hist_s"],
        K=WIN["K"],
    ):
        self.period_s, self.win_s, self.min_hist_s, self.K = (
            period_s,
            win_s,
            min_hist_s,
            K,
        )
        self.tracks: dict[int, dict] = {}

    def step(self, obs) -> list[dict]:
        t, out = (obs.t, [])
        for p in obs.persons:
            tr = self.tracks.setdefault(
                p["raw_id"],
                {"first": t, "last_call": -1000000000.0, "hist": [], "heights": []},
            )
            h = max(1.0, p["box"][3] - p["box"][1])
            tr["heights"].append(h)
            tr["hist"].append((t, p["box"], p.get("kpts")))
            while tr["hist"] and tr["hist"][0][0] < t - self.win_s - 0.5:
                tr["hist"].pop(0)
            if (
                t - tr["first"] >= self.min_hist_s - 1e-06
                and t - tr["last_call"] >= self.period_s - 1e-06
            ):
                tr["last_call"] = t
                seq, summ = window_features(
                    tr["hist"],
                    float(np.median(tr["heights"][-200:])),
                    t,
                    self.win_s,
                    self.K,
                    obs.frame_hw,
                )
                out.append(
                    {
                        "t": t,
                        "raw_id": p["raw_id"],
                        "box": list(p["box"]),
                        "index": obs.index,
                        "seq": seq,
                        "sum": summ,
                    }
                )
        for rid in [r for r, tr in self.tracks.items() if t - tr["hist"][-1][0] > 10.0]:
            del self.tracks[rid]
        return out


def _norm_kpts(k, box, H):
    """(17,3) -> (17,2) positions relative to hip centre / H, (17,) visibility mask."""
    x1, y1, x2, y2 = box
    if k is None:
        return (
            np.zeros((17, 2), np.float32),
            np.zeros(17, np.float32),
            np.array([(x1 + x2) / 2, (y1 + y2) / 2]),
        )
    vis = (k[:, 2] >= KP_CONF).astype(np.float32)
    if vis[L_HIP] and vis[R_HIP]:
        c = k[[L_HIP, R_HIP], :2].mean(0)
    elif vis[L_HIP] or vis[R_HIP]:
        c = k[L_HIP if vis[L_HIP] else R_HIP, :2]
    else:
        c = np.array([(x1 + x2) / 2, y1 + 0.55 * (y2 - y1)])
    xy = (k[:, :2] - c) / H * vis[:, None]
    return (xy.astype(np.float32), vis, c)


def window_features(hist, H, t, win_s, K, frame_hw):
    """hist: [(t, box, kpts|None)] of one track (time-ordered, all <= t). Returns (K, F) sequence and summary vector."""
    times = np.array([h[0] for h in hist])
    grid = t - win_s + win_s * (np.arange(K) + 1) / K
    rows, centres, boxes = ([], [], [])
    for g in grid:
        j = int(np.searchsorted(times, g + 1e-06, side="right")) - 1
        j = max(j, 0)
        jj = j
        while jj > 0 and hist[jj][2] is None and (times[j] - times[jj - 1] < 0.35):
            jj -= 1
        if hist[jj][2] is not None:
            j = jj
        _, box, k = hist[j]
        xy, vis, c = _norm_kpts(k, box, H)
        rows.append((xy, vis))
        centres.append(c)
        boxes.append(box)
    dt = win_s / K
    xy = np.stack([r[0] for r in rows])
    vis = np.stack([r[1] for r in rows])
    vel = np.zeros_like(xy)
    both = vis[1:] * vis[:-1]
    vel[1:] = (xy[1:] - xy[:-1]) / dt * both[..., None]
    cen = np.array(centres) / H
    cvel = np.zeros((K, 2), np.float32)
    cvel[1:] = (cen[1:] - cen[:-1]) / dt
    bh = np.array([max(1.0, b[3] - b[1]) for b in boxes]) / H
    asp = np.array([(b[2] - b[0]) / max(1.0, b[3] - b[1]) for b in boxes])
    fh, fw = frame_hw
    border = np.array(
        [
            float(
                b[0] <= 0.01 * fw
                or b[1] <= 0.01 * fh
                or b[2] >= 0.99 * fw
                or (b[3] >= 0.99 * fh)
            )
            for b in boxes
        ]
    )
    seq = np.concatenate(
        [
            xy.reshape(K, -1),
            vis,
            np.clip(vel, -8, 8).reshape(K, -1),
            np.clip(cvel, -8, 8),
            bh[:, None],
            asp[:, None],
            border[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    m = vis[..., None]
    cnt = np.maximum(vis.sum(0), 1)[:, None]
    mean = (xy * m).sum(0) / cnt
    big = 1000.0
    mn = np.where(m > 0, xy, big).min(0)
    mx = np.where(m > 0, xy, -big).max(0)
    seen = vis.sum(0) > 0
    mn[~seen], mx[~seen] = (0, 0)
    rng = mx - mn
    spd = np.linalg.norm(vel, axis=2)
    sho = xy[:, [L_SHO, R_SHO]]
    wri = xy[:, [L_WRI, R_WRI]]
    wv = vis[:, [L_WRI, R_WRI]] * vis[:, [L_SHO, R_SHO]]
    ext = np.linalg.norm(wri - sho, axis=2) * wv
    knee_y = np.where(
        vis[:, [L_KNE, R_KNE]].sum(1) > 0,
        (xy[:, [L_KNE, R_KNE], 1] * vis[:, [L_KNE, R_KNE]]).sum(1)
        / np.maximum(vis[:, [L_KNE, R_KNE]].sum(1), 1),
        0,
    )
    wrist_low = (wri[..., 1] * vis[:, [L_WRI, R_WRI]]).max(1) - knee_y
    torso = np.linalg.norm(sho.mean(1), axis=1) * vis[:, [L_SHO, R_SHO]].min(1)
    body = [
        bh.min(),
        bh.max(),
        bh[-1],
        asp.min(),
        asp.max(),
        asp[-1],
        border.max(),
        np.linalg.norm(cvel, axis=1).mean(),
        np.linalg.norm(cvel, axis=1).max(),
        cvel[:, 0].mean(),
        cvel[:, 1].mean(),
        ext.max(),
        ext.mean(),
        wrist_low.max(),
        torso.min(),
        torso.max(),
        vis.mean(),
        vis[:, ARMS].mean(),
    ]
    summ = np.concatenate(
        [
            mean.ravel(),
            mn.ravel(),
            mx.ravel(),
            rng.ravel(),
            spd.max(0),
            spd.mean(0),
            vis.mean(0),
            np.array(body),
        ]
    ).astype(np.float32)
    return (seq, np.nan_to_num(summ, nan=0.0, posinf=0.0, neginf=0.0))


class PoseActionModel:
    """Averaged probability of the ensemble members: relu(z @ W1 + b1) @ w2 + b2 -> sigmoid, z standardised."""

    def __init__(self, path=DEFAULT_WEIGHTS):
        d = np.load(path)
        self.mu, self.sd = (d["mu"], d["sd"])
        self.W1, self.b1, self.W2, self.b2 = (d["W1"], d["b1"], d["W2"], d["b2"])
        self.threshold = float(d["threshold"])

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, np.float64))
        out = np.zeros(len(X))
        for i in range(len(self.W1)):
            z = (X - self.mu[i]) / self.sd[i]
            logit = (
                np.maximum(z @ self.W1[i] + self.b1[i], 0.0) @ self.W2[i] + self.b2[i]
            )
            out += 1.0 / (1.0 + np.exp(-logit))
        return out / len(self.W1)
