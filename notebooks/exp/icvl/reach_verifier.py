"""Reach verifier: can a small keypoint model tell a littering drop from an ordinary walker at the moment reach_down
fires? reach_down's ICVL false alerts are mostly people walking with a hanging arm (a non-litterer gets an alert in a
random 7.6 s window 33 % of the time, chance.py), and its threshold cannot be tightened without losing drops.

Features: the pose_action summary of the person's last 1 s (stage1_pose_action.window_features, 205 numbers).
Training (ICVL ODD videos only): every 1 s pose window of a labelled person (TrackWindows); positive = a litterer
with the window ending in [t0, t1 + 1.5 s] of their littering incident, negative = a person not within 5 s of any
littering of theirs; plus the windows at every reach_down alert (same labels, weight 3: the decision being made).
5-seed MLP ensemble (exp/actions/pose_seq.TorchModel). The threshold is chosen on out-of-fold scores (5 folds grouped
by video): the highest one that keeps >= KEEP of the littering incidents reach_down caught.
At run time reach_down alerts with verifier score < threshold are dropped (other rules untouched).

Usage (from notebooks/):
  ../.venv/bin/python exp/icvl/reach_verifier.py train      -> weights/reach_verifier_icvl.npz + exp/icvl/reach_verifier_train.json
  ../.venv/bin/python exp/icvl/reach_verifier.py even       ICVL even half (never used for training)
  ../.venv/bin/python exp/icvl/reach_verifier.py tune       original tune / train sets (sweep.py metrics)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(HERE), str(NB / "exp" / "actions")]

from pose_seq import TorchModel  # noqa: E402
from score import (DEFAULT_RULES, WINDOW_S, GateConfig, box_hit, compact, labels_for, load_runs, on_track,  # noqa: E402
                   person_box, replay_rules, score)
from stage1_gate import intersection_over  # noqa: E402
from stage1_pose_action import TrackWindows, window_features  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
KEEP = (0.97, 0.95, 0.90)
OUT = NB / "weights" / "reach_verifier_icvl.npz"
CFG = dict(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",))


AHEAD_S = 0.0  # > 0: also use the person's pose over the AHEAD_S seconds after the alert (the gate holds the alert)


def track_window(res, rid, t, frame_hw):
    obs = [o for o in res.observations if t - 2.0 <= o.t <= t + 1e-6]
    hist = [(x.t, q["box"], q.get("kpts")) for x in obs for q in x.persons if q["raw_id"] == rid]
    if not hist:
        return None
    heights = [q["box"][3] - q["box"][1] for x in res.observations if x.t <= t for q in x.persons if q["raw_id"] == rid]
    return window_features(hist, float(np.median(heights[-200:])), t, 1.0, 8, frame_hw)[1]


def with_ahead(now, later):
    """[window at t, window at t + AHEAD_S (zeros if the track is gone), 1 if it was found]."""
    if later is None:
        return np.concatenate([now, np.zeros_like(now), [0.0]]).astype(np.float32)
    return np.concatenate([now, later, [1.0]]).astype(np.float32)


def alert_window(res, alert):
    """The pose summary of the alerted person's last 1 s at the alert time (what the gate could compute online),
    plus the second after it when AHEAD_S > 0."""
    t = alert["timestamp_s"]
    obs = [o for o in res.observations if t - 2.0 <= o.t <= t + 1e-6]
    if not obs:
        return None
    o = obs[-1]
    p = max(o.persons, key=lambda q: intersection_over(q["box"], alert["roi_box"]), default=None)
    if p is None:
        return None
    now = track_window(res, p["raw_id"], t, o.frame_hw)
    if not AHEAD_S:
        return now
    return with_ahead(now, track_window(res, p["raw_id"], t + AHEAD_S, o.frame_hw)
                      if any(x.t >= t + AHEAD_S - 0.25 for x in res.observations) else None)


def label_at(t, box, episodes, people):
    """1 = litterer at their drop, 0 = labelled person not near any littering of theirs, None = unclear / unlabelled."""
    lit = defaultdict(list)
    for ep in episodes.get("littering", []):
        lit[ep["pid"]].append(ep)
    for pid, tr in people.items():
        if not on_track({"timestamp_s": t, "roi_box": box}, tr):
            continue
        eps = lit.get(pid, [])
        if any(ep["t0"] <= t <= ep["t1"] + 1.5 for ep in eps):
            return 1
        if all(t < ep["t0"] - 5 or t > ep["t1"] + 5 for ep in eps):
            return 0
        return None
    return None


def build(runs):
    X, y, w, groups = [], [], [], []
    cfg = GateConfig(**CFG)
    for vi, (g, res) in enumerate(runs):
        episodes, people = labels_for(g.tag, g.seq.native_fps)
        tw = TrackWindows()
        for o in res.observations:
            for win in tw.step(o):
                lab = label_at(win["t"], win["box"], episodes, people)
                if lab is not None:
                    X.append(win["sum"]), y.append(lab), w.append(1.0), groups.append(vi)
        for a in replay_rules(res, cfg)[0]:
            if a["event_type"] != "reach_down":
                continue
            lab = label_at(a["timestamp_s"], a["roi_box"], episodes, people)
            f = alert_window(res, a)
            if lab is not None and f is not None:
                X.append(f), y.append(lab), w.append(3.0), groups.append(vi)
    return np.array(X), np.array(y, np.float32), np.array(w, np.float32), np.array(groups)


class Verifier:
    def __init__(self, models, thr):
        self.models, self.thr = models, thr

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def fit(X, y, w):
    # balance the classes (walkers outnumber drops)
    wb = w * np.where(y == 1, (y == 0).sum() / max(1, (y == 1).sum()), 1.0)
    return [TorchModel("mlp", seed=s).fit(X, y, wb) for s in SEEDS]


def make_filter(predict, thr):
    def filt(res, alerts):
        out = []
        for a in alerts:
            if a["event_type"] == "reach_down":
                f = alert_window(res, a)
                if f is not None and predict(f[None])[0] < thr:
                    continue
            out.append(a)
        return out
    return filt


def caught_littering(runs, alerts_by_seq):
    n = c = 0
    for g, res in runs:
        episodes, people = labels_for(g.tag, g.seq.native_fps)
        al = alerts_by_seq[res.seq_id]
        for ep in episodes.get("littering", []):
            n += 1
            c += any(on_track(a, people[ep["pid"]], ep["t0"], ep["t1"], WINDOW_S["littering"]) for a in al)
    return c, n


def train():
    runs = load_runs("odd")
    X, y, w, groups = build(runs)
    print(f"{len(y)} windows, {int(y.sum())} positive", flush=True)
    # out-of-fold scores for every reach_down alert, folds grouped by video
    folds = np.array([vi % 5 for vi in range(len(runs))])
    cfg = GateConfig(**CFG)
    alerts = {res.seq_id: replay_rules(res, cfg)[0] for _, res in runs}
    oof = {}
    for k in range(5):
        tr = folds[groups] != k
        models = fit(X[tr], y[tr], w[tr])
        pred = lambda F: np.mean([m.predict(F) for m in models], axis=0)  # noqa: E731
        for vi, (g, res) in enumerate(runs):
            if folds[vi] != k:
                continue
            for j, a in enumerate(alerts[res.seq_id]):
                if a["event_type"] == "reach_down":
                    f = alert_window(res, a)
                    oof[(res.seq_id, j)] = 1.0 if f is None else float(pred(f[None])[0])
        print(f"fold {k} done", flush=True)
    base_c, n = caught_littering(runs, alerts)
    rows = []
    for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
        kept = {sid: [a for j, a in enumerate(al) if a["event_type"] != "reach_down" or oof[(sid, j)] >= thr]
                for sid, al in alerts.items()}
        c, _ = caught_littering(runs, kept)
        rows.append({"thr": float(thr), "littering_caught": c, "of": n, "base_caught": base_c,
                     "reach_alerts": sum(a["event_type"] == "reach_down" for al in kept.values() for a in al),
                     "all_alerts": sum(len(al) for al in kept.values())})
        print(json.dumps(rows[-1]), flush=True)
    choice = {}
    for keep in KEEP:
        ok = [r for r in rows if r["littering_caught"] >= keep * base_c]
        choice[str(keep)] = max(ok, key=lambda r: r["thr"])["thr"] if ok else 0.0
    models = fit(X, y, w)
    np.savez(OUT, mu=np.stack([m.mu for m in models]), sd=np.stack([m.sd for m in models]),
             W1=np.stack([m.net[0].weight.detach().numpy().T for m in models]),
             b1=np.stack([m.net[0].bias.detach().numpy() for m in models]),
             W2=np.stack([m.net[3].weight.detach().numpy()[0] for m in models]),
             b2=np.array([float(m.net[3].bias.detach().numpy()[0]) for m in models]),
             threshold=choice[str(KEEP[1])])
    (HERE / "reach_verifier_train.json").write_text(json.dumps({"windows": len(y), "positives": int(y.sum()),
                                                                 "oof_sweep": rows, "thresholds": choice}, indent=1))
    print("thresholds", choice, "saved", OUT)


def load():
    from stage1_pose_action import PoseActionModel
    m = PoseActionModel(OUT)
    return m.predict, m.threshold


def even(thrs):
    predict, _ = load()
    runs = load_runs("even")
    cfg = GateConfig(**CFG)
    rows = [compact(score(runs, cfg), {})]
    print(json.dumps(rows[-1]), flush=True)
    for thr in thrs:
        rows.append(compact(score(runs, cfg, make_filter(predict, thr)), {"reach_verifier_thr": thr}))
        print(json.dumps(rows[-1]), flush=True)
    (HERE / "reach_verifier_even.json").write_text(json.dumps(rows, indent=1))


def tune(thrs):
    sys.path[:0] = [str(NB / "exp" / "alerts"), str(NB / "exp" / "newcam"), str(NB / "exp"), str(NB / "tools"),
                    str(NB / "exp" / "gt_review")]
    import sweep as S
    from exp_harness import baseline
    predict, _ = load()
    data = S.data()
    rows = []
    for thr in [None, *thrs]:
        filt = (lambda r, al: al) if thr is None else make_filter(predict, thr)
        orig = S.partial

        def patched(fn, cfg, _f=filt):
            return lambda r, ds: _f(r, fn(r, ds, cfg=cfg))
        S.partial = patched
        try:
            rows.append(S.score({}, *data[:4]))
        finally:
            S.partial = orig
        rows[-1]["setting"] = {"reach_verifier_thr": thr}
        print(json.dumps(rows[-1]), flush=True)
    (HERE / "reach_verifier_tune.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    thrs = [float(x) for x in sys.argv[2:]]
    {"train": lambda: train(), "even": lambda: even(thrs), "tune": lambda: tune(thrs)}[cmd]()
