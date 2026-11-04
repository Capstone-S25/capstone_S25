"""Reach verifier trained on ALL tune data, not only ICVL (the ICVL-only one, reach_verifier.py, cut ICVL walker
alerts but lost littering clips 96.5 -> 84.3 right person: it learned ICVL's "let go while walking", not bending).

Same features (pose_action summary of the person's last 1 s) and MLP ensemble. Rows, tune / train data only:
  icvl    ICVL odd videos: litterer at their drop = 1, labelled people not near their own littering = 0
  litter  littering train clips: person on a labelled littering box = 1, other people = 0
  iwdd    IWDD train: labelled dumper (iwdd_actors_train.json) in [onset - 1, onset + 5] = 1, other people there = 0,
          every person in normal videos = 0
  avenue  Avenue train normal videos (tune half): everyone = 0
Every 1 s pose window plus the window at every reach_down alert (weight 3). Each source's positives and negatives get
equal total weight. 5 folds grouped by sequence give out-of-fold scores for every reach_down alert on the tune data;
the threshold is chosen on those (sweep.py metrics + ICVL odd), then the final model (all rows) is scored ONCE on
held-out data.
Usage (from notebooks/):
  ../.venv/bin/python exp/icvl/reach_verifier_mixed.py oof      -> exp/icvl/reach_verifier_mixed_oof.json (+ scores pkl)
  ../.venv/bin/python exp/icvl/reach_verifier_mixed.py export   -> weights/reach_verifier.npz
"""
import json
import pickle
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(HERE), str(NB / "exp" / "actions"), str(NB / "exp" / "alerts"), str(NB / "exp" / "newcam"),
                str(NB / "exp"), str(NB / "tools"), str(NB / "exp" / "gt_review")]

import sweep as S  # noqa: E402
from cached import load_cached  # noqa: E402
from exp_harness import baseline  # noqa: E402
from loaders import iwdd_labels  # noqa: E402
from pose_seq import TorchModel  # noqa: E402
from reach_verifier import alert_window, label_at  # noqa: E402
from score import compact, labels_for, load_runs, score  # noqa: E402
from stage1_bench import box_hit  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, replay_rules  # noqa: E402
from stage1_pose_action import TrackWindows  # noqa: E402

CFG = GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",))
VERIFIED = ("reach_down", "pose_action")
SEEDS = (0, 1, 2, 3, 4)
import os  # noqa: E402

import reach_verifier as RV  # noqa: E402

RV.AHEAD_S = float(os.environ.get("AHEAD_S", "0"))
TAG = (f"_ahead{RV.AHEAD_S:g}" if RV.AHEAD_S else "") + os.environ.get("TAG", "")
SCORES = HERE / f"reach_verifier_mixed_oof{TAG}.pkl"
OUT = NB / "weights" / f"reach_verifier{TAG}.npz"
ACTORS = json.loads((NB / "exp" / "newcam" / "iwdd_actors_train.json").read_text())["actors"]


def gt_label(g, t, box, idx_t):
    """littering clips: 1 if box is on a labelled box at the nearest labelled frame within 1 s, else 0."""
    near = [(abs(tt - t), i) for i, tt in idx_t.items() if i in g.gt and abs(tt - t) <= 1.0]
    if not near:
        return 0
    i = min(near)[1]
    return int(any(box_hit(box, b) for b in g.gt[i]))


def labeller(source, g, r, extra):
    if source == "icvl":
        episodes, people = extra
        return lambda t, box, rid: label_at(t, box, episodes, people)
    if source == "litter":
        idx_t = {o.index: o.t for o in r.observations}
        return lambda t, box, rid: gt_label(g, t, box, idx_t)
    if source == "iwdd":
        lab = extra[r.seq_id]
        if not lab["positive"]:
            return lambda t, box, rid: 0
        actors = ACTORS.get(r.seq_id.split("/")[-1])
        if actors is None:
            return lambda t, box, rid: None
        on = lab["onset_s"]
        return lambda t, box, rid: (int(rid in actors) if on - 1 <= t <= on + 5 else None)
    return lambda t, box, rid: 0


def sources():
    T = S.splits(heldout=False)
    labels = iwdd_labels("train")
    out = [("icvl", g, r, labels_for(g.tag, g.seq.native_fps)) for g, r in load_runs("odd")]
    out += [("litter", g, r, None) for g, r in T["littering"]]
    out += [("iwdd", g, r, labels) for g, r in load_cached("iwdd_train")]
    out += [("avenue", g, r, None) for g, r in T["normal"]]
    return out


def rid_at(r, alert):
    from stage1_gate import intersection_over
    o = min(r.observations, key=lambda o: abs(o.t - alert["timestamp_s"]))
    p = max(o.persons, key=lambda q: intersection_over(q["box"], alert["roi_box"]), default=None)
    return None if p is None else p["raw_id"]


def build():
    X, y, w, src, seq = [], [], [], [], []
    alerts = {}
    for k, (source, g, r, extra) in enumerate(sources()):
        lab = labeller(source, g, r, extra)
        tw = TrackWindows()
        wins = [win for o in r.observations for win in tw.step(o)]
        by_rid = defaultdict(list)
        for win in wins:
            by_rid[win["raw_id"]].append(win)
        for win in wins:
            v = lab(win["t"], win["box"], win["raw_id"])
            if v is None:
                continue
            f = win["sum"]
            if RV.AHEAD_S:
                later = [x for x in by_rid[win["raw_id"]] if abs(x["t"] - win["t"] - RV.AHEAD_S) <= 0.3]
                f = RV.with_ahead(f, later[0]["sum"] if later else None)
            X.append(f), y.append(v), w.append(1.0), src.append(source), seq.append(k)
        al = replay_rules(r, CFG)[0]
        alerts[k] = (source, r.seq_id, al)
        for a in al:
            if a["event_type"] != "reach_down":
                continue
            f = alert_window(r, a)
            v = lab(a["timestamp_s"], a["roi_box"], rid_at(r, a))
            if f is not None and v is not None:
                X.append(f), y.append(v), w.append(3.0), src.append(source), seq.append(k)
        if k % 200 == 0:
            print(k, "sequences", len(y), "rows", flush=True)
    X, y, w, src, seq = np.array(X), np.array(y, np.float32), np.array(w, np.float32), np.array(src), np.array(seq)
    # equal total weight per source and class
    for s in set(src):
        for c in (0, 1):
            m = (src == s) & (y == c)
            if m.any():
                w[m] *= 1000.0 / w[m].sum()
    return X, y, w, src, seq, alerts


def fit(X, y, w):
    return [TorchModel("mlp", seed=s).fit(X, y, w) for s in SEEDS]


def oof():
    srcs = sources()
    X, y, w, src, seq, alerts = build()
    print({s: (int(((src == s) & (y == 1)).sum()), int(((src == s) & (y == 0)).sum())) for s in set(src)}, flush=True)
    scores = {}
    for fold in range(5):
        tr = seq % 5 != fold
        models = fit(X[tr], y[tr], w[tr])
        for k, (source, sid, al) in alerts.items():
            if k % 5 != fold:
                continue
            r = srcs[k][2]
            for a in al:
                if a["event_type"] in VERIFIED:
                    f = alert_window(r, a)
                    p = 1.0 if f is None else float(np.mean([m.predict(f[None])[0] for m in models]))
                    scores[(sid, round(a["timestamp_s"], 3), a["event_type"])] = p
        print("fold", fold, "done", flush=True)
    SCORES.write_bytes(pickle.dumps(scores))
    sweep_thresholds(scores)


def lookup_filter(scores, thr, pose_thr=None):
    """Drop reach_down alerts scoring < thr (and pose_action alerts < pose_thr) by their out-of-fold verifier score."""
    def ok(r, a):
        t = {"reach_down": thr, "pose_action": pose_thr}.get(a["event_type"])
        if t is None:
            return True
        key = (r.seq_id, round(a["timestamp_s"], 3))
        return scores.get(key + (a["event_type"],), scores.get(key, 1.0)) >= t

    def filt(r, al):
        return [a for a in al if ok(r, a)]
    return filt


def sweep_thresholds(scores=None, thrs=(0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5), pose_thrs=(None,)):
    scores = scores or pickle.loads(SCORES.read_bytes())
    data = S.data()
    icvl = load_runs("odd")
    rows = []
    for thr, pose_thr in [(a, b) for a in thrs for b in pose_thrs]:
        filt = lookup_filter(scores, thr, pose_thr)
        orig = S.partial
        S.partial = lambda fn, cfg, _f=filt: (lambda r, ds: _f(r, fn(r, ds, cfg=cfg)))
        try:
            row = S.score({}, *data[:4])
        finally:
            S.partial = orig
        row["setting"] = {"reach_verifier_thr": thr, "pose_verifier_thr": pose_thr}
        row["icvl_odd"] = compact(score(icvl, CFG, filt), {})
        rows.append(row)
        print(json.dumps(row), flush=True)
    (HERE / f"reach_verifier_mixed_oof{TAG}.json").write_text(json.dumps(rows, indent=1))


def export(thr):
    X, y, w, src, seq, _ = build()
    models = fit(X, y, w)
    np.savez(OUT, mu=np.stack([m.mu for m in models]), sd=np.stack([m.sd for m in models]),
             W1=np.stack([m.net[0].weight.detach().numpy().T for m in models]),
             b1=np.stack([m.net[0].bias.detach().numpy() for m in models]),
             W2=np.stack([m.net[3].weight.detach().numpy()[0] for m in models]),
             b2=np.array([float(m.net[3].bias.detach().numpy()[0]) for m in models]), threshold=thr)
    print("saved", OUT, "threshold", thr)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "oof":
        oof()
    elif cmd == "sweep":
        sweep_thresholds(thrs=[float(x) for x in sys.argv[2:]])
    elif cmd == "sweep_pose":   # reach threshold, then pose thresholds
        sweep_thresholds(thrs=[float(sys.argv[2])], pose_thrs=[None] + [float(x) for x in sys.argv[3:]])
    else:
        export(float(sys.argv[2]))
