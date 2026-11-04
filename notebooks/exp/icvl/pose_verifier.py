"""Pose-cue verifier: a second look at pose_action alerts, trained on what the pose cue never saw.

The pose cue (stage1_pose_action) was trained on littering clips vs Avenue walkway footage only: no smokers and no one
on a phone, yet it is our smoking detector and phone calls / texting are a fifth of ICVL's false alerts. This
verifier uses the same features (the person's last 1 s of body points, 205 numbers) and the same 5-seed MLP as the
reach verifier, trained on tune / train data only:
  icvl     ICVL odd videos: smoking (gesture = smoking at that frame) or littering ([t0, t1 + 1.5 s]) = 1; a labelled
           person not within 5 s of any incident of theirs = 0, weight x2 when ICVL says they are phoning / texting
  litter   littering train clips: on a labelled littering box = 1, other people = 0
  smoking  smoking tune clips: on a labelled smoker box = 1, other people = 0
  iwdd     IWDD train: labelled dumper around the onset = 1, other people = 0, normal videos = 0
  avenue   Avenue tune normal videos = 0
Every 1 s window plus the window at every pose_action alert (weight 3); each source's classes get equal total
weight. Five folds grouped by sequence give out-of-fold scores for every pose_action alert; thresholds are chosen on
those (ICVL odd + sweep.py metrics), with the gate's crop check on and the verifier applied before the person dedup.
The final model is then scored once on ICVL even.
Usage (from notebooks/):
  ../.venv/bin/python exp/icvl/pose_verifier.py oof              -> exp/icvl/pose_verifier_oof.pkl + sweep
  ../.venv/bin/python exp/icvl/pose_verifier.py sweep THR ...    -> exp/icvl/pose_verifier_oof.json
  ../.venv/bin/python exp/icvl/pose_verifier.py export THR       -> weights/pose_verifier.npz
  ../.venv/bin/python exp/icvl/pose_verifier.py even THR ...     ICVL even (held-out), final model
"""
import json
import pickle
import sys
from bisect import bisect_left
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(HERE), str(NB / "exp" / "actions"), str(NB / "exp" / "alerts"), str(NB / "exp" / "newcam"),
                str(NB / "exp"), str(NB / "tools"), str(NB / "exp" / "gt_review")]

import stage1_crop_check  # noqa: E402
import sweep as S  # noqa: E402
from cached import CACHE, load_cached  # noqa: E402
from fpr_sweep import evaluate  # noqa: E402
from hand_probe import dedup  # noqa: E402
from inventory import ICVL, read_labels  # noqa: E402
from loaders import iwdd_labels  # noqa: E402
from reach_verifier import alert_window  # noqa: E402
from reach_verifier_mixed import ACTORS, fit, gt_label, rid_at  # noqa: E402
from score import labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, replay_rules  # noqa: E402
from stage1_pose_action import TrackWindows  # noqa: E402

# the gate as it will run: deck settings + crop check, verifier before the dedup (so replay with the dedup off)
BASE = dict(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",), pose_action_cooldown_s=15.0, crop_check=True)
CFG = GateConfig(**BASE, person_dedup_s=0.0)
SCORES = HERE / "pose_verifier_oof.pkl"
OUT = NB / "weights" / "pose_verifier.npz"


def load_crop_scores():
    for path in (q for q in CACHE.glob("crop_scores_*.pkl") if "_v1_" not in q.name):
        stage1_crop_check.SCORES.update(pickle.loads(path.read_bytes()))


def icvl_labeller(g, extra):
    episodes, people = extra
    fps = g.seq.native_fps
    rows = defaultdict(list)
    for r in read_labels(ICVL / g.tag):
        rows[r[6]].append((r[0] / fps, r[9]))
    for v in rows.values():
        v.sort()
    mine = defaultdict(list)
    for ev in ("smoking", "littering"):
        for ep in episodes.get(ev, []):
            mine[ep["pid"]].append((ev, ep))

    def lab(t, box, rid):
        for pid, tr in people.items():
            if not on_track({"timestamp_s": t, "roi_box": box}, tr):
                continue
            r = rows[pid]
            k = min(range(len(r)), key=lambda i: abs(r[i][0] - t)) if r else None
            gesture = r[k][1] if k is not None and abs(r[k][0] - t) <= 1.0 else None
            if gesture == 2 or any(ev == "littering" and ep["t0"] <= t <= ep["t1"] + 1.5 for ev, ep in mine[pid]):
                return 1, 1.0
            if all(t < ep["t0"] - 5 or t > ep["t1"] + 5 for _, ep in mine[pid]):
                return 0, 2.0 if gesture in (1, 3) else 1.0
            return None, 0.0
        return None, 0.0
    return lab


def labeller(source, g, r, extra):
    if source == "icvl":
        return icvl_labeller(g, extra)
    if source in ("litter", "smoking"):
        idx_t = {o.index: o.t for o in r.observations}
        return lambda t, box, rid: (gt_label(g, t, box, idx_t), 1.0)
    if source == "iwdd":
        lab = extra[r.seq_id]
        if not lab["positive"]:
            return lambda t, box, rid: (0, 1.0)
        actors = ACTORS.get(r.seq_id.split("/")[-1])
        if actors is None:
            return lambda t, box, rid: (None, 0.0)
        on = lab["onset_s"]
        return lambda t, box, rid: ((int(rid in actors), 1.0) if on - 1 <= t <= on + 5 else (None, 0.0))
    return lambda t, box, rid: (0, 1.0)


def sources():
    T = S.splits(heldout=False)
    labels = iwdd_labels("train")
    out = [("icvl", g, r, labels_for(g.tag, g.seq.native_fps)) for g, r in load_runs("odd")]
    out += [("litter", g, r, None) for g, r in T["littering"]]
    out += [("smoking", g, r, None) for g, r in T["smoking"]]
    out += [("iwdd", g, r, labels) for g, r in load_cached("iwdd_train")]
    out += [("avenue", g, r, None) for g, r in T["normal"]]
    return out


def build(srcs):
    X, y, w, src, seq, alerts = [], [], [], [], [], {}
    for k, (source, g, r, extra) in enumerate(srcs):
        lab = labeller(source, g, r, extra)
        tw = TrackWindows()
        for win in (win for o in r.observations for win in tw.step(o)):
            v, wt = lab(win["t"], win["box"], win["raw_id"])
            if v is not None:
                X.append(win["sum"]), y.append(v), w.append(wt), src.append(source), seq.append(k)
        al = replay_rules(r, CFG)[0]
        alerts[k] = (r.seq_id, al)
        for a in al:
            if a["event_type"] != "pose_action":
                continue
            f = alert_window(r, a)
            v, wt = lab(a["timestamp_s"], a["roi_box"], rid_at(r, a))
            if f is not None and v is not None:
                X.append(f), y.append(v), w.append(3.0 * wt), src.append(source), seq.append(k)
        if k % 200 == 0:
            print(k, "sequences", len(y), "rows", flush=True)
    X, y, w, src, seq = np.array(X), np.array(y, np.float32), np.array(w, np.float32), np.array(src), np.array(seq)
    for s in set(src):
        for c in (0, 1):
            m = (src == s) & (y == c)
            if m.any():
                w[m] *= 1000.0 / w[m].sum()
    return X, y, w, src, seq, alerts


def key(seq_id, a):
    return (seq_id, a["event_id"])


def oof():
    load_crop_scores()
    srcs = sources()
    X, y, w, src, seq, alerts = build(srcs)
    print({s: (int(((src == s) & (y == 1)).sum()), int(((src == s) & (y == 0)).sum())) for s in sorted(set(src))}, flush=True)
    scores = {}
    for fold in range(5):
        tr = seq % 5 != fold
        models = fit(X[tr], y[tr], w[tr])
        for k, (sid, al) in alerts.items():
            if k % 5 != fold:
                continue
            r = srcs[k][2]
            for a in al:
                if a["event_type"] == "pose_action":
                    f = alert_window(r, a)
                    scores[key(sid, a)] = 1.0 if f is None else float(np.mean([m.predict(f[None])[0] for m in models]))
        print("fold", fold, "done", flush=True)
    SCORES.write_bytes(pickle.dumps(scores))
    sweep(scores)


def make_filt(scores, thr):
    """Drop pose_action alerts scoring below thr, then the person dedup (alerts are replayed with it off)."""
    def filt(res, alerts):
        return dedup([a for a in alerts if a["event_type"] != "pose_action" or thr is None
                      or scores.get(key(res.seq_id, a), 1.0) >= thr])
    return filt


def sweep(scores=None, thrs=(None, 0.1, 0.2, 0.3, 0.4, 0.5)):
    load_crop_scores()
    scores = scores or pickle.loads(SCORES.read_bytes())
    data = S.data()
    odd = load_runs("odd")
    rows = []
    for thr in thrs:
        filt = make_filt(scores, thr)
        orig = S.partial
        S.partial = lambda fn, cfg, _f=filt: (lambda r, ds: _f(r, fn(r, ds, cfg=cfg)))
        try:
            row = S.score({"pose_action_cooldown_s": 15.0, "crop_check": True, "person_dedup_s": 0.0}, *data[:4])
        finally:
            S.partial = orig
        row["setting"] = {"pose_verifier_thr": thr}
        row["icvl_odd"] = evaluate(odd, CFG, filt)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (HERE / "pose_verifier_oof.json").write_text(json.dumps(rows, indent=1))


def export(thr):
    srcs = sources()
    X, y, w, *_ = build(srcs)
    models = fit(X, y, w)
    np.savez(OUT, mu=np.stack([m.mu for m in models]), sd=np.stack([m.sd for m in models]),
             W1=np.stack([m.net[0].weight.detach().numpy().T for m in models]),
             b1=np.stack([m.net[0].bias.detach().numpy() for m in models]),
             W2=np.stack([m.net[3].weight.detach().numpy()[0] for m in models]),
             b2=np.array([float(m.net[3].bias.detach().numpy()[0]) for m in models]), threshold=thr)
    print("saved", OUT, "threshold", thr)


def even(thrs):
    """ICVL even (never used for training or thresholds), final exported model."""
    from stage1_pose_action import PoseActionModel
    load_crop_scores()
    model = PoseActionModel(OUT)
    runs = load_runs("even")
    scores = {}
    for g, r in runs:
        for a in replay_rules(r, CFG)[0]:
            if a["event_type"] == "pose_action":
                f = alert_window(r, a)
                scores[key(r.seq_id, a)] = 1.0 if f is None else float(model.predict(f[None])[0])
    for thr in [None] + thrs:
        print(json.dumps({"pose_verifier_thr": thr, **evaluate(runs, CFG, make_filt(scores, thr))}), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "oof":
        oof()
    elif cmd == "sweep":
        sweep(thrs=[None] + [float(x) for x in sys.argv[2:]])
    elif cmd == "export":
        export(float(sys.argv[2]))
    else:
        even([float(x) for x in sys.argv[2:]])
