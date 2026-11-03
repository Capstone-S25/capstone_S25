"""Retrain pose_action with the dumper labelled in IWDD train (iwdd_actors_train.json), not just the onset second.

Rows as in iwdd_pose.py (littering tune clips, Avenue tune normals, every window of IWDD train negatives), but in IWDD
positive videos the windows in [onset - 1 s, onset + 5 s] are
  1  if the track is a labelled dumper            (group iwdd_pos, every video weighted equally)
  0  if it is another tracked person in that video (group iwdd_bys, hard negatives)
and videos whose dumper is ambiguous or untracked are left out. Same features, MLP, 5 seeds x 5 folds (folds by
video), threshold for ~1 candidate/min on out-of-fold Avenue tune normals.
Compared on IWDD train out-of-fold with the current cue (which never saw IWDD):
  actor_auc   within labelled videos, does the dumper's window score above the bystanders' (pairwise AUC)?
  right_first in labelled videos, is the first candidate in [onset - 1, onset + 10] on the dumper?
plus hit@10s vs chance, littering tune and Avenue false alarms (iwdd_pose.compare).
Usage (from notebooks/): ../.venv/bin/python exp/newcam/iwdd_pose_actor.py build [--retrain]  -> iwdd_pose_actor_train.json
          ../.venv/bin/python exp/newcam/iwdd_pose_actor.py heldout  (once, after choosing on train)
          ../.venv/bin/python exp/newcam/iwdd_pose_actor.py export   -> weights/pose_action_mlp_iwdd_actor.npz
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "actions"), str(HERE)]

import iwdd_pose as IP  # noqa: E402
import pose_seq as P  # noqa: E402
import pose_seq_final as F  # noqa: E402

IP.SHARES = {"lit_pos": 0.25, "iwdd_pos": 0.25, "lit_neg": 0.5 / 4, "normal": 0.5 / 4, "iwdd_neg": 0.5 / 4,
             "iwdd_bys": 0.5 / 4}
STORE = P.CACHE / "iwdd_pose_actor.pkl"
WEIGHTS = NB / "weights" / "pose_action_mlp_iwdd_actor.npz"
ACTORS = json.loads((HERE / "iwdd_actors_train.json").read_text())["actors"]


def vid(sid):
    return sid.split("/")[-1]


def relabel(rows, win, labels):
    out = []
    for ds, sid, i, y, group in rows:
        if group != "iwdd_pos":
            out.append((ds, sid, i, y, group))
            continue
        actors = ACTORS.get(vid(sid))
        if actors is None:
            continue  # dumper ambiguous or untracked: no labels for this video
        if win["iwdd"][sid][i]["raw_id"] in actors:
            out.append((ds, sid, i, 1, "iwdd_pos"))
        else:
            out.append((ds, sid, i, 0, "iwdd_bys"))
    return out


def actor_metrics(win, scores, labels, thr):
    """Pairwise AUC of dumper vs bystander windows within each labelled video, and whether the first candidate in
    the scoring window is on the dumper."""
    pairs = wins = 0.0
    first_right = first_any = 0
    for sid, ws in win["iwdd"].items():
        actors = ACTORS.get(vid(sid))
        if actors is None or not ws:
            continue
        onset = labels[sid]["onset_s"]
        s = scores[sid]
        idx = [i for i, w in enumerate(ws) if onset - IP.PRE_S <= w["t"] <= onset + IP.POST_S]
        pos = [s[i] for i in idx if ws[i]["raw_id"] in actors]
        neg = [s[i] for i in idx if ws[i]["raw_id"] not in actors]
        for p in pos:
            for n in neg:
                pairs += 1
                wins += 1.0 if p > n else 0.5 if p == n else 0.0
        fired = [i for i, w in enumerate(ws) if onset - 1 <= w["t"] <= onset + 10 and s[i] >= thr]
        if fired:
            first_any += 1
            first_right += ws[min(fired, key=lambda i: ws[i]["t"])]["raw_id"] in actors
    return {"actor_auc": round(wins / max(pairs, 1), 3), "pairs": int(pairs),
            "videos_with_candidate": first_any, "first_candidate_on_dumper": first_right}


def build():
    T, labels, iw, win, fold, rows = IP.windows()
    rows = relabel(rows, win, labels)
    n_pos = sum(1 for r in rows if r[4] == "iwdd_pos")
    print(f"rows {len(rows)}: iwdd dumper windows {n_pos}, bystander windows {sum(r[4] == 'iwdd_bys' for r in rows)}",
          flush=True)
    allX = {ds: {s: P.feats(ws, F.VARIANT) for s, ws in d.items() if ws} for ds, d in win.items()}
    X = np.stack([allX[d][s][i] for d, s, i, _, _ in rows])
    if STORE.exists() and "--retrain" not in sys.argv:
        st = pickle.loads(STORE.read_bytes())
        oof, thr = st["oof"], st["thr"]
    else:
        oof, full = IP.train(win, fold, rows, X)
        thr = IP.threshold(win, oof, T)
        STORE.write_bytes(pickle.dumps({"ensemble": full, "thr": thr, "oof": oof}))

    old = pickle.loads(F.FROZEN.read_bytes())
    old_model = old["ensemble"]
    old_scores = {s: old_model.predict(P.feats(ws, F.VARIANT)) if ws else np.zeros(0) for s, ws in win["iwdd"].items()}
    out = IP.compare(T, labels, iw, win, {"actor-labelled cue": (oof, thr)})
    out["actor metrics (IWDD train labelled videos)"] = {
        "current cue": actor_metrics(win, old_scores, labels, old["thr"]),
        "actor-labelled cue (out-of-fold)": actor_metrics(win, oof["iwdd"], labels, thr)}
    print(json.dumps(out["actor metrics (IWDD train labelled videos)"], indent=1))
    (HERE / "iwdd_pose_actor_train.json").write_text(json.dumps(out, indent=1))


def heldout():
    """IWDD test once: rules + current cue vs rules + actor-labelled cue (full model, frozen threshold)."""
    from functools import partial

    from exp_harness import baseline, combine
    from iwdd_eval import evaluate, inview_from_avenue, load
    from loaders import iwdd_labels
    from stage1_gate import GateConfig

    st = pickle.loads(STORE.read_bytes())
    old = pickle.loads(F.FROZEN.read_bytes())
    data, labels = load("test"), iwdd_labels("test")
    rules = partial(baseline, cfg=GateConfig(inview_s=inview_from_avenue()))

    def cue(model, thr):
        def run(r, ds):
            ws = P.extract(r)
            probs = model.predict(P.feats(ws, F.VARIANT)) if ws else []
            return P._events_from([(w["t"], w["raw_id"], w["box"]) for w in ws], probs, thr, F.COOLDOWN)
        return run

    out = {"rules": evaluate(data, labels, rules),
           "rules + current cue": evaluate(data, labels, combine(rules, cue(old["ensemble"], old["thr"]))),
           "rules + actor-labelled cue": evaluate(data, labels, combine(rules, cue(st["ensemble"], st["thr"])))}
    for k, v in out.items():
        print(k, {m: v[m] for m in ("hit@10s_%", "chance@10s_%", "neg_any_%", "neg_cand_per_min")}, flush=True)
    (HERE / "iwdd_pose_actor_heldout.json").write_text(json.dumps(out, indent=1))


def export():
    IP.STORE, IP.WEIGHTS = STORE, WEIGHTS
    st = pickle.loads(STORE.read_bytes())
    STORE.write_bytes(pickle.dumps({"x": st}))
    try:
        IP.export("x")
    finally:
        STORE.write_bytes(pickle.dumps(st))


if __name__ == "__main__":
    {"build": build, "heldout": heldout, "export": export}[sys.argv[1]]()
