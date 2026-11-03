"""Retrain the pose_action cue with IWDD train added (weak labels), keep the old training data, compare on train OOF.

IWDD only gives the onset second, not who dumps. Windows of positive videos in [onset - 1 s, onset + 5 s] are
positive candidates; windows elsewhere in positive videos are dropped (preparation / aftermath are ambiguous);
every window of a negative video is 0. Two labelings:
  all   every tracked person inside the onset interval is positive (bystanders become label noise)
  mil   one refinement round: per positive video keep only the track with the highest round-0 out-of-fold score
        inside the interval as positive; other tracks there are dropped
The original data stays: littering tune clips (GT actor = 1, others = 0) and Avenue tune normal videos (all 0).
Same features (stage1_pose_action.window_features summary) and MLP as the current cue, 5 seeds x 5 folds; folds
split by clip/video so every window is scored by a model that never saw its video (OOF).
The threshold keeps the current budget: ~1 candidate/min on OOF Avenue normal (TARGET_FA), chosen without IWDD.

Usage (from notebooks/):
    ../.venv/bin/python exp/newcam/iwdd_pose.py build     # train OOF + full models (train split only), compare on train
    ../.venv/bin/python exp/newcam/iwdd_pose.py export    # write weights/pose_action_mlp_iwdd.npz for the chosen labeling
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
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "actions"), str(NB / "exp" / "gt_review"),
                str(HERE)]

import pose_seq as P  # noqa: E402
import pose_seq_final as F  # noqa: E402
from compare_models import model_splits  # noqa: E402
from exp_harness import baseline, combine  # noqa: E402
from iwdd_eval import evaluate, inview_from_avenue, load  # noqa: E402
from loaders import iwdd_labels  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import GateConfig  # noqa: E402

FOLDS = 5
PRE_S, POST_S = 1.0, 5.0
SHARES = {"lit_pos": 0.25, "iwdd_pos": 0.25, "lit_neg": 0.5 / 3, "normal": 0.5 / 3, "iwdd_neg": 0.5 / 3}
STORE = P.CACHE / "iwdd_pose.pkl"
WEIGHTS = NB / "weights" / "pose_action_mlp_iwdd.npz"


def _iwdd_folds(labels):
    fold, per = {}, defaultdict(int)
    for sid in sorted(labels):
        key = labels[sid]["kind"]
        fold[sid] = per[key] % FOLDS
        per[key] += 1
    return fold


def windows():
    T = model_splits("s", heldout=False)
    labels = iwdd_labels("train")
    iw = load("train")
    win = {"littering": {r.seq_id: P.extract(r) for _, r in T["littering"]},
           "normal": {r.seq_id: P.extract(r) for _, r in T["normal"]},
           "iwdd": {r.seq_id: P.extract(r) for _, r in iw}}
    fold = {**F._folds(T, FOLDS), **_iwdd_folds(labels)}
    rows = []  # (ds, sid, i, y, group)
    for g, r in T["littering"]:
        for i, w in enumerate(win["littering"][r.seq_id]):
            y = P._label(w, g)
            if y >= 0:
                rows.append(("littering", r.seq_id, i, y, "lit_pos" if y else "lit_neg"))
    rows += [("normal", s, i, 0, "normal") for s, ws in win["normal"].items() for i, _ in enumerate(ws)]
    for sid, ws in win["iwdd"].items():
        lab = labels[sid]
        for i, w in enumerate(ws):
            if not lab["positive"]:
                rows.append(("iwdd", sid, i, 0, "iwdd_neg"))
            elif lab["onset_s"] - PRE_S <= w["t"] <= lab["onset_s"] + POST_S:
                rows.append(("iwdd", sid, i, 1, "iwdd_pos"))
    return T, labels, iw, win, fold, rows


def weights(rows):
    """Each group gets its SHARES of the total; within iwdd_pos every video counts equally."""
    w = np.zeros(len(rows))
    groups = defaultdict(list)
    for j, row in enumerate(rows):
        groups[row[4]].append(j)
    for gname, idx in groups.items():
        if gname == "iwdd_pos":
            per_vid = defaultdict(list)
            for j in idx:
                per_vid[rows[j][1]].append(j)
            for js in per_vid.values():
                w[js] = SHARES[gname] / len(per_vid) / len(js)
        else:
            w[idx] = SHARES[gname] / len(idx)
    return w * len(rows)


def train(win, fold, rows, X):
    """OOF scores for every window of every video (by fold) and the full-data ensemble."""
    y = np.array([r[3] for r in rows], float)
    f = np.array([fold[r[1]] for r in rows])
    w = weights(rows)
    feats = {ds: {s: P.feats(ws, F.VARIANT) for s, ws in d.items() if ws} for ds, d in win.items()}
    oof = {ds: {s: np.zeros(len(ws)) for s, ws in d.items()} for ds, d in win.items()}
    full = []
    for seed in F.SEEDS:
        for k in range(FOLDS):
            m = P.TorchModel("mlp", seed=seed).fit(X[f != k], y[f != k], w[f != k])
            for ds, d in feats.items():
                for s, Xs in d.items():
                    if fold[s] == k:
                        oof[ds][s] += m.predict(Xs) / len(F.SEEDS)
        full.append(P.TorchModel("mlp", seed=seed).fit(X, y, w))
        print(f"  seed {seed} done", flush=True)
    return oof, F.Ensemble(full)


def mil_rows(rows, win, oof):
    """Keep, per positive IWDD video, only the interval windows of its highest-scoring track."""
    best = {}
    for ds, sid, i, y, gname in rows:
        if gname == "iwdd_pos":
            p, rid = oof["iwdd"][sid][i], win["iwdd"][sid][i]["raw_id"]
            if p > best.get(sid, (-1, None))[0]:
                best[sid] = (p, rid)
    return [r for r in rows if r[4] != "iwdd_pos" or win["iwdd"][r[1]][r[2]]["raw_id"] == best[r[1]][1]]


def threshold(win, oof, T):
    meta = {s: [(w["t"], w["raw_id"], w["box"]) for w in ws] for s, ws in win["normal"].items()}
    minutes = sum(r.duration_s for _, r in T["normal"]) / 60
    cand = np.unique(np.concatenate([v for v in oof["normal"].values() if len(v)]))[::-1]
    thr = 1.0
    for t in np.r_[cand, 0.0]:
        if sum(len(P._events_from(meta[s], p, t, F.COOLDOWN)) for s, p in oof["normal"].items()) / minutes > F.TARGET_FA:
            break
        thr = float(t)
    return thr


def oof_method(win, oof, ds, thr):
    def run(result, dataset):
        ws = win[ds].get(result.seq_id, [])
        return P._events_from([(w["t"], w["raw_id"], w["box"]) for w in ws], oof[ds][result.seq_id], thr, F.COOLDOWN)
    return run


def compare(T, labels, iw, win, variants):
    """Train-split OOF comparison: IWDD train, littering tune, Avenue normal tune FA."""
    rules = partial(baseline, cfg=GateConfig(inview_s=inview_from_avenue()))
    old_oof = pickle.loads(F.FROZEN.read_bytes())
    old_cue = F.frozen_method((old_oof["oof"], old_oof["meta"]))[0]
    cues = {"old cue": (old_cue, old_oof["thr"])}
    for name, (oof, thr) in variants.items():
        cues[name] = (lambda r, ds, oof=oof, thr=thr: oof_method(win, oof, "iwdd" if r.seq_id.startswith("iwdd/") else
                                                                ("littering" if ds == "littering" else "normal"),
                                                                thr)(r, ds), thr)
    out = {"rules only": {"iwdd_train": evaluate(iw, labels, rules)}}
    minutes = sum(r.duration_s for _, r in T["normal"]) / 60
    for name, (cue, thr) in cues.items():
        m = combine(rules, cue)
        lit = summarize([evaluate_run(g, r, m(r, "littering")) for g, r in T["littering"]], "l")
        alone = summarize([evaluate_run(g, r, cue(r, "littering")) for g, r in T["littering"]], "a")
        out[f"rules + {name}"] = {
            "threshold": round(thr, 3),
            "iwdd_train": evaluate(iw, labels, m),
            "littering_tune_event_%": lit["event_recall_%"], "littering_tune_crop_%": lit["crop_recall_%"],
            "littering_tune_cue_alone_crop_%": alone["crop_recall_%"],
            "avenue_normal_cue_per_min": round(sum(len(cue(r, "normal")) for _, r in T["normal"]) / minutes, 2)}
        print(name, json.dumps(out[f"rules + {name}"]), flush=True)
    return out


def build():
    T, labels, iw, win, fold, rows = windows()
    print(f"windows: {len(rows)} labelled ({sum(r[3] for r in rows)} positive)", flush=True)
    allX = {ds: {s: P.feats(ws, F.VARIANT) for s, ws in d.items() if ws} for ds, d in win.items()}
    X = np.stack([allX[d][s][i] for d, s, i, _, _ in rows])
    print("round 0 (all interval tracks positive)", flush=True)
    oof0, full0 = train(win, fold, rows, X)
    rows1 = mil_rows(rows, win, oof0)
    X1 = np.stack([allX[d][s][i] for d, s, i, _, _ in rows1])
    print(f"round 1 (MIL, {len(rows1)} rows)", flush=True)
    oof1, full1 = train(win, fold, rows1, X1)
    variants = {"iwdd all": (oof0, threshold(win, oof0, T)), "iwdd mil": (oof1, threshold(win, oof1, T))}
    STORE.write_bytes(pickle.dumps({"all": {"ensemble": full0, "thr": variants["iwdd all"][1], "oof": oof0},
                                    "mil": {"ensemble": full1, "thr": variants["iwdd mil"][1], "oof": oof1}}))
    out = compare(T, labels, iw, win, variants)
    (HERE / "iwdd_pose_train.json").write_text(json.dumps(out, indent=1))


def export(which="mil"):
    st = pickle.loads(STORE.read_bytes())[which]
    ms = st["ensemble"].models
    lin = [[layer for layer in m.net if hasattr(layer, "weight")] for m in ms]
    arr = lambda x: x.detach().numpy().astype(np.float64)  # noqa: E731
    np.savez_compressed(WEIGHTS, threshold=st["thr"], mu=np.stack([m.mu for m in ms]), sd=np.stack([m.sd for m in ms]),
                        W1=np.stack([arr(a.weight).T for a, _ in lin]), b1=np.stack([arr(a.bias) for a, _ in lin]),
                        W2=np.stack([arr(b.weight).ravel() for _, b in lin]), b2=np.array([b.bias.item() for _, b in lin]))
    from stage1_pose_action import PoseActionModel
    X = P.feats([w for _, r in load("train")[:20] for w in P.extract(r)], F.VARIANT)
    diff = np.abs(PoseActionModel(WEIGHTS).predict(X) - st["ensemble"].predict(X)).max()
    print(f"wrote {WEIGHTS} ({which}, threshold {st['thr']:.3f}); max |numpy - torch| = {diff:.2e}")


if __name__ == "__main__":
    {"build": build, "export": lambda: export(*sys.argv[2:])}[sys.argv[1]]()
