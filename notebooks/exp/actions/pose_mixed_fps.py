"""Retrain the pose_action cue on windows from both ~10 fps and 5 fps perception, so it holds up in the data
pipeline (which runs Stage 1 at 5 fps; see exp/newcam/fps_check.py).

Why: window_features resamples each 1 s window to 8 points by sample-and-hold, so at 5 fps every other velocity
is 0 and the next doubled; the current model only saw ~10 fps windows.
Same features, MLP, seeds, class weights and folds as pose_seq_final.py; a video keeps its fold at both rates.
Each rate gets half the weight of every group. The threshold is the highest that keeps out-of-fold Avenue tune
normal false alarms <= TARGET_FA per minute at BOTH rates.

Usage (from notebooks/):
    ../.venv/bin/python exp/actions/pose_mixed_fps.py freeze    # tune half only -> cache/pose_mixed_fps.pkl
    ../.venv/bin/python exp/actions/pose_mixed_fps.py heldout   # one-time held-out comparison with the current cue
    ../.venv/bin/python exp/actions/pose_mixed_fps.py export    # weights/pose_action_mlp_mixfps.npz
"""
import json
import pickle
import sys
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "gt_review"), str(HERE)]

import pose_seq as P  # noqa: E402
import pose_seq_final as F  # noqa: E402
from compare_models import model_splits  # noqa: E402
from exp_harness import baseline, combine  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig  # noqa: E402

RATES = {"10fps": NB / "bench_cache" / "models" / "s", "5fps": NB / "bench_cache" / "models" / "s_5fps"}
STORE = P.CACHE / "pose_mixed_fps.pkl"
WEIGHTS = NB / "weights" / "pose_action_mlp_mixfps.npz"


def splits(rate, heldout):
    """Littering clips and Avenue normal videos of one half at one rate (Avenue split as in compare_models)."""
    cache = RATES[rate]
    load = lambda n: pickle.loads((cache / f"{n}.pkl").read_bytes())  # noqa: E731
    avenue = load("avenue")
    want = 0 if heldout else 1
    num = lambda g: int(g.seq.seq_id.split("/")[-1])  # noqa: E731
    return {"littering": load("litter_heldout" if heldout else "litter_train"),
            "normal": [x for x in avenue if x[0].seq.seq_id.startswith("train/") and num(x[0]) % 2 == want]}


def freeze(folds=5):
    T = {rate: splits(rate, heldout=False) for rate in RATES}
    fold = F._folds(T["10fps"], folds)
    win = {(rate, ds): {r.seq_id: P.extract(r) for _, r in T[rate][ds]} for rate in RATES for ds in ("littering", "normal")}
    rows = []  # (rate, ds, seq_id, i, y)
    for rate in RATES:
        for g, r in T[rate]["littering"]:
            rows += [(rate, "littering", r.seq_id, i, y)
                     for i, w in enumerate(win[rate, "littering"][r.seq_id]) if (y := P._label(w, g)) >= 0]
        rows += [(rate, "normal", s, i, 0) for s, ws in win[rate, "normal"].items() for i, _ in enumerate(ws)]
    X = P.feats([win[rt, d][s][i] for rt, d, s, i, _ in rows], F.VARIANT)
    y = np.array([x[4] for x in rows], float)
    f = np.array([fold[x[2]] for x in rows])
    # pose_seq_final weights (positives 1, littering negatives 0.5, normal 0.5), split evenly across the rates
    w = np.zeros(len(y))
    for rate in RATES:
        rt = np.array([x[0] == rate for x in rows])
        lit = np.array([x[1] == "littering" for x in rows])
        for mask, share in (((y == 1) & rt, 1.0), ((y == 0) & lit & rt, 0.5), (~lit & rt, 0.5)):
            w[mask] = share / 2 / mask.sum()
    w *= len(y)

    oof = {key: {s: np.zeros(len(ws)) for s, ws in d.items()} for key, d in win.items()}
    full = []
    for seed in F.SEEDS:
        for k in range(folds):
            m = P.TorchModel("mlp", seed=seed).fit(X[f != k], y[f != k], w[f != k])
            for key, d in win.items():
                for s, ws in d.items():
                    if fold[s] == k and ws:
                        oof[key][s] += m.predict(P.feats(ws, F.VARIANT)) / len(F.SEEDS)
        full.append(P.TorchModel("mlp", seed=seed).fit(X, y, w))
        print(f"seed {seed} trained", flush=True)

    meta = {key: {s: [(wd["t"], wd["raw_id"], wd["box"]) for wd in ws] for s, ws in d.items()} for key, d in win.items()}
    minutes = {rate: sum(r.duration_s for _, r in T[rate]["normal"]) / 60 for rate in RATES}

    def fa(rate, thr):
        key = (rate, "normal")
        return sum(len(P._events_from(meta[key][s], p, thr, F.COOLDOWN)) for s, p in oof[key].items()) / minutes[rate]

    cand = np.unique(np.concatenate([v for (_, ds), d in oof.items() if ds == "normal" for v in d.values() if len(v)]))[::-1]
    thr = 1.0
    for t in np.r_[cand, 0.0]:
        if max(fa(rate, t) for rate in RATES) > F.TARGET_FA:
            break
        thr = float(t)
    STORE.write_bytes(pickle.dumps({"ensemble": F.Ensemble(full), "thr": thr, "oof": oof, "meta": meta}))

    out = {"threshold": thr}
    for rate in RATES:
        key = (rate, "littering")
        cue = lambda r, ds, key=key: P._events_from(meta[key][r.seq_id], oof[key][r.seq_id], thr, F.COOLDOWN)  # noqa: E731
        s = summarize([evaluate_run(g, r, cue(r, "littering")) for g, r in T[rate]["littering"]], "l")
        out[rate] = {"littering_tune_cue_alone_event_%": s["event_recall_%"], "littering_tune_cue_alone_crop_%": s["crop_recall_%"],
                     "avenue_tune_normal_cue_per_min": round(fa(rate, thr), 2)}
    print("tune out-of-fold:", json.dumps(out), flush=True)
    (HERE / "pose_mixed_fps_tune.json").write_text(json.dumps(out, indent=1))


def heldout():
    """Score the current cue and the mixed-rate cue once on held-out, at both rates, with and without the rules."""
    new = pickle.loads(STORE.read_bytes())
    old = pickle.loads(F.FROZEN.read_bytes())
    rules = partial(baseline, cfg=GateConfig(inview_s=60.0, rules=DEFAULT_RULES))
    out = {}
    for rate in RATES:
        S = splits(rate, heldout=True)
        minutes = sum(r.duration_s for _, r in S["normal"]) / 60
        for name, st in (("current cue", old), ("mixed-rate cue", new)):
            def cue(r, ds, st=st):
                ws = P.extract(r)
                probs = st["ensemble"].predict(P.feats(ws, F.VARIANT)) if ws else []
                return P._events_from([(w["t"], w["raw_id"], w["box"]) for w in ws], probs, st["thr"], F.COOLDOWN)
            row = {}
            for label, m in (("alone", cue), ("rules +", combine(rules, cue))):
                s = summarize([evaluate_run(g, r, m(r, "littering")) for g, r in S["littering"]], "l")
                row[f"{label} littering event/crop %"] = f"{s['event_recall_%']}/{s['crop_recall_%']}"
                row[f"{label} normal per min"] = round(sum(len(m(r, "normal")) for _, r in S["normal"]) / minutes, 2)
            out[f"{rate} {name}"] = row
            print(rate, name, json.dumps(row), flush=True)
    (HERE / "pose_mixed_fps_heldout.json").write_text(json.dumps(out, indent=1))


def export():
    st = pickle.loads(STORE.read_bytes())
    ms = st["ensemble"].models
    lin = [[layer for layer in m.net if hasattr(layer, "weight")] for m in ms]
    arr = lambda x: x.detach().numpy().astype(np.float64)  # noqa: E731
    np.savez_compressed(WEIGHTS, threshold=st["thr"], mu=np.stack([m.mu for m in ms]), sd=np.stack([m.sd for m in ms]),
                        W1=np.stack([arr(a.weight).T for a, _ in lin]), b1=np.stack([arr(a.bias) for a, _ in lin]),
                        W2=np.stack([arr(b.weight).ravel() for _, b in lin]), b2=np.array([b.bias.item() for _, b in lin]))
    from stage1_pose_action import PoseActionModel
    X = P.feats([w for _, r in splits("5fps", heldout=False)["littering"][:20] for w in P.extract(r)], F.VARIANT)
    diff = np.abs(PoseActionModel(WEIGHTS).predict(X) - st["ensemble"].predict(X)).max()
    print(f"wrote {WEIGHTS} (threshold {st['thr']:.3f}); max |numpy - torch| = {diff:.2e}")


if __name__ == "__main__":
    {"freeze": freeze, "heldout": heldout, "export": export}[sys.argv[1]]()
