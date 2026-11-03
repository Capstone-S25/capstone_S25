"""Freeze the pose-sequence cue (seed ensemble + one threshold) on the tune half, then score it ONCE on held-out.

Usage (from notebooks/):
    ../.venv/bin/python exp/actions/pose_seq_final.py freeze     # tune half only; writes cache/pose_seq_ensemble.pkl
    ../.venv/bin/python exp/actions/pose_seq_final.py heldout    # one-time held-out measurement with the frozen model
Follows the review in pose_seq_review.json: average 5 seeds, calibrate one threshold on the averaged out-of-fold
normal scores, and compare against "tallest person at the same moments" as well as the random-person controls.
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
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "gt_review"), str(HERE)]

import pose_seq as P  # noqa: E402
from compare_models import model_splits  # noqa: E402
from exp_harness import baseline, combine, make_event, score  # noqa: E402
from random_control import lift  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import GateConfig, calibrate_inview  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
VARIANT = "mlp_sum"
TARGET_FA = 1.0
COOLDOWN = P.DEFAULTS["cooldown_s"]
FROZEN = P.CACHE / "pose_seq_ensemble.pkl"


class Ensemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


def _folds(T, folds=5):
    fold, per_tag = {}, defaultdict(int)
    for g, r in sorted(T["littering"], key=lambda x: x[1].seq_id):
        fold[r.seq_id] = per_tag[g.tag] % folds
        per_tag[g.tag] += 1
    for i, (_, r) in enumerate(sorted(T["normal"], key=lambda x: x[1].seq_id)):
        fold[r.seq_id] = i % folds
    return fold


def freeze(folds=5):
    T = model_splits("s", heldout=False)
    win = {ds: {r.seq_id: P.extract(r) for _, r in T[ds]} for ds in ("littering", "normal")}
    fold = _folds(T, folds)
    rows = [("littering", r.seq_id, i, y, fold[r.seq_id])
            for g, r in T["littering"] for i, w in enumerate(win["littering"][r.seq_id]) if (y := P._label(w, g)) >= 0]
    rows += [("normal", r.seq_id, i, 0, fold[r.seq_id]) for _, r in T["normal"] for i, _ in enumerate(win["normal"][r.seq_id])]
    X = P.feats([win[d][s][i] for d, s, i, _, _ in rows], VARIANT)
    y = np.array([x[3] for x in rows], float)
    f = np.array([x[4] for x in rows])
    lit = np.array([x[0] == "littering" for x in rows])
    w = np.ones(len(y))
    w[y == 1] = 1.0 / y.sum()
    w[(y == 0) & lit] = 0.5 / ((y == 0) & lit).sum()
    w[~lit] = 0.5 / (~lit).sum()
    w *= len(y)

    oof = {ds: {sid: np.zeros(len(ws)) for sid, ws in win[ds].items()} for ds in win}
    full = []
    for seed in SEEDS:
        for k in range(folds):
            tr = f != k
            m = P.TorchModel("mlp", seed=seed).fit(X[tr], y[tr], w[tr])
            for ds in win:
                for sid, ws in win[ds].items():
                    if fold[sid] == k and ws:
                        oof[ds][sid] += m.predict(P.feats(ws, VARIANT)) / len(SEEDS)
        full.append(P.TorchModel("mlp", seed=seed).fit(X, y, w))
        print(f"seed {seed} trained", flush=True)

    meta = {sid: [(wd["t"], wd["raw_id"], wd["box"]) for wd in ws] for ds in win for sid, ws in win[ds].items()}
    minutes = sum(r.duration_s for _, r in T["normal"]) / 60
    cand = np.unique(np.concatenate([v for v in oof["normal"].values() if len(v)]))[::-1]
    thr = 1.0
    for t in np.r_[cand, 0.0]:
        if sum(len(P._events_from(meta[s], p, t, COOLDOWN)) for s, p in oof["normal"].items()) / minutes > TARGET_FA:
            break
        thr = float(t)
    FROZEN.write_bytes(pickle.dumps({"ensemble": Ensemble(full), "thr": thr, "seeds": SEEDS, "variant": VARIANT,
                                     "target_fa": TARGET_FA, "cooldown_s": COOLDOWN, "oof": oof, "meta": meta}))
    print(f"frozen: threshold {thr:.3f} for {TARGET_FA}/min on averaged out-of-fold tune normal scores", flush=True)
    report("tune", T, oof_scores=(oof, meta))


def frozen_method(oof_scores=None):
    st = pickle.loads(FROZEN.read_bytes())
    model, thr = st["ensemble"], st["thr"]

    def method(result, dataset):
        if oof_scores and result.seq_id in oof_scores[0].get(dataset, {}):
            return P._events_from(oof_scores[1][result.seq_id], oof_scores[0][dataset][result.seq_id], thr, COOLDOWN)
        ws = P.extract(result)
        probs = model.predict(P.feats(ws, VARIANT)) if ws else []
        return P._events_from([(wd["t"], wd["raw_id"], wd["box"]) for wd in ws], probs, thr, COOLDOWN)

    return method, thr


def tallest_same_times(method):
    """Control: at each moment the method fires, send the tallest person in that frame instead."""
    def run(result, dataset):
        by_t = {round(o.t, 2): o for o in result.observations}
        out = []
        for e in method(result, dataset):
            o = by_t.get(round(e["timestamp_s"], 2)) or min(result.observations, key=lambda o: abs(o.t - e["timestamp_s"]))
            if o.persons:
                out.append(make_event("tallest", o.t, max(o.persons, key=lambda p: p["box"][3] - p["box"][1])["box"]))
        return out
    return run


def report(half, S, oof_scores=None):
    from scoped import scoped
    cfg = GateConfig(inview_s=calibrate_inview([r for _, r in model_splits("s", heldout=False)["normal"]]))
    base = partial(baseline, cfg=cfg)
    cue, thr = frozen_method(oof_scores)
    addon = combine(base, cue)
    rows = {}
    for name, m in (("current rules", base), ("pose cue alone", cue), ("current rules + pose cue", addon)):
        s = score(m, splits=S, name=name)
        mc, cc, _ = lift(m, S["littering"], seeds=10)
        fc, fcc, _ = P.fair_lift(m, S["littering"], seeds=10)
        strict = summarize([evaluate_run(g, r, m(r, "avenue")) for g, r in scoped(S["avenue"], keep=("in",))], "s")
        rows[name] = {**{k: v for k, v in s.items() if k != "method"},
                      "littering_lift_random": round(mc - cc, 1), "littering_lift_distinct_random": round(fc - fcc, 1),
                      "loitering_strict_event_%": strict["event_recall_%"], "loitering_strict_crop_%": strict["crop_recall_%"]}
    tall = summarize([evaluate_run(g, r, tallest_same_times(cue)(r, "littering")) for g, r in S["littering"]], "t")
    alone = summarize([evaluate_run(g, r, cue(r, "littering")) for g, r in S["littering"]], "a")
    rows["pose cue alone"]["littering_crop_tallest_same_times_%"] = tall["crop_recall_%"]
    rows["pose cue alone"]["skill_over_tallest"] = round(alone["crop_recall_%"] - tall["crop_recall_%"], 1)
    per_action = {}
    for tag in sorted({g.tag for g, _ in S["littering"]}):
        sub = [(g, r) for g, r in S["littering"] if g.tag == tag]
        a = summarize([evaluate_run(g, r, cue(r, "littering")) for g, r in sub], tag)["crop_recall_%"]
        t = summarize([evaluate_run(g, r, tallest_same_times(cue)(r, "littering")) for g, r in sub], tag)["crop_recall_%"]
        b = summarize([evaluate_run(g, r, addon(r, "littering")) for g, r in sub], tag)["crop_recall_%"]
        c = summarize([evaluate_run(g, r, base(r, "littering")) for g, r in sub], tag)["crop_recall_%"]
        per_action[tag] = {"current rules": c, "with pose cue": b, "cue alone": a, "tallest same times": t}
    out = {"half": half, "threshold": thr, "results": rows, "littering_per_action_crop_%": per_action}
    (HERE / f"pose_seq_final_{half}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def export(path=NB / "weights" / "pose_action_mlp.npz"):
    """Write the frozen ensemble as numpy arrays for stage1_pose_action.PoseActionModel (no torch needed)."""
    st = pickle.loads(FROZEN.read_bytes())
    ms = st["ensemble"].models
    lin = [[l for l in m.net if hasattr(l, "weight")] for m in ms]
    arr = lambda x: x.detach().numpy().astype(np.float64)  # noqa: E731
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, threshold=st["thr"], mu=np.stack([m.mu for m in ms]), sd=np.stack([m.sd for m in ms]),
                        W1=np.stack([arr(a.weight).T for a, _ in lin]), b1=np.stack([arr(a.bias) for a, _ in lin]),
                        W2=np.stack([arr(b.weight).ravel() for _, b in lin]),
                        b2=np.array([b.bias.item() for _, b in lin]))
    from stage1_pose_action import PoseActionModel
    X = P.feats([w for _, r in model_splits("s", heldout=True)["littering"][:20] for w in P.extract(r)], VARIANT)
    diff = np.abs(PoseActionModel(path).predict(X) - st["ensemble"].predict(X)).max()
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB); max |numpy - torch| = {diff:.2e}")


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export()
    elif sys.argv[1] == "freeze":
        freeze()
    elif sys.argv[1] == "heldout":
        assert FROZEN.exists(), "freeze first"
        report("held-out", model_splits("s", heldout=True))
