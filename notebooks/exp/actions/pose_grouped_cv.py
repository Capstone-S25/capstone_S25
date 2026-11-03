"""An honest littering estimate for the pose_action cue: cross-validation grouped by SOURCE video.

The littering train / held-out split shares source videos (every held-out clip's source video also has train clips,
e.g. dropping_v01_6 vs dropping_v01_1..5), so the frozen cue's held-out 97 / 86 % is "new clips, same scenes".
Here all 330 clips (train + held-out) are split into 5 folds by (action, source video), and the Avenue tune normals
by video, exactly as pose_seq_final.freeze does otherwise (same features, MLP, weights, 5 seeds). Every clip is scored
by a model that never saw its source video; the threshold comes from out-of-fold normal scores (~1/min).
Reports cue alone and rules + cue on all 330 clips and on the 100 former held-out clips, next to the frozen cue.
Usage (from notebooks/): ../.venv/bin/python exp/actions/pose_grouped_cv.py   -> exp/actions/pose_grouped_cv.json
"""
import json
import pickle
import re
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE)]

import pose_seq as P  # noqa: E402
import pose_seq_final as F  # noqa: E402
from compare_models import model_splits  # noqa: E402
from exp_harness import baseline, combine  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import GateConfig, calibrate_inview  # noqa: E402

FOLDS = 5


def source(seq_id):
    m = re.match(r"(.+)_v(\d+)_\d+$", seq_id)
    return f"{m.group(1)}_v{m.group(2)}"


def folds(lit, normal):
    """Littering folds by source video, balanced per action; normal videos round-robin."""
    by_action = defaultdict(set)
    for g, r in lit:
        by_action[g.tag].add(source(r.seq_id))
    src_fold = {}
    for tag in sorted(by_action):
        for i, s in enumerate(sorted(by_action[tag])):
            src_fold[s] = i % FOLDS
    fold = {r.seq_id: src_fold[source(r.seq_id)] for _, r in lit}
    for i, (_, r) in enumerate(sorted(normal, key=lambda x: x[1].seq_id)):
        fold[r.seq_id] = i % FOLDS
    return fold


def main():
    T = model_splits("s", heldout=False)
    H = model_splits("s", heldout=True)
    heldout_ids = {r.seq_id for _, r in H["littering"]}
    lit = T["littering"] + H["littering"]
    normal = T["normal"]
    fold = folds(lit, normal)
    win = {"littering": {r.seq_id: P.extract(r) for _, r in lit}, "normal": {r.seq_id: P.extract(r) for _, r in normal}}
    gts = {r.seq_id: g for g, r in lit}
    rows = [("littering", s, i, y, fold[s]) for s, ws in win["littering"].items() for i, w in enumerate(ws)
            if (y := P._label(w, gts[s])) >= 0]
    rows += [("normal", s, i, 0, fold[s]) for s, ws in win["normal"].items() for i, _ in enumerate(ws)]
    X = P.feats([win[d][s][i] for d, s, i, _, _ in rows], F.VARIANT)
    y = np.array([x[3] for x in rows], float)
    f = np.array([x[4] for x in rows])
    is_lit = np.array([x[0] == "littering" for x in rows])
    w = np.ones(len(y))
    w[y == 1] = 1.0 / y.sum()
    w[(y == 0) & is_lit] = 0.5 / ((y == 0) & is_lit).sum()
    w[~is_lit] = 0.5 / (~is_lit).sum()
    w *= len(y)

    oof = {ds: {s: np.zeros(len(ws)) for s, ws in d.items()} for ds, d in win.items()}
    for seed in F.SEEDS:
        for k in range(FOLDS):
            m = P.TorchModel("mlp", seed=seed).fit(X[f != k], y[f != k], w[f != k])
            for ds, d in win.items():
                for s, ws in d.items():
                    if fold[s] == k and ws:
                        oof[ds][s] += m.predict(P.feats(ws, F.VARIANT)) / len(F.SEEDS)
        print(f"seed {seed} done", flush=True)

    meta = {s: [(x["t"], x["raw_id"], x["box"]) for x in ws] for d in win.values() for s, ws in d.items()}
    minutes = sum(r.duration_s for _, r in normal) / 60
    cand = np.unique(np.concatenate([v for v in oof["normal"].values() if len(v)]))[::-1]
    thr = 1.0
    for t in np.r_[cand, 0.0]:
        if sum(len(P._events_from(meta[s], p, t, F.COOLDOWN)) for s, p in oof["normal"].items()) / minutes > F.TARGET_FA:
            break
        thr = float(t)

    grouped = lambda r, ds: P._events_from(meta[r.seq_id], oof["littering"][r.seq_id], thr, F.COOLDOWN)  # noqa: E731
    frozen = F.frozen_method()[0]
    rules = partial(baseline, cfg=GateConfig(inview_s=calibrate_inview([r for _, r in normal])))
    out = {"threshold": round(thr, 3), "folds": "by (action, source video)"}
    for subset, clips in (("all 330 clips", lit), ("former held-out 100 clips", [x for x in lit if x[1].seq_id in heldout_ids])):
        row = {}
        for name, m in (("rules", rules), ("frozen cue alone", frozen), ("grouped-CV cue alone", grouped),
                        ("rules + frozen cue", combine(rules, frozen)), ("rules + grouped-CV cue", combine(rules, grouped))):
            s = summarize([evaluate_run(g, r, m(r, "littering")) for g, r in clips], name)
            row[name] = f"{s['event_recall_%']}/{s['crop_recall_%']}"
        out[subset] = row
        print(subset, json.dumps(row), flush=True)
    out["note"] = ("frozen cue = the shipped model; on 'all 330' it has trained on the 230 train clips (in-sample), "
                   "on the former held-out it was never trained on those clips but did see their source videos")
    (HERE / "pose_grouped_cv.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
