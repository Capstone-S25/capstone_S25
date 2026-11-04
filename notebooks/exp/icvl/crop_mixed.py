"""Crop check trained on more than ICVL: ICVL odd + littering train clips + smoking tune clips + IWDD train + Avenue
tune normals (exp/alerts/crop_rows.py), one logistic regression per rule on the before / now / after embeddings,
each source's classes given equal total weight.

Every training dataset is scored out-of-fold (5 folds grouped by sequence), so the sweep.py metrics and ICVL odd stay
out-of-sample; ICVL even (never trained on) is scored by the final model. Compared with the ICVL-only crop check.
Usage (from notebooks/):
  ../.venv/bin/python exp/icvl/crop_mixed.py eval REACH/POSE ...    -> exp/icvl/crop_mixed.json
  ../.venv/bin/python exp/icvl/crop_mixed.py export REACH POSE      -> weights/crop_check[_pose_action].npz + scores
"""
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(HERE), str(NB / "exp" / "alerts"), str(NB / "exp"), str(NB / "tools"),
                str(NB / "exp" / "newcam"), str(NB / "exp" / "gt_review"), str(NB / "exp" / "actions")]

import sweep as S  # noqa: E402
from crop_v2 import DT, feats  # noqa: E402
from fpr_sweep import evaluate  # noqa: E402
from hand_probe import NODEDUP, dedup  # noqa: E402
from score import load_runs  # noqa: E402

CACHE = NB / "bench_cache" / "models" / "s"
RULES = ("reach_down", "pose_action")
FOLDS = 5


def rows():
    odd_ids = {r.seq_id for _, r in load_runs("odd")}
    out = []
    for rule, name in (("reach_down", "crop_v2_icvl.pkl"), ("pose_action", "crop_v2_pose_action_icvl.pkl")):
        for r in pickle.loads((CACHE / name).read_bytes()):
            if r["now"] is not None:
                out.append({**r, "rule": rule, "source": "icvl" if r["seq"] in odd_ids else "icvl_even"})
    return out + pickle.loads((CACHE / "crop_rows_tune.pkl").read_bytes())


def weights(tr):
    w = np.ones(len(tr))
    src, y = np.array([r["source"] for r in tr]), np.array([r["y"] for r in tr])
    for s in set(src):
        for c in (0, 1):
            m = (src == s) & (y == c)
            if m.any():
                w[m] = 1000.0 / m.sum()
    return w


def fit(tr):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=4000).fit(np.stack([feats(r, "all") for r in tr]), [r["y"] for r in tr],
                                                        sample_weight=weights(tr))


def scores():
    """{rule: {(seq, event_id): score}}: out-of-fold for training sources, final model for ICVL even."""
    R = rows()
    out = {}
    for rule in RULES:
        rr = [r for r in R if r["rule"] == rule]
        train = [r for r in rr if r["source"] != "icvl_even"]
        seqs = sorted({r["seq"] for r in train})
        fold = {s: i % FOLDS for i, s in enumerate(seqs)}
        sc = {}
        for k in range(FOLDS):
            clf = fit([r for r in train if fold[r["seq"]] != k and r["y"] is not None])
            for r in train:
                if fold[r["seq"]] == k:
                    sc[(r["seq"], r["event_id"])] = float(clf.predict_proba(feats(r, "all")[None])[0, 1])
        clf = fit([r for r in train if r["y"] is not None])
        for r in rr:
            if r["source"] == "icvl_even":
                sc[(r["seq"], r["event_id"])] = float(clf.predict_proba(feats(r, "all")[None])[0, 1])
        out[rule] = sc
        print(rule, {s: sum(1 for r in train if r["source"] == s and r["y"] is not None) for s in sorted({r["source"] for r in train})},
              flush=True)
    return out


def make_filt(sc, thr):
    def filt(res, al):
        return dedup([a for a in al if a["event_type"] not in thr or sc[a["event_type"]].get((res.seq_id, a["event_id"]), 1.0)
                      >= thr[a["event_type"]]])
    return filt


def evaluate_all(pairs):
    sc = scores()
    data = S.data()
    runs = load_runs()
    odd, even = runs[1::2], runs[0::2]
    out = []
    for tr, tp in pairs:
        thr = {"reach_down": tr, "pose_action": tp}
        filt = make_filt(sc, thr)
        orig = S.partial
        S.partial = lambda fn, cfg, _f=filt: (lambda r, ds: _f(r, fn(r, ds, cfg=cfg)))
        try:
            row = S.score({"pose_action_cooldown_s": 15.0, "person_dedup_s": 0.0}, *data[:4])
        finally:
            S.partial = orig
        row["setting"] = thr
        row["icvl_odd_oof"] = evaluate(odd, NODEDUP, filt)
        row["icvl_even"] = evaluate(even, NODEDUP, filt)
        out.append(row)
        print(json.dumps(row), flush=True)
    (HERE / "crop_mixed.json").write_text(json.dumps(out, indent=1))


def export(tr, tp):
    from stage1_crop_check import CropCheck, weights_for
    R = rows()
    allsc = {}
    for rule, thr in (("reach_down", tr), ("pose_action", tp)):
        rr = [r for r in R if r["rule"] == rule]
        clf = fit([r for r in rr if r["source"] != "icvl_even" and r["y"] is not None])
        path = weights_for(rule)
        np.savez(path, coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
                 threshold=np.float32(thr), version=np.int32(2), dt=np.float32(DT),
                 trained_on=np.array(f"{rule} alerts: ICVL odd + littering train + smoking tune + IWDD train + Avenue tune"))
        check = CropCheck(path)
        allsc[rule] = {(r["seq"], r["event_id"]): check.score_embeddings([r["before"], r["now"], r["after"]]) for r in rr}
        print(path, len(allsc[rule]))
    # replay scores (final model; out-of-fold scores are what crop_mixed.json reports for the training sets)
    groups = defaultdict(dict)
    for rule, sc in allsc.items():
        for (seq, eid), v in sc.items():
            groups[(rule, "icvl" if seq.startswith("icvl/") else "tune")][(seq, eid)] = v
    for (rule, part), sc in groups.items():
        name = ("crop_scores" if rule == "reach_down" else f"crop_scores_{rule}") + ("_icvl.pkl" if part == "icvl" else "_tune.pkl")
        (CACHE / name).write_bytes(pickle.dumps(sc))


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        evaluate_all([(float(a), float(b)) for a, b in (x.split("/") for x in sys.argv[2:])])
    else:
        export(float(sys.argv[2]), float(sys.argv[3]))
