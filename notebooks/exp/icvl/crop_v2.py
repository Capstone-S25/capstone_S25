"""Crop check v2: does seeing the person before and after the reach help? A drop is a change (something in the hand,
then not); one crop at the alert moment cannot show that.

For every reach_down alert on ICVL (deck settings, dedup off: the check runs before it) the person is cropped at
t - 1 s, t and t + 1 s (same raw track; the box grown like the alert's roi, which adds the hand square) and embedded
with MobileCLIP2-S2 (same encoder as stage1_crop_check). Classifiers (logistic regression, trained on ICVL odd,
scored on even) on:
  now        e(t)                                  (= the current crop check, as a reference)
  now+diff   e(t), e(t+1) - e(t-1)
  all        e(t-1), e(t), e(t+1)
Holding an alert 1 s for the "after" crop is the latency cost.
Usage (from notebooks/):
  ../.venv/bin/python -I exp/icvl/crop_v2.py embed ENCODER [RULE]  -> bench_cache/models/s/crop_v2_icvl.pkl
                                                     (RULE other than reach_down: crop_v2_<RULE>_icvl.pkl)
  ../.venv/bin/python exp/icvl/crop_v2.py score              -> exp/icvl/crop_v2.json
  ../.venv/bin/python exp/icvl/crop_v2.py export x [RULE]    -> weights/crop_check[_RULE].npz + replay scores
"""
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(HERE)]

from fpr_sweep import evaluate  # noqa: E402
from hand_probe import NODEDUP, dedup, label  # noqa: E402
from inventory import ICVL, video_for  # noqa: E402
from score import labels_for, load_runs  # noqa: E402
from stage1_crop_check import crop  # noqa: E402
from stage1_gate import intersection_over, replay_rules  # noqa: E402

RULE = sys.argv[3] if len(sys.argv) > 3 else "reach_down"   # embed / export: which rule's alerts
OUT = NB / "bench_cache" / "models" / "s" / ("crop_v2_icvl.pkl" if RULE == "reach_down" else f"crop_v2_{RULE}_icvl.pkl")
DT = 1.0


def boxes_around(res, a):
    """The alert's roi at t, and the same person's box (grown like the roi) at t - DT and t + DT (None if not seen)."""
    t, roi = a["timestamp_s"], a["roi_box"]
    o = min(res.observations, key=lambda o: abs(o.t - t))
    p = max(o.persons, key=lambda q: intersection_over(q["box"], roi), default=None)
    if p is None:
        return None
    pb = p["box"]
    grow = [roi[0] - pb[0], roi[1] - pb[1], roi[2] - pb[2], roi[3] - pb[3]]
    out = {}
    for name, tt in (("before", t - DT), ("after", t + DT)):
        best = None
        for o2 in res.observations:
            if abs(o2.t - tt) <= 0.4:
                for q in o2.persons:
                    if q["raw_id"] == p["raw_id"] and (best is None or abs(o2.t - tt) < best[0]):
                        best = (abs(o2.t - tt), o2.index, [q["box"][i] + grow[i] for i in range(4)])
        out[name] = best[1:] if best else None
    out["now"] = (a["frame_index"], roi)
    return out


def embed(encoder):
    import open_clip
    import torch
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, prep = open_clip.create_model_and_transforms(encoder)
    model = model.eval().to(dev)
    rows = []
    for g, res in load_runs():
        alerts = [a for a in replay_rules(res, NODEDUP)[0] if a["event_type"] == RULE]
        if not alerts:
            continue
        eps, people = labels_for(g.tag, g.seq.native_fps)
        cap = cv2.VideoCapture(str(video_for(ICVL / g.tag)))
        for a in alerts:
            bx = boxes_around(res, a)
            if bx is None:
                continue
            emb = {}
            for name, v in bx.items():
                if v is None:
                    emb[name] = None
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, v[0])   # observation indices are native frame numbers
                ok, img = cap.read()
                c = crop(img, v[1]) if ok else None
                if c is None or c.size == 0:
                    emb[name] = None
                    continue
                with torch.no_grad():
                    f = model.encode_image(prep(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))[None].to(dev))
                emb[name] = (f / f.norm()).cpu().numpy()[0].astype(np.float32)
            rows.append({"seq": res.seq_id, "event_id": a["event_id"], "y": label(a, eps, people), **emb})
        print(res.seq_id, len(rows), flush=True)
    OUT.write_bytes(pickle.dumps(rows))


def feats(r, kind):
    now = r["now"]
    b = r["before"] if r["before"] is not None else now
    af = r["after"] if r["after"] is not None else now
    if kind == "now":
        return now
    if kind == "now+diff":
        return np.concatenate([now, af - b])
    return np.concatenate([b, now, af])


def score():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rows = [r for r in pickle.loads(OUT.read_bytes()) if r["now"] is not None]
    runs = load_runs()
    odd_ids = {r.seq_id for _, r in runs[1::2]}
    even = runs[0::2]
    out = {}
    for kind in ("now", "now+diff", "all"):
        tr = [r for r in rows if r["seq"] in odd_ids and r["y"] is not None]
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(
            np.stack([feats(r, kind) for r in tr]), [r["y"] for r in tr])
        p = {(r["seq"], r["event_id"]): float(clf.predict_proba(feats(r, kind)[None])[0, 1]) for r in rows if r["seq"] not in odd_ids}
        te = [r for r in rows if r["seq"] not in odd_ids and r["y"] is not None]
        res_k = {"auc_even": round(roc_auc_score([r["y"] for r in te], [p[(r["seq"], r["event_id"])] for r in te]), 3), "even": []}
        for thr in (0.0, 0.2, 0.3, 0.4, 0.5):
            def filt(res, alerts, thr=thr):
                return dedup([a for a in alerts if a["event_type"] != "reach_down" or p.get((res.seq_id, a["event_id"]), 1.0) >= thr])
            res_k["even"].append({"thr": thr, **evaluate(even, NODEDUP, filt)})
        out[kind] = res_k
        print(kind, json.dumps(res_k), flush=True)
    (HERE / "crop_v2.json").write_text(json.dumps(out, indent=1))


def export(thr=0.2):
    """Weights (trained on ICVL odd only) and ICVL replay scores for RULE: reach_down -> weights/crop_check.npz +
    crop_scores_icvl.pkl; other rules -> weights/crop_check_<RULE>.npz + crop_scores_<RULE>_icvl.pkl."""
    from sklearn.linear_model import LogisticRegression
    from stage1_crop_check import CropCheck, weights_for
    rows = [r for r in pickle.loads(OUT.read_bytes()) if r["now"] is not None]
    odd_ids = {r.seq_id for _, r in load_runs("odd")}
    tr = [r for r in rows if r["seq"] in odd_ids and r["y"] is not None]
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(
        np.stack([feats(r, "all") for r in tr]), [r["y"] for r in tr])
    path = weights_for(RULE)
    np.savez(path, coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
             threshold=np.float32(thr), version=np.int32(2), dt=np.float32(DT),
             trained_on=np.array(f"ICVL odd-half {RULE} alerts, crops at t - 1 s, t, t + 1 s"))
    check = CropCheck(path)
    scores = {(r["seq"], r["event_id"]): check.score_embeddings([r["before"], r["now"], r["after"]]) for r in rows}
    ref = {(r["seq"], r["event_id"]): float(clf.predict_proba(feats(r, "all")[None])[0, 1]) for r in rows[:50]}
    assert all(abs(scores[k] - v) < 1e-4 for k, v in ref.items()), "CropCheck disagrees with the sklearn classifier"
    name = "crop_scores_icvl.pkl" if RULE == "reach_down" else f"crop_scores_{RULE}_icvl.pkl"
    (OUT.parent / name).write_bytes(pickle.dumps(scores))
    print(path, len(scores), "scores")


if __name__ == "__main__":
    {"embed": lambda: embed(sys.argv[2]), "score": score, "export": export}[sys.argv[1]]()
