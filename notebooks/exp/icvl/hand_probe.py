"""Can the pixels of an alert's crop tell a real littering / smoking alert from a person carrying something or on the
phone? Frozen MobileCLIP2-S2 image features (public timm/MobileCLIP2-S2-OpenCLIP weights, from Yuchuan's CPU bundle's
offline cache) + logistic regression, trained on ICVL odd videos, scored on even.

Alerts: reach_down and pose_action under the deck's configuration (one_per_person.CFG).
  positive  on a littering incident (+-3 s) or during a smoking incident
  negative  on a labelled person with no incident in the video, or on nobody labelled
  (alerts on an incident person outside the incident are left out of training, but scored on the even half)
Crop: the alert's roi_box (person box, plus the hand square for reach_down) padded 15 %.
Usage (from notebooks/):
  ../.venv/bin/python exp/icvl/hand_probe.py embed BUNDLE   -> bench_cache/models/s/hand_probe_icvl.pkl
  ../.venv/bin/python exp/icvl/hand_probe.py score          -> exp/icvl/hand_probe.json (cut-off grid on even)
  ../.venv/bin/python exp/icvl/hand_probe.py export         -> weights/crop_check.npz + crop_scores_icvl.pkl
  ../.venv/bin/python exp/icvl/hand_probe.py select         -> exp/icvl/hand_probe_select.json (cut-offs chosen on
                                                               odd out-of-fold scores, even scored once)
"""
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(HERE)]

from fpr_sweep import evaluate  # noqa: E402
from inventory import ICVL, video_for  # noqa: E402
from one_per_person import CFG  # noqa: E402
from score import EVENTS, TOL_S, WINDOW_S, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import intersection_over, iou, replay_rules  # noqa: E402

OUT = NB / "bench_cache" / "models" / "s" / "hand_probe_icvl.pkl"
RULES = ("reach_down", "pose_action")
NODEDUP = replace(CFG, person_dedup_s=0.0)   # the filter runs before the gate's dedup, so replay without it
SNAPSHOT = "data/models/huggingface/hub/models--timm--MobileCLIP2-S2-OpenCLIP/snapshots/ac6b37c8fc40b62b623d09ed228a6f0c9bc29fe6"


def label(a, eps, people):
    for ev in ("littering", "smoking"):
        if any(on_track(a, people[ep["pid"]], ep["t0"], ep["t1"], WINDOW_S.get(ev, TOL_S)) for ep in eps.get(ev, [])):
            return 1
    pos = {ep["pid"] for ev in EVENTS for ep in eps.get(ev, [])}
    on = [pid for pid, tr in people.items() if on_track(a, tr)]
    return 0 if not on or on[0] not in pos else None


def crop(img, box, pad=0.15):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    X1, Y1 = int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h))
    X2, Y2 = int(min(img.shape[1], x2 + pad * w)), int(min(img.shape[0], y2 + pad * h))
    return img[Y1:Y2, X1:X2]


def embed(bundle):
    import open_clip
    import torch
    from PIL import Image
    model, _, prep = open_clip.create_model_and_transforms(f"local-dir:{Path(bundle) / SNAPSHOT}")
    model.eval()
    rows = pickle.loads(OUT.read_bytes()) if OUT.exists() else []
    done = {(r["seq"], r["event_id"]) for r in rows}
    for g, res in load_runs():
        alerts = [a for a in replay_rules(res, NODEDUP)[0] if a["event_type"] in RULES and (res.seq_id, a["event_id"]) not in done]
        if not alerts:
            continue
        eps, people = labels_for(g.tag, g.seq.native_fps)
        cap = cv2.VideoCapture(str(video_for(ICVL / g.tag)))
        for a in alerts:
            y = label(a, eps, people)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(a["timestamp_s"] * g.seq.native_fps)))
            ok, img = cap.read()
            if not ok:
                continue
            c = crop(img, a["roi_box"])
            with torch.no_grad():
                f = model.encode_image(prep(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))[None])
            rows.append({"seq": res.seq_id, "event_id": a["event_id"], "rule": a["event_type"], "y": y,
                         "f": (f / f.norm()).numpy()[0].astype(np.float32)})
        print(res.seq_id, len(rows), flush=True)
    OUT.write_bytes(pickle.dumps(rows))


def fit(rows):
    from sklearn.linear_model import LogisticRegression
    X = np.stack([np.concatenate([r["f"], [r["rule"] == "reach_down"]]) for r in rows])
    y = np.array([r["y"] for r in rows])
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(X, y)


def feats(r):
    return np.concatenate([r["f"], [r["rule"] == "reach_down"]])[None]


def oof(rows, folds=5):
    """Out-of-fold probe scores for rows (grouped by video), so thresholds can be chosen without the test half."""
    from sklearn.model_selection import GroupKFold
    lab = [r for r in rows if r["y"] is not None]
    seqs = sorted({r["seq"] for r in rows})
    fold = {s: i % folds for i, s in enumerate(seqs)}
    out = {}
    for k in range(folds):
        clf = fit([r for r in lab if fold[r["seq"]] != k])
        for r in rows:
            if fold[r["seq"]] == k:
                out[(r["seq"], r["event_id"])] = float(clf.predict_proba(feats(r))[0, 1])
    return out


def dedup(alerts, window_s=CFG.person_dedup_s):
    """The gate's person dedup (RuleEngine._dedup with dedup_per_rule off), applied after the probe filter."""
    sent, kept = [], []
    for e in sorted(alerts, key=lambda a: a["timestamp_s"]):
        t, box = e["timestamp_s"], e["roi_box"]
        sent = [x for x in sent if t - x[0] <= window_s]
        if any(ent == e["entity_id"] or iou(box, b) >= 0.5 or max(intersection_over(box, b), intersection_over(b, box)) >= 0.7
               for _, ent, b in sent):
            continue
        sent.append((t, e["entity_id"], box))
        kept.append(e)
    return kept


def make_filt(p, thr):
    """Score alerts replayed with NODEDUP: drop reach_down / pose_action below their cut-off, then dedup."""
    def filt(res, alerts):
        return dedup([a for a in alerts if a["event_type"] not in RULES
                      or p.get((res.seq_id, a["event_id"]), 1.0) >= thr[a["event_type"]]])
    return filt


def caught(m, ev):
    return int(m[ev].split("/")[0])


def select():
    """Choose (reach_down, pose_action) cut-offs on the odd half (out-of-fold scores): the lowest per-person FPR that
    keeps >= 98 % of the littering and >= 95 % of the smoking incidents caught. Then score the even half once."""
    rows = pickle.loads(OUT.read_bytes())
    runs = load_runs()
    odd, even = runs[1::2], runs[0::2]
    odd_ids = {r.seq_id for _, r in odd}
    p_odd = oof([r for r in rows if r["seq"] in odd_ids])
    base = evaluate(odd, NODEDUP, make_filt({}, {r: 0.0 for r in RULES}))
    best, grid = None, []
    for tr in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6):
        for tp in (0.0, 0.1, 0.2, 0.3, 0.4):
            thr = {"reach_down": tr, "pose_action": tp}
            m = evaluate(odd, NODEDUP, make_filt(p_odd, thr))
            grid.append({"thr": thr, **m})
            ok = caught(m, "littering") >= 0.98 * caught(base, "littering") and caught(m, "smoking") >= 0.95 * caught(base, "smoking")
            if ok and (best is None or m["fpr_%"] < best[1]["fpr_%"]):
                best = (thr, m)
    clf = fit([r for r in rows if r["seq"] in odd_ids and r["y"] is not None])
    p_even = {(r["seq"], r["event_id"]): float(clf.predict_proba(feats(r))[0, 1]) for r in rows if r["seq"] not in odd_ids}
    out = {"odd_base": base, "chosen": best[0], "odd_chosen": best[1],
           "even_base": evaluate(even, NODEDUP, make_filt({}, {r: 0.0 for r in RULES})),
           "even_chosen": evaluate(even, NODEDUP, make_filt(p_even, best[0])), "odd_grid": grid}
    (HERE / "hand_probe_select.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "odd_grid"}, indent=1))


def export(thr=0.3):
    """weights/crop_check.npz (trained on ICVL odd only, so the even half stays held out) and the precomputed scores
    of every embedded ICVL alert for replays (bench_cache/models/s/crop_scores_icvl.pkl)."""
    from stage1_crop_check import DEFAULT_WEIGHTS, CropCheck
    rows = pickle.loads(OUT.read_bytes())
    odd_ids = {r.seq_id for _, r in load_runs()[1::2]}
    clf = fit([r for r in rows if r["seq"] in odd_ids and r["y"] is not None])
    np.savez(DEFAULT_WEIGHTS, coef=clf.coef_[0].astype(np.float32), intercept=np.float32(clf.intercept_[0]),
             threshold=np.float32(thr), trained_on=np.array("ICVL odd-half reach_down + pose_action alerts"))
    check = CropCheck()
    scores = {(r["seq"], r["event_id"]): check.score_embedding(r["f"], r["rule"]) for r in rows}
    ref = {(r["seq"], r["event_id"]): float(clf.predict_proba(feats(r))[0, 1]) for r in rows[:50]}
    assert all(abs(scores[k] - v) < 1e-4 for k, v in ref.items()), "CropCheck disagrees with the sklearn probe"
    (OUT.parent / "crop_scores_icvl.pkl").write_bytes(pickle.dumps(scores))
    print(DEFAULT_WEIGHTS, len(scores), "scores")


def score():
    from sklearn.metrics import roc_auc_score
    rows = pickle.loads(OUT.read_bytes())
    runs = load_runs()
    odd_ids = {r.seq_id for _, r in runs[1::2]}
    train = [r for r in rows if r["seq"] in odd_ids and r["y"] is not None]
    every = [r for r in rows if r["seq"] not in odd_ids]
    test = [r for r in every if r["y"] is not None]
    clf = fit(train)
    p = {(r["seq"], r["event_id"]): float(clf.predict_proba(feats(r))[0, 1]) for r in every}
    out = {"train": {"n": len(train), "pos": int(sum(r["y"] for r in train))},
           "test": {"n": len(test), "pos": int(sum(r["y"] for r in test))},
           "auc_even": {rule: round(roc_auc_score([r["y"] for r in test if r["rule"] == rule],
                                                 [p[(r["seq"], r["event_id"])] for r in test if r["rule"] == rule]), 3)
                        for rule in RULES}, "even": []}
    even = runs[0::2]
    for thr in [(0.0, 0.0), (0.3, 0.0), (0.4, 0.0), (0.5, 0.0), (0.4, 0.1), (0.4, 0.2), (0.5, 0.1), (0.5, 0.2)]:
        def filt(res, alerts, thr=dict(zip(RULES, thr))):
            return [a for a in alerts if a["event_type"] not in RULES or p.get((res.seq_id, a["event_id"]), 1.0) >= thr[a["event_type"]]]
        out["even"].append({"probe_thr": dict(zip(RULES, thr)), **evaluate(even, CFG, filt)})
        print(json.dumps(out["even"][-1]), flush=True)
    (HERE / "hand_probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "even"}, indent=1))


if __name__ == "__main__":
    {"embed": lambda: embed(sys.argv[2]), "score": score, "select": select, "export": export}[sys.argv[1]]()
