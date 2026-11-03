"""Skeleton-sequence action classifier as a Stage 1 cue (second-stage check on tracked people).

Idea: Stage 1 already runs YOLO26s-pose on every frame. For each tracked person (tracker raw_id), about once per
second we take the last ~1 s of that person's keypoints, normalise them (relative to the hip centre, divided by the
person's body height = median box height of the track so far), resample to K steps, add velocities and a few body
cues (box height vs the track's usual height, box aspect, whole-body speed), and score the window with a small
learned model trained to separate the littering ACTOR (person box matching the GT box) from bystanders and from
people in anomaly-free footage. A person whose score passes a threshold becomes a "pose_action" candidate.

Training data (tune half only):
  * littering tune clips: windows on the GT actor = 1, windows on other people = 0 (ambiguous overlaps dropped);
  * tune normal videos (anomaly-free): every window = 0 (optional, `use_normal`).
Clip-level (and video-level) 5-fold cross-validation: every littering clip and every normal video is scored by a
model that never saw it (out-of-fold, OOF). Avenue / smoking (never used for training) are scored by the model
trained on all folds. The threshold is calibrated on the OOF normal-video scores for a target false-alarm rate
(so FA on normal footage is in-sample for the threshold, but not for the model weights).

Causal: a window at time t uses observations with time <= t of the same track only. The method never reads GT;
labels are only used in `build()` (offline training).

Variants (`variant=`):
  lr_sum   logistic regression (numpy, L2) on ~200 per-window summary features
  mlp_sum  1-hidden-layer MLP (torch) on the same summary features
  gru_seq  tiny GRU (torch) on the K-step sequence of normalised keypoints + velocities
Each also has a "_lit" twin trained without the normal-video negatives.

Usage:  python exp/actions/pose_seq.py build     (extract windows, train all variants with CV, cache scores)
        python exp/actions/pose_seq.py eval      (score variants with exp_harness and the random control)
        python exp/actions/pose_seq.py timing    (CPU ms per call)
        python exp/actions/pose_seq.py diagnose  (within-clip AUC, matched-rate controls, littering-camera FA)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np

NB = Path("/Users/ianvarella/capstone_S25/notebooks")
for p in (NB, NB / "exp", NB / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp_harness import baseline, combine, make_event  # noqa: E402
from stage1_gate import GateConfig, iou  # noqa: E402

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

from stage1_pose_action import (ARMS, KP_CONF, WIN, TrackWindows, _norm_kpts, extract,  # noqa: E402,F401
                               window_features)

DEFAULTS = dict(variant="lr_sum", target_fa=1.0, thr=None, cooldown_s=5.0)
VARIANTS = ("lr_sum", "mlp_sum", "gru_seq", "lr_sum_lit", "mlp_sum_lit", "gru_seq_lit")


# --------------------------------------------------------------------------- models

class LR:
    def __init__(self, l2=1.0, iters=400, lr=0.5):
        self.l2, self.iters, self.lr = l2, iters, lr

    def fit(self, X, y, w):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-3
        Z = (X - self.mu) / self.sd
        n, d = Z.shape
        self.W, self.b = np.zeros(d), 0.0
        # full-batch gradient descent with momentum (convex, enough for ~200 dims)
        vW, vb = np.zeros(d), 0.0
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-(Z @ self.W + self.b)))
            g = (p - y) * w
            gW = Z.T @ g / w.sum() + self.l2 * self.W / n
            gb = g.sum() / w.sum()
            vW = 0.9 * vW - self.lr * gW
            vb = 0.9 * vb - self.lr * gb
            self.W += vW
            self.b += vb
        return self

    def predict(self, X):
        return 1 / (1 + np.exp(-(((X - self.mu) / self.sd) @ self.W + self.b)))


try:
    import torch.nn as _nn

    class GRUNet(_nn.Module):
        def __init__(self, d, hidden):
            super().__init__()
            self.gru = _nn.GRU(d, hidden, batch_first=True)
            self.head = _nn.Sequential(_nn.Dropout(0.3), _nn.Linear(hidden, 1))

        def forward(self, x):
            return self.head(self.gru(x)[1][-1])
except ImportError:   # torch missing: only the numpy LR variant works
    GRUNet = None


def _GRU(d, hidden):
    return GRUNet(d, hidden)


class TorchModel:
    def __init__(self, kind, epochs=60, hidden=32, wd=1e-3, seed=0):
        self.kind, self.epochs, self.hidden, self.wd, self.seed = kind, epochs, hidden, wd, seed

    def _net(self, d):
        import torch.nn as nn

        if self.kind == "mlp":
            return nn.Sequential(nn.Linear(d, self.hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(self.hidden, 1))

        return _GRU(d, self.hidden)

    def fit(self, X, y, w):
        import torch

        torch.manual_seed(self.seed)
        ax = (0,) if X.ndim == 2 else (0, 1)
        self.mu, self.sd = X.mean(ax), X.std(ax) + 1e-3
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt, wt = torch.tensor(y, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)
        self.net = self._net(X.shape[-1])
        opt = torch.optim.AdamW(self.net.parameters(), lr=3e-3, weight_decay=self.wd)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            self.net.train()
            for idx in torch.randperm(len(Xt), generator=g).split(128):
                logit = self.net(Xt[idx]).squeeze(-1)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, yt[idx], reduction="none")
                        * wt[idx]).sum() / wt[idx].sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    def predict(self, X):
        import torch

        with torch.no_grad():
            Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            return torch.sigmoid(self.net(Xt).squeeze(-1)).numpy().astype(np.float64)


def new_model(variant):
    base = variant.replace("_lit", "")
    return {"lr_sum": lambda: LR(l2=5.0), "mlp_sum": lambda: TorchModel("mlp"),
            "gru_seq": lambda: TorchModel("gru", epochs=40)}[base]()


def feats(windows, variant):
    key = "seq" if "seq" in variant else "sum"
    return np.stack([w[key] for w in windows]) if windows else np.zeros((0, 1), np.float32)


# --------------------------------------------------------------------------- offline build (labels used here only)

def _label(w, g):
    gts = g.gt.get(w["index"], [])
    best = max((iou(w["box"], b) for b in gts), default=0.0)
    return 1 if best >= 0.4 else (0 if best < 0.15 else -1)


def build(splits=None, folds=5, variants=VARIANTS):
    from compare_models import model_splits

    splits = splits or model_splits("s", heldout=False)
    t0 = time.time()
    win = {ds: {r.seq_id: extract(r) for _, r in splits[ds]} for ds in splits}
    print(f"windows extracted in {time.time() - t0:.1f}s:", {ds: sum(map(len, v.values())) for ds, v in win.items()},
          flush=True)

    # folds: littering clips stratified by action, normal videos round-robin
    fold = {}
    per_tag = defaultdict(int)
    for g, r in sorted(splits["littering"], key=lambda x: x[1].seq_id):
        fold[r.seq_id] = per_tag[g.tag] % folds
        per_tag[g.tag] += 1
    for i, (_, r) in enumerate(sorted(splits["normal"], key=lambda x: x[1].seq_id)):
        fold[r.seq_id] = i % folds

    rows = []   # (dataset, seq_id, window idx, label, fold)
    for g, r in splits["littering"]:
        for i, w in enumerate(win["littering"][r.seq_id]):
            y = _label(w, g)
            if y >= 0:
                rows.append(("littering", r.seq_id, i, y, fold[r.seq_id]))
    for _, r in splits["normal"]:
        for i, _w in enumerate(win["normal"][r.seq_id]):
            rows.append(("normal", r.seq_id, i, 0, fold[r.seq_id]))
    ys = np.array([x[3] for x in rows])
    print(f"training rows: {len(rows)}, littering pos {int(ys.sum())}, "
          f"littering neg {sum(1 for x in rows if x[0] == 'littering' and x[3] == 0)}, "
          f"normal neg {sum(1 for x in rows if x[0] == 'normal')}", flush=True)

    scores = {}
    cv_report = {}
    for v in variants:
        t1 = time.time()
        use = [x for x in rows if x[0] == "littering" or not v.endswith("_lit")]
        X = feats([win[x[0]][x[1]][x[2]] for x in use], v)
        y = np.array([x[3] for x in use], float)
        f = np.array([x[4] for x in use])
        lit = np.array([x[0] == "littering" for x in use])
        # balance: littering positives and negatives weighted to equal mass; normal negatives = littering negatives
        w = np.ones(len(y))
        npos, nneg_l, nneg_n = y.sum(), ((y == 0) & lit).sum(), (~lit).sum()
        w[y == 1] = 1.0 / npos
        w[(y == 0) & lit] = (0.5 if nneg_n else 1.0) / nneg_l
        if nneg_n:
            w[~lit] = 0.5 / nneg_n
        w *= len(y)
        s = {ds: {sid: np.full(len(ws), np.nan) for sid, ws in win[ds].items()} for ds in win}
        oof = np.zeros(len(y))
        for k in range(folds):
            tr = f != k
            mdl = new_model(v).fit(X[tr], y[tr], w[tr])
            # OOF on every littering clip / normal video in fold k (all windows, including ambiguous ones)
            for ds in ("littering", "normal"):
                for sid, ws in win[ds].items():
                    if fold.get(sid) == k and ws:
                        s[ds][sid] = mdl.predict(feats(ws, v))
            oof[~tr] = mdl.predict(X[~tr])
        full = new_model(v).fit(X, y, w)
        for ds in ("avenue", "smoking"):
            for sid, ws in win[ds].items():
                if ws:
                    s[ds][sid] = full.predict(feats(ws, v))
        if v.endswith("_lit"):     # normal videos were not in training at all: score with the full model
            for sid, ws in win["normal"].items():
                if ws:
                    s["normal"][sid] = full.predict(feats(ws, v))
        (CACHE / f"model_{v}.pkl").write_bytes(pickle.dumps(full))
        scores[v] = s
        cv_report[v] = {"oof_auc_littering_actor_vs_bystander": round(_auc(oof[lit], y[lit]), 3),
                        "oof_auc_actor_vs_normal": round(_auc(np.r_[oof[lit & (y == 1)], oof[~lit]],
                                                              np.r_[np.ones((lit & (y == 1)).sum()), np.zeros((~lit).sum())]), 3)
                        if (~lit).any() else None,
                        "train_s": round(time.time() - t1, 1)}
        print(v, cv_report[v], flush=True)
    meta = {sid: [(w["t"], w["raw_id"], w["box"]) for w in ws] for ds in win for sid, ws in win[ds].items()}
    (CACHE / "scores.pkl").write_bytes(pickle.dumps({"scores": scores, "meta": meta, "fold": fold}))
    (CACHE / "cv_report.json").write_text(json.dumps(cv_report, indent=1))
    n_calls = {ds: sum(map(len, win[ds].values())) for ds in win}
    minutes = {ds: sum(r.duration_s for _, r in splits[ds]) / 60 for ds in splits}
    (CACHE / "calls.json").write_text(json.dumps({ds: {"calls": n_calls[ds], "minutes": round(minutes[ds], 2),
                                                       "calls_per_min": round(n_calls[ds] / minutes[ds], 1)}
                                                  for ds in win}, indent=1))
    return scores


def _auc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    npos = y.sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


# --------------------------------------------------------------------------- method

_STORE = {}


def _store():
    if "s" not in _STORE:
        _STORE["s"] = pickle.loads((CACHE / "scores.pkl").read_bytes())
    return _STORE["s"]


def _events_from(meta, probs, thr, cooldown_s):
    last, out = {}, []
    for (t, rid, box), p in zip(meta, probs):
        if p >= thr and t - last.get(rid, -1e9) >= cooldown_s:
            last[rid] = t
            out.append(make_event("pose_action", t, box, score=round(float(p), 3)))
    return out


def calibrate(variant, target_fa, cooldown_s=DEFAULTS["cooldown_s"]):
    """Threshold giving about target_fa candidates per minute on the (OOF-scored) tune normal videos."""
    st = _store()
    from compare_models import model_splits  # noqa: F401  (durations come from the cached calls.json)
    minutes = json.loads((CACHE / "calls.json").read_text())["normal"]["minutes"]
    sn = st["scores"][variant]["normal"]
    cand = np.unique(np.concatenate([v for v in sn.values() if len(v)]))[::-1]
    lo = 1.0
    for thr in np.r_[cand, 0.0]:
        n = sum(len(_events_from(st["meta"][sid], p, thr, cooldown_s)) for sid, p in sn.items())
        if n / minutes > target_fa:
            break
        lo = thr
    return float(lo)


def make_method(variant=DEFAULTS["variant"], target_fa=DEFAULTS["target_fa"], thr=DEFAULTS["thr"],
                cooldown_s=DEFAULTS["cooldown_s"]):
    """Cached path: a sequence seen in build() uses its cached (OOF where it was trained on) scores.
    Live path (any other sequence): windows are extracted causally and scored by the full model."""
    if thr is None:
        thr = calibrate(variant, target_fa, cooldown_s)
    model = None

    def method(result, dataset):
        nonlocal model
        st = _store()
        sc = st["scores"][variant].get(dataset, {})
        if result.seq_id in sc:
            return _events_from(st["meta"][result.seq_id], sc[result.seq_id], thr, cooldown_s)
        if model is None:
            model = pickle.loads((CACHE / f"model_{variant}.pkl").read_bytes())
        ws = extract(result)
        probs = model.predict(feats(ws, variant)) if ws else []
        return _events_from([(w["t"], w["raw_id"], w["box"]) for w in ws], probs, thr, cooldown_s)

    method.thr = thr
    return method


# --------------------------------------------------------------------------- evaluation

def evaluate(variants=("lr_sum", "mlp_sum", "gru_seq", "lr_sum_lit", "gru_seq_lit"), targets=(0.5, 1.0, 2.0, 4.0),
             seeds=10):
    import pandas as pd
    import exp_harness as H
    from compare_models import model_splits
    from random_control import lift

    T = model_splits("s", heldout=False)
    calls = json.loads((CACHE / "calls.json").read_text())
    def memo(m):
        cache = {}

        def run(r, ds):
            if (ds, r.seq_id) not in cache:
                cache[(ds, r.seq_id)] = m(r, ds)
            return [dict(e) for e in cache[(ds, r.seq_id)]]
        return run

    base = memo(partial(baseline, cfg=GateConfig()))
    no_reach = memo(partial(baseline, cfg=GateConfig(rules=tuple(r for r in GateConfig().rules if r != "reach_down"))))
    reach_only = memo(partial(baseline, cfg=GateConfig(rules=("reach_down",))))
    rows = []

    def add(name, mode, m, extra=None):
        row = H.score(m, splits=T, name=name)
        mc, cc, cs = lift(m, T["littering"], seeds=seeds)
        row.update(mode=mode, lit_lift=round(mc - cc, 1), lit_control=round(cc, 1), lit_control_sd=round(cs, 1),
                   **(extra or {}))
        rows.append(row)
        print({k: row[k] for k in ("method", "mode", "littering_event_%", "littering_crop_%", "lit_lift", "avenue_event_%",
                                   "avenue_crop_%", "smoking_event_%", "smoking_crop_%", "normal_FA_per_min")}, flush=True)

    add("baseline", "reference", base)
    add("reach_down only", "reference", reach_only)
    add("baseline minus reach_down", "reference", no_reach)
    for v in variants:
        for tgt in targets:
            m = make_method(variant=v, target_fa=tgt)
            ex = {"variant": v, "target_fa": tgt, "thr": round(m.thr, 4)}
            add(f"{v}@{tgt}", "alone", m, ex)
            add(f"{v}@{tgt}", "add-on", combine(base, m), ex)
            add(f"{v}@{tgt}", "replace reach_down", combine(no_reach, m), ex)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE.parent / "pose_seq_results.csv", index=False)
    (CACHE / "calls_summary.json").write_text(json.dumps(calls, indent=1))
    return df


def timing(variants=("lr_sum", "mlp_sum", "gru_seq"), n=500):
    """CPU ms per call = window feature extraction + one model forward, single-threaded."""
    import torch
    from compare_models import model_splits

    torch.set_num_threads(1)
    T = model_splits("s", heldout=False)
    r = max((r for _, r in T["avenue"]), key=lambda r: sum(len(o.persons) for o in r.observations))
    tw = TrackWindows()
    # warm the extractor on a prefix, then time feature extraction for real calls
    ws, fe = [], []
    for o in r.observations:
        t0 = time.perf_counter()
        out = tw.step(o)
        dt = (time.perf_counter() - t0) * 1000
        if out:
            fe.append(dt / len(out))
            ws += out
    out = {"feature_ms_per_call(incl. per-frame bookkeeping)": round(float(np.median(fe)), 3)}
    for v in variants:
        mdl = pickle.loads((CACHE / f"model_{v}.pkl").read_bytes())
        X = feats(ws[:n], v)
        for i in range(20):
            mdl.predict(X[i:i + 1])
        t0 = time.perf_counter()
        for i in range(len(X)):
            mdl.predict(X[i:i + 1])
        out[f"{v}_model_ms_per_call"] = round((time.perf_counter() - t0) * 1000 / len(X), 3)
    (CACHE / "timing.json").write_text(json.dumps(out, indent=1))
    print(out)
    return out


# --------------------------------------------------------------------------- diagnostics (honest skill checks)

def control_norep(method, seed):
    """Like random_control.control, but events sharing a timestamp go to DISTINCT random people. The stock control
    samples with replacement, so "fire once on every person" alone gets a +22 lift there (it is +4 here)."""
    import random
    rng = random.Random(seed)

    def run(r, ds):
        by_t = {round(o.t, 2): o for o in r.observations}
        groups = defaultdict(int)
        for e in method(r, ds):
            groups[round(e["timestamp_s"], 2)] += 1
        out = []
        for t, n in groups.items():
            o = by_t.get(t) or min(r.observations, key=lambda o: abs(o.t - t))
            if o.persons:
                ps = list(o.persons)
                rng.shuffle(ps)
                out += [make_event("control", o.t, ps[k % len(ps)]["box"]) for k in range(n)]
        return out
    return run


def fair_lift(method, data, seeds=10):
    from random_control import crop
    m = crop(method, data)
    c = [crop(control_norep(method, s), data) for s in range(seeds)]
    return m, float(np.mean(c)), float(np.std(c))


def diagnose(configs=(("mlp_sum", 1.0), ("mlp_sum", 2.0), ("gru_seq", 2.0), ("gru_seq", 4.0), ("lr_sum", 4.0))):
    from compare_models import model_splits
    from random_control import lift

    T = model_splits("s", heldout=False)
    L = T["littering"]
    G = {r.seq_id: g for g, r in L}
    st = _store()
    W = {r.seq_id: extract(r) for _, r in L}
    labs = {sid: np.array([_label(w, G[sid]) for w in ws]) for sid, ws in W.items()}
    out = {}

    def within_auc(score_fn):
        num = den = 0.0
        for sid, ws in W.items():
            y, sc = labs[sid], score_fn(sid, ws)
            a, b = sc[y == 1], sc[y == 0]
            num += (a[:, None] > b[None]).sum() + 0.5 * (a[:, None] == b[None]).sum()
            den += len(a) * len(b)
        return round(num / den, 3)

    out["within_clip_auc_actor_vs_bystander"] = {v: within_auc(lambda sid, ws: st["scores"][v]["littering"][sid])
                                                 for v in VARIANTS}
    out["within_clip_auc_actor_vs_bystander"]["trivial: box height"] = within_auc(
        lambda sid, ws: np.array([w["box"][3] - w["box"][1] for w in ws]))
    mins = sum(r.duration_s for _, r in L) / 60
    nmin = sum(r.duration_s for _, r in T["normal"]) / 60

    def fire_once(r, ds):
        ws = extract(r)
        return _events_from([(w["t"], w["raw_id"], w["box"]) for w in ws], np.ones(len(ws)), 0.5, DEFAULTS["cooldown_s"])

    def tallest(r, ds):
        by_t = {o.t: o for o in r.observations}
        ws = extract(r)
        sc = [(w["box"][3] - w["box"][1]) / max(p["box"][3] - p["box"][1] for p in by_t[w["t"]].persons) for w in ws]
        return _events_from([(w["t"], w["raw_id"], w["box"]) for w in ws], np.array(sc), 1.0, DEFAULTS["cooldown_s"])

    rows = {}
    refs = [("reach_down only", partial(baseline, cfg=GateConfig(rules=("reach_down",)))),
            ("baseline", partial(baseline, cfg=GateConfig())),
            ("fire once on every person", fire_once), ("tallest person in frame", tallest)]
    refs += [(f"{v}@{tgt} alone", make_method(variant=v, target_fa=tgt)) for v, tgt in configs]
    for name, m in refs:
        mc, c, _ = fair_lift(m, L)
        sc, scc, _ = lift(m, L, seeds=10)
        ev = {sid: m(r, "littering") for sid, r in ((r.seq_id, r) for _, r in L)}
        nb = 0
        for g, r in L:
            for e in ev[r.seq_id]:
                idx = min(r.observations, key=lambda o: abs(o.t - e["timestamp_s"])).index
                if max((iou(e["roi_box"], b) for b in g.gt.get(idx, [])), default=0.0) < 0.15:
                    nb += 1
        rows[name] = {"crop_%": mc, "stock_lift": round(sc - scc, 1), "norep_lift": round(mc - c, 1),
                      "cand_per_clip": round(sum(map(len, ev.values())) / len(L), 2),
                      "littering_bystander_cands_per_min": round(nb / mins, 1),
                      "normal_FA_per_min": round(sum(len(m(r, "normal")) for _, r in T["normal"]) / nmin, 2)}
        print(name, rows[name], flush=True)
    out["matched_rate_controls"] = rows
    (CACHE / "diagnostics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["within_clip_auc_actor_vs_bystander"], indent=1))
    return out


if __name__ == "__main__":
    import pose_seq as _self   # run through the importable module so pickled models load from anywhere
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    build, evaluate, timing, diagnose = _self.build, _self.evaluate, _self.timing, _self.diagnose
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "eval":
        import pandas as pd
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 30)
        print(evaluate())
    elif cmd == "timing":
        timing()
    elif cmd == "diagnose":
        diagnose()
