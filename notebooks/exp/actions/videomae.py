"""VideoMAE (Kinetics-400) action cue on tracked-person tubes, as a second-stage check for Stage 1.

How it works
* Tracks: the same TrackStitcher ids the rule engine uses (RuleEngine with no rules, replayed on cached observations).
* Schedule: each tracked person is checked at most once per `period_s` (1 s), once the track has `min_hist_s` of history.
* Tube: the person's last `tube_s` seconds (1.6 s at >= 5 fps; 4 s on the ~1 fps smoking camera), only frames <= t.
  16 frames are sampled evenly from that history (frames are repeated when there are fewer than 16). Each frame is cut
  as a square of side `crop_scale` x the largest person-box side in the tube, centred on that frame's box (so the crop
  follows the person; arm and body motion stays visible), resized to 224, ImageNet-normalised, RGB.
* Model: MCG-NJU/videomae-base-finetuned-kinetics. The cue is the softmax mass on a class group (see GROUPS).
* Cache: `cache/videomae/<dataset>.pkl` holds every scheduled call: (seq_id, t, cid, box, n_frames, logits[400] float16).
  Threshold sweeps only read the cache. `build_cache()` is the offline run; `python videomae.py` builds all four
  tune-half caches. The method itself never reads ground truth and only uses frames <= t.

make_method(**params) -> method(result, dataset) -> events (exp_harness.make_event).
"""
from __future__ import annotations

import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
for p in (str(NB), str(NB / "exp"), str(NB / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402

from stage1_gate import GateConfig, RuleEngine  # noqa: E402

CACHE = HERE / "cache" / "videomae"   # own subfolder: xclip.py / pose_seq.py also write into cache/
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# Kinetics-400 class ids (from the model's id2label).
GROUPS = {
    "smoke": [316, 317],                                   # smoking, smoking hookah
    "throw": [48, 49, 50, 148, 166, 356, 357, 358, 361],   # catching/throwing baseball/frisbee/softball, hammer, javelin,
                                                           # throwing axe/ball/discus, tossing coin
    "kick": [105, 152, 174, 175, 302],                     # drop kicking, high kick, kicking field goal/soccer ball, side kick
    "pick": [136, 214],                                    # garbage collecting, picking fruit
}
GROUPS["litter"] = GROUPS["throw"] + GROUPS["kick"] + GROUPS["pick"]

DEFAULTS = dict(
    groups=("litter",),     # class groups whose softmax mass is the cue (a list fires on any group)
    thr=0.10,               # mass threshold; one value, or a dict {group: thr}
    cooldown_s=3.0,         # the same person is not re-sent by this cue within cooldown_s
    min_frames=1,           # minimum distinct frames in the tube (1 lets a brand-new track be checked)
    pad_s=2.5,              # clip window half-width (same as the rules)
)

SCHEDULE = dict(period_s=1.0, min_hist_s=0.5, tube_s_fast=1.6, tube_s_slow=4.0, n_frames=16, crop_scale=1.3, side=224)


# --------------------------------------------------------------------------- model

_MODEL = {}


def load_model(device: str = "cpu"):
    if device not in _MODEL:
        import torch
        from transformers import VideoMAEForVideoClassification
        m = VideoMAEForVideoClassification.from_pretrained(MODEL_ID, local_files_only=True).eval().to(device)
        if device == "mps":             # offline cache runs only; softmax differs from fp32 by < 0.002
            m = m.half()
        _MODEL[device] = m
    return _MODEL[device]


def run_model(tubes: np.ndarray, device: str = "cpu") -> np.ndarray:
    """tubes: (B, 16, 224, 224, 3) float32 normalised RGB -> logits (B, 400)."""
    import torch
    m = load_model(device)
    x = torch.from_numpy(np.ascontiguousarray(tubes.transpose(0, 1, 4, 2, 3))).to(device)  # B T C H W
    x = x.to(next(m.parameters()).dtype)
    with torch.inference_mode():
        return m(pixel_values=x).logits.float().cpu().numpy()


# --------------------------------------------------------------------------- tubes

def crop_square(img, cx, cy, side, out=224):
    h, w = img.shape[:2]
    s = int(round(side))
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    pad = max(0, -x0, -y0, x0 + s - w, y0 + s - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x0, y0 = x0 + pad, y0 + pad
    c = img[y0:y0 + s, x0:x0 + s]
    return cv2.resize(c, (out, out), interpolation=cv2.INTER_AREA if s > out else cv2.INTER_LINEAR)


def make_tube(hist, frames, sched=SCHEDULE):
    """hist: list of (t, obs_i, box) for one person, oldest first, all at times <= now.
    frames: obs_i -> BGR image. Returns (16, 224, 224, 3) normalised RGB float32."""
    n = sched["n_frames"]
    pick = np.linspace(0, len(hist) - 1, n).round().astype(int)
    side = sched["crop_scale"] * max(max(b[2] - b[0], b[3] - b[1]) for _, _, b in hist)
    side = max(side, 32)
    crops = {}
    out = np.empty((n, sched["side"], sched["side"], 3), np.float32)
    for k, j in enumerate(pick):
        if j not in crops:
            _, oi, b = hist[j]
            c = crop_square(frames[oi], (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, side, sched["side"])
            crops[j] = (cv2.cvtColor(c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
        out[k] = crops[j]
    return out


class FrameReader:
    """Frames for one cached sequence, read in order: image paths, or a video read sequentially by native index."""

    def __init__(self, g, r):
        self.seq, self.r = g.seq, r
        self.cap = cv2.VideoCapture(g.seq.path) if hasattr(g.seq, "path") and not hasattr(g.seq, "paths") else None
        self.pos = 0

    def get(self, obs):
        if self.cap is None:
            return cv2.imread(obs.path)
        target = obs.index
        if target < self.pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.pos = target
        img = None
        while self.pos <= target:
            ok, img = self.cap.read()
            self.pos += 1
            if not ok:
                return None
        return img

    def close(self):
        if self.cap is not None:
            self.cap.release()


def scheduled_calls(r, sched=SCHEDULE):
    """Yields (obs_i, t, cid, box, hist) for every scheduled check, causally. hist only holds frames <= t."""
    eng = RuleEngine(GateConfig(rules=()), r.fps)
    slow = r.fps < 5
    tube_s = sched["tube_s_slow"] if slow else sched["tube_s_fast"]
    hists, last_call = {}, {}
    for oi, o in enumerate(r.observations):
        eng.step(o)
        for cid, p in zip(eng.last_cids, o.persons):
            h = hists.setdefault(cid, deque())
            h.append((o.t, oi, list(p["box"])))
            while h and h[0][0] < o.t - tube_s:
                h.popleft()
            span = o.t - h[0][0]
            ready = slow or span >= sched["min_hist_s"]
            if ready and o.t - last_call.get(cid, -1e9) >= sched["period_s"] - 1e-6:
                last_call[cid] = o.t
                yield oi, o.t, cid, list(p["box"]), list(h)


def build_cache(dataset: str, items, device: str = "mps", batch: int = 16, verbose: bool = True) -> dict:
    """Offline: run VideoMAE on every scheduled call of every sequence. Saves cache/<dataset>.pkl."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{dataset}.pkl"
    out = pickle.loads(path.read_bytes()) if path.exists() else {}
    t_start, n_calls = time.time(), 0
    for si, (g, r) in enumerate(items):
        if r.seq_id in out:
            continue
        reader = FrameReader(g, r)
        frames, recs, pend = {}, [], []
        keep_s = max(SCHEDULE["tube_s_slow"], SCHEDULE["tube_s_fast"]) + 0.5

        def flush():
            if pend:
                lg = run_model(np.stack([x[1] for x in pend]), device)
                for (rec, _), l in zip(pend, lg):
                    rec["logits"] = l.astype(np.float16)
                    recs.append(rec)
                pend.clear()

        calls = scheduled_calls(r)
        nxt = next(calls, None)
        for oi, o in enumerate(r.observations):
            img = reader.get(o)
            if img is not None:
                frames[oi] = img
            for k in [k for k in frames if r.observations[k].t < o.t - keep_s]:
                del frames[k]
            while nxt is not None and nxt[0] == oi:
                _, t, cid, box, hist = nxt
                hist = [x for x in hist if x[1] in frames]
                if hist:
                    tube = make_tube(hist, frames)
                    pend.append(({"t": t, "obs_i": oi, "cid": cid, "box": box,
                                  "n_frames": len({x[1] for x in hist})}, tube))
                    if len(pend) >= batch:
                        flush()
                nxt = next(calls, None)
        flush()
        reader.close()
        out[r.seq_id] = recs
        n_calls += len(recs)
        if verbose and (si % 20 == 0 or si == len(items) - 1):
            print(f"{dataset} {si + 1}/{len(items)} calls {n_calls} {time.time() - t_start:.0f}s", flush=True)
            path.write_bytes(pickle.dumps(out))
    path.write_bytes(pickle.dumps(out))
    return out


# --------------------------------------------------------------------------- cue + method

_CACHES: dict[str, dict] = {}


def cache_for(dataset: str) -> dict:
    if dataset not in _CACHES:
        _CACHES[dataset] = pickle.loads((CACHE / f"{dataset}.pkl").read_bytes())
    return _CACHES[dataset]


def softmax(logits):
    z = np.asarray(logits, np.float32)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def group_mass(rec, group):
    if "_p" not in rec:
        rec["_p"] = softmax(rec["logits"])
    return float(rec["_p"][GROUPS[group]].sum())


def make_method(**params):
    P = {**DEFAULTS, **params}
    groups = tuple(P["groups"])
    thr = P["thr"] if isinstance(P["thr"], dict) else {g: P["thr"] for g in groups}

    def method(result, dataset):
        recs = cache_for(dataset).get(result.seq_id)
        if recs is None:
            raise KeyError(f"no VideoMAE cache for {dataset}/{result.seq_id}; run build_cache first")
        last, out = {}, []
        for rec in recs:                          # recs are in time order; each uses frames <= rec['t'] only
            if rec["n_frames"] < P["min_frames"]:
                continue
            for g in groups:
                s = group_mass(rec, g)
                if s >= thr[g] and rec["t"] - last.get(rec["cid"], -1e9) >= P["cooldown_s"]:
                    last[rec["cid"]] = rec["t"]
                    out.append(__import__("exp_harness").make_event(f"action_{g}", rec["t"], rec["box"],
                                                                     pad=P["pad_s"], score=round(s, 3)))
                    break
        return out

    return method


# --------------------------------------------------------------------------- latency

def cpu_latency(n: int = 10, threads: int | None = None) -> dict:
    """CPU ms per model call (batch 1, 16x224x224) and ms per tube build (crop + resize + normalise)."""
    import torch
    if threads:
        torch.set_num_threads(threads)
    m = load_model("cpu")
    x = np.random.randn(1, 16, 224, 224, 3).astype(np.float32)
    run_model(x, "cpu")
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        run_model(x, "cpu")
        ts.append((time.perf_counter() - t0) * 1000)
    img = np.random.randint(0, 255, (1080, 1920, 3), np.uint8)
    hist = [(i / 10, i, [900 + i, 400, 1000 + i, 700]) for i in range(16)]
    frames = {i: img for i in range(16)}
    t0 = time.perf_counter()
    for _ in range(5):
        make_tube(hist, frames)
    tube_ms = (time.perf_counter() - t0) * 200
    return {"model_ms_median": float(np.median(ts)), "model_ms_p90": float(np.percentile(ts, 90)),
            "tube_ms": tube_ms, "threads": torch.get_num_threads(), "params": sum(p.numel() for p in m.parameters())}


if __name__ == "__main__":
    from compare_models import model_splits
    TUNE = model_splits("s", heldout=False)
    for ds in sys.argv[1:] or ["littering", "avenue", "normal", "smoking"]:
        build_cache(ds, TUNE[ds])
