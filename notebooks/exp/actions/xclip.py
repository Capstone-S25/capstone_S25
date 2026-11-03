"""X-CLIP zero-shot action cue on person tubes (Stage 1 second-stage check).

For each tracked person (tracker raw_id), at most once per `CALL_EVERY_S` seconds, build a "tube": the last
`TUBE_S` seconds of that person's observations (at least 8 / fps seconds on low-fps sources, capped at 8 s),
8 frames sampled evenly (repeated if fewer), each cropped to the same square around the union of the person's
boxes in the tube padded by 40 %, resized to 224 px. microsoft/xclip-base-patch32 scores the tube against text
prompts; a candidate fires when the best positive prompt's softmax probability beats every negative prompt's by
`margin`.

Causal: a call at time t only uses observations / frames with time <= t (the tube ends at the current frame).
No ground truth is read.

Caching: the expensive part (vision tower + MIT) is run once per call and cached in cache/<dataset>.pt as
video_embeds (512) + mean patch features (49x512, fp16), so any prompt set can be scored later in milliseconds
(the X-CLIP prompt generator conditions each text on the video's patch features, so text logits are computed
from the cached features, not from a fixed bank).

Usage:
    python exp/actions/xclip.py build [mps|cpu]   # extract features for all tune-half sets
    python exp/actions/xclip.py bench             # CPU ms per call
    python exp/actions/xclip.py eval              # sweep and report
"""
from __future__ import annotations

import sys
import time
from functools import lru_cache, partial
from pathlib import Path

import cv2
import numpy as np

NB = Path("/Users/ianvarella/capstone_S25/notebooks")
sys.path[:0] = [str(NB), str(NB / "exp"), str(NB / "tools")]

from exp_harness import make_event  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
MODEL_ID = "microsoft/xclip-base-patch32"
N_FRAMES = 8
TUBE_S = 1.5
MAX_TUBE_S = 8.0
PAD = 0.4
CALL_EVERY_S = 1.0
MIN_OBS = 3            # a track needs this many observations before its first call
SIZE = 224

# ----------------------------------------------------------------------------------------------- prompts
POS = {
    "trash_ground": "a person throwing trash on the ground",
    "drop_litter": "a person dropping litter",
    "bag_air": "a person throwing a bag in the air",
    "running": "a person running",
    "smoking": "a person smoking a cigarette",
    "toss": "a person tossing something away",
    "put_down": "a person putting something down on the ground",
    "dump": "a person dumping garbage",
    "throw_obj": "a person throwing an object",
    "bend_pick": "a person bending down to the ground",
    "jump": "a person jumping",
    "smoke2": "a man holding a cigarette to his mouth",
}
NEG = {
    "walking": "a person walking",
    "standing": "a person standing",
    "carry_bag": "a person carrying a bag",
    "walk_street": "people walking on a sidewalk",
    "phone": "a person looking at a phone",
    "talking": "a person talking",
    "working": "a person working",
    "backpack": "a person walking with a backpack",
}
PROMPTS = {**POS, **NEG}

PROMPT_SETS = {
    # the brief's set
    "brief": (["trash_ground", "drop_litter", "bag_air", "running", "smoking"], ["walking", "standing", "carry_bag"]),
    # littering only (for the reach_down replacement)
    "litter": (["trash_ground", "drop_litter", "toss", "put_down", "dump", "throw_obj"],
               ["walking", "standing", "carry_bag"]),
    # wide: all positives, all negatives
    "wide": (list(POS), list(NEG)),
    # brief positives + more negatives (to push false alarms down)
    "brief_negs": (["trash_ground", "drop_litter", "bag_air", "running", "smoking"], list(NEG)),
    "litter_negs": (["trash_ground", "drop_litter", "toss", "put_down", "dump", "throw_obj"], list(NEG)),
}

# Chosen on the tune half (eval_xclip.py sweep): littering prompts vs 8 negatives, margin 0.2.
DEFAULTS = dict(prompt_set="litter_negs", margin=0.2, cooldown_s=5.0, min_h_px=0.0)


# ----------------------------------------------------------------------------------------------- schedule
def plan_calls(result, call_every_s: float = CALL_EVERY_S, tube_s: float = TUBE_S):
    """Causal call schedule. Returns a list of dicts: pos (observation position), t, raw_id, box (current),
    tube = [(obs position, box), ...] (8 entries, time order, all <= t)."""
    fps = result.fps or 10.0
    win = min(MAX_TUBE_S, max(tube_s, N_FRAMES / fps))
    hist: dict[int, list] = {}
    last: dict[int, float] = {}
    calls = []
    for pos, o in enumerate(result.observations):
        for p in o.persons:
            rid = p["raw_id"]
            h = hist.setdefault(rid, [])
            h.append((pos, o.t, p["box"]))
            while h and h[0][1] < o.t - win - 1e-6:
                h.pop(0)
            if len(h) < MIN_OBS or o.t - last.get(rid, -1e9) < call_every_s - 1e-6:
                continue
            last[rid] = o.t
            sel = np.linspace(0, len(h) - 1, N_FRAMES).round().astype(int)
            calls.append({"pos": pos, "t": o.t, "raw_id": rid, "box": list(map(float, p["box"])),
                          "tube": [(h[i][0], h[i][2]) for i in sel]})
    return calls


def tube_crop_box(boxes, hw):
    b = np.array(boxes, dtype=float)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = max(x2 - x1, y2 - y1) * (1 + PAD)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(2, int(round(side)))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    return [x0, y0, x0 + side, y0 + side]


def crop_square(img, box):
    """Crop with zero padding outside the image, resize to SIZE, BGR -> RGB."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    side = max(1, x2 - x1)
    out = np.zeros((side, side, 3), dtype=np.uint8)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = img[sy1:sy2, sx1:sx2]
    out = cv2.resize(out, (SIZE, SIZE), interpolation=cv2.INTER_AREA if side > SIZE else cv2.INTER_LINEAR)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


class FrameSource:
    """Random access to a sequence's frames by observation position, read forward (video) with a small window."""

    def __init__(self, seq, result):
        self.seq, self.obs = seq, result.observations
        self.cache: dict[int, np.ndarray] = {}
        self.cap = None
        self.cur = -1
        if hasattr(seq, "paths"):
            self.idx2path = dict(zip(seq.indices, seq.paths))

    def get(self, pos):
        if pos in self.cache:
            return self.cache[pos]
        idx = self.obs[pos].index
        if hasattr(self.seq, "paths"):
            img = cv2.imread(self.idx2path[idx])
        else:
            if self.cap is None:
                self.cap = cv2.VideoCapture(self.seq.path)
            if idx <= self.cur:     # should not happen with sorted calls; re-seek
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self.cur = idx - 1
            while self.cur < idx - 1:
                self.cap.grab()
                self.cur += 1
            ok, img = self.cap.read()
            self.cur += 1
        self.cache[pos] = img
        return img

    def evict_before(self, pos):
        for k in [k for k in self.cache if k < pos]:
            del self.cache[k]


def build_tubes(seq, result, calls):
    src = FrameSource(seq, result)
    hw = result.observations[0].frame_hw if result.observations else (1, 1)
    for c in calls:
        cb = tube_crop_box([b for _, b in c["tube"]], hw)
        frames = [crop_square(src.get(p), cb) for p, _ in c["tube"]]
        src.evict_before(c["pos"] - 150)   # keep a window: other people's tubes may start earlier
        yield frames


# ----------------------------------------------------------------------------------------------- model
@lru_cache(maxsize=2)
def load_model(device: str = "cpu"):
    import torch
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval().to(device)
    return proc, model


def video_features(pixel_values, device="cpu"):
    """pixel_values: B x 8 x 3 x 224 x 224 -> (video_embeds B x 512 normalised, img_features B x 49 x 512)."""
    import torch
    _, m = load_model(device)
    with torch.no_grad():
        B, F = pixel_values.shape[:2]
        pv = pixel_values.to(device).reshape(-1, *pixel_values.shape[2:])
        vo = m.vision_model(pixel_values=pv)
        ve = m.visual_projection(vo[1]).view(B, F, -1)
        ve = m.mit(ve)[1]
        img = m.prompts_visual_layernorm(vo[0][:, 1:, :]) @ m.prompts_visual_projection
        img = img.view(B, F, -1, ve.shape[-1]).mean(dim=1)
        ve = ve / ve.norm(dim=-1, keepdim=True)
    return ve, img


@lru_cache(maxsize=4)
def text_embeds(device: str = "cpu"):
    import torch
    proc, m = load_model(device)
    tok = proc.tokenizer(list(PROMPTS.values()), padding=True, return_tensors="pt")
    with torch.no_grad():
        te = m.text_projection(m.text_model(**{k: v.to(device) for k, v in tok.items()})[1])
    return te


def prompt_logits(ve, img, device="cpu", chunk=256):
    """Logits over PROMPTS (in PROMPTS order) for cached features."""
    import torch
    _, m = load_model(device)
    te0 = text_embeds(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(ve), chunk):
            v = ve[i:i + chunk].to(device).float()
            f = img[i:i + chunk].to(device).float()
            te = te0.unsqueeze(0).expand(len(v), -1, -1)
            te = te + m.prompts_generator(te, f)
            te = te / te.norm(dim=-1, keepdim=True)
            out.append(torch.einsum("bd,bkd->bk", v, m.logit_scale.exp() * te).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, len(PROMPTS)), np.float32)


def preprocess(frames):
    proc, _ = load_model("cpu")
    return proc.image_processor(frames, return_tensors="pt")["pixel_values"]  # 1 x 8 x 3 x 224 x 224


# ----------------------------------------------------------------------------------------------- cache
def extract(dataset: str, items, device: str = "mps", batch: int = 16):
    import torch
    path = CACHE / f"{dataset}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    meta, VE, IMG = [], [], []
    buf, t0 = [], time.time()

    def flush():
        if buf:
            ve, img = video_features(torch.cat(buf), device)
            VE.append(ve.cpu())
            IMG.append(img.cpu().half())
            buf.clear()

    for k, (g, r) in enumerate(items):
        calls = plan_calls(r)
        for c, frames in zip(calls, build_tubes(g.seq, r, calls)):
            meta.append({"seq_id": r.seq_id, "t": c["t"], "raw_id": c["raw_id"], "box": c["box"], "pos": c["pos"]})
            buf.append(preprocess(frames))
            if len(buf) >= batch:
                flush()
        if k % 20 == 0:
            print(f"  {dataset} {k}/{len(items)} calls={len(meta)} {time.time() - t0:.0f}s", flush=True)
    flush()
    data = {"meta": meta, "ve": torch.cat(VE) if VE else torch.zeros(0, 512),
            "img": torch.cat(IMG) if IMG else torch.zeros(0, 49, 512).half()}
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print(f"{dataset}: {len(meta)} calls in {time.time() - t0:.0f}s", flush=True)
    return data


@lru_cache(maxsize=8)
def bank(dataset: str):
    """Per-sequence list of calls with logits over PROMPTS (cached in cache/<dataset>_logits.npz)."""
    import torch
    lp = CACHE / f"{dataset}_logits.npz"
    if lp.exists():
        z = np.load(lp, allow_pickle=True)
        meta, logits = list(z["meta"]), z["logits"]
        if list(z["prompts"]) != list(PROMPTS):
            lp.unlink()
            return bank.__wrapped__(dataset)
    else:
        d = torch.load(CACHE / f"{dataset}.pt", weights_only=False)
        meta = d["meta"]
        logits = prompt_logits(d["ve"], d["img"], "cpu")
        np.savez(lp, meta=np.array(meta, dtype=object), logits=logits, prompts=np.array(list(PROMPTS)))
    by_seq: dict[str, list] = {}
    for m, lg in zip(meta, logits):
        by_seq.setdefault(m["seq_id"], []).append((m, lg))
    return by_seq


# ----------------------------------------------------------------------------------------------- method
def make_method(prompt_set="brief", margin=0.2, cooldown_s=5.0, min_h_px=0.0, positives=None, negatives=None,
                event_type="xclip_action"):
    pos, neg = PROMPT_SETS[prompt_set] if positives is None else (positives, negatives)
    keys = list(PROMPTS)
    cols = [keys.index(k) for k in pos + neg]
    npos = len(pos)

    def method(result, dataset):
        calls = bank(dataset).get(result.seq_id, [])
        out, last = [], {}
        for m, lg in calls:
            if m["box"][3] - m["box"][1] < min_h_px:
                continue
            z = lg[cols]
            p = np.exp(z - z.max())
            p /= p.sum()
            gap = p[:npos].max() - p[npos:].max()
            if gap >= margin and m["t"] - last.get(m["raw_id"], -1e9) >= cooldown_s:
                last[m["raw_id"]] = m["t"]
                out.append(make_event(event_type, m["t"], m["box"], prompt=pos[int(p[:npos].argmax())],
                                      xclip_gap=round(float(gap), 3)))
        return out
    return method


# ----------------------------------------------------------------------------------------------- CLI
def tune_splits():
    from compare_models import model_splits
    return model_splits("s", heldout=False)


def cmd_build(device="mps"):
    S = tune_splits()
    for ds in ("littering", "avenue", "normal", "smoking"):
        extract(ds, S[ds], device=device)
        bank(ds)


def cmd_bench(n=20):
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    S = tune_splits()
    g, r = S["littering"][0]
    calls = plan_calls(r)
    tubes = list(build_tubes(g.seq, r, calls[:1]))
    load_model("cpu")
    pv = preprocess(tubes[0])
    for _ in range(3):
        video_features(pv, "cpu")
    t = time.perf_counter()
    for _ in range(n):
        video_features(pv, "cpu")
    model_ms = (time.perf_counter() - t) / n * 1000
    t = time.perf_counter()
    for _ in range(n):
        pv = preprocess(list(build_tubes(g.seq, r, calls[:1]))[0])
    prep_ms = (time.perf_counter() - t) / n * 1000
    ve, img = video_features(pv, "cpu")
    t = time.perf_counter()
    for _ in range(n):
        prompt_logits(ve, img, "cpu")
    text_ms = (time.perf_counter() - t) / n * 1000
    print(f"CPU threads={torch.get_num_threads()} vision+MIT {model_ms:.0f} ms/call, prompt head ({len(PROMPTS)} prompts) "
          f"{text_ms:.1f} ms, crop+preprocess (incl. jpg reads, 1080p) {prep_ms:.0f} ms")
    return model_ms, text_ms, prep_ms


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        cmd_build(sys.argv[2] if len(sys.argv) > 2 else "mps")
    elif cmd == "bench":
        cmd_bench()
