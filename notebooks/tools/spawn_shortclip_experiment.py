"""Short-clip test of the 'something spawned near a person' idea.

The first frame of a ~3 s window stands in for a background model that has already learned the
empty scene. A candidate is a region that differs from that reference, has stayed unchanged for
`stable_s`, is not covered by any person (now or in the reference frame), and had a person
standing next to it in the last `owner_s` seconds.
"""
import itertools
import json
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks")
from stage1_bench import box_hit  # noqa: E402
from stage1_gate import expand_box  # noqa: E402

WIDTH = 480


def small_gray(img):
    s = WIDTH / img.shape[1]
    g = cv2.resize(img, (WIDTH, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32), s


def person_mask(shape, boxes, s, grow):
    m = np.zeros(shape, np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = (np.array(expand_box(b, grow)) * s).astype(int)
        m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 1
    return m


def spawn_events(frames, times, persons_per_frame, p):
    """frames: list of (small grey image, scale); persons_per_frame: lists of xyxy boxes (full res)."""
    grays = [g for g, _ in frames]
    scale = frames[0][1]
    ref = grays[0]
    ref_people = person_mask(ref.shape, persons_per_frame[0], scale, p["grow"])
    h, w = ref.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    events, flagged = [], []
    for i in range(1, len(grays)):
        t = times[i]
        j = next((k for k in range(i, -1, -1) if t - times[k] >= p["stable_s"]), None)
        if j is None:
            continue
        diff = (np.abs(grays[i] - ref) > p["diff_thr"]).astype(np.uint8)
        steady = (np.abs(grays[i] - grays[j]) < p["steady_thr"]).astype(np.uint8)
        busy = person_mask(ref.shape, persons_per_frame[i], scale, p["grow"]) | ref_people
        cand = cv2.morphologyEx(diff & steady & (1 - busy), cv2.MORPH_OPEN, kernel)
        n, _, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
        for x, y, bw, bh, area in stats[1:]:
            frac = area / (h * w)
            if not (p["min_area"] <= frac <= p["max_area"]):
                continue
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2 or max(bw, bh) > 4 * max(1, min(bw, bh)):
                continue
            box = [x / scale, y / scale, (x + bw) / scale, (y + bh) / scale]
            if any(abs(box[0] - f[0]) + abs(box[1] - f[1]) < 40 / scale for f in flagged):
                continue
            c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            owner = None
            for k in range(i, -1, -1):
                if t - times[k] > p["owner_s"]:
                    break
                for pb in persons_per_frame[k]:
                    ph = pb[3] - pb[1]
                    pc = ((pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2)
                    if min(np.hypot(c[0] - pc[0], c[1] - pc[1]), np.hypot(c[0] - pc[0], c[1] - pb[3])) <= p["owner_h"] * ph:
                        owner = pb
                        break
                if owner is not None:
                    break
            if owner is None:
                continue
            flagged.append(box)
            roi = [min(box[0], owner[0]), min(box[1], owner[1]), max(box[2], owner[2]), max(box[3], owner[3])]
            events.append({"t": t, "roi": roi, "object": box})
    return events


def load_litter(pkl):
    out = []
    for g, r in pickle.load(open(pkl, "rb")):
        frames = [small_gray(cv2.imread(p)) for p in g.seq.paths]
        times = [o.t for o in r.observations]
        persons = [[p["box"] for p in o.persons] for o in r.observations]
        out.append((g, frames, times, persons))
    return out


def load_avenue_normal(pkl, window_s=3.0):
    out = []
    for g, r in pickle.load(open(pkl, "rb")):
        if not g.seq.seq_id.startswith("train/"):
            continue
        imgs = {fr.index: small_gray(fr.image) for fr in g.seq}
        obs = r.observations
        start = 0
        while start < len(obs):
            chunk = [o for o in obs[start:] if o.t - obs[start].t < window_s]
            if len(chunk) >= 10:
                out.append(([imgs[o.index] for o in chunk], [o.t for o in chunk],
                            [[p["box"] for p in o.persons] for o in chunk]))
            start += len(chunk)
    return out


def score(litter, avenue, p):
    hits = reached = 0
    for g, frames, times, persons in litter:
        ev = spawn_events(frames, times, persons, p)
        reached += bool(ev)
        hits += any(any(box_hit(e["roi"], b) for i, bs in g.gt.items() if abs(times[i] - e["t"]) <= 1.0 for b in bs) for e in ev)
    fa = sum(len(spawn_events(f, t, ps, p)) for f, t, ps in avenue)
    minutes = len(avenue) * 3.0 / 60
    return {"clips": len(litter), "reached_%": round(100 * reached / len(litter), 1),
            "crop_hit_%": round(100 * hits / len(litter), 1), "FA_per_min": round(fa / minutes, 1)}


if __name__ == "__main__":
    S = sys.argv[1]
    litter_tune = load_litter(f"{S}/litter_clips_train.pkl")
    avenue = load_avenue_normal("/Users/ianvarella/capstone_S25/notebooks/bench_cache/avenue.pkl")
    print("tuning clips", len(litter_tune), "| avenue normal 3s windows", len(avenue), flush=True)
    base = {"diff_thr": 25, "steady_thr": 10, "stable_s": 0.5, "grow": 0.25, "min_area": 2e-4, "max_area": 0.02,
            "owner_s": 2.0, "owner_h": 1.0}
    rows = []
    for diff_thr, stable_s, min_area, owner_h in itertools.product([20, 30], [0.3, 0.6], [1e-4, 4e-4], [0.75, 1.5]):
        p = {**base, "diff_thr": diff_thr, "stable_s": stable_s, "min_area": min_area, "owner_h": owner_h}
        r = {**{k: p[k] for k in ("diff_thr", "stable_s", "min_area", "owner_h")}, **score(litter_tune, avenue, p)}
        rows.append(r)
        print(r, flush=True)
    json.dump(rows, open(f"{S}/spawn_sweep.json", "w"), indent=1)
