"""Does each Stage 1 rule beat chance?  Rate-matched baselines on every cached dataset, with a lenient and a strict
crop test.

For a method's candidates on a sequence, two baselines with exactly the same number of candidates:
  same_times   same timestamps, ROI replaced by a randomly chosen tracked person in that frame
  uniform      timestamps spread uniformly at random over the sequence, ROI = random tracked person there
Hit tests (candidate within 1 s of a labelled frame):
  lenient  the current box_hit (ROI padded 40%, covers >= 30% of the GT box, no size limit)
  strict   padded ROI covers >= 50% of the GT box AND the padded ROI is at most 16x the GT box area
Recall is per GT event (labelled runs merged with a 1 s gap, as in evaluate_run); smoking also per episode (labels
<= 30 s apart). Baselines are averaged over SEEDS. Per rule type: events caught by that rule's candidates.
Usage (from notebooks/): ../.venv/bin/python exp/chance.py
"""
import json
import pickle
import random
import sys
import zlib
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np

NB = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "actions"), str(NB / "exp" / "newcam"),
                str(NB / "exp" / "misses"), str(NB / "exp" / "gt_review")]

from compare_models import model_splits  # noqa: E402
from exp_harness import baseline, combine  # noqa: E402
from stage1_bench import gt_events  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, calibrate_inview, intersection_over  # noqa: E402

SEEDS = 10
CACHE = NB / "bench_cache" / "models" / "s"


def padded(roi, pad=0.4):
    x1, y1, x2, y2 = roi
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return [x1 - px, y1 - py, x2 + px, y2 + py]


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def hit_lenient(roi, gt):
    return intersection_over(gt, padded(roi)) >= 0.3


def hit_strict(roi, gt):
    p = padded(roi)
    return intersection_over(gt, p) >= 0.5 and area(p) <= 16.0 * max(1.0, area(gt))


def events_of(g, r, mode):
    times, idxs = [o.t for o in r.observations], [o.index for o in r.observations]
    if mode == "episode":
        pos = [(t, i) for t, i in zip(times, idxs) if i in g.gt]
        out, cur = [], None
        for t, i in pos:
            if cur and t - cur[-1][0] <= 30.0:
                cur.append((t, i))
            else:
                cur = [(t, i)]
                out.append(cur)
        return out
    return [ge["frames"] for ge in gt_events(times, idxs, g.gt, r.fps, 1.0)]


def caught(evs, g, cands, hit):
    """cands: [(t, roi, type)] -> (number of events caught, {type: events caught by that type})."""
    n, by = 0, defaultdict(int)
    for ev in evs:
        types = {c[2] for c in cands if any(abs(t - c[0]) <= 1.0 and any(hit(c[1], b) for b in g.gt[i]) for t, i in ev)}
        n += bool(types)
        for k in types:
            by[k] += 1
    return n, by


def baselines(r, cands, rng):
    obs = r.observations
    ts = np.array([o.t for o in obs])
    same, unif = [], []
    for t, roi, typ in cands:
        o = obs[int(np.argmin(np.abs(ts - t)))]
        if o.persons:
            same.append((t, rng.choice(o.persons)["box"], typ))
        o2 = obs[rng.randrange(len(obs))]
        if o2.persons:
            unif.append((o2.t, rng.choice(o2.persons)["box"], typ))
    return same, unif


def table(data, method, ds, mode="event"):
    """data: [(g, r)]; method(r, ds) -> events."""
    totals = defaultdict(float)
    per_type = defaultdict(lambda: defaultdict(float))
    n_events = 0
    n_cands = 0
    minutes = sum(r.duration_s for _, r in data) / 60
    for g, r in data:
        if not g.gt:
            continue
        evs = events_of(g, r, mode)
        n_events += len(evs)
        cands = [(e["timestamp_s"], e["roi_box"], e["event_type"]) for e in method(r, ds)]
        n_cands += len(cands)
        for hname, hit in (("lenient", hit_lenient), ("strict", hit_strict)):
            n, by = caught(evs, g, cands, hit)
            totals[f"method_{hname}"] += n
            for k, v in by.items():
                per_type[k][f"method_{hname}"] += v
            for seed in range(SEEDS):
                rng = random.Random(zlib.crc32(f"{r.seq_id}#{seed}".encode()))
                same, unif = baselines(r, cands, rng)
                for bname, bc in (("same_times", same), ("uniform", unif)):
                    n2, by2 = caught(evs, g, bc, hit)
                    totals[f"{bname}_{hname}"] += n2 / SEEDS
                    for k, v in by2.items():
                        per_type[k][f"{bname}_{hname}"] += v / SEEDS
    pct = lambda x: round(100 * x / max(1, n_events), 1)  # noqa: E731
    row = {"events": n_events, "candidates_per_min": round(n_cands / max(minutes, 1e-9), 2)}
    for k, v in totals.items():
        row[k + "_%"] = pct(v)
    for h in ("lenient", "strict"):
        row[f"above_same_times_{h}"] = round(row.get(f"method_{h}_%", 0) - row.get(f"same_times_{h}_%", 0), 1)
        row[f"above_uniform_{h}"] = round(row.get(f"method_{h}_%", 0) - row.get(f"uniform_{h}_%", 0), 1)
    row["per_rule"] = {k: {kk: pct(vv) for kk, vv in v.items()} for k, v in per_type.items()}
    return row


def main():
    import pose_seq_final as F
    from loaders import meva_relabel
    from static_v2 import aboda_gt
    T = model_splits("s", heldout=False)
    H = model_splits("s", heldout=True)
    inview = calibrate_inview([r for _, r in T["normal"]])
    rules = GateConfig(inview_s=inview)
    rules_pose = GateConfig(inview_s=inview, rules=DEFAULT_RULES + ("pose_action",))
    st = pickle.loads(F.FROZEN.read_bytes())
    oof_cue = F.frozen_method((st["oof"], st["meta"]))[0]
    methods = {
        "rules": lambda half: partial(baseline, cfg=rules),
        "rules+pose": lambda half: (combine(partial(baseline, cfg=rules), oof_cue) if half == "tune" else partial(baseline, cfg=rules_pose)),
    }
    meva = pickle.loads((CACHE / "meva.pkl").read_bytes())
    sets = [
        ("littering", "tune", T["littering"], "event"), ("littering", "held-out", H["littering"], "event"),
        ("loitering", "tune", T["avenue"], "event"), ("loitering", "held-out", H["avenue"], "event"),
        ("smoking", "tune", T["smoking"], "episode"), ("smoking", "held-out", H["smoking"], "episode"),
        ("meva_puts_down", "unseen", [x for x in meva_relabel(meva, ("person_puts_down_object",)) if x[0].gt], "event"),
        ("meva_picks_up", "unseen", [x for x in meva_relabel(meva, ("person_picks_up_object",)) if x[0].gt], "event"),
        ("caviar_left_bag", "unseen", [x for x in pickle.loads((CACHE / "caviar.pkl").read_bytes()) if x[0].gt], "event"),
        ("aboda_left_bag", "unseen", [x for x in aboda_gt(pickle.loads((CACHE / "aboda.pkl").read_bytes())) if x[0].gt], "event"),
    ]
    out = {}
    for ds, half, data, mode in sets:
        for mname, mk in methods.items():
            if ds.startswith("littering") is False and mname == "rules+pose" and half == "tune" and ds != "littering":
                m = partial(baseline, cfg=rules_pose)
            else:
                m = mk(half)
            key = f"{ds}/{half}/{mname}"
            out[key] = table(data, m, "littering" if ds == "littering" else ds, mode)
            r = out[key]
            print(f"{key:40s} n={r['events']:3d} cand/min={r['candidates_per_min']:5.1f} | lenient {r['method_lenient_%']:5.1f} "
                  f"(same-times {r['same_times_lenient_%']:5.1f}, uniform {r['uniform_lenient_%']:5.1f}) | strict {r['method_strict_%']:5.1f} "
                  f"(same-times {r['same_times_strict_%']:5.1f}, uniform {r['uniform_strict_%']:5.1f})", flush=True)
    (NB / "exp" / "chance_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
