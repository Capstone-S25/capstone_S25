"""Why does reach_down fire on ICVL walkers and cyclists? Per reach_down alert, simple features from the cached track:
sideways foot speed and size change (walking toward / away from the camera), both in body heights per second over
the last 1 s and 2 s, person height in the 640 px detector input, how far the low wrist is below the knee, whether
the person was the same track for the whole window. Grouped by what the labelled person was doing (littering hit
within 3 s, walking, cycling, ...). ODD videos only (the ICVL tune half; even videos are kept for the final check).
Usage (from notebooks/): ../.venv/bin/python exp/icvl/reach_features.py   -> exp/icvl/reach_features.json
"""
import json
import math
import pickle
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from inventory import ICVL, MOVE, POSTURE, read_labels  # noqa: E402
from score import (CACHE, DEFAULT_RULES, TOL_S, WINDOW_S, GateConfig, box_hit, labels_for, on_track,  # noqa: E402
                   person_box, replay_rules)
from stage1_gate import L_KNEE, L_WRIST, R_KNEE, R_WRIST, intersection_over  # noqa: E402


def track_feats(obs_list, k, raw_id, h_now, span):
    t = obs_list[k].t
    pts = []
    for o in obs_list[max(0, k - 40):k + 1]:
        if o.t < t - span:
            continue
        for p in o.persons:
            if p["raw_id"] == raw_id:
                b = p["box"]
                pts.append((o.t, (b[0] + b[2]) / 2, b[3], max(1.0, b[3] - b[1])))
    if len(pts) < 2 or pts[-1][0] - pts[0][0] < 0.5 * span:
        return None, None
    a, b = pts[0], pts[-1]
    dt = b[0] - a[0]
    hm = float(np.median([q[3] for q in pts]))
    return math.hypot(b[1] - a[1], b[2] - a[2]) / dt / hm, abs(math.log(b[3] / a[3])) / dt


def main():
    cfg = GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",))
    rows = []
    for vi, (g, res) in enumerate(pickle.loads(CACHE.read_bytes())):
        if vi % 2 == 0:
            continue
        fps = g.seq.native_fps
        alerts = [a for a in replay_rules(res, cfg)[0] if a["event_type"] == "reach_down"]
        if not alerts:
            continue
        episodes, people = labels_for(g.tag, fps)
        lab = defaultdict(list)
        for r in read_labels(ICVL / g.tag):
            lab[r[6]].append((r[0] / fps, r))
        ts = [o.t for o in res.observations]
        for a in alerts:
            t = a["timestamp_s"]
            k = min(range(len(ts)), key=lambda i: abs(ts[i] - t)) if len(ts) < 50 else \
                min(range(max(0, bisect_left(ts, t) - 1), min(len(ts), bisect_left(ts, t) + 1)), key=lambda i: abs(ts[i] - t))
            o = res.observations[k]
            p = max(o.persons, key=lambda q: intersection_over(q["box"], a["roi_box"]), default=None)
            if p is None:
                continue
            h = max(1.0, p["box"][3] - p["box"][1])
            kp = p.get("kpts")
            margin = None
            if kp is not None:
                knees = [kp[i, 1] for i in (L_KNEE, R_KNEE) if kp[i, 2] >= 0.5]
                wr = [kp[i, 1] for i in (L_WRIST, R_WRIST) if kp[i, 2] >= 0.5]
                if knees and wr:
                    margin = float((max(wr) - float(np.mean(knees))) / h)
            v1, s1 = track_feats(res.observations, k, p["raw_id"], h, 1.0)
            v2, s2 = track_feats(res.observations, k, p["raw_id"], h, 2.0)
            if any(on_track(a, people[ep["pid"]], ep["t0"], ep["t1"], WINDOW_S["littering"]) for ep in episodes.get("littering", [])):
                group = "littering"
            else:
                group = "nobody labelled"
                for pid, pts in lab.items():
                    tt = [q[0] for q in pts]
                    lo, hi = bisect_left(tt, t - TOL_S), bisect_right(tt, t + TOL_S)
                    near = [q[1] for q in pts[lo:hi] if box_hit(a["roi_box"], person_box(q[1]))]
                    if near:
                        r = near[0]
                        group = "other labelled event" if (r[9] == 2 or r[10]) else f"{POSTURE.get(r[7])}/{MOVE.get(r[8])}"
                        break
            rows.append({"group": group, "video": res.seq_id, "t": round(t, 2), "h640": float(h * 640 / max(o.frame_hw)),
                         "margin": margin, "v1": v1, "s1": s1, "v2": v2, "s2": s2})
    out = {}
    for grp in sorted({r["group"] for r in rows}, key=lambda x: -sum(r["group"] == x for r in rows)):
        sel = [r for r in rows if r["group"] == grp]
        d = {"n": len(sel)}
        for f in ("h640", "margin", "v1", "s1", "v2", "s2"):
            vals = [r[f] for r in sel if r[f] is not None]
            d[f] = [round(float(np.percentile(vals, q)), 2) for q in (25, 50, 75)] if vals else None
            d[f + "_missing"] = len(sel) - len(vals)
        out[grp] = d
    print(json.dumps(out, indent=1))
    (Path(__file__).parent / "reach_features.json").write_text(json.dumps({"summary": out, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
