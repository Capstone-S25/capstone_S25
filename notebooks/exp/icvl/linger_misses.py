"""Which ICVL "lingering 20s+" incidents does sustained_presence (stays put 20 s) miss, and why?
Per miss: labelled duration, how much of it YOLO tracked the person (a cached person box on the labelled box),
how many tracker ids that took, the longest single-id stretch, and the person's movement (norm radius, the rule's
own measure over the labelled feet points) -> exp/icvl/linger_misses.json
Usage (from notebooks/): ../.venv/bin/python exp/icvl/linger_misses.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from score import TOL_S, box_hit, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, replay_rules  # noqa: E402

EV = "lingering 20s+"


def coverage(res, ep):
    """Fraction of the incident's sampled frames with a YOLO person on the labelled box, ids used, longest id run."""
    ts, boxes = np.array(ep["ts"]), ep["boxes"]
    seen, ids, runs, cur = 0, set(), [], None
    obs = [o for o in res.observations if ep["t0"] <= o.t <= ep["t1"]]
    for o in obs:
        k = int(np.argmin(np.abs(ts - o.t)))
        if abs(ts[k] - o.t) > TOL_S:
            continue
        hit = [p for p in o.persons if box_hit(p["box"], boxes[k])]
        if hit:
            seen += 1
            rid = hit[0]["raw_id"]
            ids.add(rid)
            if cur and cur[0] == rid:
                cur[2] = o.t
            else:
                cur = [rid, o.t, o.t]
                runs.append(cur)
    longest = max((r[2] - r[1] for r in runs), default=0.0)
    return (seen / len(obs) if obs else 0.0), len(ids), longest


def radius(ep):
    """The rule's stillness measure on the labelled feet: max distance from the mean, in person heights."""
    b = np.array(ep["boxes"])
    xy = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1)
    h = np.median(b[:, 3] - b[:, 1])
    return float(np.max(np.hypot(*(xy - xy.mean(0)).T)) / h), float(h)


def main():
    cfg = GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",))
    rows, n = [], 0
    for g, res in load_runs():
        episodes, people = labels_for(g.tag, g.seq.native_fps)
        eps = episodes.get(EV, [])
        if not eps:
            continue
        alerts = replay_rules(res, cfg)[0]
        stay = [a for a in alerts if a["event_type"] == "sustained_presence"]
        for ep in eps:
            n += 1
            tr = people[ep["pid"]]
            if any(on_track(a, tr, ep["t0"], ep["t1"]) for a in stay):
                continue
            cov, nid, longest = coverage(res, ep)
            rad, h = radius(ep)
            rows.append({"seq": res.seq_id, "tag": g.tag, "pid": ep["pid"], "t0": round(ep["t0"], 1),
                         "t1": round(ep["t1"], 1), "dur_s": round(ep["t1"] - ep["t0"], 1),
                         "tracked_%": round(100 * cov), "track_ids": nid, "longest_id_s": round(longest, 1),
                         "feet_radius_h": round(rad, 2), "person_h_px": round(h),
                         "other_rules": sorted({a["event_type"] for a in alerts
                                                if on_track(a, tr, ep["t0"], ep["t1"])})})
    out = {"incidents": n, "missed": len(rows), "misses": rows}
    (HERE / "linger_misses.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(json.dumps(r))
    print(n, "incidents", len(rows), "missed by sustained_presence")


if __name__ == "__main__":
    main()
