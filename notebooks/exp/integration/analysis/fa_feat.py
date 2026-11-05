"""Default gate, ICVL even and odd: share of false alerts by simple at-alert features."""
import sys, pickle
from collections import defaultdict
import numpy as np
sys.path[:0] = ["exp/icvl", "."]
from fpr_attrib import CFG, owner
from score import EVENTS, TOL_S, WINDOW_S, labels_for, load_runs, on_track
from stage1_gate import replay_rules
rows = []
for half in ("odd", "even"):
    for g, res in load_runs(half):
        alerts = replay_rules(res, CFG)[0]
        if not alerts: continue
        eps, people = labels_for(g.tag, g.seq.native_fps)
        H, W = res.observations[0].frame_hw if hasattr(res.observations[0], "frame_hw") else res.observations[0].shape
        first = {}
        for o in res.observations:
            for p in o.persons:
                first.setdefault(p.get("track_id", p.get("id", p.get("raw_id"))), o.t)
        n_people = {round(o.t, 2): len(o.persons) for o in res.observations}
        for a in alerts:
            o = owner(a, people)
            ok = o is not None and any(ep["pid"] == o and on_track(a, people[o], ep["t0"], ep["t1"], WINDOW_S.get(ev, TOL_S)) for ev in EVENTS for ep in eps.get(ev, []))
            x0, y0, x1, y1 = a["roi_box"]
            rows.append(dict(half=half, rule=a["event_type"], false=not ok, t=a["timestamp_s"], dur=res.duration_s,
                             edge=min(x0, y0, W - x1, H - y1) / H, h=(y1 - y0) / H, bottom=y1 / H, seq=res.seq_id))
pickle.dump(rows, open(sys.argv[1], "wb"))
def show(name, key, bins):
    print(name)
    for rule in ("reach_down", "pose_action", "sustained_presence", "long_presence"):
        r = [x for x in rows if x["rule"] == rule]
        parts = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = [x for x in r if lo <= key(x) < hi]
            if b: parts.append(f"[{lo},{hi}) {len(b)} {100*np.mean([x['false'] for x in b]):.0f}%")
        print(" ", rule, " | ".join(parts))
show("alert time in video (s)", lambda x: x["t"], [0, 1, 2, 3, 5, 10, 1e9])
show("gap to frame edge (x frame height)", lambda x: x["edge"], [-1, 0.01, 0.03, 0.06, 0.1, 9])
show("box height (x frame height)", lambda x: x["h"], [0, 0.15, 0.25, 0.35, 0.5, 9])
show("time left in video (s)", lambda x: x["dur"] - x["t"], [-9, 1, 2, 3, 5, 1e9])
