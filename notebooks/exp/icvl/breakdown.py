"""Which rule caught each ICVL incident, and what labelled people were doing when an alert was on them but on no
labelled event (posture / movement / gesture layers at the alert time). Same matching as score.py.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/breakdown.py [odd|even] ['<json GateConfig overrides>']
  -> exp/icvl/breakdown.json
"""
import json
import pickle
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from inventory import GESTURE, ICVL, MOVE, POSTURE, read_labels  # noqa: E402
from score import (DEFAULT_RULES, EVENTS, TOL_S, WINDOW_S, GateConfig, box_hit, labels_for, load_runs,  # noqa: E402
                   on_track, person_box, replay_rules)


def main():
    half = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("odd", "even") else None
    kw = json.loads(sys.argv[-1]) if len(sys.argv) > 1 and sys.argv[-1].startswith("{") else {}
    cfg = GateConfig(**{"inview_s": 60.0, "rules": DEFAULT_RULES + ("pose_action",), **kw})
    caught_by = {ev: Counter() for ev in EVENTS}
    doing = defaultdict(Counter)
    for g, res in load_runs(half):
        alerts = replay_rules(res, cfg)[0]
        fps = g.seq.native_fps
        episodes, people = labels_for(g.tag, fps)
        rows = defaultdict(list)
        for r in read_labels(ICVL / g.tag):
            rows[r[6]].append((r[0] / fps, r))
        on_event = [False] * len(alerts)
        for ev in EVENTS:
            for ep in episodes.get(ev, []):
                rules = set()
                for k, a in enumerate(alerts):
                    if on_track(a, people[ep["pid"]], ep["t0"], ep["t1"], WINDOW_S.get(ev, TOL_S)):
                        rules.add(a["event_type"])
                        on_event[k] = True
                for rule in rules:
                    caught_by[ev][rule] += 1
                caught_by[ev]["incidents"] += 1
        for k, a in enumerate(alerts):
            if on_event[k]:
                continue
            t = a["timestamp_s"]
            for pid, pts in rows.items():
                ts = [p[0] for p in pts]
                lo, hi = bisect_left(ts, t - TOL_S), bisect_right(ts, t + TOL_S)
                near = [p[1] for p in pts[lo:hi] if box_hit(a["roi_box"], person_box(p[1]))]
                if near:
                    r = min(near, key=lambda r: abs(r[0] / fps - t))
                    doing[a["event_type"]][f"{POSTURE.get(r[7])} / {MOVE.get(r[8])} / {GESTURE.get(r[9])}"] += 1
                    break
            else:
                doing[a["event_type"]]["nobody labelled"] += 1
    out = {"caught_by_rule": {ev: dict(c) for ev, c in caught_by.items()},
           "off_event_alerts_by_what_person_was_doing": {k: dict(v.most_common(8)) for k, v in doing.items()}}
    (Path(__file__).parent / "breakdown.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
