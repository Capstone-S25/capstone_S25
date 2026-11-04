"""Pick ICVL example incidents for the presentation videos: per event, caught and missed incidents on the EVEN half
(never used for tuning), with the rule that caught it and, for misses, how well YOLO saw the person.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/examples.py -> exp/icvl/examples.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from linger_misses import coverage  # noqa: E402
from score import WINDOW_S, TOL_S, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, replay_rules  # noqa: E402

CFG = GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",))


def main():
    rows = []
    for g, res in load_runs("even"):
        episodes, people = labels_for(g.tag, g.seq.native_fps)
        alerts = replay_rules(res, CFG)[0]
        for ev in ("littering", "smoking", "lingering 20s+"):
            for ep in episodes.get(ev, []):
                tr = people[ep["pid"]]
                hits = [a for a in alerts if on_track(a, tr, ep["t0"], ep["t1"], WINDOW_S.get(ev, TOL_S))]
                cov, nid, longest = coverage(res, ep)
                b = np.array(ep["boxes"])
                rows.append({"seq": res.seq_id, "event": ev, "pid": ep["pid"], "t0": round(ep["t0"], 1),
                             "t1": round(ep["t1"], 1), "caught": bool(hits),
                             "rules": sorted({a["event_type"] for a in hits}),
                             "first_alert": min((a["timestamp_s"] for a in hits), default=None),
                             "tracked_%": round(100 * cov), "ids": nid,
                             "h_px": round(float(np.median(b[:, 3] - b[:, 1]))),
                             "others_in_view": len(people) - 1, "minutes": round(res.duration_s / 60, 1),
                             "alerts_in_video": len(alerts)})
    (HERE / "examples.json").write_text(json.dumps(rows, indent=1))
    for ev in ("littering", "smoking", "lingering 20s+"):
        r = [x for x in rows if x["event"] == ev]
        print(ev, sum(x["caught"] for x in r), "/", len(r))


if __name__ == "__main__":
    main()
