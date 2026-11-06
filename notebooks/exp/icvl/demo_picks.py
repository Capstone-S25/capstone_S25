"""Candidate clips for the one-example-per-activity demo (ICVL even half, current gate with crop check, one alert ->
one person): casual littering (dropped while walking), dumping (put down while standing still), smoking, and loitering
(in view 60 s+, moving around). Each candidate is caught; listed with the rules that caught it, the labelled person's
median height (share of frame height) and the clip length. Pick by eye from the top rows.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/demo_picks.py   -> exp/icvl/demo_picks.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from fpr_attrib import CFG, owner  # noqa: E402  (also loads the cached crop-check scores)
from inventory import ICVL, read_labels  # noqa: E402
from litter_style import style  # noqa: E402
from score import TOL_S, WINDOW_S, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import replay_rules  # noqa: E402


def main():
    picks = defaultdict(list)
    for g, res in load_runs("even"):
        fps, H = g.seq.native_fps, res.observations[0].frame_hw[0] if hasattr(res.observations[0], "frame_hw") else None
        alerts = replay_rules(res, CFG)[0]
        eps, people = labels_for(g.tag, fps)
        own = [owner(a, people) for a in alerts]
        rows = defaultdict(list)
        for r in read_labels(ICVL / g.tag):
            rows[r[6]].append(r)

        def caught(pid, t0, t1, w):
            ks = [k for k, a in enumerate(alerts) if own[k] == pid and on_track(a, people[pid], t0, t1, w)]
            return sorted({alerts[k]["event_type"] for k in ks}), [round(alerts[k]["timestamp_s"], 1) for k in ks]

        def height(pid):
            hs = [b[3] - b[1] for b in people[pid]["boxes"]]
            return round(float(np.median(hs)), 0)

        base = {"seq": res.seq_id.split("/", 1)[1], "video_s": round(res.duration_s, 1)}
        for ep in eps.get("littering", []):
            mine = [r for r in rows[ep["pid"]] if r[10] == 1 and ep["t0"] - 1e-6 <= r[0] / fps <= ep["t1"] + 1e-6]
            kind = "casual littering" if style(mine).endswith("walking") else "dumping"
            rules, ts = caught(ep["pid"], ep["t0"], ep["t1"], WINDOW_S["littering"])
            if rules:
                picks[kind].append({**base, "pid": ep["pid"], "t0": round(ep["t0"], 1), "t1": round(ep["t1"], 1),
                                    "rules": rules, "alert_t": ts, "height_px": height(ep["pid"])})
        for ep in eps.get("smoking", []):
            rules, ts = caught(ep["pid"], ep["t0"], ep["t1"], WINDOW_S.get("smoking", TOL_S))
            if rules:
                picks["smoking"].append({**base, "pid": ep["pid"], "t0": round(ep["t0"], 1), "t1": round(ep["t1"], 1),
                                         "rules": rules, "alert_t": ts, "height_px": height(ep["pid"])})
        still = {ep["pid"] for ep in eps.get("lingering 20s+", [])}
        for pid, tr in people.items():
            t0, t1 = tr["ts"][0], tr["ts"][-1]
            if t1 - t0 >= 60:
                rules, ts = caught(pid, t0, t1, TOL_S)
                if "long_presence" in rules or "sustained_presence" in rules:
                    picks["loitering"].append({**base, "pid": pid, "t0": round(t0, 1), "t1": round(t1, 1), "rules": rules,
                                               "alert_t": ts, "height_px": height(pid), "ever_still_20s": pid in still})
    for k in picks:
        picks[k].sort(key=lambda p: (-p["height_px"], p["video_s"]))
    out = {k: v[:12] for k, v in picks.items()}
    print(json.dumps({k: len(v) for k, v in picks.items()}))
    for k, v in out.items():
        print("==", k)
        for p in v[:8]:
            print(json.dumps(p))
    (HERE / "demo_picks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
