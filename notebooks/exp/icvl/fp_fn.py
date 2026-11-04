"""Why are incidents missed, and why are people doing nothing we care about flagged? All of ICVL, current settings
(15 s pose-cue gap, as in the deck).

Misses (each labelled incident with no alert on that person), first reason that applies:
  not seen        YOLO had a person on the labelled box in under half the incident's frames
  filter          switching off one gate filter alone catches it (which one is listed)
  never fired     seen, and no single filter is to blame: the body points never looked like it
plus a flag when an alert did land on that person within 10 s outside the scoring window (a timing near-miss).

False alarms (alerts on no labelled incident):
  near a real incident    on a person who has an incident in that video, within 10 s of it (e.g. bending before a drop)
  same person, other time on a person with an incident, but not near it
  nothing we flag         on a person with no incident: what ICVL says they were doing (posture / movement / gesture)
  nobody labelled         on someone ICVL did not label
And the per-person false positive rate split by what the person was doing.

Usage (from notebooks/): ../.venv/bin/python exp/icvl/fp_fn.py   -> exp/icvl/fp_fn.json
"""
import json
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from inventory import GESTURE, ICVL, MOVE, POSTURE, read_labels  # noqa: E402
from linger_misses import coverage  # noqa: E402
from one_per_person import CFG  # noqa: E402
from score import EVENTS, TOL_S, WINDOW_S, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import replay_rules  # noqa: E402

NEAR_S = 10.0
OFF = {
    "littering": {"walking fast": dict(reach_max_speed_h=None), "frame edge": dict(reach_skip_border=False),
                  "overlap": dict(reach_skip_overlap=None), "seated": dict(reach_skip_seated_s=None),
                  "rider": dict(reach_skip_riders=False), "verifier": dict(reach_verifier=False),
                  "keypoint confidence 0.3": dict(reach_kp_conf=0.3), "wrist margin -0.3": dict(reach_margin_h=-0.3),
                  "dedup across rules": dict(dedup_per_rule=True)},
    "smoking": {"pose-cue gap 5 s": dict(pose_action_cooldown_s=5.0), "pose-cue threshold 0.5": dict(pose_action_thr=0.5),
                "dedup across rules": dict(dedup_per_rule=True)},
    "lingering 20s+": {"stays-put repeat 20 s": dict(loiter_repeat_s=20.0), "stillness radius 0.9": dict(loiter_radius_h=0.9),
                       "dedup across rules": dict(dedup_per_rule=True)},
}
OFF["lingering 60s+"] = OFF["lingering 20s+"]


def doing_at(rows, t):
    """ICVL's posture / movement / gesture for one person at time t (nearest labelled frame)."""
    k = min(range(len(rows)), key=lambda i: abs(rows[i][0] - t))
    _, p, m, gst, ev = rows[k]
    return POSTURE.get(p, p), MOVE.get(m, m), GESTURE.get(gst, gst), ev


def main():
    misses, fp, fpr = [], Counter(), defaultdict(Counter)
    fp_by_rule = defaultdict(Counter)
    for g, res in load_runs():
        fps = g.seq.native_fps
        alerts = replay_rules(res, CFG)[0]
        eps, people = labels_for(g.tag, fps)
        rows = defaultdict(list)
        for r in read_labels(ICVL / g.tag):
            rows[r[6]].append((r[0] / fps, r[7], r[8], r[9], r[10]))
        pos = {ep["pid"] for ev in EVENTS for ep in eps.get(ev, [])}
        on_any = [False] * len(alerts)
        replays = {}
        for ev in EVENTS:
            w = WINDOW_S.get(ev, TOL_S)
            for ep in eps.get(ev, []):
                tr = people[ep["pid"]]
                hits = [k for k, a in enumerate(alerts) if on_track(a, tr, ep["t0"], ep["t1"], w)]
                for k in hits:
                    on_any[k] = True
                if hits:
                    continue
                seen = coverage(res, ep)[0]
                near = [round(a["timestamp_s"] - ep["t0"], 1) for a in alerts
                        if on_track(a, tr, ep["t0"] - NEAR_S, ep["t1"] + NEAR_S, 0)]
                fixes = []
                for name, kw in OFF[ev].items():
                    if name not in replays:
                        replays[name] = replay_rules(res, replace(CFG, **kw))[0]
                    if any(on_track(a, tr, ep["t0"], ep["t1"], w) for a in replays[name]):
                        fixes.append(name)
                h = sorted(b[3] - b[1] for b in ep["boxes"])[len(ep["boxes"]) // 2]
                reason = "not seen" if seen < 0.5 else "filter" if fixes else "never fired"
                misses.append({"event": ev, "seq": res.seq_id, "pid": ep["pid"], "t0": round(ep["t0"], 1),
                               "camera": res.seq_id.split("/")[1], "seen": round(seen, 2), "height_px": round(h),
                               "reason": reason, "fixed_by": fixes, "alert_nearby_s": near})
        for k, a in enumerate(alerts):
            if on_any[k]:
                continue
            on = [pid for pid, tr in people.items() if on_track(a, tr)]
            rule = a["event_type"]
            if not on:
                kind = "nobody labelled"
            elif on[0] in pos:
                t = a["timestamp_s"]
                near = any(ep["pid"] == on[0] and ep["t0"] - NEAR_S <= t <= ep["t1"] + NEAR_S
                           for ev in EVENTS for ep in eps.get(ev, []))
                kind = "near a real incident (same person, within 10 s)" if near else "same person, other time"
            else:
                p, m, gst, ev = doing_at(rows[on[0]], a["timestamp_s"])
                kind = f"nothing we flag: {p} / {m}" + (f" / {gst}" if gst != "nothing" else "") + (" / FALLING" if ev == 2 else "")
            fp[kind] += 1
            fp_by_rule[rule][kind.split(":")[0]] += 1
        # per-person false positive rate, split by what the person mostly did
        for pid, tr in people.items():
            if pid in pos:
                continue
            r = rows[pid]
            moves = Counter(MOVE.get(x[2], x[2]) for x in r)
            gest = {GESTURE.get(x[3], x[3]) for x in r} - {"nothing"}
            postures = {POSTURE.get(x[1], x[1]) for x in r}
            flagged = any(on_track(a, tr) for a in alerts)
            groups = {f"mostly {moves.most_common(1)[0][0]}"} | {f"any {x}" for x in gest} | \
                     {f"any {x}" for x in postures & {"sitting", "bending", "lying"}} | {"all"}
            if any(x[4] == 2 for x in r):
                groups.add("any falling")
            for grp in groups:
                fpr[grp]["people"] += 1
                fpr[grp]["flagged"] += flagged

    out = {"misses": {}, "false_alarms": dict(fp.most_common()), "false_alarms_by_rule": {k: dict(v) for k, v in fp_by_rule.items()},
           "fpr_by_activity": {k: {**v, "rate_%": round(100 * v["flagged"] / v["people"], 1)}
                               for k, v in sorted(fpr.items(), key=lambda kv: -kv[1]["people"])},
           "miss_list": misses}
    for ev in EVENTS:
        m = [x for x in misses if x["event"] == ev]
        out["misses"][ev] = {"missed": len(m), "reason": dict(Counter(x["reason"] for x in m)),
                             "fixed_by": dict(Counter(f for x in m for f in x["fixed_by"])),
                             "alert_within_10s_outside_window": sum(bool(x["alert_nearby_s"]) for x in m),
                             "by_camera": dict(Counter(x["camera"] for x in m).most_common())}
    (HERE / "fp_fn.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "miss_list"}, indent=1))


if __name__ == "__main__":
    main()
