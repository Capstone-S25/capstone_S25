"""Loitering stand-in by time in view rather than stillness, on cached ICVL runs (default gate, crop check on,
one alert -> one person as guards.py).
"In view N s+" = a labelled person whose first and last labelled frames are at least N s apart, however they move.
Caught = an alert owned by that person between their first and last labelled frame (+- TOL_S); counted for any rule
and for the two dwell rules alone (stays put, long in view), since anyone in view that long tends to get some alert.
Normal people = no incident under any definition (score.py events plus in view 20 s+).
Usage (from notebooks/): ../.venv/bin/python exp/icvl/inview.py [odd|even]   -> exp/icvl/inview_<half>.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from fpr_attrib import CFG, owner  # noqa: E402  (also loads the cached crop-check scores)
from score import EVENTS, TOL_S, WINDOW_S, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import replay_rules  # noqa: E402

INVIEW_S = (20.0, 60.0)
DWELL_RULES = ("sustained_presence", "long_presence")


def main():
    half = sys.argv[1] if len(sys.argv) > 1 else "even"
    caught = {ev: [0, 0] for ev in EVENTS}
    inview = {n: {"incidents": 0, "any_rule": 0, "dwell_rules": 0, "never_still_20s": 0} for n in INVIEW_S}
    neg = flagged = neg_old = flagged_old = 0
    for g, res in load_runs(half):
        alerts = replay_rules(res, CFG)[0]
        eps, people = labels_for(g.tag, g.seq.native_fps)
        own = [owner(a, people) for a in alerts]
        pos = set()
        for ev in EVENTS:
            w = WINDOW_S.get(ev, TOL_S)
            for ep in eps.get(ev, []):
                pos.add(ep["pid"])
                hit = any(own[k] == ep["pid"] and on_track(a, people[ep["pid"]], ep["t0"], ep["t1"], w) for k, a in enumerate(alerts))
                caught[ev][0] += hit
                caught[ev][1] += 1
        still20 = {ep["pid"] for ep in eps.get("lingering 20s+", [])}
        long_pos = set()
        for pid, tr in people.items():
            t0, t1 = tr["ts"][0], tr["ts"][-1]
            for n in INVIEW_S:
                if t1 - t0 < n:
                    continue
                long_pos.add(pid)
                ks = [k for k, a in enumerate(alerts) if own[k] == pid and on_track(a, tr, t0, t1)]
                c = inview[n]
                c["incidents"] += 1
                c["any_rule"] += bool(ks)
                c["dwell_rules"] += any(alerts[k]["event_type"] in DWELL_RULES for k in ks)
                c["never_still_20s"] += pid not in still20
        for pid in people:
            if pid not in pos:
                neg_old += 1
                flagged_old += pid in own
                if pid not in long_pos:
                    neg += 1
                    flagged += pid in own
    out = {"half": half, **{ev: f"{c}/{n}" for ev, (c, n) in caught.items()},
           **{f"in view {int(n)}s+": c for n, c in inview.items()},
           "normal_flagged_old": f"{flagged_old}/{neg_old} ({100 * flagged_old / neg_old:.1f} %)",
           "normal_flagged_new": f"{flagged}/{neg} ({100 * flagged / neg:.1f} %)"}
    print(json.dumps(out, indent=1))
    (HERE / f"inview_{half}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
