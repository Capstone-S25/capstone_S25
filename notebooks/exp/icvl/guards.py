"""Person-rule guards (GateConfig still_border_margin / still_min_h / reach_min_h / hand_min_age_s /
pose_border_margin / reach_border_margin) on cached ICVL runs, one alert -> one person (fpr_attrib.owner).
Per variant and half: incidents caught, normal people flagged, alerts and the share of them that are false.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/guards.py [odd|even] [VARIANT ...]   -> exp/icvl/guards_<half>.json
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from fpr_attrib import CFG, owner  # noqa: E402  (also loads the cached crop-check scores)
from score import EVENTS, TOL_S, WINDOW_S, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import replay_rules  # noqa: E402

VARIANTS = {
    "default": {},
    **{f"still_border_{m}": {"still_border_margin": m} for m in (0.01, 0.03, 0.05)},
    **{f"still_min_h_{h}": {"still_min_h": h} for h in (0.1, 0.15, 0.2)},
    **{f"reach_border_{m}": {"reach_border_margin": m} for m in (0.03, 0.05)},
    **{f"pose_border_{m}": {"pose_border_margin": m} for m in (0.01, 0.03)},
    **{f"reach_min_h_{h}": {"reach_min_h": h} for h in (0.1, 0.15)},
    **{f"hand_age_{a}": {"hand_min_age_s": a} for a in (0.5, 1.0, 2.0)},
}


def evaluate(runs, cfg):
    caught = {ev: [0, 0] for ev in EVENTS}
    neg = flagged = alerts_n = false_n = 0
    by_rule = {}
    for g, res in runs:
        alerts = replay_rules(res, cfg)[0]
        eps, people = labels_for(g.tag, g.seq.native_fps)
        own = [owner(a, people) for a in alerts]
        hit = [False] * len(alerts)
        pos = set()
        for ev in EVENTS:
            w = WINDOW_S.get(ev, TOL_S)
            for ep in eps.get(ev, []):
                pos.add(ep["pid"])
                ks = [k for k, a in enumerate(alerts) if own[k] == ep["pid"] and on_track(a, people[ep["pid"]], ep["t0"], ep["t1"], w)]
                caught[ev][1] += 1
                caught[ev][0] += bool(ks)
                for k in ks:
                    hit[k] = True
        for pid in people:
            if pid not in pos:
                neg += 1
                flagged += pid in own
        for a, h in zip(alerts, hit):
            r = by_rule.setdefault(a["event_type"], [0, 0])
            r[0] += 1
            r[1] += not h
        alerts_n += len(alerts)
        false_n += hit.count(False)
    return {**{ev: f"{c}/{n}" for ev, (c, n) in caught.items()}, "fpr_%": round(100 * flagged / neg, 1),
            "alerts": alerts_n, "false_%": round(100 * false_n / max(1, alerts_n), 1),
            "by_rule_false": {k: f"{f}/{n}" for k, (n, f) in by_rule.items()}}


def main():
    half = sys.argv[1] if len(sys.argv) > 1 else "odd"
    only = sys.argv[2:]
    runs, out = load_runs(half), {}
    for name, kw in VARIANTS.items():
        if only and name not in only:
            continue
        out[name] = evaluate(runs, replace(CFG, **kw))
        print(name, json.dumps(out[name]), flush=True)
    (HERE / f"guards_{half}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
