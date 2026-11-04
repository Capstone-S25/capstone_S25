"""Is ICVL littering caught by luck? reach_down also fires on ordinary walkers, so a litterer could be "caught" by an
alert that has nothing to do with the drop. For each littering incident: actual = an alert on that person within the
incident window (score.py); chance = the same person's window slid to every other time the person is in view (all
labelled times outside the incident +- 5 s), i.e. how often an equally long window at a random moment of that person's
visit has an alert on them. Same alerts, same matching. ODD half (tune) by default.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/chance.py [odd|even|all] ['<json GateConfig overrides>']
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from score import DEFAULT_RULES, WINDOW_S, GateConfig, labels_for, load_runs, on_track, replay_rules  # noqa: E402


def main():
    half = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("odd", "even") else None
    kw = json.loads(sys.argv[-1]) if sys.argv[-1].startswith("{") else {}
    cfg = GateConfig(**{"inview_s": 60.0, "rules": DEFAULT_RULES + ("pose_action",), **kw})
    w = WINDOW_S["littering"]
    actual, chance, n, m, actual_m = 0, 0.0, 0, 0, 0
    by_rule_actual, by_rule_chance = {}, {}
    for g, res in load_runs(half):
        episodes, people = labels_for(g.tag, g.seq.native_fps)
        eps = episodes.get("littering", [])
        if not eps:
            continue
        alerts = replay_rules(res, cfg)[0]
        for ep in eps:
            tr = people[ep["pid"]]
            n += 1
            on = [a for a in alerts if on_track(a, tr, ep["t0"], ep["t1"], w)]
            actual += bool(on)
            for rule in {a["event_type"] for a in on}:
                by_rule_actual[rule] = by_rule_actual.get(rule, 0) + 1
            span = ep["t1"] - ep["t0"]
            ts = np.array(tr["ts"])
            starts = [s for s in np.arange(ts[0], ts[-1] - span + 1e-6, 0.5)
                      if s + span + w < ep["t0"] - 5 or s - w > ep["t1"] + 5]
            if not starts:
                continue
            m += 1
            actual_m += bool(on)
            on_person = [a for a in alerts if on_track(a, tr)]
            hits = [any(s - w <= a["timestamp_s"] <= s + span + w for a in on_person) for s in starts]
            chance += float(np.mean(hits))
            for rule in {a["event_type"] for a in on_person}:
                rh = [any(s - w <= a["timestamp_s"] <= s + span + w for a in on_person if a["event_type"] == rule) for s in starts]
                by_rule_chance[rule] = by_rule_chance.get(rule, 0.0) + float(np.mean(rh))
    print(json.dumps({"incidents": n, "caught": actual, "caught_%": round(100 * actual / n, 1),
                      "incidents_with_time_in_view_outside_the_drop": m, "caught_of_those_%": round(100 * actual_m / max(m, 1), 1),
                      "chance_of_those_%": round(100 * chance / max(m, 1), 1),
                      "by_rule_caught_%": {k: round(100 * v / n, 1) for k, v in by_rule_actual.items()},
                      "by_rule_chance_%": {k: round(100 * v / max(m, 1), 1) for k, v in by_rule_chance.items()}}, indent=1))


if __name__ == "__main__":
    main()


def walkers(half="odd", kw=None, event="littering"):
    """Chance from other people: in videos with the event, slide an event-length window (median incident span +-
    the event's window) along every labelled person who never has that event; share of windows with an alert on them."""
    from score import TOL_S
    cfg = GateConfig(**{"inview_s": 60.0, "rules": DEFAULT_RULES + ("pose_action",), **(kw or {})})
    w = WINDOW_S.get(event, TOL_S)
    spans, hits = [], []
    runs = load_runs(half)
    for g, res in runs:
        episodes, _ = labels_for(g.tag, g.seq.native_fps)
        spans += [ep["t1"] - ep["t0"] for ep in episodes.get(event, [])]
    span = float(np.median(spans))
    for g, res in runs:
        episodes, people = labels_for(g.tag, g.seq.native_fps)
        if not episodes.get(event):
            continue
        litterers = {ep["pid"] for ep in episodes[event]}
        alerts = replay_rules(res, cfg)[0]
        for pid, tr in people.items():
            if pid in litterers or tr["ts"][-1] - tr["ts"][0] < span:
                continue
            on_person = [a["timestamp_s"] for a in alerts if on_track(a, tr)]
            for s in np.arange(tr["ts"][0], tr["ts"][-1] - span + 1e-6, 1.0):
                hits.append(any(s - w <= t <= s + span + w for t in on_person))
    return {"window_s": round(span + 2 * w, 1), "walker_windows": len(hits), "chance_%": round(100 * float(np.mean(hits)), 1)}
