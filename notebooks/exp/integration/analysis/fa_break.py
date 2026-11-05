"""Default gate, ICVL even: false alerts by rule and by kind (one alert -> one person)."""
import sys, json
from collections import Counter
sys.path[:0] = ["exp/icvl", "."]
from fpr_attrib import CFG, owner
from score import EVENTS, TOL_S, WINDOW_S, labels_for, load_runs, on_track
c = Counter(); tot = Counter()
for g, res in load_runs("even"):
    from stage1_gate import replay_rules
    alerts = replay_rules(res, CFG)[0]
    eps, people = labels_for(g.tag, g.seq.native_fps)
    for a in alerts:
        o = owner(a, people); rule = a["event_type"]; tot[rule] += 1
        if o is None:
            c[(rule, "on nobody labelled")] += 1; continue
        hit = [ev for ev in EVENTS for ep in eps.get(ev, []) if ep["pid"] == o and on_track(a, people[o], ep["t0"], ep["t1"], WINDOW_S.get(ev, TOL_S))]
        if hit: continue
        has = [ev for ev in EVENTS for ep in eps.get(ev, []) if ep["pid"] == o]
        c[(rule, "person has an incident, but not now" if has else "person with no incident")] += 1
print("alerts", sum(tot.values()), "false", sum(c.values()))
for rule in tot:
    f = {k[1]: v for k, v in c.items() if k[0] == rule}
    print(f"{rule:20s} alerts {tot[rule]:4d} false {sum(f.values()):4d}", f)
