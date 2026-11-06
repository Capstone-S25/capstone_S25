"""Is the stays-put rule (sustained_presence, same spot 20 s) worth keeping? ICVL held-out half, default gate,
crop check on, one alert -> one person (as inview.py), with and without it.
Per variant: smoking / littering caught, in view 20 s+ / 60 s+ caught (any alert), normal people flagged (no incident
under any definition), alerts and alerts per minute; plus incidents caught ONLY by stays put in the default gate.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/staysput.py [odd|even]   -> exp/icvl/staysput_<half>.json
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from fpr_attrib import CFG, owner  # noqa: E402  (also loads the cached crop-check scores)
from score import TOL_S, WINDOW_S, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import replay_rules  # noqa: E402

RULE = "sustained_presence"   # the event type the "loiter" rule (stays put) sends


def incidents(eps, people):
    """(name, pid, t0, t1, window) for smoking, littering and in view 20 / 60 s+."""
    out = [(ev, ep["pid"], ep["t0"], ep["t1"], WINDOW_S.get(ev, TOL_S)) for ev in ("smoking", "littering") for ep in eps.get(ev, [])]
    for pid, tr in people.items():
        t0, t1 = tr["ts"][0], tr["ts"][-1]
        out += [(f"in view {n}s+", pid, t0, t1, TOL_S) for n in (20, 60) if t1 - t0 >= n]
    return out


def main():
    half = sys.argv[1] if len(sys.argv) > 1 else "even"
    variants = {"default": CFG, "without stays put": replace(CFG, rules=tuple(r for r in CFG.rules if r != "loiter"))}
    tot = {v: {"caught": {}, "n": {}, "neg": 0, "flagged": 0, "alerts": 0} for v in variants}
    only = {}
    minutes = 0.0
    for g, res in load_runs(half):
        eps, people = labels_for(g.tag, g.seq.native_fps)
        inc = incidents(eps, people)
        pos = {i[1] for i in inc}
        minutes += res.duration_s / 60
        for v, cfg in variants.items():
            alerts = replay_rules(res, cfg)[0]
            own = [owner(a, people) for a in alerts]
            t = tot[v]
            t["alerts"] += len(alerts)
            for name, pid, t0, t1, w in inc:
                ks = [k for k, a in enumerate(alerts) if own[k] == pid and on_track(a, people[pid], t0, t1, w)]
                t["n"][name] = t["n"].get(name, 0) + 1
                t["caught"][name] = t["caught"].get(name, 0) + bool(ks)
                if v == "default" and ks and all(alerts[k]["event_type"] == RULE for k in ks):
                    only[name] = only.get(name, 0) + 1
            for pid in people:
                if pid not in pos:
                    t["neg"] += 1
                    t["flagged"] += pid in own
    out = {"half": half}
    for v, t in tot.items():
        out[v] = {**{k: f'{t["caught"][k]}/{n}' for k, n in t["n"].items()},
                  "normal_flagged": f'{t["flagged"]}/{t["neg"]} ({100 * t["flagged"] / t["neg"]:.1f} %)',
                  "alerts": t["alerts"], "alerts_per_min": round(t["alerts"] / minutes, 2)}
    out["caught only by stays put (default)"] = only
    print(json.dumps(out, indent=1))
    (HERE / f"staysput_{half}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
