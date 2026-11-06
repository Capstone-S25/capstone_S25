"""Does the pose cue (learned skeleton model) earn its place, or would the handwritten hand-to-face rule ("smoking":
wrist within hand_face_h body heights of the face, smoke_min_hits times in smoke_window_s) do the same?
ICVL, default gate + crop check, one alert -> one person (as staysput.py). Variants: pose cue (default), neither,
hand-to-face rule instead, hand-to-face rule with a few looser / stricter settings. Also counts, in the default
gate, incidents caught ONLY by the pose cue.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/pose_vs_rule.py [odd|even]   -> exp/icvl/pose_vs_rule_<half>.json
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from fpr_attrib import CFG, owner  # noqa: E402  (also loads the cached crop-check scores)
from score import load_runs, labels_for, on_track  # noqa: E402
from staysput import incidents  # noqa: E402
from stage1_gate import replay_rules  # noqa: E402

BASE = tuple(r for r in CFG.rules if r != "pose_action")
VARIANTS = {
    "pose cue (default)": CFG,
    "neither": replace(CFG, rules=BASE),
    "hand-to-face rule": replace(CFG, rules=BASE + ("smoking",)),
    "hand-to-face, 1 hit": replace(CFG, rules=BASE + ("smoking",), smoke_min_hits=1),
    "hand-to-face, 0.15 h, 3 hits": replace(CFG, rules=BASE + ("smoking",), hand_face_h=0.15, smoke_min_hits=3),
    "pose cue + hand-to-face": replace(CFG, rules=CFG.rules + ("smoking",)),
    **{f"hand-to-face, {h} h, {n} hits": replace(CFG, rules=BASE + ("smoking",), hand_face_h=h, smoke_min_hits=n)
       for h, n in ((0.12, 3), (0.12, 4), (0.10, 3), (0.10, 4), (0.08, 3))},
}


def main():
    half = sys.argv[1] if len(sys.argv) > 1 else "even"
    tot = {v: {"caught": {}, "n": {}, "neg": 0, "flagged": 0, "alerts": 0, "rule_alerts": 0} for v in VARIANTS}
    only = {}
    minutes = 0.0
    for g, res in load_runs(half):
        eps, people = labels_for(g.tag, g.seq.native_fps)
        inc = incidents(eps, people)
        pos = {i[1] for i in inc}
        minutes += res.duration_s / 60
        for v, cfg in VARIANTS.items():
            alerts = replay_rules(res, cfg)[0]
            own = [owner(a, people) for a in alerts]
            t = tot[v]
            t["alerts"] += len(alerts)
            t["rule_alerts"] += sum(a["event_type"] in ("pose_action", "smoking") for a in alerts)
            for name, pid, t0, t1, w in inc:
                ks = [k for k, a in enumerate(alerts) if own[k] == pid and on_track(a, people[pid], t0, t1, w)]
                t["n"][name] = t["n"].get(name, 0) + 1
                t["caught"][name] = t["caught"].get(name, 0) + bool(ks)
                if v == "pose cue (default)" and ks and all(alerts[k]["event_type"] == "pose_action" for k in ks):
                    only[name] = only.get(name, 0) + 1
            for pid in people:
                if pid not in pos:
                    t["neg"] += 1
                    t["flagged"] += pid in own
    out = {"half": half}
    for v, t in tot.items():
        out[v] = {**{k: f'{t["caught"][k]}/{n}' for k, n in t["n"].items() if k != "in view 20s+"},
                  "normal_flagged": f'{100 * t["flagged"] / t["neg"]:.1f} %', "alerts_per_min": round(t["alerts"] / minutes, 2),
                  "pose_or_hand_alerts": t["rule_alerts"]}
        print(v, json.dumps(out[v]), flush=True)
    out["caught only by the pose cue (default)"] = only
    print(json.dumps(only))
    (HERE / f"pose_vs_rule_{half}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
