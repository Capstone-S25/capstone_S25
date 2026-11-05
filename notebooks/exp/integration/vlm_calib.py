"""Gate + VLM end to end on ICVL, with the VLM's thresholds re-chosen on the odd half and judged on the held-out even
half (vlm_cands.py -> vlm_scores.py -> this). One alert -> one person (fpr_attrib.owner), as the deck counts.

What an operator sees is a candidate the VLM calls an anomaly on any task asked. Per setting:
  caught       incidents (per labelled person, as score.py) with a candidate on that person that the VLM confirmed on
               the matching task (littering / smoking / loitering for "lingering 20s+" and "60s+")
  fpr_%        people with no incident who raise at least one operator alarm
  false_per_h  operator alarms on no incident, per camera-hour of video
Settings: "bundle" (Yuchuan's thresholds as shipped), then for each share q, every task's threshold set so that a share
q of the odd half's false candidates asked that task pass it; unattended_items / rough_sleeping have no ICVL labels so
their thresholds come from the same rule. Routing variants drop tasks per rule.
Usage (from notebooks/): ../.venv/bin/python exp/integration/vlm_calib.py OUT_DIR   (OUT_DIR from vlm_cands.py)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[1] / "exp" / "icvl")]

from fpr_attrib import owner  # noqa: E402
from score import EVENTS, TOL_S, WINDOW_S, labels_for, load_runs, on_track  # noqa: E402

TASK_FOR = {"littering": "littering", "smoking": "smoking", "lingering 20s+": "loitering", "lingering 60s+": "loitering"}
PERSON_RULES = ("reach_down", "pose_action", "sustained_presence", "long_presence")
LIVE = {"reach_down": {"littering", "unattended_items"}, "pose_action": {"littering", "unattended_items"},
        "sustained_presence": {"loitering", "smoking", "rough_sleeping"}, "long_presence": {"loitering", "rough_sleeping"}}
ROUTINGS = {
    "live": LIVE,
    "live, no unattended on put-down": {**LIVE, "reach_down": {"littering"}, "pose_action": {"littering"}},
    "+ smoking on pose cue": {**LIVE, "reach_down": {"littering"}, "pose_action": {"littering", "smoking"}},
    "+ smoking on pose cue, no rough sleeping": {"reach_down": {"littering"}, "pose_action": {"littering", "smoking"},
                                                 "sustained_presence": {"loitering", "smoking"}, "long_presence": {"loitering"}},
}


def load(out, half):
    """Per candidate: owner, the incidents it is on, scores; plus per-sequence incident and negative-person lists."""
    rows = [json.loads(l) for f in sorted((Path(out) / half).glob("scores_*.jsonl")) for l in f.read_text().splitlines()]
    rows = list({r["event_id"]: r for r in rows}.values())
    by_seq = defaultdict(list)
    for r in rows:
        by_seq[r["seq"]].append(r)
    incidents, negatives, hours = [], [], 0.0
    for g, res in load_runs(half):
        hours += res.duration_s / 3600
        eps, people = labels_for(g.tag, g.seq.native_fps)
        pos = set()
        for ev in EVENTS:
            for n, ep in enumerate(eps.get(ev, [])):
                pos.add(ep["pid"])
                incidents.append((res.seq_id, ev, n))
        negatives += [(res.seq_id, pid) for pid in people if pid not in pos]
        for r in by_seq.get(res.seq_id, []):
            o = owner(r, people)
            r["owner"] = o
            r["episodes"] = {(ev, n) for ev in EVENTS for n, ep in enumerate(eps.get(ev, [])) if ep["pid"] == o and
                             on_track(r, people[o], ep["t0"], ep["t1"], WINDOW_S.get(ev, TOL_S))} if o is not None else set()
            r["real"] = {ev for ev, _ in r["episodes"]}
    return rows, incidents, negatives, hours


def evaluate(rows, incidents, negatives, hours, yes):
    """yes(row) -> set of tasks the VLM confirms."""
    caught, alarmed, false = set(), set(), 0
    for r in rows:
        y = yes(r)
        for ev, n in r["episodes"]:
            if TASK_FOR[ev] in y:
                caught.add((r["seq"], ev, n))
        if y:
            alarmed.add((r["seq"], r["owner"]))
            false += not r["real"]
    out = {}
    for ev in EVENTS:
        tot = [i for i in incidents if i[1] == ev]
        out[ev] = f"{sum(i in caught for i in tot)}/{len(tot)}"
    out["fpr_%"] = round(100 * sum(n in alarmed for n in negatives) / len(negatives), 1)
    out["false_per_h"] = round(false / hours, 1)
    out["alarms_per_h"] = round(sum(bool(yes(r)) for r in rows) / hours, 1)
    return out


def gate_only(rows, incidents, negatives, hours):
    return evaluate(rows, incidents, negatives, hours, lambda r: {TASK_FOR[ev] for ev in EVENTS} | {"any"})


def main(out):
    odd, even = load(out, "odd"), load(out, "even")
    res = {"gate only (every candidate shown)": gate_only(*even),
           "bundle thresholds, live routing": evaluate(*even, lambda r: {t for t, d in r["decisions"].items()
                                                                          if d == "anomaly" and t in LIVE.get(r["event_type"], ())})}
    for name, routing in ROUTINGS.items():
        for q in (0.02, 0.05, 0.1, 0.2, 0.3):
            thr = {}
            for t in ("littering", "smoking", "loitering", "rough_sleeping", "unattended_items"):
                neg = [r["scores"][t] for r in odd[0] if not r["real"] and r["scores"].get(t) is not None
                       and t in routing.get(r["event_type"], ())]
                thr[t] = float(np.quantile(neg, 1 - q)) if neg else float("inf")
            yes = lambda r, thr=thr, routing=routing: {t for t in routing.get(r["event_type"], ())
                                                       if r["scores"].get(t) is not None and r["scores"][t] >= thr[t]}
            res[f"{name}, q={q}"] = {"odd": evaluate(*odd, yes), "even": evaluate(*even, yes)}
    (HERE / "vlm_calib.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, json.dumps(v.get("even", v)))


if __name__ == "__main__":
    main(sys.argv[1])
