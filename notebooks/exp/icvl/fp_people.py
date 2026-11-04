"""Who are the people with no labelled incident that the gate flags, and why? All of ICVL, current defaults
(15 s pose-cue gap, as in the deck).

A person counts as falsely flagged when they have no littering / smoking / still-20 s+ incident in their video and at
least one alert lands on them (same matching as score.py). For each: which rules flagged them, how many alerts, and
what ICVL says they were doing at each alert (posture / movement / gesture).
Usage (from notebooks/):
  ../.venv/bin/python exp/icvl/fp_people.py            -> exp/icvl/fp_people.json
  ../.venv/bin/python exp/icvl/fp_people.py sheet OUT  -> contact sheet of sample alerts per rule (person crops)
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from fp_fn import doing_at  # noqa: E402
from inventory import ICVL, read_labels, video_for  # noqa: E402
from one_per_person import CFG  # noqa: E402
from score import EVENTS, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import replay_rules  # noqa: E402

RULE = {"reach_down": "reach-down", "pose_action": "pose cue", "sustained_presence": "stays put",
        "long_presence": "long in view", "object_appeared": "item appeared", "unattended_object": "bag left alone"}


def flagged_people():
    out = []
    for g, res in load_runs():
        fps = g.seq.native_fps
        alerts = replay_rules(res, CFG)[0]
        eps, people = labels_for(g.tag, fps)
        pos = {ep["pid"] for ev in EVENTS for ep in eps.get(ev, [])}
        rows = defaultdict(list)
        for r in read_labels(ICVL / g.tag):
            rows[r[6]].append((r[0] / fps, r[7], r[8], r[9], r[10]))
        for pid, tr in people.items():
            if pid in pos:
                continue
            mine = [a for a in alerts if on_track(a, tr)]
            if not mine:
                continue
            h = sorted(b[3] - b[1] for b in tr["boxes"])[len(tr["boxes"]) // 2]
            out.append({"seq": res.seq_id, "tag": g.tag, "fps": fps, "pid": pid, "camera": res.seq_id.split("/")[1],
                        "height_px": round(h), "in_view_s": round(tr["ts"][-1] - tr["ts"][0], 1),
                        "alerts": [{"t": a["timestamp_s"], "rule": RULE.get(a["event_type"], a["event_type"]),
                                    "roi_box": a["roi_box"], "doing": " / ".join(
                                        x for x in doing_at(rows[pid], a["timestamp_s"])[:3] if x != "nothing")}
                                   for a in mine]})
    return out


def summary(people):
    first = Counter(p["alerts"][0]["rule"] for p in people)
    combo = Counter(" + ".join(sorted({a["rule"] for a in p["alerts"]})) for p in people)
    by_rule_doing = defaultdict(Counter)
    for p in people:
        for a in p["alerts"]:
            by_rule_doing[a["rule"]][a["doing"]] += 1
    n_alerts = Counter(min(len(p["alerts"]), 5) for p in people)
    return {"people": len(people), "alerts": sum(len(p["alerts"]) for p in people),
            "people_by_rules_that_flagged_them": dict(combo.most_common()),
            "people_by_first_rule": dict(first.most_common()),
            "alerts_per_person": {("5+" if k == 5 else str(k)): v for k, v in sorted(n_alerts.items())},
            "alerts_by_rule_and_activity": {r: dict(c.most_common()) for r, c in by_rule_doing.items()},
            "people_by_camera": dict(Counter(p["camera"] for p in people).most_common()),
            "median_height_px": sorted(p["height_px"] for p in people)[len(people) // 2]}


def sheet(people, path, per_rule=6, seed=0):
    rng = random.Random(seed)
    picks = []
    for rule in ("reach-down", "pose cue", "stays put", "long in view"):
        pool = [(p, a) for p in people for a in p["alerts"] if a["rule"] == rule]
        picks += rng.sample(pool, min(per_rule, len(pool)))
    tiles = []
    for p, a in picks:
        cap = cv2.VideoCapture(str(video_for(ICVL / p["tag"])))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(a["t"] * p["fps"])))
        ok, im = cap.read()
        if not ok:
            continue
        x1, y1, x2, y2 = a["roi_box"]
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 140, 255), 3)
        # crop around the person, 1.6x their box, at least 360 px tall, so the hands are visible
        cx, cy, hh = (x1 + x2) / 2, (y1 + y2) / 2, max(360, 1.6 * (y2 - y1))
        ww = hh * 0.8
        X1, Y1 = int(max(0, cx - ww / 2)), int(max(0, cy - hh / 2))
        crop = im[Y1:int(min(im.shape[0], cy + hh / 2)), X1:int(min(im.shape[1], cx + ww / 2))]
        crop = cv2.resize(crop, (288, 360))
        cv2.rectangle(crop, (0, 0), (288, 44), (20, 20, 20), -1)
        cv2.putText(crop, a["rule"], (6, 18), 0, 0.55, (90, 190, 255), 2)
        cv2.putText(crop, f'{a["doing"] or "-"}  cam {p["camera"]}', (6, 38), 0, 0.45, (235, 235, 235), 1)
        tiles.append(crop)
    rows = [np.hstack(tiles[i:i + per_rule]) for i in range(0, len(tiles), per_rule) if len(tiles[i:i + per_rule]) == per_rule]
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    people = flagged_people()
    if len(sys.argv) > 2 and sys.argv[1] == "sheet":
        sheet(people, sys.argv[2])
        return
    out = summary(people)
    (HERE / "fp_people.json").write_text(json.dumps({**out, "list": people}, indent=1, default=float))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
