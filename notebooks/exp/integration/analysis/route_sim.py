import json, sys
from collections import Counter
from pathlib import Path
sys.path[:0] = ["exp/icvl", "."]
from fpr_attrib import owner
from score import EVENTS, TOL_S, WINDOW_S, labels_for, load_runs, on_track
out = Path(sys.argv[1])
run = json.loads((out / "run.json").read_text())
seq_of = {s["session_id"]: s["seq"] for s in run["sessions"]}
runs = {r.seq_id: g for g, r in load_runs("even")}
cands = [json.loads(l) for f in out.glob("*/candidates.jsonl") for l in f.read_text().splitlines()]
print("EVENTS", EVENTS)
DROPS = {"now": {}, "no unattended on reach/pose": {"reach_down": {"unattended_items"}, "pose_action": {"unattended_items"}},
         "no unattended anywhere on person rules": {"reach_down": {"unattended_items"}, "pose_action": {"unattended_items"}, "sustained_presence": set()}}
for name, drop in DROPS.items():
    c = Counter()
    for cand in cands:
        if cand["status"] != "completed": continue
        g = runs[seq_of[cand["session_id"]]]
        eps, people = labels_for(g.tag, g.seq.native_fps)
        a = {"timestamp_s": float(cand["timestamp_s"]), "roi_box": cand["roi_box"]}
        o = owner(a, people)
        real = {ev for ev in EVENTS for ep in eps.get(ev, []) if ep["pid"] == o and on_track(a, people[o], ep["t0"], ep["t1"], WINDOW_S.get(ev, TOL_S))} if o is not None else set()
        skip = drop.get(cand["event_type"], set())
        yes = {t for t, v in cand["vlm"].items() if v.get("decision") == "anomaly" and t not in skip}
        c["real" if real else "false"] += 1
        c["real & VLM yes" if real else "false & VLM yes"] += bool(yes)
        c[f"{cand['event_type']} false&yes"] += (not real) and bool(yes)
    mins = run["stream_s"] / 60 * len(run["plans"])
    print(name, dict(c), "false alarms/cam-hr", round(60 * c["false & VLM yes"] / mins, 1))
