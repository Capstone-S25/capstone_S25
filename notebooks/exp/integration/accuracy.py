"""What would an operator see? Match an integration run's candidates (run_integration.py OUT) to ICVL labels with the
one-alert-one-person rule (fpr_attrib.owner) and count, per routed VLM question: candidates on a real labelled
incident and whether the VLM confirmed them, and candidates on no incident that the VLM still called an anomaly
(the false alarms an operator would get).
Usage (from notebooks/): ../.venv/bin/python exp/integration/accuracy.py OUT
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parents[1] / "exp" / "icvl")]

from fpr_attrib import owner  # noqa: E402
from score import EVENTS, TOL_S, WINDOW_S, labels_for, load_runs, on_track  # noqa: E402

TASK_EVENT = {"littering": "littering", "smoking": "smoking", "loitering": "lingering 20s+"}


def main(out):
    out = Path(out)
    run = json.loads((out / "run.json").read_text())
    seq_of = {s["session_id"]: s["seq"] for s in run["sessions"]}
    runs = {r.seq_id: g for g, r in load_runs("even")}
    cands = [json.loads(l) for f in out.glob("*/candidates.jsonl") for l in f.read_text().splitlines()]
    c = Counter()
    for cand in cands:
        if cand["status"] != "completed":
            continue
        g = runs[seq_of[cand["session_id"]]]
        eps, people = labels_for(g.tag, g.seq.native_fps)
        a = {"timestamp_s": float(cand["timestamp_s"]), "roi_box": cand["roi_box"]}
        o = owner(a, people)
        real = {ev for ev in EVENTS for ep in eps.get(ev, []) if ep["pid"] == o and
                on_track(a, people[o], ep["t0"], ep["t1"], WINDOW_S.get(ev, TOL_S))} if o is not None else set()
        yes = {t for t, v in cand["vlm"].items() if v.get("decision") == "anomaly"}
        c["candidates"] += 1
        c["on a real incident" if real else "on no incident"] += 1
        if not real and yes:
            c["false candidate the VLM called an anomaly (operator false alarm)"] += 1
        if not real and not yes:
            c["false candidate the VLM rejected"] += 1
        for task, ev in TASK_EVENT.items():
            if ev in real and task in cand["tasks"]:
                c[f"{task}: real incident asked"] += 1
                c[f"{task}: real incident confirmed"] += task in yes
    minutes = run["stream_s"] / 60 * len(run["plans"])
    res = dict(c)
    res["operator false alarms per camera-hour"] = round(60 * c["false candidate the VLM called an anomaly (operator false alarm)"] / minutes, 1)
    res["camera-minutes"] = round(minutes, 1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
