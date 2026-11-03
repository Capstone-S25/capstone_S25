"""Raw counts per event: incidents, caught (recall), alerts sent and how many were on the incident (precision).

Current GateConfig defaults, rules + pose cue, footage not used for tuning:
  littering   held-out clips (every 3 s clip contains littering, so precision is optimistic)
  throwing    Avenue held-out test + normal videos, scored on the label review's in-scope THROWING incidents only
              (running is not one of the project's events; its frames count as normal footage). Avenue has no
              labelled loitering, so loitering is not measured here.
  smoking     held-out camera-days
  dumping     IWDD test: 50 dumping + 50 normal videos; an alert is "on the incident" if it is in a dumping video in
              [onset - 1 s, onset + 10 s]; caught = any such alert
  bags        ABODA: an alert is "on the incident" if its crop covers a labelled bag between put-down - 2 s and
              removal - 2 s; caught = any such alert
"Right person" counts an incident only when an alert's crop is on the labelled person (littering / loitering /
smoking); dumping and bags have no person boxes.
Usage (from notebooks/): ../.venv/bin/python exp/counts_precision.py   -> exp/counts_precision.json
"""
import json
import pickle
import sys
from functools import partial
from pathlib import Path

NB = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "newcam"), str(NB / "exp" / "alerts"),
                str(NB / "exp" / "gt_review")]

from compare_models import model_splits  # noqa: E402
from cached import load_cached, splits  # noqa: E402
from exp_harness import baseline  # noqa: E402
from iwdd_eval import in_window  # noqa: E402
from loaders import iwdd_labels  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, intersection_over  # noqa: E402

sys.path.insert(0, str(NB / "exp" / "gt_review"))
from scoped import scoped  # noqa: E402


def throwing_labels():
    """events.csv with only the in-scope, non-running Avenue incidents kept (as a group decision)."""
    import csv
    import tempfile
    rows = list(csv.DictReader(open(NB / "exp" / "gt_review" / "events.csv")))
    for r in rows:
        r["group_decision"] = "in" if (r["agreed_tag"] == "in" and "running" not in r["rule"]) else "out"
    path = Path(tempfile.mkdtemp()) / "events_throwing.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def from_bench(rows, minutes):
    s = summarize(rows, "x")
    return {"incidents": s["gt_events"], "caught": round(s["event_recall_%"] * s["gt_events"] / 100),
            "right_person": round(s["crop_recall_%"] * s["gt_events"] / 100), "alerts": s["candidates"],
            "alerts_on_incident": round(s["candidate_precision_%"] * s["candidates"] / 100), "minutes": round(minutes, 1)}


def main():
    m = partial(baseline, cfg=GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",)))
    H = splits(heldout=True)
    out = {}
    lit = [evaluate_run(g, r, m(r, "littering")) for g, r in H["littering"]]
    out["littering"] = from_bench(lit, sum(r.duration_s for _, r in H["littering"]) / 60)
    throwing = scoped(H["avenue"], keep=("in",), path=throwing_labels())
    av = [evaluate_run(g, r, m(r, "avenue")) for g, r in throwing] + \
         [evaluate_run(g, r, m(r, "normal")) for g, r in H["normal"]]
    out["littering: throwing on a walkway (Avenue)"] = from_bench(
        av, sum(r.duration_s for _, r in H["avenue"] + H["normal"]) / 60)
    sm = [evaluate_run(g, r, m(r, "smoking")) for g, r in H["smoking"]]
    out["smoking"] = from_bench(sm, sum(r.duration_s for _, r in H["smoking"]) / 60)

    lab = iwdd_labels("test")
    inc = caught = alerts = on = 0
    mins = 0.0
    for _, r in load_cached("iwdd_test"):
        evs, L = m(r, "littering"), lab[r.seq_id]
        alerts += len(evs)
        mins += r.duration_s / 60
        if L["positive"]:
            inc += 1
            hits = [e for e in evs if in_window(e["timestamp_s"], L["onset_s"], 10)]
            on += len(hits)
            caught += bool(hits)
    out["illegal dumping"] = {"incidents": inc, "caught": caught, "right_person": None, "alerts": alerts,
                              "alerts_on_incident": on, "minutes": round(mins, 1)}

    blab = json.loads((NB / "exp/newcam/aboda_labels.json").read_text())
    inc = caught = alerts = on = 0
    mins = 0.0
    for _, r in load_cached("aboda"):
        evs = m(r, "aboda")
        alerts += len(evs)
        mins += r.duration_s / 60
        bags = blab[r.seq_id.split("/")[-1]]["events"]
        good = set()
        for b in bags:
            inc += 1
            end = b["t_removed"] - 2 if b["t_removed"] is not None else float("inf")
            hits = [k for k, e in enumerate(evs) if b["t_put_down"] - 2 <= e["timestamp_s"] <= end and
                    intersection_over(b["box"], e["roi_box"]) >= 0.3]
            caught += bool(hits)
            good.update(hits)
        on += len(good)
    out["unattended items"] = {"incidents": inc, "caught": caught, "right_person": None, "alerts": alerts,
                               "alerts_on_incident": on, "minutes": round(mins, 1)}

    for k, v in out.items():
        v["recall_%"] = round(100 * v["caught"] / v["incidents"])
        v["precision_%"] = round(100 * v["alerts_on_incident"] / v["alerts"]) if v["alerts"] else None
        print(k, json.dumps(v), flush=True)
    (NB / "exp" / "counts_precision.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
