"""Re-scoped Avenue ("loitering") labels: keep only the labelled events that match the project's anomaly definition.

events.csv lists every labelled event with the reviewers' tags and a group_decision column. An event's final tag is the
group decision when filled in, otherwise the reviewers' agreed tag ("disagree" when they differ). Frames of dropped
events are treated as normal footage, so firing on them still counts as a false alarm.
"""
import csv
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent


def final_tags(path=HERE / "events.csv") -> dict[str, dict]:
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            row["final"] = (row.get("group_decision") or "").strip() or row["agreed_tag"]
            out[row["event_id"]] = row
    return out


def scoped(results, keep=("in",), path=HERE / "events.csv"):
    """Copy of avenue (g, r) results whose ground truth only contains events whose final tag is in `keep`."""
    tags = final_tags(path)
    by_video = {}
    for row in tags.values():
        by_video.setdefault(row["video"], []).append(row)
    out = []
    for g, r in results:
        drop = [(float(e["t0"]), float(e["t1"])) for e in by_video.get(g.seq.seq_id, []) if e["final"] not in keep]
        times = {o.index: o.t for o in r.observations}
        gt = {i: b for i, b in g.gt.items() if not any(a - 1e-6 <= times.get(i, -1) <= c + 1e-6 for a, c in drop)}
        out.append((replace(g, gt=gt), r))
    return out
