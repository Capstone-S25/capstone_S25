"""Why does the gate miss IWDD dumping events?  Train split only (test stays held out).

For every positive video, around the onset ([onset - 1 s, onset + 10 s]):
  persons_seen   frames with >= 1 tracked person / frames in the window
  person_h       median height of the tallest person as a fraction of frame height
  hit            a candidate inside the window (current rules + pose)
Grouped by hit/miss, kind and resolution, so a perception miss (no person found) is separated from a rule miss
(person found, no rule fired).
Usage (from notebooks/): ../.venv/bin/python exp/newcam/iwdd_diag.py
"""
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from iwdd_eval import WINDOWS, in_window, inview_from_avenue, load, methods  # noqa: E402
from loaders import iwdd_labels  # noqa: E402


def main():
    data, labels = load("train"), iwdd_labels("train")
    m = methods(inview_from_avenue())["rules+pose"]
    groups = defaultdict(list)
    for g, r in data:
        lab = labels[r.seq_id]
        if not lab["positive"]:
            continue
        win = [o for o in r.observations if in_window(o.t, lab["onset_s"], WINDOWS["10s"])]
        seen = [o for o in win if o.persons]
        heights = [max(p["box"][3] - p["box"][1] for p in o.persons) / o.frame_hw[0] for o in seen]
        hit = any(in_window(e["timestamp_s"], lab["onset_s"], WINDOWS["10s"]) for e in m(r, "littering"))
        h = win[0].frame_hw[0] if win else 0
        row = {"seen": len(seen) / max(1, len(win)), "h": statistics.median(heights) if heights else 0.0}
        cause = "hit" if hit else ("miss: no person" if row["seen"] < 0.2 else "miss: person, no rule")
        for key in (cause, f"{cause} | {lab['kind']}", f"{cause} | {'<=480p' if h <= 480 else '>480p'}"):
            groups[key].append(row)
    for k in sorted(groups):
        rows = groups[k]
        print(f"{k:40s} n={len(rows):3d}  persons_seen={statistics.mean(r['seen'] for r in rows):.2f}  "
              f"person_h={statistics.median(r['h'] for r in rows):.2f}")


if __name__ == "__main__":
    main()
