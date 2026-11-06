"""How are ICVL's littering incidents done, and which kinds does the gate catch?
Each littering episode (score.py) is typed by the labels on its own frames: bent over or not (posture == bending in any
frame) and moving or not (movement walking / running in most frames). Caught as guards.py (default gate, crop check
on, one alert -> one person, littering window 3 s), with the rules that caught it.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/litter_style.py [odd|even|all]   -> exp/icvl/litter_style_<half>.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from fpr_attrib import CFG, owner  # noqa: E402  (also loads the cached crop-check scores)
from inventory import ICVL, read_labels  # noqa: E402
from score import WINDOW_S, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import replay_rules  # noqa: E402


def style(rows):
    bent = any(r[7] == 3 for r in rows)
    moving = sum(r[8] in (1, 2) for r in rows) > len(rows) / 2
    return ("bends down" if bent else "stays upright") + (", walking" if moving else ", standing still")


def main():
    half = sys.argv[1] if len(sys.argv) > 1 else "even"
    runs = load_runs(None if half == "all" else half)
    kinds = defaultdict(lambda: {"incidents": 0, "caught": 0, "by_rule": Counter()})
    for g, res in runs:
        fps = g.seq.native_fps
        alerts = replay_rules(res, CFG)[0]
        eps, people = labels_for(g.tag, fps)
        own = [owner(a, people) for a in alerts]
        rows = defaultdict(list)
        for r in read_labels(ICVL / g.tag):
            if r[10] == 1:
                rows[r[6]].append(r)
        for ep in eps.get("littering", []):
            mine = [r for r in rows[ep["pid"]] if ep["t0"] - 1e-6 <= r[0] / fps <= ep["t1"] + 1e-6]
            k = kinds[style(mine)]
            ks = [i for i, a in enumerate(alerts)
                  if own[i] == ep["pid"] and on_track(a, people[ep["pid"]], ep["t0"], ep["t1"], WINDOW_S["littering"])]
            k["incidents"] += 1
            k["caught"] += bool(ks)
            k["by_rule"].update({alerts[i]["event_type"] for i in ks})
    out = {kind: {**v, "caught_%": round(100 * v["caught"] / v["incidents"]), "by_rule": dict(v["by_rule"])}
           for kind, v in sorted(kinds.items(), key=lambda kv: -kv[1]["incidents"])}
    print(json.dumps(out, indent=1))
    (HERE / f"litter_style_{half}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
