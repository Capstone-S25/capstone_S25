"""Numbers for the deck's "What was caught" slide, all from one run: ICVL held-out half, default gate + pose cue,
crop check on, one alert -> one person (as staysput.py).
Per event (littering, split into put down standing still / dropped while walking; smoking; loitering = in view 60 s+):
incidents, caught, median seconds from the incident's start to the first alert on that person (0 if it fired in the
run-up). Overall: normal people flagged (no incident of these kinds), share of alerts on a person during one of their
incidents, alerts per minute on all footage and on footage with no incident.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/caught_table.py [odd|even]   -> exp/icvl/caught_table_<half>.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from fpr_attrib import CFG, owner  # noqa: E402  (also loads the cached crop-check scores)
from inventory import ICVL, read_labels  # noqa: E402
from litter_style import style  # noqa: E402
from score import TOL_S, WINDOW_S, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import replay_rules  # noqa: E402


def main():
    half = sys.argv[1] if len(sys.argv) > 1 else "even"
    rows = defaultdict(lambda: {"incidents": 0, "caught": 0, "delays": []})
    neg = flagged = alerts_n = on_incident = 0
    minutes = quiet_min = quiet_alerts = 0.0
    for g, res in load_runs(half):
        fps = g.seq.native_fps
        eps, people = labels_for(g.tag, fps)
        alerts = replay_rules(res, CFG)[0]
        own = [owner(a, people) for a in alerts]
        lit_rows = defaultdict(list)
        for r in read_labels(ICVL / g.tag):
            if r[10] == 1:
                lit_rows[r[6]].append(r)
        inc = []
        for ep in eps.get("littering", []):
            mine = [r for r in lit_rows[ep["pid"]] if ep["t0"] - 1e-6 <= r[0] / fps <= ep["t1"] + 1e-6]
            kind = "littering: dropped while walking" if style(mine).endswith("walking") else "littering: put down standing still"
            inc.append((kind, ep["pid"], ep["t0"], ep["t1"], WINDOW_S["littering"]))
        inc += [("smoking", ep["pid"], ep["t0"], ep["t1"], TOL_S) for ep in eps.get("smoking", [])]
        for pid, tr in people.items():
            if tr["ts"][-1] - tr["ts"][0] >= 60:
                inc.append(("loitering: in view 60 s+", pid, tr["ts"][0], tr["ts"][-1], TOL_S))
        hit = [False] * len(alerts)
        for name, pid, t0, t1, w in inc:
            ks = [k for k, a in enumerate(alerts) if own[k] == pid and on_track(a, people[pid], t0, t1, w)]
            for k in ks:
                hit[k] = True
            for key in (name, "littering") if name.startswith("littering") else (name,):
                row = rows[key]
                row["incidents"] += 1
                if ks:
                    row["caught"] += 1
                    row["delays"].append(max(0.0, min(alerts[k]["timestamp_s"] for k in ks) - t0))
        pos = {i[1] for i in inc}
        for pid in people:
            if pid not in pos:
                neg += 1
                flagged += pid in own
        alerts_n += len(alerts)
        on_incident += sum(hit)
        minutes += res.duration_s / 60
        if not inc:
            quiet_min += res.duration_s / 60
            quiet_alerts += len(alerts)
    out = {"half": half, "events": {k: {"incidents": v["incidents"], "caught": v["caught"],
                                        "caught_%": round(100 * v["caught"] / v["incidents"]),
                                        "median_s_to_alert": round(float(np.median(v["delays"])), 1)}
                                    for k, v in sorted(rows.items())},
           "normal_flagged": f"{flagged}/{neg} ({100 * flagged / neg:.1f} %)",
           "alerts_on_an_incident": f"{on_incident}/{alerts_n} ({100 * on_incident / alerts_n:.0f} %)",
           "alerts_per_min": round(alerts_n / minutes, 2),
           "alerts_per_min_no_incident_footage": round(quiet_alerts / quiet_min, 2), "no_incident_minutes": round(quiet_min, 1)}
    print(json.dumps(out, indent=1))
    (HERE / f"caught_table_{half}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
