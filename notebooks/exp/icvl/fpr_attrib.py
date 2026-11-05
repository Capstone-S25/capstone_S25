"""Re-score ICVL with one alert -> one person: each alert belongs to the labelled person it is mainly on at its own
time (largest share of their labelled box inside the alert box, among people inside the VLM crop as score.py's
on_track counts them). Catches and the per-person false positive rate both use this, so neither side keeps the credit
or blame of a bystander standing inside the same crop.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/fpr_attrib.py [even|odd|all]   -> exp/icvl/fpr_attrib.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(HERE)]

import stage1_crop_check  # noqa: E402
from one_per_person import CFG  # noqa: E402
from score import EVENTS, TOL_S, WINDOW_S, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import intersection_over, replay_rules  # noqa: E402

for f in ("crop_scores_icvl.pkl", "crop_scores_pose_action_icvl.pkl"):
    stage1_crop_check.SCORES.update(pickle.loads((NB / "bench_cache" / "models" / "s" / f).read_bytes()))


def owner(a, people):
    """The labelled person the alert is mainly on, among those on_track counts as inside its crop (None if nobody)."""
    t, roi = a["timestamp_s"], a["roi_box"]
    best, best_cov = None, -1.0
    for pid, tr in people.items():
        if not on_track(a, tr):
            continue
        ts = np.asarray(tr["ts"])
        i = int(np.argmin(np.abs(ts - t)))
        cov = intersection_over(tr["boxes"][i], roi) if abs(ts[i] - t) <= TOL_S else 0.0
        if cov > best_cov:
            best, best_cov = pid, cov
    return best


def rescore(runs, cfg):
    caught = {ev: {"loose": 0, "owner": 0, "incidents": 0} for ev in EVENTS}
    neg = flagged_loose = flagged_owner = 0
    for g, res in runs:
        alerts = replay_rules(res, cfg)[0]
        eps, people = labels_for(g.tag, g.seq.native_fps)
        own = [owner(a, people) for a in alerts]
        pos = set()
        for ev in EVENTS:
            w = WINDOW_S.get(ev, TOL_S)
            for ep in eps.get(ev, []):
                pos.add(ep["pid"])
                tr = people[ep["pid"]]
                hits = [k for k, a in enumerate(alerts) if on_track(a, tr, ep["t0"], ep["t1"], w)]
                caught[ev]["incidents"] += 1
                caught[ev]["loose"] += bool(hits)
                caught[ev]["owner"] += any(own[k] == ep["pid"] for k in hits)
        for pid, tr in people.items():
            if pid in pos:
                continue
            neg += 1
            flagged_loose += any(on_track(a, tr) for a in alerts)
            flagged_owner += any(o == pid for o in own)
    return {"people_without_incident": neg,
            "fpr_loose_%": round(100 * flagged_loose / neg, 1), "fpr_owner_%": round(100 * flagged_owner / neg, 1),
            "flagged_loose": flagged_loose, "flagged_owner": flagged_owner,
            **{ev: {"loose": f'{c["loose"]}/{c["incidents"]}', "owner": f'{c["owner"]}/{c["incidents"]}'}
               for ev, c in caught.items()}}


if __name__ == "__main__":
    half = sys.argv[1] if len(sys.argv) > 1 else "even"
    out = rescore(load_runs(None if half == "all" else half), CFG)
    (HERE / "fpr_attrib.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
