"""Per-person false positive rate vs incidents caught, for GateConfig settings on one ICVL half.

FPR = share of labelled people with no littering / smoking / still-20 s+ incident in their video who get at least
one alert. Settings are applied on top of the deck's configuration (one_per_person.CFG: 15 s pose-cue gap).
Usage (from notebooks/): ../.venv/bin/python exp/icvl/fpr_sweep.py NAME HALF '<json list of settings>'
  -> exp/icvl/fpr_<NAME>.json
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

from one_per_person import CFG  # noqa: E402
from score import EVENTS, TOL_S, WINDOW_S, labels_for, load_runs, on_track  # noqa: E402
from stage1_gate import replay_rules  # noqa: E402


def evaluate(runs, cfg, filt=None):
    neg = flagged = alerts_n = quiet_alerts = 0
    quiet_min = 0.0
    caught = {ev: [0, 0] for ev in EVENTS}
    for g, res in runs:
        alerts = replay_rules(res, cfg)[0]
        if filt is not None:
            alerts = filt(res, alerts)
        eps, people = labels_for(g.tag, g.seq.native_fps)
        alerts_n += len(alerts)
        if not any(eps.get(ev) for ev in EVENTS):
            quiet_alerts += len(alerts)
            quiet_min += res.duration_s / 60
        pos = set()
        for ev in EVENTS:
            for ep in eps.get(ev, []):
                pos.add(ep["pid"])
                caught[ev][1] += 1
                caught[ev][0] += any(on_track(a, people[ep["pid"]], ep["t0"], ep["t1"], WINDOW_S.get(ev, TOL_S))
                                     for a in alerts)
        for pid, tr in people.items():
            if pid not in pos:
                neg += 1
                flagged += any(on_track(a, tr) for a in alerts)
    return {"fpr_%": round(100 * flagged / neg, 1), "flagged": f"{flagged}/{neg}", "alerts": alerts_n,
            "quiet_alerts_min": round(quiet_alerts / quiet_min, 2),
            **{ev: f"{c}/{n}" for ev, (c, n) in caught.items()}}


def main():
    import pickle
    import stage1_crop_check
    for path in (HERE.parents[1] / "bench_cache" / "models" / "s").glob("crop_scores_icvl.pkl"):
        stage1_crop_check.SCORES.update(pickle.loads(path.read_bytes()))   # so crop_check can be replayed
    name, half, settings = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    runs, rows = load_runs(None if half == "all" else half), []
    for kw in settings:
        rows.append({"setting": kw, **evaluate(runs, replace(CFG, **kw))})
        print(json.dumps(rows[-1]), flush=True)
    (HERE / f"fpr_{name}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
