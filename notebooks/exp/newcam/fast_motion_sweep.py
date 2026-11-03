"""fast_motion is the largest alert source at dump sites (IWDD normal videos ~7.4/min). Tighten or drop it?

Swept on TUNE / TRAIN data only (current defaults otherwise, rules + pose cue): speed threshold speed_h_per_s
(body heights per second over speed_window_s) and the rule off. Scored on littering tune clips, Avenue tune incidents
(some are people running) and normals, and IWDD train (hit within 10 s vs rate-matched chance, alerts per minute on
its normal videos). Also each setting's fast_motion-only catches.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/fast_motion_sweep.py   -> exp/newcam/fast_motion_tune.json
"""
import json
import pickle
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE)]

from compare_models import model_splits  # noqa: E402
from exp_harness import baseline  # noqa: E402
from iwdd_eval import evaluate  # noqa: E402
from loaders import iwdd_labels  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig  # noqa: E402

RULES = DEFAULT_RULES + ("pose_action",)
SETTINGS = [("current 1.2", {}), ("1.5", {"speed_h_per_s": 1.5}), ("2.0", {"speed_h_per_s": 2.0}),
            ("2.5", {"speed_h_per_s": 2.5}), ("off", {"rules": tuple(r for r in RULES if r != "speed")})]


def main():
    S = model_splits("s", heldout=False)
    iw = pickle.loads((NB / "bench_cache/models/s/iwdd_train.pkl").read_bytes())
    labels = iwdd_labels("train")
    mins = sum(r.duration_s for _, r in S["normal"]) / 60
    rows = []
    for name, kw in SETTINGS:
        cfg = GateConfig(inview_s=60.0, **{"rules": RULES, **kw})
        m = partial(baseline, cfg=cfg)
        fast = lambda r, ds, m=m: [e for e in m(r, ds) if e["event_type"] == "fast_motion"]  # noqa: E731
        lit = summarize([evaluate_run(g, r, m(r, "littering")) for g, r in S["littering"]], "l")
        av = summarize([evaluate_run(g, r, m(r, "avenue")) for g, r in S["avenue"]], "a")
        av_fast = summarize([evaluate_run(g, r, fast(r, "avenue")) for g, r in S["avenue"]], "f")
        e_all, e_fast = evaluate(iw, labels, m), evaluate(iw, labels, fast)
        row = {"fast_motion": name,
               "littering_tune": f"{lit['event_recall_%']}/{lit['crop_recall_%']}",
               "avenue_tune": f"{av['event_recall_%']}/{av['crop_recall_%']}",
               "avenue_tune_fast_alone": f"{av_fast['event_recall_%']}/{av_fast['crop_recall_%']}",
               "avenue_normal_alerts_per_min": round(sum(len(m(r, "normal")) for _, r in S["normal"]) / mins, 2),
               "iwdd_train_hit_vs_chance": f"{e_all['hit@10s_%']} vs {e_all['chance@10s_%']}",
               "iwdd_train_lead": round(e_all["hit@10s_%"] - e_all["chance@10s_%"], 1),
               "iwdd_train_normal_alerts_per_min": e_all["neg_cand_per_min"],
               "iwdd_train_fast_alone_vs_chance": f"{e_fast['hit@10s_%']} vs {e_fast['chance@10s_%']}"}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (HERE / "fast_motion_tune.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
