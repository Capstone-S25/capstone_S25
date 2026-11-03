"""Fewer alerts without losing incidents: one alert per person per episode, and slower re-sends of "stays put".

person_dedup_s: after an alert on a person, drop further alerts on that person (any rule) for this long (2.5 s now).
loiter_repeat_s: re-send "stays put" while the person stays (30 s now).
Swept on TUNE / TRAIN data only (current defaults otherwise, rules + pose cue): littering tune clips, Avenue tune
throwing incidents + normals, IWDD train (hit within 10 s vs chance, alerts / min on normal videos), smoking tune.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/alert_budget.py   -> exp/newcam/alert_budget_tune.json
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
from counts_precision import throwing_labels  # noqa: E402
from exp_harness import baseline  # noqa: E402
from iwdd_eval import evaluate  # noqa: E402
from loaders import iwdd_labels  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig  # noqa: E402

sys.path.insert(0, str(NB / "exp" / "gt_review"))
from scoped import scoped  # noqa: E402

SETTINGS = [{}, {"person_dedup_s": 5.0}, {"person_dedup_s": 10.0}, {"person_dedup_s": 20.0}, {"person_dedup_s": 30.0},
            {"loiter_repeat_s": 60.0}, {"loiter_repeat_s": 120.0},
            {"person_dedup_s": 10.0, "loiter_repeat_s": 60.0}, {"person_dedup_s": 20.0, "loiter_repeat_s": 120.0}]


def main():
    S = model_splits("s", heldout=False)
    throwing = scoped(S["avenue"], keep=("in",), path=throwing_labels())
    iw = pickle.loads((NB / "bench_cache/models/s/iwdd_train.pkl").read_bytes())
    labels = iwdd_labels("train")
    mins = sum(r.duration_s for _, r in S["normal"]) / 60
    rows = []
    for kw in SETTINGS:
        m = partial(baseline, cfg=GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",), **kw))
        lit = summarize([evaluate_run(g, r, m(r, "littering")) for g, r in S["littering"]], "l")
        thr = summarize([evaluate_run(g, r, m(r, "avenue")) for g, r in throwing], "t")
        sm = summarize([evaluate_run(g, r, m(r, "smoking")) for g, r in S["smoking"]], "s")
        e = evaluate(iw, labels, m)
        row = {"setting": kw or "current",
               "littering_tune": f"{lit['event_recall_%']}/{lit['crop_recall_%']} ({lit['candidates']} alerts)",
               "avenue_throwing_tune": f"{thr['event_recall_%']}/{thr['crop_recall_%']}",
               "walkway_alerts_per_min": round(sum(len(m(r, "normal")) for _, r in S["normal"]) / mins, 2),
               "smoking_tune": f"{sm['event_recall_%']}/{sm['crop_recall_%']} ({sm['candidates']} alerts)",
               "iwdd_train_hit_vs_chance": f"{e['hit@10s_%']} vs {e['chance@10s_%']}",
               "iwdd_train_lead": round(e["hit@10s_%"] - e["chance@10s_%"], 1),
               "dump_site_alerts_per_min": e["neg_cand_per_min"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (HERE / "alert_budget_tune.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
