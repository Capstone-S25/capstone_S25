"""How low must the wrist be for reach_down? (user saw it fire on people standing with arms down, ABODA video6)

reach_margin_h: the lowest wrist must be within this many body heights ABOVE the knee (negative = above).
Swept on tune / train data only, current defaults otherwise, rules + pose cue: littering tune, Avenue tune and
normals, IWDD train (hit within 10 s vs chance, reach_down alarms per minute on its normal videos).
Usage (from notebooks/): ../.venv/bin/python exp/newcam/reach_margin.py margin   -> exp/newcam/reach_margin_tune.json
          ../.venv/bin/python exp/newcam/reach_margin.py           -> reach_relative_tune.json (relative wrist drop)
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

MARGINS = (-0.15, -0.10, -0.05, 0.0)
DROPS = (None, 0.05, 0.10, 0.15)  # reach_relative_drop_h at the current margin (-0.15)


def main():
    S = model_splits("s", heldout=False)
    iw = pickle.loads((NB / "bench_cache/models/s/iwdd_train.pkl").read_bytes())
    labels = iwdd_labels("train")
    mins = sum(r.duration_s for _, r in S["normal"]) / 60
    rows = []
    settings = [{"reach_margin_h": mh} for mh in MARGINS] if "margin" in sys.argv else \
        [{"reach_relative_drop_h": d} for d in DROPS]
    for kw in settings:
        cfg = GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",), **kw)
        m = partial(baseline, cfg=cfg)
        only = lambda r, ds, m=m: [e for e in m(r, ds) if e["event_type"] == "reach_down"]  # noqa: E731
        lit = summarize([evaluate_run(g, r, m(r, "littering")) for g, r in S["littering"]], "l")
        alone = summarize([evaluate_run(g, r, only(r, "littering")) for g, r in S["littering"]], "a")
        av = summarize([evaluate_run(g, r, m(r, "avenue")) for g, r in S["avenue"]], "v")
        e_all, e_r = evaluate(iw, labels, m), evaluate(iw, labels, only)
        row = {**kw,
               "littering_tune": f"{lit['event_recall_%']}/{lit['crop_recall_%']}",
               "littering_tune_reach_alone": f"{alone['event_recall_%']}/{alone['crop_recall_%']}",
               "avenue_tune": f"{av['event_recall_%']}/{av['crop_recall_%']}",
               "avenue_normal_FA": round(sum(len(m(r, "normal")) for _, r in S["normal"]) / mins, 2),
               "avenue_normal_reach_FA": round(sum(len(only(r, "normal")) for _, r in S["normal"]) / mins, 2),
               "iwdd_train_hit_vs_chance": f"{e_all['hit@10s_%']} vs {e_all['chance@10s_%']}",
               "iwdd_train_reach_alone_vs_chance": f"{e_r['hit@10s_%']} vs {e_r['chance@10s_%']}",
               "iwdd_train_normal_reach_per_min": e_r["neg_cand_per_min"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    name = "reach_margin_tune.json" if "margin" in sys.argv else "reach_relative_tune.json"
    (HERE / name).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
