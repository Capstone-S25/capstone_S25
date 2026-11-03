"""Score GateConfig settings on TUNE / TRAIN data: incidents kept vs alerts sent.

Each setting (JSON dict of GateConfig fields, applied on top of the current defaults with rules + pose cue) is scored on
  littering tune clips (caught / right person), Avenue tune throwing incidents (caught / right person),
  walkway alerts per minute (Avenue tune normal videos), smoking tune episodes,
  IWDD train: dumping caught within 10 s vs rate-matched chance and alerts per minute on its normal videos.
Bicycle / motorcycle boxes from two_wheelers.py are attached to cached observations when available, so
reach_skip_riders can be replayed.
Usage (from notebooks/): ../.venv/bin/python exp/alerts/sweep.py NAME '<json list of settings>'
  -> exp/alerts/sweep_<NAME>.json
"""
import json
import pickle
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "newcam"), str(NB / "exp" / "gt_review")]

from counts_precision import throwing_labels  # noqa: E402
from exp_harness import baseline  # noqa: E402
from iwdd_eval import evaluate  # noqa: E402
from loaders import iwdd_labels  # noqa: E402
from scoped import scoped  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig  # noqa: E402

from cached import CACHE, load_cached, splits  # noqa: E402

def data():
    S = splits(heldout=False)
    iw = load_cached("iwdd_train")
    have = {k: (CACHE / f"twowheel_{k}.pkl").exists() for k in ("litter_train", "avenue", "iwdd_train")}
    return S, scoped(S["avenue"], keep=("in",), path=throwing_labels()), iw, iwdd_labels("train"), have


def score(kw, S, throwing, iw, labels):
    m = partial(baseline, cfg=GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",), **kw))
    lit = summarize([evaluate_run(g, r, m(r, "littering")) for g, r in S["littering"]], "l")
    thr = summarize([evaluate_run(g, r, m(r, "avenue")) for g, r in throwing], "t")
    sm = summarize([evaluate_run(g, r, m(r, "smoking")) for g, r in S["smoking"]], "s")
    mins = sum(r.duration_s for _, r in S["normal"]) / 60
    e = evaluate(iw, labels, m)
    return {"setting": kw or "current",
            "littering": f"{lit['event_recall_%']}/{lit['crop_recall_%']}",
            "walkway_throws": f"{thr['event_recall_%']}/{thr['crop_recall_%']}",
            "smoking": f"{sm['event_recall_%']}/{sm['crop_recall_%']}",
            "walkway_alerts_min": round(sum(len(m(r, "normal")) for _, r in S["normal"]) / mins, 2),
            "dumping_hit_vs_chance": f"{e['hit@10s_%']} vs {e['chance@10s_%']}",
            "dumping_lead": round(e["hit@10s_%"] - e["chance@10s_%"], 1),
            "dump_site_alerts_min": e["neg_cand_per_min"],
            "dump_site_videos_with_alert_%": e["neg_any_%"]}


def main():
    name, settings = sys.argv[1], json.loads(sys.argv[2])
    S, throwing, iw, labels, have = data()
    print("two-wheeler boxes attached:", have, flush=True)
    rows = []
    for kw in settings:
        rows.append(score(kw, S, throwing, iw, labels))
        print(json.dumps(rows[-1]), flush=True)
    (HERE / f"sweep_{name}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
