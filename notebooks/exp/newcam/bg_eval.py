"""Background subtraction for abandoned objects that YOLO cannot see (small bags on low-resolution cameras).

Replays the existing "litter" rule (static_object_left: a new stationary blob, away from people, that someone was
next to shortly before it became still) on cached runs with background blobs attached (bg_blobs.py), sweeping how
long the blob must stay still and how strict the novelty/stability checks are.

Reports: CAVIAR left-object recall (unseen camera), static_object_left candidates per minute on CAVIAR normal videos,
Avenue tune normal videos (known camera) and Avenue tune test videos.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/bg_eval.py
"""
import pickle
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE)]

from bg_blobs import CACHE, with_blobs  # noqa: E402
from compare_models import model_splits  # noqa: E402
from exp_harness import baseline  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, calibrate_inview  # noqa: E402


def load_bg(name):
    return pickle.loads((CACHE / f"bg_{name}.pkl").read_bytes())


def per_min(results, method, etype="static_object_left"):
    n = sum(1 for _, r in results for e in method(r, "x") if e["event_type"] == etype)
    return round(n / (sum(r.duration_s for _, r in results) / 60), 2)


def main():
    T = model_splits("s", heldout=False)
    inview = calibrate_inview([r for _, r in T["normal"]])
    caviar = with_blobs(pickle.loads((CACHE / "caviar.pkl").read_bytes()), load_bg("caviar"))
    av = with_blobs(T["avenue"] + T["normal"], load_bg("avenue_tune"))
    av_test = [x for x in av if x[1].seq_id.startswith("test/")]
    av_norm = [x for x in av if x[1].seq_id.startswith("train/")]
    cav_pos = [x for x in caviar if x[0].gt]
    cav_neg = [x for x in caviar if x[0].tag == "normal"]
    rows = []
    for static_s in (3.0, 5.0, 10.0):
        for novelty, change in ((15.0, 8.0), (25.0, 6.0), (10.0, 10.0)):
            for owner in (True, False):
                cfg = GateConfig(inview_s=inview, rules=DEFAULT_RULES + ("litter",), litter_static_s=static_s,
                                 litter_min_novelty=novelty, litter_max_change=change, litter_min_area=2e-4,
                                 litter_owner_lookback_s=4.0 if owner else 1e6, litter_owner_dist_h=1.0 if owner else 1e6)
                m = partial(baseline, cfg=cfg)
                only = lambda r, ds, m=m: [e for e in m(r, ds) if e["event_type"] == "static_object_left"]  # noqa: E731
                s_all = summarize([evaluate_run(g, r, m(r, "caviar")) for g, r in cav_pos], "all")
                s_bg = summarize([evaluate_run(g, r, only(r, "caviar")) for g, r in cav_pos], "bg")
                rows.append({"static_s": static_s, "novelty": novelty, "max_change": change, "needs_owner": owner,
                             "caviar_left_obj_crop_%_bg_only": s_bg["crop_recall_%"],
                             "caviar_left_obj_crop_%_all_rules": s_all["crop_recall_%"],
                             "caviar_bg_per_min_leftbag_videos": per_min(cav_pos, m),
                             "caviar_bg_per_min_normal": per_min(cav_neg, m),
                             "avenue_bg_per_min_normal": per_min(av_norm, m),
                             "avenue_bg_per_min_test": per_min(av_test, m),
                             "avenue_crop_%_all_rules": summarize([evaluate_run(g, r, m(r, "avenue")) for g, r in av_test], "a")["crop_recall_%"]})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(HERE / "bg_sweep.csv", index=False)


if __name__ == "__main__":
    main()
