"""Tune the reach_down rule on IWDD train (test stays held out), with guardrails on the original tune data.

IWDD misses where the dumper IS tracked: people cut off by the frame edge (reach_skip_border skips them), unloading
a car boot at hip height (wrist never near the knee; a more negative reach_margin_h allows higher hands), bending
with a bag while keypoints are uncertain (reach_kp_conf 0.5). Grid over those three settings; a setting is scored by its lead over chance on IWDD train
(hit@10s - chance@10s, rules + pose) and must not raise Avenue tune normal false alarms by more than MAX_FA_RISE
per minute or lower littering tune event recall.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/iwdd_tune.py   -> iwdd_tune_train.csv
"""
import itertools
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE)]

from compare_models import model_splits  # noqa: E402
from exp_harness import baseline  # noqa: E402
from iwdd_eval import evaluate, inview_from_avenue, load  # noqa: E402
from loaders import iwdd_labels  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig  # noqa: E402

MAX_FA_RISE = 0.5
GRID = {"reach_kp_conf": (0.5, 0.4, 0.3), "reach_skip_border": (True, False), "reach_margin_h": (-0.15, -0.3, -0.45)}


def row(cfg, T, iw, labels):
    m = partial(baseline, cfg=cfg)
    ev = evaluate(iw, labels, m)
    lit = summarize([evaluate_run(g, r, m(r, "littering")) for g, r in T["littering"]], "l")
    fa = summarize([evaluate_run(g, r, m(r, "normal")) for g, r in T["normal"]], "n")["candidates_per_min"]
    return {"iwdd_hit@10s_%": ev["hit@10s_%"], "iwdd_chance@10s_%": ev["chance@10s_%"],
            "iwdd_lead": round(ev["hit@10s_%"] - ev["chance@10s_%"], 1), "iwdd_neg_any_%": ev["neg_any_%"],
            "iwdd_neg_cand_per_min": ev["neg_cand_per_min"], "littering_tune_event_%": lit["event_recall_%"],
            "littering_tune_crop_%": lit["crop_recall_%"], "avenue_normal_FA_per_min": fa}


def main():
    T = model_splits("s", heldout=False)
    iw, labels = load("train"), iwdd_labels("train")
    base = GateConfig(inview_s=inview_from_avenue(), rules=DEFAULT_RULES + ("pose_action",))
    rows = []
    for vals in itertools.product(*GRID.values()):
        kw = dict(zip(GRID, vals))
        rows.append({**kw, **row(replace(base, **kw), T, iw, labels)})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    ref = df.iloc[0]
    df["ok"] = (df["avenue_normal_FA_per_min"] <= ref["avenue_normal_FA_per_min"] + MAX_FA_RISE) & \
               (df["littering_tune_event_%"] >= ref["littering_tune_event_%"])
    df.to_csv(HERE / "iwdd_tune_train.csv", index=False)
    print(df.sort_values("iwdd_lead", ascending=False).to_string())


if __name__ == "__main__":
    main()
