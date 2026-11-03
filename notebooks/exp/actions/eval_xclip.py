"""Sweep and report the X-CLIP cue (tune half only). Needs cache/ from `xclip.py build`.

    python exp/actions/eval_xclip.py [sweep|report]
"""
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import pandas as pd  # noqa: E402

import xclip as X  # noqa: E402
from exp_harness import baseline, combine, score  # noqa: E402
from random_control import lift  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig  # noqa: E402

TUNE = X.tune_splits()


def memo(method):
    cache = {}

    def run(r, ds):
        k = (ds, r.seq_id)
        if k not in cache:
            cache[k] = method(r, ds)
        return [dict(e) for e in cache[k]]
    return run


BASE = memo(partial(baseline, cfg=GateConfig()))
NOREACH = memo(partial(baseline, cfg=GateConfig(rules=tuple(r for r in DEFAULT_RULES if r != "reach_down"))))
MIN = {ds: sum(r.duration_s for _, r in TUNE[ds]) / 60 for ds in TUNE}


def row(name, mode, method, do_lift=True):
    s = score(method, splits=TUNE, name=name)
    out = {"variant": name, "mode": mode, "litter_ev": s["littering_event_%"], "litter_crop": s["littering_crop_%"],
           "avenue_ev": s["avenue_event_%"], "avenue_crop": s["avenue_crop_%"],
           "smoke_ev": s["smoking_event_%"], "smoke_crop": s["smoking_crop_%"], "FA_min": s["normal_FA_per_min"]}
    if do_lift:
        m, c, sd = lift(method, TUNE["littering"], seeds=10)
        out["litter_ctrl"] = round(c, 1)
        out["lift"] = round(m - c, 1)
    return out


def alone_quick(method):
    """Method alone: event counts per minute on each set (cheap pre-filter for the sweep)."""
    return {ds: round(sum(len(method(r, ds)) for _, r in TUNE[ds]) / MIN[ds], 2) for ds in TUNE}


def sweep():
    rows = []
    for ps in X.PROMPT_SETS:
        for mg in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7):
            m = memo(X.make_method(prompt_set=ps, margin=mg))
            r = row(f"{ps} m={mg}", "alone", m, do_lift=True)
            r.update({f"rate_{k}": v for k, v in alone_quick(m).items()})
            rows.append(r)
            print(r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "sweep_alone.csv", index=False)
    return df


def report(variants):
    rows = [row("baseline", "reference", BASE), row("baseline - reach_down", "reference", NOREACH)]
    for name, kw in variants:
        m = memo(X.make_method(**kw))
        rows.append(row(name, "alone", m))
        rows.append(row(name, "add-on", combine(BASE, m)))
        rows.append(row(name, "replace reach_down", combine(NOREACH, m)))
        print(rows[-3:], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "report.csv", index=False)
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if cmd == "sweep":
        print(sweep().to_string())
