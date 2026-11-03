"""Two fixes suggested by the checkpoint review, each scored against the rate-matched chance baseline (exp/chance.py).

stitch:  smoking frames are ~1 fps with gaps often > 8 s, so tracks break and the dwell rule rarely runs. Longer
         re-link windows (stitch_max_gap_s) and dwell times, tuned on the TUNE half (smoking episodes, Avenue normal
         false alarms); the chosen setting is then measured once on held-out.
dedup:   dedup_per_rule=True so a pose_action candidate does not block a reach_down candidate on the same person.
Usage (from notebooks/): ../.venv/bin/python exp/fixes.py stitch|dedup
"""
import json
import pickle
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

NB = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "newcam"), str(NB / "exp" / "gt_review")]

from chance import CACHE, table  # noqa: E402
from compare_models import model_splits  # noqa: E402
from exp_harness import baseline  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, calibrate_inview  # noqa: E402

OUT = NB / "exp"


def short(row):
    return {k: row[k] for k in ("events", "candidates_per_min", "method_strict_%", "same_times_strict_%", "above_same_times_strict",
                                "method_lenient_%", "above_same_times_lenient")}


def fa(normal, cfg):
    return summarize([evaluate_run(g, r, baseline(r, "normal", cfg)) for g, r in normal], "n")["candidates_per_min"]


def stitch():
    T, H = model_splits("s", heldout=False), model_splits("s", heldout=True)
    inview = calibrate_inview([r for _, r in T["normal"]])
    base = GateConfig(inview_s=inview, rules=DEFAULT_RULES + ("pose_action",))
    rows = []
    for gap in (8.0, 15.0, 30.0):
        for loiter in (20.0, 15.0, 10.0):
            for walk in (1.0, 0.3):
                cfg = replace(base, stitch_max_gap_s=gap, loiter_s=loiter, stitch_walk_h_per_s=walk)
                sm = table(T["smoking"], partial(baseline, cfg=cfg), "smoking", "episode")
                dw = table(T["smoking"], partial(baseline, cfg=replace(cfg, rules=("loiter",))), "smoking", "episode")
                row = {"stitch_gap_s": gap, "loiter_s": loiter, "walk_h_per_s": walk, "smoking": short(sm),
                       "dwell_rule_only": short(dw), "avenue_normal_FA_per_min": fa(T["normal"], cfg)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    (OUT / "fix_stitch_tune.json").write_text(json.dumps(rows, indent=1))
    # choose on tune: dwell-only skill above chance (strict) with Avenue normal FA <= baseline + 1/min
    ref = rows[0]["avenue_normal_FA_per_min"]
    ok = [r for r in rows if r["avenue_normal_FA_per_min"] <= ref + 1.0]
    best = max(ok, key=lambda r: (r["dwell_rule_only"]["above_same_times_strict"], r["smoking"]["above_same_times_strict"]))
    print("chosen on tune:", best["stitch_gap_s"], best["loiter_s"], best["walk_h_per_s"], flush=True)
    cfg = replace(base, stitch_max_gap_s=best["stitch_gap_s"], loiter_s=best["loiter_s"], stitch_walk_h_per_s=best["walk_h_per_s"])
    held = {}
    for name, c in (("current", base), ("chosen", cfg)):
        held[name] = {"smoking": short(table(H["smoking"], partial(baseline, cfg=c), "smoking", "episode")),
                      "dwell_rule_only": short(table(H["smoking"], partial(baseline, cfg=replace(c, rules=("loiter",))), "smoking", "episode")),
                      "loitering": short(table(H["avenue"], partial(baseline, cfg=c), "avenue")),
                      "avenue_normal_FA_per_min": fa(H["normal"], c)}
        print("held-out", name, json.dumps(held[name]), flush=True)
    (OUT / "fix_stitch_heldout.json").write_text(json.dumps({"chosen": best, "held_out": held}, indent=1))


def dedup():
    from loaders import meva_relabel
    T, H = model_splits("s", heldout=False), model_splits("s", heldout=True)
    inview = calibrate_inview([r for _, r in T["normal"]])
    meva = pickle.loads((CACHE / "meva.pkl").read_bytes())
    sets = {"littering/held-out": (H["littering"], "littering", "event"), "loitering/held-out": (H["avenue"], "avenue", "event"),
            "smoking/held-out": (H["smoking"], "smoking", "episode"),
            "meva_puts_down": ([x for x in meva_relabel(meva, ("person_puts_down_object",)) if x[0].gt], "meva", "event")}
    out = {}
    for per_rule in (False, True):
        cfg = GateConfig(inview_s=inview, rules=DEFAULT_RULES + ("pose_action",), dedup_per_rule=per_rule)
        m = partial(baseline, cfg=cfg)
        for name, (data, ds, mode) in sets.items():
            out[f"{name}/per_rule={per_rule}"] = short(table(data, m, ds, mode))
            print(name, per_rule, out[f"{name}/per_rule={per_rule}"], flush=True)
        out[f"normal_FA/per_rule={per_rule}"] = {"tune": fa(T["normal"], cfg), "held-out": fa(H["normal"], cfg)}
        print("normal FA", per_rule, out[f"normal_FA/per_rule={per_rule}"], flush=True)
    (OUT / "fix_dedup.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"stitch": stitch, "dedup": dedup}[sys.argv[1]]()
