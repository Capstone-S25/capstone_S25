"""Two follow-up sweeps suggested by miss_analysis.py / deep_dive.py (tune half only).

dwell:   shorter sustained_presence (loiter_s) for smoking. Smoking is scored per EPISODE (labels <= 30 s apart in one
         sequence merged; the per-frame labels flicker), alongside loitering recall (all / agreed labels) and false
         alarms per minute on normal footage. reach_down/pose_action stay on, as in the deployed config.
context: littering candidates often land on a bystander next to the actor. Widen pose/reach crops to include the
         people standing within `r` body heights of the person who fired, and compare with the same widening applied
         to the "tallest person at the same moments" control (a bigger box should not win just by being bigger).
"""
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "gt_review"), str(NB / "exp" / "actions"), str(HERE)]

from compare_models import model_splits  # noqa: E402
from deep_dive import episodes  # noqa: E402
from exp_harness import baseline, combine  # noqa: E402
from scoped import scoped  # noqa: E402
from stage1_bench import box_hit, evaluate_run, summarize  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, calibrate_inview, center, union_box  # noqa: E402


def episode_recall(results, method, ds="smoking"):
    n = hit = 0
    for g, r in results:
        ev = method(r, ds)
        for ep in episodes(g, r):
            n += 1
            hit += any(abs(t - e["timestamp_s"]) <= 1.0 and any(box_hit(e["roi_box"], b) for b in g.gt[i]) for e in ev for t, i in ep)
    return round(100 * hit / max(n, 1), 1)


def dwell(T):
    inview = calibrate_inview([r for _, r in T["normal"]])
    rows = []
    for loiter in (20, 15, 12, 10, 8, 6):
        for radius in (0.6, 0.9):
            cfg = GateConfig(inview_s=inview, rules=DEFAULT_RULES + ("pose_action",), loiter_s=float(loiter), loiter_radius_h=radius,
                             loiter_release_h=max(0.9, radius + 0.3))
            m = partial(baseline, cfg=cfg)
            sm = summarize([evaluate_run(g, r, m(r, "smoking")) for g, r in T["smoking"]], "s")
            av = summarize([evaluate_run(g, r, m(r, "avenue")) for g, r in T["avenue"]], "a")
            st = summarize([evaluate_run(g, r, m(r, "avenue")) for g, r in scoped(T["avenue"])], "st")
            nm = summarize([evaluate_run(g, r, m(r, "normal")) for g, r in T["normal"]], "n")
            rows.append({"loiter_s": loiter, "radius_h": radius, "smoking_episode_%": episode_recall(T["smoking"], m),
                         "smoking_crop_%": sm["crop_recall_%"], "smoking_cand_per_min": sm["candidates_per_min"],
                         "loitering_crop_%": av["crop_recall_%"], "loitering_agreed_crop_%": st["crop_recall_%"],
                         "normal_FA_per_min": nm["candidates_per_min"], "loitering_skipped_%": av["review_reduction_%"]})
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def widen(method, r_h, types=("pose_action", "reach_down", "tallest")):
    def run(result, ds):
        by_t = {round(o.t, 2): o for o in result.observations}
        out = []
        for e in method(result, ds):
            o = by_t.get(round(e["timestamp_s"], 2))
            if o is None or e["event_type"] not in types or r_h <= 0:
                out.append(e)
                continue
            box = e["roi_box"]
            h = max(1.0, box[3] - box[1])
            cx, cy = center(box)
            for p in o.persons:
                px, py = center(p["box"])
                if ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 <= r_h * h:
                    box = union_box(box, p["box"])
            out.append({**e, "roi_box": [int(v) for v in box]})
        return out
    return run


def context(T):
    import pose_seq_final as F
    st = F.pickle.loads(F.FROZEN.read_bytes())
    cue = F.frozen_method((st["oof"], st["meta"]))[0]
    inview = calibrate_inview([r for _, r in T["normal"]])
    base = partial(baseline, cfg=GateConfig(inview_s=inview))
    method = combine(base, cue)
    control = combine(base, F.tallest_same_times(cue))
    rows = []
    for r_h in (0.0, 0.5, 0.75, 1.0, 1.5):
        row = {"radius_body_heights": r_h}
        for name, m in (("rules+pose", method), ("control", control)):
            w = widen(m, r_h)
            s = summarize([evaluate_run(g, r, w(r, "littering")) for g, r in T["littering"]], name)
            areas = [((e["roi_box"][2] - e["roi_box"][0]) * (e["roi_box"][3] - e["roi_box"][1])) / (r.observations[0].frame_hw[0] * r.observations[0].frame_hw[1])
                     for g, r in T["littering"] for e in w(r, "littering") if e["event_type"] in ("pose_action", "reach_down", "tallest")]
            row[f"{name}_crop_%"] = s["crop_recall_%"]
            row[f"{name}_crop_area_%frame"] = round(100 * float(np.median(areas)), 1)
        # the same widening on the other datasets: candidate crops get bigger there too
        w = widen(partial(baseline, cfg=GateConfig(inview_s=inview, rules=DEFAULT_RULES + ("pose_action",))), r_h)
        row["loitering_crop_%"] = summarize([evaluate_run(g, r, w(r, "avenue")) for g, r in T["avenue"]], "a")["crop_recall_%"]
        row["smoking_episode_%"] = episode_recall(T["smoking"], w)
        row["skill_over_control"] = round(row["rules+pose_crop_%"] - row["control_crop_%"], 1)
        rows.append(row)
        print(row, flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    T = model_splits("s", heldout=False)
    which = sys.argv[1:] or ["dwell", "context"]
    if "context" in which:
        context(T).to_csv(HERE / "sweep_context_tune.csv", index=False)
    if "dwell" in which:
        dwell(T).to_csv(HERE / "sweep_dwell_tune.csv", index=False)
