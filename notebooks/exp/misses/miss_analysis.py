"""Why does Stage 1 miss what it misses?  (tune half only; the held-out half stays untouched)

For every ground-truth event (a run of labelled frames) that the gate does not catch with a good crop, decide:
  not_detected   no detected person box in any of the event's frames would have given a good crop (box_hit),
                 i.e. no rule could have caught it with the current perception
  no_candidate   the person was detected, but no candidate was sent while the event was happening
  wrong_box      a candidate was sent during the event but its box does not cover the labelled region
Method = current default rules + pose_action (the frozen out-of-fold scores on tune littering clips).
Writes exp/misses/misses_tune.csv (one row per GT event) and prints a breakdown per dataset.
"""
import csv
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "gt_review"), str(NB / "exp" / "actions")]

from compare_models import model_splits  # noqa: E402
from exp_harness import baseline, combine  # noqa: E402
from stage1_bench import box_hit, gt_events, merge_windows  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, calibrate_inview  # noqa: E402

TOL = 1.0


def methods(T):
    import pose_seq_final as F
    oof = F.pickle.loads(F.FROZEN.read_bytes())
    cfg = GateConfig(inview_s=calibrate_inview([r for _, r in T["normal"]]))
    lit = combine(partial(baseline, cfg=cfg), F.frozen_method((oof["oof"], oof["meta"]))[0])
    gate = partial(baseline, cfg=GateConfig(inview_s=cfg.inview_s, rules=DEFAULT_RULES + ("pose_action",)))
    return {"littering": lit, "avenue": gate, "smoking": gate}


def analyse(ds, g, r, events):
    obs = r.observations
    times, idxs = [o.t for o in obs], [o.index for o in obs]
    by_idx = {o.index: o for o in obs}
    merged = merge_windows([e["clip_window_s"] for e in events])
    rows = []
    for k, ge in enumerate(gt_events(times, idxs, g.gt, r.fps, 1.0)):
        def good(roi):
            return any(any(box_hit(roi, b) for b in g.gt[i]) for _, i in ge["frames"])
        hitters = [e for e in events if any(abs(t - e["timestamp_s"]) <= TOL and any(box_hit(e["roi_box"], b) for b in g.gt[i])
                                            for t, i in ge["frames"])]
        in_time = [e for e in events if ge["t0"] - TOL <= e["timestamp_s"] <= ge["t1"] + TOL]
        det_frames = sum(any(any(box_hit(p["box"], b) for b in g.gt[i]) for p in by_idx[i].persons) for _, i in ge["frames"])
        gt_boxes = [b for _, i in ge["frames"] for b in g.gt[i]]
        fh, fw = obs[0].frame_hw
        hs = [b[3] - b[1] for b in gt_boxes]
        cause = "caught" if hitters else ("not_detected" if det_frames == 0 else ("wrong_box" if in_time else "no_candidate"))
        n_people = [len(by_idx[i].persons) for _, i in ge["frames"]]
        rows.append({
            "dataset": ds, "sequence": r.seq_id, "tag": g.tag, "event": k, "t0": round(ge["t0"], 2), "t1": round(ge["t1"], 2),
            "duration_s": round(ge["t1"] - ge["t0"], 2), "frames": len(ge["frames"]), "cause": cause,
            "temporal_hit": any(a <= ge["t1"] and b >= ge["t0"] for a, b in merged),
            "frames_person_detected_%": round(100 * det_frames / len(ge["frames"]), 1),
            "gt_box_h_px": round(float(np.median(hs)), 1), "gt_box_h_frac": round(float(np.median(hs)) / fh, 3),
            "gt_box_area_frac": round(float(np.median([(b[2] - b[0]) * (b[3] - b[1]) for b in gt_boxes])) / (fh * fw), 4),
            "frame_h": fh, "people_in_frame": round(float(np.mean(n_people)), 1),
            "fired_in_time": ";".join(sorted({e["event_type"] for e in in_time})),
            "caught_by": ";".join(sorted({e["event_type"] for e in hitters})),
        })
    return rows


def main():
    T = model_splits("s", heldout=False)
    M = methods(T)
    rows = []
    for ds in ("littering", "avenue", "smoking"):
        for g, r in T[ds]:
            rows += analyse(ds, g, r, M[ds](r, ds))
    try:
        from scoped import final_tags
        tags = final_tags()
        for row in rows:
            if row["dataset"] == "avenue":
                hit = [e for e in tags.values() if e["video"] == row["sequence"] and float(e["t0"]) <= row["t1"] + 1e-6
                       and float(e["t1"]) >= row["t0"] - 1e-6]
                row["label_review"] = ";".join(sorted({e["final"] for e in hit}))
                row["what_happens"] = " | ".join(e["what_happens"] for e in hit)
    except Exception as ex:  # noqa: BLE001
        print("label review join failed:", ex)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(HERE / "misses_tune.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    for ds in ("littering", "avenue", "smoking"):
        sub = [r for r in rows if r["dataset"] == ds]
        c = Counter(r["cause"] for r in sub)
        print(f"\n{ds}: {len(sub)} events  " + "  ".join(f"{k} {v} ({100 * v / len(sub):.0f}%)" for k, v in c.most_common()))
        for cause in ("not_detected", "no_candidate", "wrong_box", "caught"):
            s = [r for r in sub if r["cause"] == cause]
            if s:
                print(f"  {cause:13s} median GT box h {np.median([r['gt_box_h_px'] for r in s]):6.1f}px "
                      f"({np.median([r['gt_box_h_frac'] for r in s]):.3f} of frame), median duration "
                      f"{np.median([r['duration_s'] for r in s]):5.1f}s, people/frame {np.mean([r['people_in_frame'] for r in s]):.1f}, "
                      f"fired in time: {Counter(x for r in s for x in r['fired_in_time'].split(';') if x).most_common(4)}")
        if ds == "littering":
            for tag in sorted({r["tag"] for r in sub}):
                print("   ", tag, dict(Counter(r["cause"] for r in sub if r["tag"] == tag)))
        if ds == "avenue":
            for r in sub:
                if r["cause"] != "caught":
                    print(f"    {r['sequence']} {r['t0']}-{r['t1']}s {r['cause']} [{r.get('label_review', '')}] {r.get('what_happens', '')[:110]}")


if __name__ == "__main__":
    main()
