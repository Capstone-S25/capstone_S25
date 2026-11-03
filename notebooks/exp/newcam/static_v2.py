"""Abandoned objects from background subtraction, version 2: persistence instead of an unbroken still-timer.

The original static_object_left rule needs a blob to stay "static" in every frame for litter_static_s and to barely
change since it went still. On noisy, compressed low-resolution video (CAVIAR) the bag's pixels drop out now and
then, so the timer keeps resetting (max ~6 s) and the change check fails. Here a spot counts once a new-object blob
(high novelty vs the long-term background, not touching anyone) keeps showing up there in at least `min_frac` of the
frames over the last `hold_s` seconds. One candidate per spot; the crop is the blob padded with context.

Evaluates (frozen thresholds chosen on CAVIAR + Avenue tune only, then ABODA measured):
  CAVIAR left objects (unseen camera, tiny bags) and CAVIAR normal videos,
  Avenue tune normal footage (known camera, false alarms),
  ABODA (labels from aboda_labels.json; unseen cameras), vs the YOLO-based unattended_object rule.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/static_v2.py
"""
import json
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
from exp_harness import baseline, combine, make_event  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import GateConfig, calibrate_inview, center, expand_box, intersection_over, iou  # noqa: E402


def static_v2(hold_s=8.0, min_frac=0.4, min_novelty=30.0, person_margin=0.25, min_area=2e-4, max_area=0.02, pad=1.5,
              burst_s=2.0, burst_max=None):
    def run(result, dataset):
        spots, out, fired = [], [], []
        times = []
        for o in result.observations:
            t = o.t
            times.append(t)
            people = [expand_box(p["box"], person_margin) for p in o.persons]
            for b in o.blobs:
                if b["novelty"] < min_novelty or not (min_area <= b["area_frac"] <= max_area):
                    continue
                if any(intersection_over(b["box"], q) > 0 for q in people):
                    continue
                c = center(b["box"])
                s = next((s for s in spots if iou(s["box"], b["box"]) >= 0.2 or
                          abs(center(s["box"])[0] - c[0]) + abs(center(s["box"])[1] - c[1]) <= max(8.0, 0.5 * (b["box"][2] - b["box"][0] + b["box"][3] - b["box"][1]))), None)
                if s is None:
                    if any(intersection_over(b["box"], expand_box(f, 0.5)) > 0.3 for f in fired):
                        continue
                    s = {"box": b["box"], "first": t, "hits": [], "fired": False}
                    spots.append(s)
                if not s["hits"] or s["hits"][-1] != t:
                    s["hits"].append(t)
                s["box"] = b["box"]
            for s in spots:
                s["hits"] = [h for h in s["hits"] if h >= t - hold_s]
                if s["fired"] or t - s["first"] < hold_s or not s["hits"]:
                    continue
                n_frames = sum(1 for x in times if x >= t - hold_s)
                if len(s["hits"]) >= min_frac * n_frames:
                    s["fired"] = True
                    fired.append(s["box"])
                    # one left object does not create several new still regions at once: a burst of spots that
                    # appeared within burst_s of each other is a camera move / lighting change, not objects
                    burst = sum(1 for q in spots if abs(q["first"] - s["first"]) <= burst_s)
                    if burst_max is None or burst <= burst_max:
                        out.append(make_event("static_object", t, expand_box(s["box"], pad)))
            spots = [s for s in spots if s["fired"] or (s["hits"] and t - s["hits"][-1] <= hold_s)]
        return out
    return run


def aboda_gt(results, include_ambiguous=False):
    lab = json.loads((HERE / "aboda_labels.json").read_text())
    out = []
    for g, r in results:
        v = r.seq_id.split("/")[1]
        evs = lab[v]["events"] + (lab[v].get("ambiguous_events", []) if include_ambiguous else [])
        gt = {}
        for o in r.observations:
            bs = [e["box"] for e in evs if e["t_unattended"] <= o.t <= (e["t_removed"] if e["t_removed"] is not None else 1e9)]
            if bs:
                gt[o.index] = [list(map(float, b)) for b in bs]
        out.append((replace(g, gt=gt), r))
    return out


def rate(results, method, etype):
    n = sum(1 for _, r in results for e in method(r, "x") if e["event_type"] == etype)
    return round(n / (sum(r.duration_s for _, r in results) / 60), 2)


def score(results, method):
    s = summarize([evaluate_run(g, r, method(r, "x")) for g, r in results if g.gt], "x")
    return s["event_recall_%"], s["crop_recall_%"], s["gt_events"]


def main():
    T = model_splits("s", heldout=False)
    inview = calibrate_inview([r for _, r in T["normal"]])
    cav = with_blobs(pickle.loads((CACHE / "caviar.pkl").read_bytes()), pickle.loads((CACHE / "bg_caviar.pkl").read_bytes()))
    av = with_blobs(T["avenue"] + T["normal"], pickle.loads((CACHE / "bg_avenue_tune.pkl").read_bytes()))
    av_norm = [x for x in av if x[1].seq_id.startswith("train/")]
    cav_pos, cav_neg = [x for x in cav if x[0].gt], [x for x in cav if x[0].tag == "normal"]
    rows = []
    for hold in (5.0, 8.0, 12.0):
        for frac in (0.25, 0.4, 0.6):
            for nov in (20.0, 30.0, 45.0):
                m = static_v2(hold_s=hold, min_frac=frac, min_novelty=nov)
                ev, cr, n = score(cav_pos, m)
                rows.append({"hold_s": hold, "min_frac": frac, "min_novelty": nov, "caviar_event_%": ev, "caviar_crop_%": cr,
                             "caviar_normal_per_min": rate(cav_neg, m, "static_object"),
                             "caviar_leftbag_videos_per_min": rate(cav_pos, m, "static_object"),
                             "avenue_normal_per_min": rate(av_norm, m, "static_object")})
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "static_v2_sweep.csv", index=False)
    # pick on CAVIAR + Avenue only: best CAVIAR crop recall with Avenue normal <= 0.5/min, then fewest alarms
    ok = df[df.avenue_normal_per_min <= 0.5].sort_values(["caviar_crop_%", "avenue_normal_per_min", "caviar_normal_per_min"], ascending=[False, True, True])
    best = ok.iloc[0].to_dict()
    print("\nchosen (CAVIAR + Avenue tune only):", best)
    m = static_v2(hold_s=best["hold_s"], min_frac=best["min_frac"], min_novelty=best["min_novelty"])
    yolo = partial(baseline, cfg=GateConfig(inview_s=inview))
    aboda = pickle.loads((CACHE / "aboda.pkl").read_bytes())
    aboda_bg = with_blobs(aboda, pickle.loads((CACHE / "bg_aboda.pkl").read_bytes()))
    summary = {"chosen": best}
    for amb in (False, True):
        A = aboda_gt(aboda_bg, include_ambiguous=amb)
        key = "aboda_with_ambiguous" if amb else "aboda"
        only_yolo = lambda r, ds: [e for e in yolo(r, ds) if e["event_type"] in ("unattended_object", "object_appeared")]  # noqa: E731
        summary[key] = {
            "yolo_bag_rules (event%, crop%, n)": score(A, only_yolo),
            "static_v2 (event%, crop%, n)": score(A, m),
            "both (event%, crop%, n)": score(A, combine(only_yolo, m)),
            "all_rules + static_v2 (event%, crop%, n)": score(A, combine(yolo, m)),
            "static_v2_per_min": rate(A, m, "static_object"),
            "yolo_bag_rules_per_min": round(sum(len(only_yolo(r, "x")) for _, r in A) / (sum(r.duration_s for _, r in A) / 60), 2),
        }
    summary["caviar_yolo_bag_rules (event%, crop%, n)"] = score(cav_pos, lambda r, ds: [e for e in yolo(r, ds) if e["event_type"] in ("unattended_object", "object_appeared")])
    (HERE / "static_v2_results.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
