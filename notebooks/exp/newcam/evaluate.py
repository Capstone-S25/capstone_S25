"""Unseen-camera evaluation: the frozen Stage 1 config (rules + pose_action, thresholds from the tune half of the
original datasets, nothing re-tuned) on CAVIAR, MEVA and ABODA.

Usage (from notebooks/): ../.venv/bin/python exp/newcam/evaluate.py [caviar meva aboda]
"""
import json
import pickle
import sys
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE)]

from compare_models import model_splits  # noqa: E402
from exp_harness import baseline  # noqa: E402
from stage1_bench import box_hit, evaluate_run, summarize  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, calibrate_inview, intersection_over  # noqa: E402

CACHE = NB / "bench_cache" / "models" / "s"


def configs():
    inview = calibrate_inview([r for _, r in model_splits("s", heldout=False)["normal"]])
    return {"current rules": GateConfig(inview_s=inview),
            "rules + pose_action": GateConfig(inview_s=inview, rules=DEFAULT_RULES + ("pose_action",))}


def load(name):
    return pickle.loads((CACHE / f"{name}.pkl").read_bytes())


def object_seen(g, r, min_cover=0.3):
    """Share of GT frames where YOLO reported any item covering the GT object box."""
    by = {o.index: o for o in r.observations}
    fr = [i for i in g.gt if i in by]
    seen = sum(any(intersection_over(b, it["box"]) >= min_cover or intersection_over(it["box"], b) >= min_cover
                   for b in g.gt[i] for it in by[i].items) for i in fr)
    return round(100 * seen / max(1, len(fr)), 1)


def caviar(cfgs):
    res = load("caviar")
    out = {}
    for name, cfg in cfgs.items():
        m = partial(baseline, cfg=cfg)
        pos = [(g, r) for g, r in res if g.gt]
        neg = [(g, r) for g, r in res if g.tag == "normal"]
        s = summarize([evaluate_run(g, r, m(r, "caviar")) for g, r in pos], name)
        n = summarize([evaluate_run(g, r, m(r, "caviar")) for g, r in neg], name)
        types = Counter(e["event_type"] for g, r in res for e in m(r, "caviar"))
        per = {}
        for g, r in pos:
            ev = m(r, "caviar")
            ro = evaluate_run(g, r, ev)
            per[r.seq_id] = {"caught": ro["ev_spatial"] > 0, "in_time": ro["ev_temporal"] > 0,
                             "object_detected_%frames": object_seen(g, r),
                             "gt_object_h_px": round(float(np.median([b[3] - b[1] for bs in g.gt.values() for b in bs])), 1),
                             "fired": dict(Counter(e["event_type"] for e in ev))}
        out[name] = {"left_object_event_%": s["event_recall_%"], "left_object_crop_%": s["crop_recall_%"],
                     "left_object_events": s["gt_events"], "normal_candidates_per_min": n["candidates_per_min"],
                     "all_candidates_per_min": round(sum(types.values()) / (sum(r.duration_s for _, r in res) / 60), 1),
                     "candidate_types": dict(types), "per_video": per}
    return out


def meva(cfgs):
    from loaders import meva_relabel
    res = load("meva")
    minutes = sum(r.duration_s for _, r in res) / 60
    out = {"minutes": round(minutes, 1)}
    acts = {"puts_down": ("person_puts_down_object",), "picks_up": ("person_picks_up_object",),
            "transfers": ("person_transfers_object",), "sits_down": ("person_sits_down",),
            "talks_to_person": ("person_talks_to_person",)}
    for name, cfg in cfgs.items():
        m = partial(baseline, cfg=cfg)
        events = {r.seq_id: m(r, "meva") for _, r in res}
        row = {"candidates_per_min": round(sum(map(len, events.values())) / minutes, 2),
               "candidate_types_per_min": {k: round(v / minutes, 2) for k, v in Counter(e["event_type"] for ev in events.values() for e in ev).items()}}
        for a, types in acts.items():
            rel = [(g, r) for g, r in meva_relabel(res, types) if g.gt]
            if not rel:
                continue
            rows = [evaluate_run(g, r, events[r.seq_id]) for g, r in rel]
            s = summarize(rows, a)
            row[a] = {"events": s["gt_events"], "event_%": s["event_recall_%"], "crop_%": s["crop_recall_%"],
                      "caught_by": dict(sum((Counter(x["caught_by"]) for x in rows), Counter()))}
        out[name] = row
    return out


if __name__ == "__main__":
    cfgs = configs()
    for name in sys.argv[1:] or ["caviar", "meva"]:
        result = {"caviar": caviar, "meva": meva}[name](cfgs)
        (HERE / f"results_{name}.json").write_text(json.dumps(result, indent=1, default=str))
        print(name, json.dumps(result, indent=1, default=str))
