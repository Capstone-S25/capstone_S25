"""Held-out / not-used-for-tuning numbers for the current GateConfig defaults (rules + pose cue), scored once.

Littering held-out clips, Avenue held-out (even test videos) and normal (even training videos), smoking held-out
camera-days, IWDD test (hit within 10 s vs rate-matched chance), ABODA bags caught (candidate on the bag between
put-down - 2 s and removal - 2 s). Old = reach_down without the three 2026-11-04 options.
Usage (from notebooks/): ../.venv/bin/python exp/final_defaults.py   -> exp/final_defaults.json
"""
import json
import pickle
import sys
from functools import partial
from pathlib import Path

NB = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "newcam")]

from compare_models import model_splits  # noqa: E402
from exp_harness import baseline  # noqa: E402
from iwdd_eval import evaluate  # noqa: E402
from loaders import iwdd_labels  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, intersection_over  # noqa: E402

OLD = dict(reach_hand_pad_h=None, reach_refire_move_h=None, reach_max_speed_h=None)


def score(cfg):
    m = partial(baseline, cfg=cfg)
    H = model_splits("s", heldout=True)
    out = {}
    for ds, name in (("littering", "littering"), ("avenue", "loitering"), ("smoking", "smoking")):
        s = summarize([evaluate_run(g, r, m(r, ds)) for g, r in H[ds]], ds)
        out[f"{name} caught / right person %"] = f"{s['event_recall_%']} / {s['crop_recall_%']}"
    mins = sum(r.duration_s for _, r in H["normal"]) / 60
    out["false alarms on normal video / min"] = round(sum(len(m(r, "normal")) for _, r in H["normal"]) / mins, 2)
    iw = evaluate(pickle.loads((NB / "bench_cache/models/s/iwdd_test.pkl").read_bytes()), iwdd_labels("test"), m)
    out["IWDD test dumping caught within 10 s vs chance %"] = f"{iw['hit@10s_%']} vs {iw['chance@10s_%']}"
    out["IWDD test normal videos candidates / min"] = iw["neg_cand_per_min"]
    lab = json.loads((NB / "exp/newcam/aboda_labels.json").read_text())
    caught = 0
    for _, r in pickle.loads((NB / "bench_cache/models/s/aboda.pkl").read_bytes()):
        evs = m(r, "aboda")
        for e in lab[r.seq_id.split("/")[-1]]["events"]:
            end = e["t_removed"] - 2 if e["t_removed"] is not None else float("inf")
            caught += any(e["t_put_down"] - 2 <= x["timestamp_s"] <= end and intersection_over(e["box"], x["roi_box"]) >= 0.3
                          for x in evs)
    out["ABODA bags caught"] = f"{caught}/{sum(len(v['events']) for v in lab.values())}"
    return out


def main():
    rules = DEFAULT_RULES + ("pose_action",)
    out = {"old reach_down": score(GateConfig(inview_s=60.0, rules=rules, **OLD)),
           "current defaults": score(GateConfig(inview_s=60.0, rules=rules))}
    for k, v in out.items():
        print(k, json.dumps(v), flush=True)
    (NB / "exp" / "final_defaults.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
