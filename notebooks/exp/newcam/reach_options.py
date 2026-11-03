"""Two reach_down options for "person put something down" (ABODA misses): a crop that includes the low hand
(reach_hand_pad_h) and re-firing when the person has moved since their last reach (reach_refire_move_h).

Chosen on TUNE data only (littering train clips, Avenue odd test / training videos), rules + pose cue: a setting must
not lower littering or Avenue event / crop recall or raise normal false alarms by more than 0.5 / min. The chosen
setting is then scored once on ABODA (footage not used for tuning) and the held-out halves.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/reach_options.py              -> reach_options_tune.json
          ../.venv/bin/python exp/newcam/reach_options.py aboda '<json setting>'  -> reach_options_aboda.json
"""
import json
import pickle
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE)]

from compare_models import model_splits  # noqa: E402
from exp_harness import baseline  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, intersection_over  # noqa: E402

RULES = DEFAULT_RULES + ("pose_action",)
GRID = [{}, {"reach_hand_pad_h": 0.25}, {"reach_hand_pad_h": 0.4},
        {"reach_refire_move_h": 0.75}, {"reach_refire_move_h": 1.0},
        {"reach_hand_pad_h": 0.25, "reach_refire_move_h": 1.0}, {"reach_hand_pad_h": 0.4, "reach_refire_move_h": 1.0}]


def row(S, kw):
    m = partial(baseline, cfg=GateConfig(inview_s=60.0, rules=RULES, **kw))
    lit = summarize([evaluate_run(g, r, m(r, "littering")) for g, r in S["littering"]], "l")
    av = summarize([evaluate_run(g, r, m(r, "avenue")) for g, r in S["avenue"]], "a")
    mins = sum(r.duration_s for _, r in S["normal"]) / 60
    return {"littering": f"{lit['event_recall_%']}/{lit['crop_recall_%']}",
            "avenue": f"{av['event_recall_%']}/{av['crop_recall_%']}",
            "normal_FA": round(sum(len(m(r, "normal")) for _, r in S["normal"]) / mins, 2),
            "_lit": (lit["event_recall_%"], lit["crop_recall_%"]), "_av": (av["event_recall_%"], av["crop_recall_%"])}


def tune():
    S = model_splits("s", heldout=False)
    rows = [{"setting": kw, **row(S, kw)} for kw in GRID]
    base = rows[0]
    for r in rows:
        r["ok"] = (r["_lit"][0] >= base["_lit"][0] and r["_lit"][1] >= base["_lit"][1] and r["_av"][0] >= base["_av"][0]
                   and r["_av"][1] >= base["_av"][1] and r["normal_FA"] <= base["normal_FA"] + 0.5)
        print(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}), flush=True)
    (HERE / "reach_options_tune.json").write_text(
        json.dumps([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], indent=1))


def aboda(kw):
    """ABODA bags caught (same test as yoloe_gate.py) and held-out halves, current vs the chosen setting."""
    lab = json.loads((HERE / "aboda_labels.json").read_text())
    data = pickle.loads((NB / "bench_cache" / "models" / "s" / "aboda.pkl").read_bytes())
    H = model_splits("s", heldout=True)
    out = {}
    for name, setting in (("current", {}), ("chosen", kw)):
        m = partial(baseline, cfg=GateConfig(inview_s=60.0, rules=RULES, **setting))
        caught, per = 0, {}
        for g, r in data:
            v = r.seq_id.split("/")[-1]
            for e in lab[v]["events"]:
                end = e["t_removed"] - 2 if e["t_removed"] is not None else float("inf")  # not the pick-up
                hit = [x for x in m(r, "aboda") if e["t_put_down"] - 2 <= x["timestamp_s"] <= end and
                       intersection_over(e["box"], x["roi_box"]) >= 0.3]
                caught += bool(hit)
                per[f"{v}@{e['t_put_down']}"] = sorted({x["event_type"] for x in hit})
        out[name] = {"setting": setting, "aboda_bags_caught": f"{caught}/{sum(len(l['events']) for l in lab.values())}",
                     "heldout": {k: v for k, v in row(H, setting).items() if not k.startswith("_")}, "per_bag": per}
        print(name, json.dumps({k: v for k, v in out[name].items() if k != "per_bag"}), flush=True)
    (HERE / "reach_options_aboda.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["aboda"]:
        aboda(json.loads(sys.argv[2]))
    else:
        tune()
