"""Stage 1 with the once-a-second YOLOE item pass (GateConfig.item_detector) vs the current gate, end to end.

Perception is re-run with the pass on (people on YOLO26s every frame, YOLOE-26s every 1 s for objects) on ABODA
(footage not used for tuning) and Avenue (held-out test and normal videos), and scored with rules + pose cue:
ABODA bags caught (a candidate on the labelled bag from 2 s before put-down onward) and whether the bag was ever
detected; Avenue held-out events; false alarms and bag alarms per minute on normal video; CPU ms per frame.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/yoloe_gate.py   -> exp/newcam/yoloe_gate.json
"""
import json
import pickle
import sys
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
D = NB.parent / "data"
sys.path[:0] = [str(NB), str(NB / "exp"), str(HERE)]

from exp_harness import baseline  # noqa: E402
from loaders import aboda_sequences  # noqa: E402
from stage1_bench import avenue_sequences, evaluate_run, run_perception, summarize  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, Stage1Gate, intersection_over  # noqa: E402

ORIGINAL = NB / "bench_cache" / "models" / "s"
YOLOE_CACHE = NB / "bench_cache" / "models" / "s_yoloe"
ITEM_DETECTOR = str(NB / "yoloe-26s-seg.pt")
BAG_EVENTS = ("unattended_object", "object_appeared")


def perceive():
    YOLOE_CACHE.mkdir(parents=True, exist_ok=True)
    cfg = GateConfig(det_weights=str(NB / "yolo26s.pt"), pose_weights=str(NB / "yolo26s-pose.pt"), device="mps",
                     item_detector=ITEM_DETECTOR)
    jobs = {"aboda": aboda_sequences,
            "avenue": lambda: avenue_sequences(D / "loitering", D / "loitering" / "avenue_gt" / "ground_truth_demo" /
                                               "testing_label_mask", stride=2)}
    for name, build in jobs.items():
        if not (YOLOE_CACHE / f"{name}.pkl").exists():
            (YOLOE_CACHE / f"{name}.pkl").write_bytes(pickle.dumps(run_perception(Stage1Gate(cfg), build(), verbose=False)))
            print("perception", name, "done", flush=True)


def score(cache, cfg):
    m = partial(baseline, cfg=cfg)
    lab = json.loads((HERE / "aboda_labels.json").read_text())
    load = lambda n: pickle.loads((cache / f"{n}.pkl").read_bytes())  # noqa: E731
    caught, seen, total, per_video = 0, 0, 0, {}
    for g, r in load("aboda"):
        v = r.seq_id.split("/")[-1]
        evs = m(r, "aboda")
        for e in lab[v]["events"]:
            total += 1
            b = e["box"]
            hit = [x for x in evs if x["timestamp_s"] >= e["t_put_down"] - 2 and intersection_over(b, x["roi_box"]) >= 0.3]
            ever = any(o.t >= e["t_unattended"] and any(intersection_over(b, it["box"]) >= 0.3 or
                                                       intersection_over(it["box"], b) >= 0.3 for it in o.items)
                       for o in r.observations)
            caught += bool(hit)
            seen += ever
            per_video[f"{v}@{e['t_put_down']}"] = {"caught_by": sorted({x["event_type"] for x in hit}), "bag_detected": ever}
    num = lambda g: int(g.seq.seq_id.split("/")[-1])  # noqa: E731
    av = load("avenue")
    normal = [x for x in av if x[0].seq.seq_id.startswith("train/")]
    test = [x for x in av if x[0].seq.seq_id.startswith("test/") and num(x[0]) % 2 == 0]
    a = summarize([evaluate_run(g, r, m(r, "avenue")) for g, r in test], "a")
    mins = sum(r.duration_s for _, r in normal) / 60
    lat = [r.mean_latency_ms for _, r in normal]
    return {"aboda_bags_caught": f"{caught}/{total}", "aboda_bag_ever_detected": f"{seen}/{total}",
            "avenue_heldout_event/crop_%": f"{a['event_recall_%']}/{a['crop_recall_%']}",
            "avenue_normal_FA_per_min": round(sum(len(m(r, "normal")) for _, r in normal) / mins, 2),
            "avenue_normal_bag_alarms_per_min": round(sum(len([e for e in m(r, "normal") if e["event_type"] in BAG_EVENTS])
                                                          for _, r in normal) / mins, 2),
            "perception_ms_per_frame_mps": round(sum(lat) / len(lat), 1), "per_bag": per_video}


def main():
    perceive()
    rules = DEFAULT_RULES + ("pose_action",)
    out = {"current gate": score(ORIGINAL, GateConfig(inview_s=60.0, rules=rules)),
           "with YOLOE item pass": score(YOLOE_CACHE, GateConfig(inview_s=60.0, rules=rules, item_detector=ITEM_DETECTOR))}
    for k, v in out.items():
        print(k, json.dumps({x: y for x, y in v.items() if x != "per_bag"}), flush=True)
    (HERE / "yoloe_gate.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
