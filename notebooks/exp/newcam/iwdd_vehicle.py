"""IWDD: a "person at a stopped vehicle" cue for dumping from cars and vans.

Many tracked-but-missed dumpers stand at an open boot or van door, unloading at hip height, so reach_down never
fires. Vehicles are not kept by the gate (only people and COCO_ITEMS), so YOLO26s (COCO) runs once a second here
and keeps car / motorcycle / bus / truck boxes. A vehicle is stopped when a box in the previous pass overlaps it
(IoU >= STOP_IOU). The cue fires for a tracked person whose box touches a stopped vehicle (vehicle box widened by
`margin` x its width) in `dwell` consecutive passes; one candidate per person per COOLDOWN_S, ROI = person box.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/iwdd_vehicle.py run train|test  -> bench_cache/models/s/veh_iwdd_<split>.pkl
          ../.venv/bin/python exp/newcam/iwdd_vehicle.py eval train
"""
import json
import pickle
import sys
import time
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE)]

from iwdd_eval import evaluate, inview_from_avenue, load  # noqa: E402
from loaders import iwdd_labels, iwdd_sequences  # noqa: E402
from stage1_gate import expand_box, intersection_over, iou  # noqa: E402

VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERIOD_S = 1.0
CONF = 0.3
STOP_IOU = 0.7
COOLDOWN_S = 10.0
CACHE = NB / "bench_cache" / "models" / "s"


def run(split):
    from ultralytics import YOLO
    model = YOLO(str(NB / "yolo26s.pt"))
    gts = {g.seq.seq_id: g for g in iwdd_sequences(split)}
    out, t0 = {}, time.time()
    for _, r in load(split):
        want, nxt = set(), 0.0
        for o in r.observations:
            if o.t >= nxt - 1e-6:
                want.add(o.index)
                nxt = o.t + PERIOD_S
        dets = {}
        for fr in gts[r.seq_id].seq:
            if fr.index in want:
                res = model.predict(fr.image, imgsz=640, conf=CONF, classes=list(VEHICLES), verbose=False, device="mps")[0]
                dets[fr.index] = [[float(v) for v in b] for b in res.boxes.xyxy.cpu().numpy()]
        out[r.seq_id] = dets
    (CACHE / f"veh_iwdd_{split}.pkl").write_bytes(pickle.dumps(out))
    print(f"saved veh_iwdd_{split}.pkl ({time.time() - t0:.0f}s)", flush=True)


def vehicle_cue(vehicles, dwell=2, margin=0.25):
    """method(result, dataset) -> events; vehicles = {seq_id: {frame index: [xyxy]}} from run()."""
    from exp_harness import make_event

    def run_cue(result, dataset):
        v = vehicles.get(result.seq_id, {})
        prev, streak, last, out = [], {}, {}, []
        for o in result.observations:
            if o.index not in v:
                continue
            cur = v[o.index]
            stopped = [b for b in cur if any(iou(b, p) >= STOP_IOU for p in prev)]
            prev = cur
            near = set()
            for p in o.persons:
                if any(intersection_over(p["box"], expand_box(b, margin)) > 0 for b in stopped):
                    rid = p["raw_id"]
                    near.add(rid)
                    streak[rid] = streak.get(rid, 0) + 1
                    if streak[rid] >= dwell and o.t - last.get(rid, -1e9) >= COOLDOWN_S:
                        last[rid] = o.t
                        out.append(make_event("person_at_vehicle", o.t, p["box"]))
            streak = {k: n for k, n in streak.items() if k in near}
        return out
    return run_cue


def evaluate_split(split):
    from exp_harness import baseline, combine
    from stage1_gate import DEFAULT_RULES, GateConfig
    data, labels = load(split), iwdd_labels(split)
    vehicles = pickle.loads((CACHE / f"veh_iwdd_{split}.pkl").read_bytes())
    rules = partial(baseline, cfg=GateConfig(inview_s=inview_from_avenue(), rules=DEFAULT_RULES + ("pose_action",)))
    out = {"rules + pose": evaluate(data, labels, rules)}
    for dwell in (1, 2, 3):
        for margin in (0.1, 0.25, 0.5):
            cue = vehicle_cue(vehicles, dwell, margin)
            a, b = evaluate(data, labels, cue), evaluate(data, labels, combine(rules, cue))
            key = f"dwell={dwell} margin={margin}"
            out[key] = {"alone": a, "with rules": b}
            print(key, "alone hit/chance", a["hit@10s_%"], a["chance@10s_%"], "neg_any", a["neg_any_%"],
                  "| with rules", b["hit@10s_%"], b["chance@10s_%"], "neg_any", b["neg_any_%"], b["neg_cand_per_min"],
                  flush=True)
    (HERE / f"iwdd_vehicle_{split}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"run": run, "eval": evaluate_split}[sys.argv[1]](sys.argv[2])
