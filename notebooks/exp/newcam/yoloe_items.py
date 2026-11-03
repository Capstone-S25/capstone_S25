"""Open-vocabulary objects for the bag rules: YOLOE-26s prompted with object words, run once per second.

People stay on YOLO26s every frame (cached runs). Abandoned objects are still, so an object pass once a second is
enough; its detections replace obs.items on those frames (other frames carry no items), with simple IoU tracking to
give them ids. The unattended_object / object_appeared rules then run unchanged (appear_classes extended with the
prompt words).

Usage (from notebooks/): ../.venv/bin/python exp/newcam/yoloe_items.py [aboda caviar avenue_tune]  -> bench_cache/models/s/yoloe_<name>.pkl
          ../.venv/bin/python exp/newcam/yoloe_items.py eval
"""
import json
import pickle
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE)]

from bg_blobs import CACHE, sources  # noqa: E402
from stage1_gate import GateConfig, RunResult, iou  # noqa: E402

PROMPTS = ["bag", "backpack", "handbag", "suitcase", "box", "tote bag", "luggage", "case", "package"]
PERIOD_S = 1.0
CONF = 0.2


def run_items(results, prompts=tuple(PROMPTS)):
    from ultralytics import YOLOE
    model = YOLOE(str(NB / "yoloe-26s-seg.pt"))
    model.set_classes(list(prompts))
    out = {}
    for g, r in results:
        want = {}
        nxt = 0.0
        for o in r.observations:
            if o.t >= nxt - 1e-6:
                want[o.index] = o.t
                nxt = o.t + PERIOD_S
        items, tracks, next_id = {}, [], 0
        for fr in g.seq:
            if fr.index not in want:
                continue
            res = model.predict(fr.image, imgsz=640, conf=CONF, verbose=False, device="cpu")[0]
            dets = []
            for b, c, k in zip(res.boxes.xyxy.numpy(), res.boxes.conf.numpy(), res.boxes.cls.numpy().astype(int)):
                box = [float(v) for v in b]
                tr = max(tracks, key=lambda q: iou(q["box"], box), default=None)
                if tr is None or iou(tr["box"], box) < 0.3:
                    tr = {"id": 100000 + next_id, "box": box}
                    next_id += 1
                    tracks.append(tr)
                tr["box"] = box
                dets.append({"raw_id": tr["id"], "box": box, "conf": float(c), "cls": prompts[k]})
            items[fr.index] = dets
        out[r.seq_id] = items
        print(f"  {r.seq_id}: {len(items)} object passes", flush=True)
    return out


def with_items(results, items):
    out = []
    for g, r in results:
        it = items[r.seq_id]
        obs = [replace(o, items=it.get(o.index, [])) for o in r.observations]
        out.append((g, RunResult(r.seq_id, r.source, r.fps, r.n_frames, r.duration_s, r.events, obs, r.wall_s, r.n_stitched)))
    return out


def evaluate():
    from compare_models import model_splits
    from exp_harness import baseline
    from static_v2 import aboda_gt, rate, score
    from stage1_gate import calibrate_inview
    T = model_splits("s", heldout=False)
    inview = calibrate_inview([r for _, r in T["normal"]])
    bag_types = ("unattended_object", "object_appeared")
    base_cfg = GateConfig(inview_s=inview)
    ye_cfg = GateConfig(inview_s=inview, item_min_frames=2, appear_classes=tuple(PROMPTS))
    out = {}
    av = T["avenue"] + T["normal"]
    sets = {
        "aboda": aboda_gt(pickle.loads((CACHE / "aboda.pkl").read_bytes())),
        "caviar": pickle.loads((CACHE / "caviar.pkl").read_bytes()),
        "avenue_normal": [x for x in av if x[1].seq_id.startswith("train/")],
        "avenue_test": [x for x in av if x[1].seq_id.startswith("test/")],
    }
    ye = {"aboda": "aboda", "caviar": "caviar", "avenue_normal": "avenue_tune", "avenue_test": "avenue_tune"}
    for name, res in sets.items():
        items = pickle.loads((CACHE / f"yoloe_{ye[name]}.pkl").read_bytes())
        for label, data, cfg in (("coco_every_frame", res, base_cfg), ("yoloe_1fps", with_items(res, items), ye_cfg)):
            m = partial(baseline, cfg=cfg)
            only = lambda r, ds, m=m: [e for e in m(r, ds) if e["event_type"] in bag_types]  # noqa: E731
            row = {"bag_alarms_per_min": round(sum(len(only(r, "x")) for _, r in data) / (sum(r.duration_s for _, r in data) / 60), 2)}
            if any(g.gt for g, _ in data) and name in ("aboda", "caviar"):
                row["bag_rules (event%, crop%, n)"] = score([x for x in data if x[0].gt], only)
            out[f"{name}/{label}"] = row
            print(name, label, row, flush=True)
    (HERE / "yoloe_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["eval"]:
        evaluate()
    else:
        for name in sys.argv[1:]:
            path = CACHE / f"yoloe_{name}.pkl"
            if not path.exists():
                t0 = time.time()
                path.write_bytes(pickle.dumps(run_items(sources(name))))
                print(f"{name} saved ({time.time() - t0:.0f}s)", flush=True)
