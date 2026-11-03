"""IWDD: can open-vocabulary objects (YOLOE once a second, as in yoloe_items.py) catch dumping the person cues miss?

Static dumping leaves waste behind, often while the dumper is hidden by a car or pixelated. YOLOE-26s is prompted
with waste words; its 1 fps detections replace obs.items and the existing object rules (object_appeared,
unattended_object) run unchanged with appear_classes = the prompts.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/iwdd_objects.py run train|test   -> bench_cache/models/s/yoloe_iwdd_<split>.pkl
          ../.venv/bin/python exp/newcam/iwdd_objects.py eval train|test     (YOLOE object rules)
          ../.venv/bin/python exp/newcam/iwdd_objects.py bg train|test       (background-subtraction static objects, needs bg_blobs.py iwdd_<split>)
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
from yoloe_items import PROMPTS, run_items, with_items  # noqa: E402

WASTE = tuple(PROMPTS) + ("trash bag", "garbage", "rubbish", "tire", "mattress", "furniture", "bucket", "sack")
CACHE = NB / "bench_cache" / "models" / "s"
OBJECT_RULES = ("unattended_object", "object_appeared")


def run(split):
    path = CACHE / f"yoloe_iwdd_{split}.pkl"
    gts = {g.seq.seq_id: g for g in iwdd_sequences(split)}
    data = [(gts[r.seq_id], r) for _, r in load(split)]   # cached results hold the video handles again
    t0 = time.time()
    path.write_bytes(pickle.dumps(run_items(data, WASTE)))
    print(f"saved {path.name} ({time.time() - t0:.0f}s)", flush=True)


def evaluate_split(split):
    from exp_harness import baseline, combine
    from stage1_gate import GateConfig
    data, labels = load(split), iwdd_labels(split)
    items = pickle.loads((CACHE / f"yoloe_iwdd_{split}.pkl").read_bytes())
    ye = with_items(data, items)
    inview = inview_from_avenue()
    rules = partial(baseline, cfg=GateConfig(inview_s=inview))
    ye_cfg = GateConfig(inview_s=inview, item_min_frames=2, appear_classes=WASTE)
    only = lambda r, ds: [e for e in baseline(r, ds, cfg=ye_cfg) if e["event_type"] in OBJECT_RULES]  # noqa: E731
    by_id = {r.seq_id: r for _, r in ye}
    ye_only = lambda r, ds: only(by_id[r.seq_id], ds)  # noqa: E731
    out = {"rules (COCO items)": evaluate(data, labels, rules),
           "YOLOE object rules alone": evaluate(data, labels, ye_only),
           "rules + YOLOE object rules": evaluate(data, labels, combine(rules, ye_only))}
    for k, v in out.items():
        print(k, json.dumps(v), flush=True)
    (HERE / f"iwdd_objects_{split}.json").write_text(json.dumps(out, indent=1))


def evaluate_bg(split):
    """Background-subtraction static objects (static_v2, settings frozen on CAVIAR + Avenue) on top of the rules."""
    from bg_blobs import with_blobs
    from exp_harness import baseline, combine
    from stage1_gate import GateConfig
    from static_v2 import static_v2
    data, labels = load(split), iwdd_labels(split)
    blobs = pickle.loads((CACHE / f"bg_iwdd_{split}.pkl").read_bytes())
    by_id = {r.seq_id: r for _, r in with_blobs(data, blobs)}
    rules = partial(baseline, cfg=GateConfig(inview_s=inview_from_avenue()))
    out = {}
    for name, kw in (("static_v2", {}), ("static_v2 burst guard", {"burst_max": 1})):
        rule = static_v2(**kw)
        alone = lambda r, ds, rule=rule: rule(by_id[r.seq_id], ds)  # noqa: E731
        out[f"{name} alone"] = evaluate(data, labels, alone)
        out[f"rules + {name}"] = evaluate(data, labels, combine(rules, alone))
    for k, v in out.items():
        print(k, json.dumps(v), flush=True)
    (HERE / f"iwdd_bg_{split}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"run": run, "eval": evaluate_split, "bg": evaluate_bg}[sys.argv[1]](sys.argv[2])
