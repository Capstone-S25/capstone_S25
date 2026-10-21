import sys
from dataclasses import replace
from functools import partial
sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks"); sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks/exp")
from exp_harness import SPLITS, baseline, score
from stage1_bench import evaluate_run
from stage1_gate import GateConfig, calibrate_inview
base = GateConfig(inview_s=calibrate_inview([r for _, r in SPLITS["normal"]]))
old = replace(base, reach_min_hits=1, pose_min_h_px=0, reach_skip_border=False)
def show(name, cfg):
    s = score(partial(baseline, cfg=cfg), name=name)
    print(f"{name:38s} littering {s['littering_event_%']:5.1f}/{s['littering_crop_%']:5.1f} | loitering {s['avenue_event_%']:5.1f}/{s['avenue_crop_%']:5.1f} skipped {s['avenue_reduction_%']:5.1f}% | smoking {s['smoking_event_%']:5.1f}/{s['smoking_crop_%']:5.1f} | FA/min {s['normal_FA_per_min']:5.2f}", flush=True)
show("old reach + new bag logic", old)
show("old reach, no bag rule at all", replace(old, rules=tuple(r for r in old.rules if r != "unattended")))
show("+ skip border", replace(old, reach_skip_border=True))
show("+ min height 60 px", replace(old, pose_min_h_px=60))
show("+ min height 40 px", replace(old, pose_min_h_px=40))
show("+ 2 frames", replace(old, reach_min_hits=2))
# which Avenue events are lost with border skipping?
for name, cfg in [("old", old), ("border", replace(old, reach_skip_border=True))]:
    lost = []
    for g, r in SPLITS["avenue"]:
        row = evaluate_run(g, r, baseline(r, "avenue", cfg))
        lost += [(r.seq_id, m["t0"]) for m in row["missed"]]
    print(name, "missed avenue events:", lost)
# object_appeared / unattended counts
for ds in ("avenue", "normal"):
    from collections import Counter
    c = Counter(e["event_type"] for g, r in SPLITS[ds] for e in baseline(r, ds, base))
    print(ds, dict(c))
