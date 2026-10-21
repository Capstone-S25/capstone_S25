import sys
from dataclasses import replace
from functools import partial
sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks"); sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks/exp")
from exp_harness import SPLITS, baseline, score
from random_control import lift
from stage1_gate import GateConfig, calibrate_inview
NORMAL = [r for _, r in SPLITS["normal"]]
base = GateConfig(inview_s=calibrate_inview(NORMAL))
def row(name, cfg):
    s = score(partial(baseline, cfg=cfg), name=name)
    m, c, sd = lift(partial(baseline, cfg=cfg), SPLITS["littering"], seeds=5)
    print(f"{name:42s} littering {s['littering_event_%']:5.1f}/{s['littering_crop_%']:5.1f} (random {c:4.1f}, lift {m-c:+5.1f}) | "
          f"loitering {s['avenue_event_%']:5.1f}/{s['avenue_crop_%']:5.1f} skipped {s['avenue_reduction_%']:5.1f}% | "
          f"smoking {s['smoking_event_%']:5.1f}/{s['smoking_crop_%']:5.1f} | FA/min {s['normal_FA_per_min']:5.2f}", flush=True)
row("old reach_down (1 frame, any size)", replace(base, reach_min_hits=1, pose_min_h_px=0))
for hits in (2, 3):
    for margin in (-0.15, -0.10, -0.05, 0.0):
        row(f"reach: {hits} frames, margin {margin}", replace(base, reach_min_hits=hits, reach_margin_h=margin))
