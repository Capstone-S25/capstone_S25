import sys, numpy as np
sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks"); sys.path.insert(0, ".")
exec(open("combo_eval.py").read().split("combos = {")[0])
from exp_harness import SPLITS
for name, m in [("baseline", baseline), ("act(A)", LA_A), ("motion", MOT), ("crop", CROP), ("dwell(A_lean)", SD_A), ("poseanom(A)", PA_A)]:
    fr = []
    for ds in ("littering", "avenue", "smoking", "normal"):
        for g, r in SPLITS[ds]:
            h, w = r.observations[0].frame_hw if r.observations else (1, 1)
            for e in m(r, ds):
                x1, y1, x2, y2 = e["roi_box"]
                fr.append(max(0, x2 - x1) * max(0, y2 - y1) / (w * h))
    fr = np.array(fr)
    print(f"{name:14s} n={len(fr):5d} roi area % of frame: median {100*np.median(fr):.1f}  p90 {100*np.percentile(fr,90):.1f}  max {100*fr.max():.1f}")
