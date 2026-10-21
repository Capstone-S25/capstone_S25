import sys
from dataclasses import replace
from functools import partial
sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks"); sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks/exp")
from exp_harness import SPLITS, baseline, score
from random_control import lift
from stage1_gate import GateConfig, calibrate_inview
base = GateConfig(inview_s=calibrate_inview([r for _, r in SPLITS["normal"]]), reach_min_hits=1)
def show(name, cfg):
    s = score(partial(baseline, cfg=cfg), name=name)
    m, c, _ = lift(partial(baseline, cfg=cfg), SPLITS["littering"], seeds=5)
    reach_fa = sum(1 for g, r in SPLITS["normal"] for e in baseline(r, "normal", cfg) if e["event_type"] == "reach_down")
    print(f"{name:34s} littering {s['littering_event_%']:5.1f}/{s['littering_crop_%']:5.1f} (lift {m-c:+5.1f}) | loitering {s['avenue_event_%']:5.1f}/{s['avenue_crop_%']:5.1f} | smoking {s['smoking_event_%']:5.1f}/{s['smoking_crop_%']:5.1f} | FA/min {s['normal_FA_per_min']:5.2f} (reach_down FAs: {reach_fa})", flush=True)
#show("1 frame, border skip (current)", base)
for kc in ():
    show(f"+ keypoint conf {kc}", replace(base, reach_kp_conf=kc))
for br in ():
    show(f"+ bending (box <= {br} x usual)", replace(base, reach_bend_ratio=br))
print("--- combinations with keypoint conf 0.5")
k5 = replace(base, reach_kp_conf=0.5)
show("kp0.5, no border skip, no min height", replace(k5, reach_skip_border=False, pose_min_h_px=0))
show("kp0.5 + border skip", replace(k5, pose_min_h_px=0))
show("kp0.5 + min height 60", replace(k5, reach_skip_border=False))
