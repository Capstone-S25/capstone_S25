import sys
from collections import Counter
from functools import partial
sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks"); sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks/exp")
from exp_harness import SPLITS, baseline, score
from stage1_gate import GateConfig, calibrate_inview
cfg = GateConfig(inview_s=calibrate_inview([r for _, r in SPLITS["normal"]]))
s = score(partial(baseline, cfg=cfg), name="x")
print({k: s[k] for k in ("littering_event_%", "littering_crop_%", "avenue_event_%", "avenue_crop_%", "avenue_reduction_%", "smoking_event_%", "smoking_crop_%", "normal_FA_per_min")})
for ds in ("normal", "avenue"):
    print(ds, dict(Counter(e["event_type"] for g, r in SPLITS[ds] for e in baseline(r, ds, cfg))))
print("test/09 bag events:", [(e["timestamp_s"], e["event_type"]) for g, r in SPLITS["avenue"] if g.seq.seq_id == "test/09" for e in baseline(r, "avenue", cfg) if e["event_type"] in ("object_appeared", "unattended_object", "reach_down")])
