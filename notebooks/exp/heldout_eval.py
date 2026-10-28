"""ONE-TIME held-out measurement (STAGE1_HELDOUT=1). Thresholds were fixed on the tune half beforehand."""
import os, sys, json
os.environ["STAGE1_HELDOUT"] = "1"
sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from exp_harness import SPLITS, score, combine, baseline, dedup, make_event, load_splits
import motion_anomaly, littering_act
from random_control import lift
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

assert all(not g.seq.seq_id.startswith(("train/01", "train/03")) for g, _ in SPLITS["normal"]), "held-out normal must be even videos"
tune_normal = load_splits(heldout=False)["normal"]
H99 = float(np.percentile([p["box"][3] - p["box"][1] for _, r in tune_normal for o in r.observations for p in o.persons], 99))

def big_box(r, ds):
    if ds not in ("avenue", "normal"):
        return []
    out, last = [], {}
    for o in r.observations:
        for p in o.persons:
            if p["box"][3] - p["box"][1] > H99 and o.t - last.get(p["raw_id"], -1e9) >= 3.0:
                last[p["raw_id"]] = o.t
                out.append(make_event("near_camera", o.t, p["box"]))
    return out

MOT = motion_anomaly.make_method(**motion_anomaly.DEFAULTS)
LA = littering_act.make_method(**{**littering_act.DEFAULTS, "extend_h": 0.18})
rows = [score(dedup(baseline), name="current rules"),
        score(dedup(combine(baseline, MOT)), name="+ unusual-place prior (motion module)"),
        score(dedup(combine(baseline, big_box)), name="+ very-tall-box rule (control)"),
        score(dedup(combine(baseline, LA)), name="+ littering act cues (rejected)")]
df = pd.DataFrame(rows).set_index("method")
print(df.T)
m, c, s = lift(baseline, SPLITS["littering"])
ml, cl, sl = lift(dedup(combine(baseline, LA)), SPLITS["littering"])
ctrl = {"baseline": [m, c, s], "with_act_cues": [ml, cl, sl], "H99_px": H99}
print(f"held-out littering crop recall: current rules {m:.1f} vs random {c:.1f}+-{s:.1f} (lift {m-c:+.1f}); "
      f"with act cues {ml:.1f} vs random {cl:.1f}+-{sl:.1f} (lift {ml-cl:+.1f})")
df.to_csv("heldout_results.csv"); json.dump(ctrl, open("heldout_control.json", "w"))
