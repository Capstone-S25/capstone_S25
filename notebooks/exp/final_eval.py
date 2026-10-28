"""Final configuration numbers on both halves (thresholds fixed beforehand; calibration uses tune normal footage only)."""
import sys
from functools import partial
sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks")
import pandas as pd
from exp_harness import load_splits, baseline, dedup
from stage1_bench import evaluate_run, summarize
from stage1_gate import DEFAULT_RULES, GateConfig, calibrate_inview, calibrate_near_camera

TUNE, TEST = load_splits(heldout=False), load_splits(heldout=True)
normal_fit = [r for _, r in TUNE["normal"]]
LOITER = dict(near_camera_px=calibrate_near_camera(normal_fit), inview_s=calibrate_inview(normal_fit))
print("loitering camera calibration:", {k: round(v, 1) for k, v in LOITER.items()})
OLD = tuple(r for r in DEFAULT_RULES if r not in ("near_camera", "in_view"))
configs = {
    "old rules": (GateConfig(rules=OLD), GateConfig(rules=OLD)),
    "+ near_camera": (GateConfig(rules=OLD + ("near_camera",), **LOITER), GateConfig(rules=OLD)),
    "+ near_camera + in_view": (GateConfig(**LOITER), GateConfig()),
}
rows = []
for half, S in (("tune", TUNE), ("held-out", TEST)):
    for name, (loiter_cfg, other_cfg) in configs.items():
        row = {"half": half, "config": name}
        for ds, cfg in (("littering", other_cfg), ("avenue", loiter_cfg), ("smoking", other_cfg)):
            s = summarize([evaluate_run(g, r, baseline(r, ds, cfg)) for g, r in S[ds]], ds)
            key = "loitering" if ds == "avenue" else ds
            row[f"{key} events %"] = s["event_recall_%"]
            row[f"{key} right person %"] = s["crop_recall_%"]
            if ds != "littering":
                row[f"{key} video skipped %"] = s["review_reduction_%"]
        n = sum(len(baseline(r, "normal", loiter_cfg)) for _, r in S["normal"])
        row["false alarms / min"] = round(n / (sum(r.duration_s for _, r in S["normal"]) / 60), 1)
        rows.append(row)
df = pd.DataFrame(rows).set_index(["half", "config"])
df.to_csv("exp/final_results.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df.T)
