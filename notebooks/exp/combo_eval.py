import sys, importlib, json
sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks"); sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks/exp")
import pandas as pd
from exp_harness import score, combine, baseline, dedup, per_action
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
M = {n: importlib.import_module(n) for n in ["littering_act", "pose_anomaly", "motion_anomaly", "smoking_dwell", "crop_policy"]}
def op(mod, name=None, **kw):
    params = dict(M[mod].DEFAULTS)
    if name:
        params.update(M[mod].OPERATING_POINTS[name] if hasattr(M[mod], "OPERATING_POINTS") and name in M[mod].OPERATING_POINTS else {})
    params.update(kw)
    return M[mod].make_method(**params)
for mod in M:
    print(mod, "OPERATING_POINTS:", list(getattr(M[mod], "OPERATING_POINTS", {}).keys()))

LA_A = op("littering_act", extend_h=0.18)
LA_C = op("littering_act")
MOT = op("motion_anomaly")
CROP = op("crop_policy", trigger_wrist=None)
SD_A = op("smoking_dwell", "A_lean")
SD_B = op("smoking_dwell", "B_mid")
PA_A = op("pose_anomaly", "A")
combos = {
    "base": [baseline],
    "base+act(A)": [baseline, LA_A],
    "base+motion": [baseline, MOT],
    "base+crop": [baseline, CROP],
    "base+act(A)+motion+crop": [baseline, LA_A, MOT, CROP],
    "base+act(C)+motion+crop": [baseline, LA_C, MOT, CROP],
    "...(A)+dwell(A_lean)": [baseline, LA_A, MOT, CROP, SD_A],
    "...(A)+dwell(B_mid)": [baseline, LA_A, MOT, CROP, SD_B],
    "...(A)+poseanom(A)": [baseline, LA_A, MOT, CROP, PA_A],
    "...(A)+poseanom(A)+dwell(A_lean)": [baseline, LA_A, MOT, CROP, PA_A, SD_A],
}
rows = []
for name, ms in combos.items():
    rows.append(score(dedup(combine(*ms)), name=name)); print(rows[-1], flush=True)
df = pd.DataFrame(rows).set_index("method")
df.to_csv("combo_tune.csv"); print(df)
