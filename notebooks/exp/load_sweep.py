import sys
sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks"); sys.path.insert(0, ".")
exec(open("combo_eval.py").read().split("combos = {")[0])
rows = []
def shorten(method, pad):
    def run(r, ds):
        out = []
        for e in method(r, ds):
            t = e["timestamp_s"]; e = dict(e); e["clip_window_s"] = [max(0.0, t - pad), t + pad]; out.append(e)
        return out
    return run
for rep in (3.0, 10.0, 30.0):
    MOTr = op("motion_anomaly", repeat_s=rep)
    for pad in (2.5, 1.5):
        for name, ms in {"act+motion+crop": [baseline, LA_A, MOTr, CROP],
                         "act+motion+crop+dwell": [baseline, LA_A, MOTr, CROP, SD_A]}.items():
            rows.append(score(shorten(dedup(combine(*ms)), pad), name=f"{name} rep={rep} pad={pad}")); print(rows[-1], flush=True)
pd.DataFrame(rows).set_index("method").to_csv("load_sweep.csv")
print(pd.DataFrame(rows).set_index("method")[["littering_crop_%","avenue_event_%","avenue_crop_%","avenue_reduction_%","smoking_event_%","smoking_crop_%","smoking_reduction_%","normal_FA_per_min"]])
