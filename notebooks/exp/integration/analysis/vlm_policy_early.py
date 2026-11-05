"""End-to-end policies on ICVL even, every threshold from odd. Put-down alerts: fused (crop + VLM littering + verifier)
logistic, threshold keeping share R of odd real-littering candidates. Stillness alerts: VLM loitering at odd quantile,
or passed straight through. Smoking: VLM can't tell (AUC 0.45), so 'smoking' follows the stillness decision."""
import json, sys, glob
import numpy as np
sys.path[:0] = ["exp/integration", "exp/icvl", "."]
from sklearn.linear_model import LogisticRegression
from vlm_calib import load, evaluate
S = sys.argv[1]
ALL = {"littering", "smoking", "loitering"}
PUT, STILL = ("reach_down", "pose_action"), ("sustained_presence", "long_presence")
def prep(half):
    rows, inc, neg, hrs = load(S, half)
    cand = {json.loads(l)["event_id"]: json.loads(l) for f in glob.glob(f"{S}/{half}/candidates_*.jsonl") for l in open(f)}
    for r in rows:
        c = cand[r["event_id"]]
        r["x"] = [c.get("crop_score") if c.get("crop_score") is not None else 1.0, r["scores"].get("littering") or 0.0, c.get("verifier_score") or 0.0]
    return rows, inc, neg, hrs
odd, even = prep("odd"), prep("even")
tr = [r for r in odd[0] if r["event_type"] in PUT]
m = LogisticRegression().fit([r["x"] for r in tr], ["littering" in r["real"] for r in tr])
for d in (odd, even):
    for r in d[0]:
        r["p"] = m.predict_proba([r["x"]])[0, 1] if r["event_type"] in PUT else None
pos = sorted(r["p"] for r in tr if "littering" in r["real"])
lo = sorted(r["scores"]["loitering"] for r in odd[0] if r["event_type"] in STILL and r["scores"].get("loitering") is not None and not r["real"])
print("gate only", json.dumps(evaluate(*even, lambda r: {"littering", "smoking", "loitering"})))
for R in (0.98, 0.95, 0.9, 0.8):
    tp = pos[int((1 - R) * len(pos))]
    for still in ("pass", 0.5, 0.3):
        tl = -np.inf if still == "pass" else np.quantile(lo, 1 - still)
        def yes(r, tp=tp, tl=tl):
            if r["event_type"] in PUT:
                return ALL if r["p"] >= tp else set()
            s = r["scores"].get("loitering")
            return ALL if (s is None and tl == -np.inf) or (s is not None and s >= tl) else set()
        print(f"keep {R} of litter, stillness {still}", json.dumps(evaluate(*even, yes)))
