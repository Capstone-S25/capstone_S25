import json, sys, glob
import numpy as np
sys.path[:0] = ["exp/integration", "exp/icvl", "."]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from vlm_calib import load
S = sys.argv[1]
def data(half):
    rows, *_ = load(S, half)
    cand = {json.loads(l)["event_id"]: json.loads(l) for f in glob.glob(f"{S}/{half}/candidates_*.jsonl") for l in open(f)}
    X, y = [], []
    for r in rows:
        c = cand[r["event_id"]]
        if r["event_type"] not in ("reach_down", "pose_action") or r["scores"].get("littering") is None or c.get("crop_score") is None: continue
        X.append([c["crop_score"], r["scores"]["littering"], c.get("verifier_score") or 0]); y.append("littering" in r["real"])
    return np.array(X), np.array(y)
Xo, yo = data("odd"); Xe, ye = data("even")
for cols, name in [([0], "crop only"), ([1], "VLM only"), ([0, 1], "crop + VLM"), ([0, 1, 2], "crop + VLM + verifier")]:
    m = LogisticRegression().fit(Xo[:, cols], yo)
    print(name, "even AUC", round(roc_auc_score(ye, m.predict_proba(Xe[:, cols])[:, 1]), 3), "n odd/even", len(yo), len(ye))
