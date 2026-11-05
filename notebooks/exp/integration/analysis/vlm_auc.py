import json, sys, glob
sys.path[:0] = ["exp/integration", "exp/icvl", "."]
from sklearn.metrics import roc_auc_score
from vlm_calib import load
S = sys.argv[1]
rows, *_ = load(S, "even")
cand = {}
for f in glob.glob(f"{S}/even/candidates_*.jsonl"):
    for l in open(f):
        c = json.loads(l); cand[c["event_id"]] = c
for task, ev, rules in [("littering", "littering", ("reach_down", "pose_action")), ("smoking", "smoking", ("reach_down", "pose_action", "sustained_presence", "long_presence")),
                        ("loitering", "lingering 20s+", ("sustained_presence", "long_presence")), ("unattended_items", "littering", ("reach_down", "pose_action"))]:
    rs = [r for r in rows if r["event_type"] in rules and r["scores"].get(task) is not None]
    y = [ev in r["real"] for r in rs]
    out = {"n": len(rs), "real": sum(y), "VLM AUC": round(roc_auc_score(y, [r["scores"][task] for r in rs]), 3)}
    cs = [(cand[r["event_id"]].get("crop_score"), yy) for r, yy in zip(rs, y) if cand[r["event_id"]].get("crop_score") is not None]
    if cs and len({b for _, b in cs}) == 2:
        out["gate crop_score AUC (same rows)"] = round(roc_auc_score([b for _, b in cs], [a for a, _ in cs]), 3)
    print(task, "vs", ev, out)
