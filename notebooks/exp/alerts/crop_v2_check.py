"""Does the before/now/after crop classifier (exp/icvl/crop_v2.py, "all" features, trained on ICVL odd only) hold up
on the tune / train data sweep.py scores (littering clips, Avenue, smoking clips, IWDD train)?

Every reach_down alert (deck settings, dedup off) is cropped at t - 1 s, t, t + 1 s, embedded with MobileCLIP2-S2 and
scored; sweep.py metrics are then computed with alerts below the cut-off dropped before the person dedup, next to the
current crop check (crop_check=True) and no check.
Usage (from notebooks/):
  ../.venv/bin/python -I exp/alerts/crop_v2_check.py embed ENCODER [RULE]
      -> bench_cache/models/s/crop_scores_<name>.pkl (RULE pose_action: crop_scores_pose_action_<name>.pkl)
  ../.venv/bin/python exp/alerts/crop_v2_check.py sweep REACH/POSE ...  (e.g. 0.2/none 0.2/0.2)
      -> exp/alerts/sweep_crop_v2.json
"""
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE), str(NB / "exp" / "icvl")]

import sweep as S  # noqa: E402
from cached import CACHE, load_cached, splits  # noqa: E402
from crop_v2 import OUT as ICVL_ROWS, boxes_around, feats  # noqa: E402
from hand_probe import dedup  # noqa: E402
from score import load_runs  # noqa: E402
from stage1_crop_check import crop  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, replay_rules  # noqa: E402

CFG = GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",), pose_action_cooldown_s=15.0, person_dedup_s=0.0)
NAMES = ("littering", "avenue", "smoking", "iwdd_train")
RULE = "reach_down"   # embed: "pose_action" scores the pose-cue alerts with a classifier trained on ICVL odd pose alerts


def classifier(rule="reach_down"):
    from sklearn.linear_model import LogisticRegression
    src = ICVL_ROWS if rule == "reach_down" else ICVL_ROWS.with_name(f"crop_v2_{rule}_icvl.pkl")
    rows = [r for r in pickle.loads(src.read_bytes()) if r["now"] is not None]
    odd_ids = {r.seq_id for _, r in load_runs("odd")}
    tr = [r for r in rows if r["seq"] in odd_ids and r["y"] is not None]
    return LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(
        np.stack([feats(r, "all") for r in tr]), [r["y"] for r in tr])


def embed(encoder, rule="reach_down"):
    import open_clip
    import torch
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, prep = open_clip.create_model_and_transforms(encoder)
    model = model.eval().to(dev)
    clf = classifier(rule)
    S_ = splits(heldout=False)
    groups = {"littering": S_["littering"], "avenue": S_["avenue"] + S_["normal"], "smoking": S_["smoking"],
              "iwdd_train": load_cached("iwdd_train")}
    for name in NAMES:
        path = CACHE / (f"crop_scores_{name}.pkl" if rule == "reach_down" else f"crop_scores_{rule}_{name}.pkl")
        if path.exists():
            continue
        scores = {}
        for g, res in groups[name]:
            want = {}
            for a in replay_rules(res, CFG)[0]:
                if a["event_type"] == rule:
                    bx = boxes_around(res, a)
                    if bx is None:
                        continue
                    for part, v in bx.items():
                        if v is not None:
                            want.setdefault(v[0], []).append((a["event_id"], part, v[1]))
                    scores[a["event_id"]] = {"before": None, "now": None, "after": None}
            if not want:
                continue
            for fr in g.seq:
                for eid, part, box in want.pop(fr.index, []):
                    c = crop(fr.image, box)
                    if c.size:
                        with torch.no_grad():
                            f = model.encode_image(prep(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))[None].to(dev))
                        scores[eid][part] = (f / f.norm()).cpu().numpy()[0].astype(np.float32)
                if not want:
                    break
            for eid in [e for e in scores if isinstance(scores[e], dict)]:
                emb = scores.pop(eid)
                if emb["now"] is not None:
                    scores[(res.seq_id, eid)] = float(clf.predict_proba(feats(emb, "all")[None])[0, 1])
        path.write_bytes(pickle.dumps(scores))
        print(name, len(scores), rule, "alerts scored", flush=True)


def sweep(pairs):
    """pairs: (reach_down cut-off, pose_action cut-off or None); both applied before the person dedup."""
    load = lambda f: pickle.loads((CACHE / f).read_bytes())
    reach, pose = {}, {}
    for name in NAMES:
        reach.update(load(f"crop_scores_{name}.pkl"))
        if (CACHE / f"crop_scores_pose_action_{name}.pkl").exists():
            pose.update(load(f"crop_scores_pose_action_{name}.pkl"))
    data = S.data()
    rows = [S.score({"pose_action_cooldown_s": 15.0}, *data[:4])]
    rows[0]["setting"] = "none"
    print(json.dumps(rows[0]), flush=True)
    for tr, tp in pairs:
        def filt(r, al, tr=tr, tp=tp):
            def keep(a):
                if a["event_type"] == "reach_down":
                    return reach.get((r.seq_id, a["event_id"]), 1.0) >= tr
                if a["event_type"] == "pose_action" and tp is not None:
                    return pose.get((r.seq_id, a["event_id"]), 1.0) >= tp
                return True
            return dedup([a for a in al if keep(a)])
        orig = S.partial
        S.partial = lambda fn, cfg, _f=filt: (lambda r, ds: _f(r, fn(r, ds, cfg=cfg)))
        try:
            row = S.score({"pose_action_cooldown_s": 15.0, "person_dedup_s": 0.0}, *data[:4])
        finally:
            S.partial = orig
        row["setting"] = f"reach {tr} / pose {tp}"
        rows.append(row)
        print(json.dumps(row), flush=True)
    (HERE / "sweep_crop_v2.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed(sys.argv[2], *sys.argv[3:4])
    else:
        sweep([(float(a), None if b == "none" else float(b)) for a, b in (x.split("/") for x in sys.argv[2:])])
