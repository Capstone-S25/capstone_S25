"""Labelled before / now / after crop embeddings for the reach_down and pose_action alerts on the tune / train data
(littering train clips, smoking tune clips, IWDD train, Avenue tune), so the crop check can be trained on more than
ICVL's cameras (exp/icvl/crop_mixed.py). Same crops and encoder as exp/icvl/crop_v2.py; labels as the pose verifier
(exp/icvl/pose_verifier.labeller): on a labelled littering / smoker box or the IWDD dumper = 1, other people = 0,
Avenue normal = 0.
Usage (from notebooks/): ../.venv/bin/python -I exp/alerts/crop_rows.py ENCODER
  -> bench_cache/models/s/crop_rows_tune.pkl  [{source, seq, event_id, rule, y, before, now, after}]
"""
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE), str(NB / "exp" / "icvl"), str(NB / "exp" / "actions"),
                str(NB / "exp" / "newcam"), str(NB / "exp" / "gt_review")]

from cached import CACHE, load_cached, splits  # noqa: E402
from crop_v2 import boxes_around  # noqa: E402
from loaders import iwdd_labels  # noqa: E402
from pose_verifier import labeller  # noqa: E402
from reach_verifier_mixed import rid_at  # noqa: E402
from stage1_crop_check import crop  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, replay_rules  # noqa: E402

CFG = GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",), pose_action_cooldown_s=15.0, person_dedup_s=0.0)
OUT = CACHE / "crop_rows_tune.pkl"
RULES = ("reach_down", "pose_action")


def main(encoder):
    import open_clip
    import torch
    from PIL import Image
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, prep = open_clip.create_model_and_transforms(encoder)
    model = model.eval().to(dev)
    T = splits(heldout=False)
    labels = iwdd_labels("train")
    groups = [("litter", T["littering"], None), ("smoking", T["smoking"], None), ("iwdd", load_cached("iwdd_train"), labels),
              ("avenue", T["avenue"] + T["normal"], None)]
    rows = []
    for source, runs, extra in groups:
        for g, res in runs:
            lab = labeller(source if source != "avenue" else "avenue", g, res, extra)
            want, pending = {}, {}
            for a in replay_rules(res, CFG)[0]:
                if a["event_type"] not in RULES:
                    continue
                y = lab(a["timestamp_s"], a["roi_box"], rid_at(res, a))[0]
                if source == "avenue" and res.seq_id.startswith("test/"):
                    y = None   # Avenue test videos have throws but no person-level labels here: score only
                bx = boxes_around(res, a)
                if bx is None:
                    continue
                pending[a["event_id"]] = {"source": source, "seq": res.seq_id, "event_id": a["event_id"],
                                          "rule": a["event_type"], "y": y, "before": None, "now": None, "after": None}
                for part, v in bx.items():
                    if v is not None:
                        want.setdefault(v[0], []).append((a["event_id"], part, v[1]))
            if not want:
                continue
            for fr in g.seq:
                for eid, part, box in want.pop(fr.index, []):
                    c = crop(fr.image, box)
                    if c.size:
                        with torch.no_grad():
                            f = model.encode_image(prep(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))[None].to(dev))
                        pending[eid][part] = (f / f.norm()).cpu().numpy()[0].astype(np.float32)
                if not want:
                    break
            rows += [r for r in pending.values() if r["now"] is not None]
        print(source, len(rows), "rows so far", flush=True)
    OUT.write_bytes(pickle.dumps(rows))


if __name__ == "__main__":
    main(sys.argv[1])
