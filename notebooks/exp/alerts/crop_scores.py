"""Precomputed crop-check scores for the tune / train data used by sweep.py, so crop_check can be replayed on cached
runs (which have no frames). Every reach_down alert of the replayed gate (dedup off: the crop check runs before it) is
cropped from its frame, embedded with the crop check's encoder and scored with weights/crop_check.npz.
Usage (from notebooks/): ../.venv/bin/python -I exp/alerts/crop_scores.py ENCODER
  ENCODER: open_clip id or "local-dir:<MobileCLIP2-S2 snapshot folder>"
  -> bench_cache/models/s/crop_scores_{littering,avenue,smoking,iwdd_train}.pkl
"""
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE)]

from cached import CACHE, load_cached, splits  # noqa: E402
from stage1_crop_check import CropCheck  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig, replay_rules  # noqa: E402

CFG = GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",), pose_action_cooldown_s=15.0, person_dedup_s=0.0)


def score_runs(runs, check, rules=("reach_down",)):
    out = {}
    for g, res in runs:
        want = {}
        for a in replay_rules(res, CFG)[0]:
            if a["event_type"] in rules:
                want.setdefault(a["frame_index"], []).append(a)
        if not want:
            continue
        for fr in g.seq:
            for a in want.pop(fr.index, []):
                try:
                    out[(res.seq_id, a["event_id"])] = check.score(fr.image, a["roi_box"], a["event_type"])
                except ValueError:
                    pass
            if not want:
                break
    return out


def main():
    check = CropCheck(encoder=sys.argv[1])
    S = splits(heldout=False)
    for name, runs in (("littering", S["littering"]), ("avenue", S["avenue"] + S["normal"]), ("smoking", S["smoking"]),
                       ("iwdd_train", load_cached("iwdd_train"))):
        path = CACHE / f"crop_scores_{name}.pkl"
        if path.exists():
            continue
        scores = score_runs(runs, check)
        path.write_bytes(pickle.dumps(scores))
        print(name, len(scores), "alerts scored", flush=True)


if __name__ == "__main__":
    main()
