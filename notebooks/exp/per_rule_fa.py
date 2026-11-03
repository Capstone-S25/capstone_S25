"""False alarms per rule on held-out normal footage (current GateConfig defaults, rules + pose cue).

Avenue held-out normal videos (even training videos: a campus walkway) and IWDD test normal videos (streets and dump
sites, no dumping). Candidates per minute of video, by rule, and the share of videos with at least one.
Usage (from notebooks/): ../.venv/bin/python exp/per_rule_fa.py   -> exp/per_rule_fa.json
"""
import json
import pickle
import sys
from collections import Counter
from functools import partial
from pathlib import Path

NB = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(NB / "exp" / "newcam"), str(NB / "exp" / "alerts"),
                str(NB / "exp" / "gt_review")]

from compare_models import model_splits  # noqa: E402
from cached import load_cached, splits  # noqa: E402
from exp_harness import baseline  # noqa: E402
from loaders import iwdd_labels  # noqa: E402
from stage1_gate import DEFAULT_RULES, GateConfig  # noqa: E402


def per_rule(videos, m, ds):
    mins = sum(r.duration_s for r in videos) / 60
    c, any_v = Counter(), Counter()
    for r in videos:
        evs = m(r, ds)
        c.update(e["event_type"] for e in evs)
        any_v.update({e["event_type"] for e in evs})
    return {"minutes": round(mins, 1), "videos": len(videos), "total_per_min": round(sum(c.values()) / mins, 2),
            "per_min": {k: round(v / mins, 2) for k, v in c.most_common()},
            "videos_with_any_%": {k: round(100 * v / len(videos)) for k, v in any_v.most_common()}}


def main():
    m = partial(baseline, cfg=GateConfig(inview_s=60.0, rules=DEFAULT_RULES + ("pose_action",)))
    avenue = [r for _, r in splits(heldout=True)["normal"]]
    lab = iwdd_labels("test")
    iwdd = [r for _, r in load_cached("iwdd_test")
            if not lab[r.seq_id]["positive"]]
    out = {"avenue held-out normal": per_rule(avenue, m, "normal"), "IWDD test normal": per_rule(iwdd, m, "littering")}
    print(json.dumps(out, indent=1))
    (NB / "exp" / "per_rule_fa.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
