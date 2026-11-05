"""pose_every_n=2 on cached runs: blank keypoints on skipped frames, re-score with the one-person rule."""
import copy, json, re, sys
sys.path[:0] = ["exp/icvl", "."]
import stage1_crop_check
from fpr_attrib import CFG, rescore
from score import load_runs

class Near(dict):
    def get(self, key, default=None):
        if key in self: return self[key]
        seq, eid = key
        m = re.match(r"(.*_f)(\d+)$", eid)
        if not m: return default
        for d in (1, -1, 2, -2, 3, -3):
            k2 = (seq, f"{m.group(1)}{int(m.group(2)) + d}")
            if k2 in self:
                self.miss_hit += 1; return self[k2]
        self.miss += 1
        return default
near = Near(stage1_crop_check.SCORES); near.miss = near.miss_hit = 0
stage1_crop_check.SCORES = near
import stage1_gate
runs = load_runs("even")
print("n=1", json.dumps(rescore(runs, CFG)), flush=True)
near.miss = near.miss_hit = 0
for g, r in runs:
    for o in r.observations:
        if o.index % 2:   # gate skips pose on frames where n % 2 != 0
            for p in o.persons: p["kpts"] = None
print("n=2", json.dumps(rescore(runs, CFG)), "nearest-score lookups", near.miss_hit, "unscored", near.miss, flush=True)
