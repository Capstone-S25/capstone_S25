"""Rate-matched random control for littering clips: same number of candidates per clip at the same times,
each on a random person visible in that frame. Lift = method crop recall - control crop recall."""
import sys, random
sys.path.insert(0, "/Users/ianvarella/capstone_S25/notebooks"); sys.path.insert(0, ".")
import numpy as np
from exp_harness import SPLITS, baseline, make_event
from stage1_bench import evaluate_run, summarize


def control(method, seed):
    rng = random.Random(seed)
    def run(r, ds):
        by_t = {round(o.t, 2): o for o in r.observations}
        out = []
        for e in method(r, ds):
            o = by_t.get(round(e["timestamp_s"], 2)) or min(r.observations, key=lambda o: abs(o.t - e["timestamp_s"]))
            if o.persons:
                out.append(make_event("control", o.t, rng.choice(o.persons)["box"]))
        return out
    return run


def crop(method, data):
    return summarize([evaluate_run(g, r, method(r, "littering")) for g, r in data], "x")["crop_recall_%"]


def lift(method, data, seeds=10):
    m = crop(method, data)
    c = [crop(control(method, s), data) for s in range(seeds)]
    return m, float(np.mean(c)), float(np.std(c))


if __name__ == "__main__":
    data = SPLITS["littering"]
    multi = [(g, r) for g, r in data if max(len(o.persons) for o in r.observations) >= 2]
    print(f"tune clips {len(data)}, with >= 2 people {len(multi)}")
    for name, sub in [("all clips", data), ("clips with >= 2 people", multi)]:
        m, c, s = lift(baseline, sub)
        print(f"baseline rules on {name}: crop recall {m:.1f} vs random control {c:.1f} +- {s:.1f} -> lift {m - c:+.1f}")
