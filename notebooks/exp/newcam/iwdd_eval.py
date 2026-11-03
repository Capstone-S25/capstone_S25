"""IWDD (MIVIA-IWDD-500 illegal waste dumping): does the Stage 1 gate send a candidate while the dumping happens?

Labels give one onset second per positive video and no boxes, so this is a temporal test only (crop checks are done
visually, see iwdd_crops.py). Per method:
  hit@W        positive videos with a candidate in [onset - 1 s, onset + W]   (W = 5 s, 10 s, end of video)
  chance@W     same number of candidates per positive video at uniform random times, averaged over SEEDS
  neg_any      negative videos with any candidate (each would be one VLM query at least)
  cand/min     candidates per minute on negative videos
Breakdowns by disposal kind (static/dynamic), night, and by rule type (which rules fired inside the 10 s window).
Usage (from notebooks/): ../.venv/bin/python exp/newcam/iwdd_eval.py test|train
"""
import json
import pickle
import random
import sys
import zlib
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE)]

from loaders import iwdd_labels  # noqa: E402

SEEDS = 20
WINDOWS = {"5s": 5.0, "10s": 10.0, "end": float("inf")}
PRE_S = 1.0


def load(split: str) -> list:
    return pickle.loads((NB / "bench_cache" / "models" / "s" / f"iwdd_{split}.pkl").read_bytes())


def in_window(t: float, onset: float, w: float) -> bool:
    return onset - PRE_S <= t <= onset + w


def evaluate(data: list, labels: dict, method, ds: str = "littering") -> dict:
    """data: [(g, r)] from load(); method(r, ds) -> events."""
    hits = {k: 0 for k in WINDOWS}
    chance = {k: 0.0 for k in WINDOWS}
    by_kind = defaultdict(lambda: [0, 0])
    rule_hits, rule_neg = Counter(), Counter()
    n_pos = n_neg = neg_any = 0
    neg_cands = 0
    neg_minutes = 0.0
    for g, r in data:
        lab = labels[r.seq_id]
        ts = [(e["timestamp_s"], e["event_type"]) for e in method(r, ds)]
        if not lab["positive"]:
            n_neg += 1
            neg_any += bool(ts)
            neg_cands += len(ts)
            neg_minutes += r.duration_s / 60
            rule_neg.update({k for _, k in ts})
            continue
        n_pos += 1
        onset = lab["onset_s"]
        for k, w in WINDOWS.items():
            hits[k] += any(in_window(t, onset, w) for t, _ in ts)
            rng = random.Random(zlib.crc32(f"{r.seq_id}#{k}".encode()))
            for _ in range(SEEDS):
                chance[k] += any(in_window(rng.uniform(0, r.duration_s), onset, w) for _ in ts) / SEEDS
        hit10 = any(in_window(t, onset, WINDOWS["10s"]) for t, _ in ts)
        for key in (lab["kind"], "night" if lab["night"] else "day"):
            by_kind[key][0] += hit10
            by_kind[key][1] += 1
        rule_hits.update({k for t, k in ts if in_window(t, onset, WINDOWS["10s"])})
    pct = lambda x, n: round(100 * x / max(1, n), 1)  # noqa: E731
    row = {"positives": n_pos, "negatives": n_neg}
    for k in WINDOWS:
        row[f"hit@{k}_%"] = pct(hits[k], n_pos)
        row[f"chance@{k}_%"] = pct(chance[k], n_pos)
    row["neg_any_%"] = pct(neg_any, n_neg)
    row["neg_cand_per_min"] = round(neg_cands / max(neg_minutes, 1e-9), 2)
    row["hit@10s_by"] = {k: f"{v[0]}/{v[1]} ({pct(*v)}%)" for k, v in sorted(by_kind.items())}
    row["rules_in_10s_window_%"] = {k: pct(v, n_pos) for k, v in rule_hits.most_common()}
    row["rules_on_negatives_%"] = {k: pct(v, n_neg) for k, v in rule_neg.most_common()}
    return row


def methods(inview: float) -> dict:
    from exp_harness import baseline
    from stage1_gate import DEFAULT_RULES, GateConfig
    return {"rules": partial(baseline, cfg=GateConfig(inview_s=inview)),
            "rules+pose": partial(baseline, cfg=GateConfig(inview_s=inview, rules=DEFAULT_RULES + ("pose_action",)))}


def inview_from_avenue() -> float:
    from compare_models import model_splits
    from stage1_gate import calibrate_inview
    return calibrate_inview([r for _, r in model_splits("s", heldout=False)["normal"]])


def main(split: str):
    data, labels = load(split), iwdd_labels(split)
    out = {}
    for name, m in methods(inview_from_avenue()).items():
        out[name] = evaluate(data, labels, m)
        print(name, json.dumps(out[name]), flush=True)
    (HERE / f"iwdd_results_{split}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
