"""Compare YOLO26 n / s / m (detection + pose) with the same Stage 1 rules, on the tune and held-out halves.

Usage (from notebooks/):  ../.venv/bin/python tools/compare_models.py
Writes exp/model_comparison.csv. Needs bench_cache/models/{s,m}/*.pkl from tools/run_models.py.
"""
import pickle
import sys
from functools import partial
from pathlib import Path

NB = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(NB), str(NB / "exp")]

import pandas as pd  # noqa: E402

from exp_harness import baseline, load_splits  # noqa: E402
from random_control import lift  # noqa: E402
from stage1_bench import evaluate_run, summarize  # noqa: E402
from stage1_gate import GateConfig, calibrate_inview  # noqa: E402


def model_splits(size: str, heldout: bool) -> dict:
    if size == "n":
        return load_splits(heldout=heldout)
    cache = NB / "bench_cache" / "models" / size
    load = lambda name: pickle.loads((cache / f"{name}.pkl").read_bytes())
    want = 0 if heldout else 1
    num = lambda g: int(g.seq.seq_id.split("/")[-1])
    avenue, smoke = load("avenue"), load("smoke")
    days = sorted({g.seq.seq_id.split("#")[0] for g, _ in smoke})
    day = lambda g: days.index(g.seq.seq_id.split("#")[0]) % 2
    return {"avenue": [x for x in avenue if x[0].seq.seq_id.startswith("test/") and num(x[0]) % 2 == want],
            "normal": [x for x in avenue if x[0].seq.seq_id.startswith("train/") and num(x[0]) % 2 == want],
            "smoking": [x for x in smoke if day(x[0]) == (1 if heldout else 0)],
            "littering": load("litter_heldout" if heldout else "litter_train")}


def score(size: str) -> list[dict]:
    tune = model_splits(size, heldout=False)
    cfg = GateConfig(inview_s=calibrate_inview([r for _, r in tune["normal"]]))
    method = partial(baseline, cfg=cfg)
    rows = []
    for half, S in (("tune", tune), ("held-out", model_splits(size, heldout=True))):
        row = {"model": f"YOLO26{size}", "half": half}
        for ds, name in (("littering", "littering"), ("avenue", "loitering"), ("smoking", "smoking")):
            s = summarize([evaluate_run(g, r, method(r, ds)) for g, r in S[ds]], ds)
            row[f"{name} events %"] = s["event_recall_%"]
            row[f"{name} right person %"] = s["crop_recall_%"]
            if ds == "avenue":
                row["loitering video skipped %"] = s["review_reduction_%"]
        m, c, _ = lift(method, S["littering"], seeds=5)
        row["littering lift over random"] = round(m - c, 1)
        n = sum(len(method(r, "normal")) for _, r in S["normal"])
        row["false alarms / min"] = round(n / (sum(r.duration_s for _, r in S["normal"]) / 60), 1)
        rows.append(row)
        print(row, flush=True)
    return rows


if __name__ == "__main__":
    sizes = sys.argv[1:] or ["n", "s", "m"]
    df = pd.DataFrame([row for size in sizes for row in score(size)]).set_index(["half", "model"]).sort_index()
    df.to_csv(NB / "exp" / "model_comparison.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(df.T)
