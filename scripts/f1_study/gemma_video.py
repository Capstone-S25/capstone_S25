"""Does Gemma do better on denser video than on the selected frames?

Scores the same held-out windows under several frame-count / token-budget
conditions with the Yes/No answer margin (no generation, no test labels):

  selected@280   the views the pipeline already selects (4 or 8 frames)
  dense4@560, dense8@280, dense16@140, dense32@70   same visual-token budget
  dense16@280, dense32@140, dense32@280             2x and 4x the budget

Run with scripts/gemma4_gpu.sh's environment (.venv-gemma4).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anomaly_eval.gemma4.backends import TransformersBackend
from anomaly_eval.gemma4.protocol import Policy

OUT = ROOT / "results/f1_study_20261111/gemma_video"
CONDITIONS = [
    ("selected@280", None, 280),
    ("dense4@560", 4, 560),
    ("dense8@280", 8, 280),
    ("dense16@140", 16, 140),
    ("dense32@70", 32, 70),
    ("dense16@280", 16, 280),
    ("dense32@140", 32, 140),
    ("dense32@280", 32, 280),
]


def case(row, frames):
    if frames is None:
        views, times = row["selected"], row["selected_times"]
    else:
        idx = np.rint(np.linspace(0, len(row["dense"]) - 1, frames)).astype(int)
        views = [row["dense"][i] for i in idx]
        times = [row["dense_times"][i] for i in idx]
    return {"task": row["task"], "views": views, "timestamps": times}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--tag", default="e4b")
    parser.add_argument("--prompt", default="evidence")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in (OUT / "manifest.jsonl").read_text().splitlines()
    ]
    if args.limit:
        rows = rows[:: max(1, len(rows) // args.limit)]
    target = OUT / f"scores_{args.tag}.json"
    done = json.loads(target.read_text()) if target.exists() else {}
    hub = ROOT / "data/models/gemma4_hub"
    snapshots = hub / f"models--{args.model.replace('/', '--')}" / "snapshots"
    backend = TransformersBackend(str(next(snapshots.iterdir())))
    for name, frames, tokens in CONDITIONS:
        policy = Policy(name, args.prompt, tokens, "native", 64)
        scores = done.setdefault(name, {"scores": {}, "seconds": {}})
        started = time.perf_counter()
        for i, row in enumerate(rows):
            if row["id"] in scores["scores"]:
                continue
            t = time.perf_counter()
            scores["scores"][row["id"]] = backend.score([case(row, frames)], policy)[0][
                "score"
            ]
            scores["seconds"][row["id"]] = time.perf_counter() - t
            if i % 100 == 0:
                print(name, i, f"{time.perf_counter() - started:.0f}s", flush=True)
                target.write_text(json.dumps(done))
        target.write_text(json.dumps(done))
    backend.close()


if __name__ == "__main__":
    main()
