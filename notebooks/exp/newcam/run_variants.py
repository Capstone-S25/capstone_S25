"""Perception variants for the unseen cameras: bigger input size and/or a lower confirmation threshold.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/run_variants.py
Writes bench_cache/models/s/<dataset>__<variant>.pkl
"""
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]
from loaders import caviar_sequences, meva_sequences  # noqa: E402
from stage1_bench import run_perception  # noqa: E402
from stage1_gate import GateConfig, Stage1Gate  # noqa: E402

OUT = HERE.parents[1] / "bench_cache" / "models" / "s"
BASE = dict(det_weights="yolo26s.pt", pose_weights="yolo26s-pose.pt", device="mps")
VARIANTS = {"conf25": dict(person_conf=0.25), "img1280": dict(imgsz=1280), "img1280_conf25": dict(imgsz=1280, person_conf=0.25)}
JOBS = [("caviar", caviar_sequences, v) for v in ("img1280", "conf25", "img1280_conf25")] + \
       [("meva", meva_sequences, v) for v in ("conf25", "img1280", "img1280_conf25")]
for name, build, var in JOBS:
    path = OUT / f"{name}__{var}.pkl"
    if path.exists():
        continue
    t0, res = time.time(), []
    cfg = GateConfig(**BASE, **VARIANTS[var])
    for g in build():
        res += run_perception(Stage1Gate(cfg), [g], verbose=False)
        print(f"{name}__{var} {g.seq.seq_id} ({time.time() - t0:.0f}s)", flush=True)
    path.write_bytes(pickle.dumps(res))
    print(f"{name}__{var} saved ({time.time() - t0:.0f}s)", flush=True)
