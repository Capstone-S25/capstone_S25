"""Run YOLO26s + pose perception on the unseen-camera datasets (offline, MPS) and cache it.
Usage (from notebooks/): ../.venv/bin/python exp/newcam/run_perception.py [caviar aboda meva iwdd_test iwdd_train]
Writes bench_cache/models/s/<name>.pkl
"""
import pickle
import sys
import time
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]
from loaders import aboda_sequences, caviar_sequences, iwdd_sequences, meva_sequences  # noqa: E402
from stage1_bench import run_perception  # noqa: E402
from stage1_gate import GateConfig, Stage1Gate  # noqa: E402

OUT = HERE.parents[1] / "bench_cache" / "models" / "s"
JOBS = {"caviar": caviar_sequences, "aboda": aboda_sequences, "meva": meva_sequences,
        "iwdd_test": partial(iwdd_sequences, "test"), "iwdd_train": partial(iwdd_sequences, "train")}
cfg = GateConfig(det_weights="yolo26s.pt", pose_weights="yolo26s-pose.pt", device="mps")
for name in sys.argv[1:] or list(JOBS):
    if (OUT / f"{name}.pkl").exists():
        continue
    t0 = time.time()
    res = []
    for g in JOBS[name]():
        res += run_perception(Stage1Gate(cfg), [g], verbose=False)
        print(f"{name} {g.seq.seq_id} done ({time.time() - t0:.0f}s)", flush=True)
    pickle.dump(res, open(OUT / f"{name}.pkl", "wb"))
    print(f"{name} saved ({len(res)} sequences, {time.time() - t0:.0f}s)", flush=True)
