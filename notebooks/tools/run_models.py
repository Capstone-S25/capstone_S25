"""Run perception for bigger YOLO models on every dataset (offline, on the GPU) and cache the results.

Usage (from notebooks/):  ../.venv/bin/python tools/run_models.py s m
Writes bench_cache/models/<size>/{litter_train,litter_heldout,avenue,smoke}.pkl
"""
import pickle
import sys
from pathlib import Path

NB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NB))
from stage1_bench import avenue_sequences, littering_sequences, run_perception, smoking_sequences  # noqa: E402
from stage1_gate import GateConfig, Stage1Gate  # noqa: E402

D = NB.parent / "data"
for size in sys.argv[1:]:
    out = NB / "bench_cache" / "models" / size
    out.mkdir(parents=True, exist_ok=True)
    cfg = GateConfig(det_weights=f"yolo26{size}.pt", pose_weights=f"yolo26{size}-pose.pt", device="mps")
    jobs = {
        "litter_train": lambda: littering_sequences(D / "littering", splits=("train",)),
        "litter_heldout": lambda: littering_sequences(D / "littering", splits=("valid", "test")),
        "avenue": lambda: avenue_sequences(D / "loitering", D / "loitering" / "avenue_gt" / "ground_truth_demo" / "testing_label_mask", stride=2),
        "smoke": lambda: smoking_sequences(D / "smoking", splits=("train", "valid")),
    }
    for name, build in jobs.items():
        if (out / f"{name}.pkl").exists():
            continue
        res = run_perception(Stage1Gate(cfg), build(), verbose=False)
        pickle.dump(res, open(out / f"{name}.pkl", "wb"))
        print(f"{size} {name} done ({len(res)} sequences)", flush=True)
