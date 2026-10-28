"""CPU latency of Stage 1 per model size (run with nothing else using the CPU).

Usage (from notebooks/):  ../.venv/bin/python tools/latency.py bench_cache/latency.json n s m
"""
import json
import sys
from pathlib import Path

import numpy as np

NB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NB))
from stage1_gate import GateConfig, Stage1Gate, load_sequence  # noqa: E402

D = NB.parent / "data"
SOURCES = [
    ("Loitering video (640x360)", D / "loitering" / "testing_videos" / "01.avi", 200),
    ("Littering clip (1920x1080)", D / "littering" / "test" / "dumping_v01_5", 30),
    ("Smoking CCTV (640x480)", D / "smoking" / "valid", 60),
]
out_path, sizes = sys.argv[1], sys.argv[2:] or ["n"]
rows = []
for size in sizes:
    cfg = GateConfig(det_weights=f"yolo26{size}.pt", pose_weights=f"yolo26{size}-pose.pt", device="cpu")
    gate = Stage1Gate(cfg)
    for name, src, n in SOURCES:
        gate.run(load_sequence(src, fps=10.0, max_frames=5))
        res = gate.run(load_sequence(src, fps=10.0, max_frames=n))
        lat = [o.latency_ms for o in res.observations]
        mean = lambda key: round(float(np.mean([x[key] for x in lat if key in x])), 1) if any(key in x for x in lat) else None
        rows.append({"model": f"YOLO26{size}", "source": name, "frames": len(lat),
                     "detect + track ms": mean("det_ms"), "pose ms": mean("pose_ms"),
                     "total ms / frame": mean("total_ms"),
                     "fps": round(1000.0 / float(np.mean([x["total_ms"] for x in lat])), 1)})
        print(rows[-1], flush=True)
json.dump(rows, open(out_path, "w"), indent=1)
