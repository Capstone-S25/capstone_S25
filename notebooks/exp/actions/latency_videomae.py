"""CPU latency of one VideoMAE call (batch 1, 16x224x224) -> cache/videomae/cpu_latency.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import videomae as V  # noqa: E402

if __name__ == "__main__":
    r = V.cpu_latency(n=8, threads=int(sys.argv[1]) if len(sys.argv) > 1 else None)
    print(r)
    (V.CACHE / "cpu_latency.json").write_text(json.dumps(r, indent=1))
