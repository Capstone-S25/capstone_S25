"""Stage 1 perception on ICVL (real surveillance video) at the deployment rate, cached for scoring.

Every ICVL video with an annotation file, sampled at ~5 fps (stride = round(native fps / 5)), YOLO26s + pose +
ByteTrack with the current GateConfig defaults (motorcycle boxes kept for reach_skip_riders), on the Mac GPU.
ICVL was never used for tuning, so everything scored on it is a held-out result.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/run.py   -> bench_cache/models/s/icvl.pkl
"""
import pickle
import sys
import time
from pathlib import Path

import cv2

NB = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(NB), str(Path(__file__).parent)]

from inventory import ICVL, video_for  # noqa: E402
from stage1_bench import GTSequence, run_perception  # noqa: E402
from stage1_gate import GateConfig, Stage1Gate, VideoSequence  # noqa: E402

TARGET_FPS = 5.0
OUT = NB / "bench_cache" / "models" / "s" / "icvl.pkl"


def icvl_sequences():
    seqs = []
    for txt in sorted(ICVL.glob("*/*.txt")):
        vid = video_for(txt)
        if vid is None:
            continue
        cap = cv2.VideoCapture(str(vid))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        stride = max(1, round(fps / TARGET_FPS))
        seq_id = f"icvl/{vid.parent.name}/{vid.stem}"
        seqs.append(GTSequence("icvl", VideoSequence(vid, stride=stride, seq_id=seq_id), {}, tag=str(txt.relative_to(ICVL))))
    return seqs


def main():
    cfg = GateConfig(det_weights=str(NB / "yolo26s.pt"), pose_weights=str(NB / "yolo26s-pose.pt"), device="mps")
    seqs = icvl_sequences()
    print(f"{len(seqs)} videos", flush=True)
    t0, out = time.time(), []
    gate = Stage1Gate(cfg)
    for k in range(0, len(seqs), 50):
        out += run_perception(gate, seqs[k:k + 50], verbose=False)
        print(f"{len(out)}/{len(seqs)} done ({time.time() - t0:.0f}s)", flush=True)
        OUT.write_bytes(pickle.dumps(out))
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
