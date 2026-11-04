"""Static-change blobs (the gate's StaticForeground, default settings) for every cached ICVL observation frame, so a
"did something new land on the ground after a reach-down?" check can be replayed. CPU only, no YOLO.
Usage (from notebooks/): ../.venv/bin/python exp/icvl/ground_blobs.py   -> bench_cache/models/s/blobs_icvl.pkl
  ({seq_id: {frame_index: [blob dicts]}})
"""
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(HERE)]

from run import icvl_sequences  # noqa: E402
from stage1_gate import GateConfig, StaticForeground  # noqa: E402

OUT = NB / "bench_cache" / "models" / "s" / "blobs_icvl.pkl"


def main():
    cfg, out, t0 = GateConfig(), {}, time.time()
    seqs = icvl_sequences()
    for i, g in enumerate(seqs):
        bg, last, blobs = None, None, {}
        for fr in g.seq:
            if bg is None:
                bg = StaticForeground(cfg, g.seq.fps, fr.image.shape[:2])
            b = bg.apply(fr.image, (fr.t - last) if last is not None else 1.0 / g.seq.fps)
            last = fr.t
            if b:
                blobs[fr.index] = b
        out[g.seq.seq_id] = blobs
        if i % 50 == 0:
            print(f"{i}/{len(seqs)} ({time.time() - t0:.0f}s)", flush=True)
    OUT.write_bytes(pickle.dumps(out))
    print(f"done: {len(out)} sequences ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
