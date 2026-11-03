"""Background-subtraction pass (no YOLO) to add static-foreground blobs to cached perception runs.

The blobs depend only on the frames and the background-model settings (bg_long_s, bg_short_s, bg_width), so they
are computed once here and the "litter" / static-object rule is then replayed under many settings.
bg_long_s = 60 s so that an object left for a while is still "new" to the slow model.

Usage (from notebooks/): ../.venv/bin/python exp/newcam/bg_blobs.py caviar aboda avenue_tune
Writes bench_cache/models/s/bg_<name>.pkl = {seq_id: {frame index: [blob, ...]}}
"""
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parents[1]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp"), str(HERE)]

from stage1_gate import GateConfig, RunResult, StaticForeground  # noqa: E402

CACHE = NB / "bench_cache" / "models" / "s"
BG_CFG = GateConfig(bg_long_s=60.0, bg_short_s=1.0, litter_static_s=10.0)


def blobs_for(seq):
    bg, out, last = None, {}, None
    for fr in seq:
        if bg is None:
            bg = StaticForeground(BG_CFG, seq.fps, fr.image.shape[:2])
        dt = (fr.t - last) if last is not None else 1.0 / seq.fps
        last = fr.t
        out[fr.index] = bg.apply(fr.image, dt)
    return out


def with_blobs(results, blobs):
    """Copies of cached (g, RunResult) pairs whose observations carry the background blobs."""
    out = []
    for g, r in results:
        b = blobs[r.seq_id]
        obs = [replace(o, blobs=b.get(o.index, [])) for o in r.observations]
        out.append((g, RunResult(r.seq_id, r.source, r.fps, r.n_frames, r.duration_s, r.events, obs, r.wall_s, r.n_stitched)))
    return out


def sources(name):
    if name == "avenue_tune":
        from compare_models import model_splits
        T = model_splits("s", heldout=False)
        return T["avenue"] + T["normal"]
    return pickle.loads((CACHE / f"{name}.pkl").read_bytes())


if __name__ == "__main__":
    for name in sys.argv[1:]:
        path = CACHE / f"bg_{name}.pkl"
        if path.exists():
            continue
        t0, res = time.time(), {}
        for g, r in sources(name):
            res[r.seq_id] = blobs_for(g.seq)
            print(f"{name} {r.seq_id} ({time.time() - t0:.0f}s)", flush=True)
        path.write_bytes(pickle.dumps(res))
        print(f"{name} saved", flush=True)
