"""Cached Stage 1 runs with their bicycle / motorcycle boxes (exp/alerts/two_wheelers.py) attached, so options that
need them (reach_skip_riders) can be replayed."""
import pickle
import sys
from pathlib import Path

NB = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(NB), str(NB / "tools"), str(NB / "exp")]

from compare_models import model_splits  # noqa: E402

CACHE = NB / "bench_cache" / "models" / "s"


def attach_two_wheelers(results, name):
    path = CACHE / f"twowheel_{name}.pkl"
    if not path.exists():
        return False
    boxes = pickle.loads(path.read_bytes())
    for g, r in results:
        b = boxes.get(g.seq.seq_id, {})
        for o in r.observations:
            o.two_wheelers = b.get(o.index, [])
    return True


def load_cached(name):
    """A cached run (bench_cache/models/s/<name>.pkl) with its two-wheeler boxes attached when available."""
    results = pickle.loads((CACHE / f"{name}.pkl").read_bytes())
    attach_two_wheelers(results, name)
    return results


def splits(heldout):
    """compare_models.model_splits with two-wheeler boxes attached (littering and Avenue halves)."""
    S = model_splits("s", heldout=heldout)
    attach_two_wheelers(S["littering"], "litter_heldout" if heldout else "litter_train")
    attach_two_wheelers(S["avenue"] + S["normal"], "avenue")
    return S
