"""Rate-matched random-call control for the X-CLIP cue on every set: same number of X-CLIP calls fired per
sequence, chosen at random among that sequence's scheduled calls (same people, same 1 s schedule)."""
import random
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import numpy as np  # noqa: E402

import eval_xclip as E  # noqa: E402
import xclip as X  # noqa: E402
from exp_harness import combine, make_event, per_action  # noqa: E402


def random_calls(method, seed):
    def run(r, ds):
        n = len(method(r, ds))
        calls = X.bank(ds).get(r.seq_id, [])
        rng = random.Random(f"{seed}{r.seq_id}")
        pick = rng.sample(calls, min(n, len(calls)))
        return [make_event("rand_call", m["t"], m["box"]) for m, _ in pick]
    return run


if __name__ == "__main__":
    kw = dict(prompt_set="litter_negs", margin=float(sys.argv[1]) if len(sys.argv) > 1 else 0.2)
    m = E.memo(X.make_method(**kw))
    rows = []
    for s in range(5):
        c = random_calls(m, s)
        rows.append((E.row("ctrl", "alone", c, do_lift=False), E.row("ctrl", "add-on", combine(E.BASE, c), do_lift=False),
                     E.row("ctrl", "replace", combine(E.NOREACH, c), do_lift=False)))
    for i, mode in enumerate(("alone", "add-on", "replace reach_down")):
        keys = [k for k in rows[0][i] if k not in ("variant", "mode")]
        print(mode, {k: f"{np.mean([r[i][k] for r in rows]):.1f}+-{np.std([r[i][k] for r in rows]):.1f}" for k in keys})
    print(per_action(m, E.TUNE).join(per_action(E.BASE, E.TUNE), rsuffix="_base").join(
        per_action(combine(E.NOREACH, m), E.TUNE), rsuffix="_replace"))
