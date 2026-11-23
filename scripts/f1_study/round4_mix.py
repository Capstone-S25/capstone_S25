"""Round 4: combine two fitted runs task by task.

    round4_mix.py r4_mixSL r4_icvl smoking littering

Writes runs/<tag>_round3.pkl with the named tasks from the second run
(r4_icvl: fit-side ICVL cameras included) and every other task from
r4_base (the release recipe). `r4_mixSL` is what ensemble-20261116 ships;
`r4_mixS` takes only smoking.
"""

from __future__ import annotations

import os
import pickle
import sys

from common import OUT


def main():
    tag, source, *tasks = sys.argv[1:]
    runs = OUT / "runs"
    # F1_MIX_BASE=r5_base mixes onto the leak-free base of the team split.
    base_tag = os.environ.get("F1_MIX_BASE", "r4_base")
    with open(runs / f"{base_tag}_round3.pkl", "rb") as f:
        base = pickle.load(f)
    with open(runs / f"{source}_round3.pkl", "rb") as f:
        other = pickle.load(f)
    if not set(tasks) <= set(base):
        raise ValueError(f"Unknown task in {tasks}")
    mixed = {t: other[t] if t in tasks else base[t] for t in base}
    with open(runs / f"{tag}_round3.pkl", "wb") as f:
        pickle.dump(mixed, f)
    print(tag, "from", source, tasks)


if __name__ == "__main__":
    main()
