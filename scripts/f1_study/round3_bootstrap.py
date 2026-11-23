"""Paired camera bootstrap of the new-site summary between two reports.

    round3_bootstrap.py r3_release "r3_final+r3_det_n02"

Reads the `<spec>_site.json` files of round3_report.py. Within every
(task, source) cell whole cameras/clips are resampled with replacement, the
same draw for both models; the summary is the mean recall over positive
sources and the mean share of ordinary footage flagged over certain-label
ordinary sources.
"""

import json
import sys

import numpy as np
from common import OUT

NORMAL = {"virat", "avenue", "meva", "xrf55"}


def load(spec):
    return json.loads((OUT / "runs" / f"{spec}_site.json").read_text())["site"]


def summary(cells, picks):
    recall, flagged = [], []
    for key, cell in cells.items():
        groups = np.array(cell["groups"])
        flag, y = np.array(cell["flag"]), np.array(cell["y"])
        idx = np.concatenate([np.flatnonzero(groups == g) for g in picks[key]])
        if y[idx].sum():
            recall.append(flag[idx][y[idx] == 1].mean())
        source = key.split("|")[1]
        if source in NORMAL and not cell["uncertain"] and (y[idx] == 0).sum():
            flagged.append(flag[idx][y[idx] == 0].mean())
    return np.mean(recall), np.mean(flagged)


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    rng = np.random.default_rng(0)
    names = {k: sorted(set(v["groups"])) for k, v in a.items()}
    same = {k: names[k] for k in a}
    base = [summary(a, same), summary(b, same)]
    diffs = []
    for _ in range(2000):
        picks = {k: rng.choice(v, len(v)) for k, v in names.items()}
        ra, fa = summary(a, picks)
        rb, fb = summary(b, picks)
        diffs.append((rb - ra, fb - fa))
    diffs = np.array(diffs)
    for i, label in enumerate(["new-site recall", "ordinary footage flagged"]):
        lo, hi = np.quantile(diffs[:, i], [0.025, 0.975])
        print(
            f"{label}: {base[0][i]:.3f} -> {base[1][i]:.3f}, change "
            f"{base[1][i] - base[0][i]:+.3f} (95% interval {lo:+.3f} to {hi:+.3f})"
        )


if __name__ == "__main__":
    main()
