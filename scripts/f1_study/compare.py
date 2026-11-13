"""Side-by-side per-source F1 (and AP) for saved runs: python compare.py tag:protocol ..."""

import json
import sys

from common import OUT, TEST_SOURCES, macro

field = "f1"
args = sys.argv[1:]
if args and args[0] in {"ap", "f1", "fpr", "best_threshold_f1", "recall", "precision"}:
    field, args = args[0], args[1:]
runs = {}
for spec in args:
    tag, protocol = spec.split(":")
    if tag == "baseline":
        data = json.loads((OUT / "baseline.json").read_text())
        runs[spec] = data[protocol]
        continue
    data = json.loads((OUT / "runs" / f"{tag}.json").read_text())["results"]
    runs[spec] = data[f"{tag}:{protocol}"]
width = max(len(s) for s in runs) + 1
print(f"{field:<30}" + "".join(f"{s[-width:]:>{width}}" for s in runs))
for name in TEST_SOURCES.values():
    cells = "".join(
        f"{runs[s][name][field]:>{width}.3f}" if name in runs[s] else f"{'—':>{width}}"
        for s in runs
    )
    print(f"{name:<30}{cells}")
print(
    f"{'macro':<30}" + "".join(f"{macro(r, field):>{width}.3f}" for r in runs.values())
)
