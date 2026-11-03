# Event and small-item improvements

The [17 October findings](../reports/event_improvements_findings_20261017.md)
and [full run](../results/event_improvements_20261017/report.md) add source-aware
smoking thresholds/ensembles, causal item-state/memory ablations, and a GPU tiled
detector comparison. The [consolidated slides](../reports/slides/model_evaluation_progress_20261017.pdf)
cover the entire evaluation and training history.

The completed `results/event_improvements_20261017/` run preserves earlier runs.
All policy comparisons on smoking, CAVIAR and AVSS are retrospective. The new
highway-debris clips support component detection only; their duration and labels
do not establish owner-associated unattended events. No latency test or new
visual-backbone fine-tune was performed in this round.

## Reproduction

The dated component scripts use the default data/output directories below and
refuse overwriting completed scored artifacts. Use a separate workspace for a
full clean reconstruction. The policy script accepts a fresh `--output` path.
Report/verification and presentation builds can be repeated from saved outputs.

```bash
export PYTHONPATH="$PWD"
.venv/bin/python scripts/datasets/fetch_item_component.py
PYTHONPATH="/tmp/capstone_eval_deps:$PWD" .venv/bin/python scripts/datasets/prepare_item_component.py
.venv/bin/python scripts/video/run_event_improvements.py --output results/event_improvements_reproduction

export LD_LIBRARY_PATH="$PWD/data/models/nvidia_runtime/extracted${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv-gpu/bin/python scripts/video/benchmark_item_tiles.py
.venv/bin/python scripts/video/report_event_improvements.py
```

The fetcher verifies the pinned author archive hash and only extracts data files;
it checks cached files without downloading again. The component preparation
preserves original VoTT boxes and timestamps, checks actual FPS and decoded frame
counts, and excludes annotation timestamps outside the stream. Three explicit
ABODA source aliases are excluded from new-data counts. Existing CAVIAR training
references are used only for diagnostics, never to choose inference tiles.

The detector uses the existing pinned Grounding DINO tiny weights, generic
queries and thresholds. It encodes a full image and four fixed overlapping tiles.
The reference annotations are partial; scores report reference-box recall, not
precision. Raw per-tile detections and merged predictions allow independent
recomputation. A tuple/list filtering error in the initial aggregation was caught
and corrected by replaying identical cached outputs; invalid initial metric
exports are explicitly preserved and labelled separately. Verification rebuilds
NMS merges from raw serialized detections to prevent that error recurring.

Smoking policies average sigmoid-transformed saved-head scores with fixed equal
weights. The values are not calibrated probabilities. One cutoff must satisfy
the empirical false-positive budget in every development source; the route
maximizes worst-source recall, then mean recall and AP ties. All 38 policy scores,
cutoffs and selections are independently recomputed from original dev predictions.

The new tracker leaves the original baseline unchanged. Its trailing-median
stationarity and per-item spatial alarm memory are tested as separate ablations.
They provide no primary AVSS accuracy gain; the simpler state baseline remains
the reference. Unit tests cover causality, simultaneous items, fragmented IDs,
missing evidence, owner return, stationary jitter, sustained motion and threshold
ties. Per-item alert traces are replayed separately from model metrics.

The next validation must use independently reviewed events/normal intervals and
previously untouched cameras. Zero observed alerts over 2.76 normal minutes does
not establish one false alert per hour: the conditional one-sided 95% Poisson
upper bound is 65.1 per hour. Source diversity and temporal label completeness
remain the main evidence gaps.
