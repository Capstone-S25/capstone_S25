# Public-source generalization pilot (20 October 2026)

The report and CSVs are in [reports/generalization_pilot_20261020](../reports/generalization_pilot_20261020/report.md). This pilot checks whether adding a new public action source helps smoking classification and whether the earlier tiled item detector helps the complete causal unattended-item pipeline. It also evaluates the original SmolVLM against its existing smoking LoRA route on the new source.

XRF55 is a **scripted action/gesture proxy**: training-only inspection did not independently confirm a visible cigarette. It is not new verified CCTV smoking truth. The other source search decisions, including unavailable and rejected sources, are in `data/generalization_pilot_20261020/source_audit.json`. No new littering, loitering or public-sleep results are claimed in this round.

## Runtime

Use the existing Python 3.12 CPU and CUDA environments. Neural accuracy inference uses CUDA, float32 and no pacing. The convex logistic-regression fits and numerical metric replay use CPU. There is no new inference-latency benchmark.

The workspace's driver/library mismatch requires the already downloaded matching NVIDIA runtime. In Bash:

```bash
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="$PWD/data/models/nvidia_runtime/extracted${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

GPU commands need device access in the execution environment. Do not replace the system driver. Model weights and dependencies are already cached locally. The optional SmolVLM PEFT dependencies are in `data/tools/smol_training_deps`; append that directory to `PYTHONPATH` for its inference command. OpenCV used for video preparation is in `/tmp/capstone_eval_deps` in this workspace; an ordinary reproduction can install `opencv-python-headless` in its own virtual environment instead.

## Acquisition and immutable protocols

The default paths are date-specific. Freeze/fit/evaluate stages refuse to replace scored outputs. Use separate roots in a copied configuration/script when intentionally starting a new study; retain the existing outputs.

```bash
.venv/bin/python scripts/datasets/fetch_xrf55_pilot.py plan
.venv/bin/python scripts/datasets/fetch_xrf55_pilot.py fetch
PYTHONPATH="/tmp/capstone_eval_deps:$PWD" .venv/bin/python scripts/datasets/fetch_xrf55_pilot.py prepare
.venv/bin/python scripts/generalization/train_source_generalization.py freeze
.venv/bin/python scripts/generalization/evaluate_smolvlm_source_pilot.py freeze
PYTHONPATH="/tmp/capstone_eval_deps:$PWD" .venv/bin/python scripts/generalization/run_generalization_pilot.py prepare
```

The acquisition plan and split were frozen before downloads. The executable workflow subsequently includes the recorded URL-encoding transport amendment and the training-only proxy-label qualification. The original saved protocol drafts remain alongside those amendments. These do not authorize changing splits or labels after test scoring.

## Smoking feature extraction, selection and evaluation

Automatic crop detections cover every new frame; no anomaly labels are supplied to the detector. The wrapper sets a separate output root while reusing the exact existing strategy feature implementation. Alpha zero means the pinned pretrained visual encoder; it does not use the previous fine-tuned visual parameters.

```bash
.venv-gpu/bin/python scripts/strategy/detect_strategy_regions.py \
  --manifest data/generalization_pilot_20261020/xrf55/samples.jsonl \
  --cache data/generalization_pilot_20261020/xrf55/detections.jsonl \
  --batch-size 16

for encoder in mobileclip2-s2 siglip2-b16-224; do
  .venv-gpu/bin/python scripts/strategy/extract_strategy_features.py \
  --data-dir data/generalization_pilot_20261020/xrf55 \
    --manifest data/generalization_pilot_20261020/xrf55/samples.jsonl \
    --tag xrf55 --stage full --alphas 0 --model "$encoder"
  .venv-gpu/bin/python scripts/strategy/extract_strategy_features.py \
  --data-dir data/generalization_pilot_20261020/xrf55 \
    --manifest data/generalization_pilot_20261020/xrf55/samples.jsonl \
    --detections data/generalization_pilot_20261020/xrf55/detections.jsonl \
    --tag xrf55 --stage crop --model "$encoder"
done

# Freeze both encoders' development selections before any held-out evaluation.
.venv/bin/python scripts/generalization/train_source_generalization.py fit --model mobileclip2-s2
.venv/bin/python scripts/generalization/train_source_generalization.py fit --model siglip2-b16-224
.venv/bin/python scripts/generalization/train_source_generalization.py evaluate --model mobileclip2-s2
.venv/bin/python scripts/generalization/train_source_generalization.py evaluate --model siglip2-b16-224
```

Thirty-six head fits vary source family (old/new/mixed), full/full-plus-crop representation and C (0.1/1/10). All families receive the same combined development-source supervision for selection and thresholding. A single global cutoff satisfies empirical FPR at most 5% in every development source. The new source's scene-4 test has three held-out numeric subject aliases, 60 publisher-positive clips and 180 controls. Historical source tests are retrospective.

```bash
PYTHONPATH="$PWD/data/tools/smol_training_deps:$PWD" \
  .venv-gpu/bin/python scripts/generalization/evaluate_smolvlm_source_pilot.py run
```

SmolVLM uses eight uniformly selected views for both original and prior LoRA routes. Both development calibrations are frozen before any test forward pass. There is no new adapter training or source-test-based route selection. Its eight-view prompt differs from the CLIP sixteen-view recipe; interpret the matched comparison within each family.

## Stream ablation

```bash
.venv-gpu/bin/python scripts/generalization/run_generalization_pilot.py detect --batch-size 12
.venv/bin/python scripts/generalization/run_generalization_pilot.py evaluate
```

The three arms use one immutable set of 12,560 raw detector views across ten complete recordings. Full-frame people/items, full people plus tiled items, and all tiled proposals feed the same causal tracker and alarm rule. There is no event-driven frame selection or state tuning. Published AVSS EASY/MEDIUM, right-censored AVSS HARD and provisional ABODA intervals remain separate in results. Unknown-time alarms are counted separately.

The primary full-stream normal exposure adds forty seconds relative to the older t >= 20 tables. Aligned per-recording metrics are saved; the extra seconds are existing footage, not new data acquisition.

## Verification and report build

```bash
.venv/bin/python scripts/generalization/verify_generalization_pilot.py --stage streams
.venv-gpu/bin/python scripts/generalization/validate_source_pilot_gpu.py
.venv/bin/python scripts/generalization/verify_generalization_pilot.py
.venv/bin/python scripts/reporting/build_generalization_pilot_report.py
XDG_CACHE_HOME="$PWD/data/tools/tectonic/cache" \
  data/tools/tectonic/tectonic --keep-logs \
  reports/generalization_pilot_20261020/runthrough.tex
PYTHONPATH="/tmp/capstone_slide_deps:$PWD" .venv/bin/python \
  reports/slides/build_generalization_pilot_20261020.py
PYTHONPATH="/tmp/capstone_slide_deps:$PWD" .venv/bin/python \
  scripts/reporting/validate_generalization_pilot_documents.py --render
```

Replay verifies image identities, disjoint subject aliases/pixels, all candidate dev cutoffs and selections, selected and reference test scores, stream traces/alarms and event metrics. Separate native CUDA forwards validate both new CLIP feature caches and each SmolVLM route. The report builder requires passed replay and native checks. The report's source manifest binds reporting inputs and outputs; raw data, predictions and weights remain local under `data/` and `results/`.

The full historical paper and deck through 19 October remain intact. The 20 October PDF is a focused addendum, rather than a replacement for those experiments.
