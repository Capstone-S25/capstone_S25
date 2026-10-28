# Train and use the CLIP anomaly classifiers

The supervised study trains **MobileCLIP2-S2** and **SigLIP2-B/16-224** with the
existing smoking, littering, loitering, rough-sleeping and unattended-item
definitions. The [protocol](../reports/clip_training_protocol_20261015.md) records
the data qualifications and development-only selection plan. The resulting models
classify scenes/clips; they do not output object bounding boxes.

## Data and splits

The frozen research manifest is
`data/clip_training_20261015/samples.jsonl`. Each row contains ordered image views,
one known binary target, a source group, evidence, decoded pixel hashes and a
split. Unknown targets do not contribute loss. Decoded pixels are rechecked before
training; camera/recording/author groups cannot cross splits.

The study uses 7,498 examples: 3,258 training, 1,633 development, 1,801 test and
806 additional Stanford40 official test photographs. There are 12,478 distinct
view images. Frames, views and adjacent windows are not independent events.

| Target | Training source | Held-out evidence and main limitation |
| --- | --- | --- |
| Smoking | CCTV camera c4 plus Stanford40 official-training photos | Camera c1 and official photo test; historical CCTV results were inspected |
| Littering | Author-labelled MIVIA-IWDD videos and hard negatives | All 100 official test videos; video-level waste dumping is broader than small littering |
| Loitering | Street Scene track/type annotations | Whole-recording supervised re-split; a single camera |
| Rough sleeping | New Commons photos and matched controls | Author/perceptual-group holdout; provisional assistant labels, six positive test photos |
| Unattended items | CAVIAR and ABODA lobby intervals | Classroom/walkway families; provisional assistant labels, retrospective source holdout |

498/500 IWDD videos were acquired. `vid0325.mp4` and `vid0346.mp4` were unavailable
from the public training release. Another 63 acquired training videos were excluded
because their perceptual-source clusters touched official test videos. The final
IWDD manifest has 274 train, 61 development and 100 test videos. Nothing is
silently relabelled or moved from the official test into training.

Rough-sleeping annotations describe visible apparent sleep/bedding and public
context. Housing status is not inferred. The old synthetic lying dataset and
person-only loitering exports are not treated as verified anomaly supervision.

## Run training

Use a CUDA PyTorch environment with the repository requirements and cached model
revisions. Source-video preparation additionally needs OpenCV; this workspace has
it under `/tmp/capstone_eval_deps`. The existing GPU helper uses the isolated
environment and matching NVIDIA libraries without changing the system driver.

```bash
# One model; every output directory must be new.
scripts/evaluate_gpu.sh train-clip \
  --manifest data/clip_training_20261015/samples.jsonl \
  --model siglip2-b16-224 --output results/my_new_training_run \
  --batch-size 32 --workers 6 --threads 8 --epochs 10 \
  --learning-rates 1e-5 3e-5 --seed 42
```

Repeat with `--model mobileclip2-s2` for the other backbone. Two jobs fit this
workspace's GPU. No pacing or latency experiment is part of training.
`--prepare-features-only` creates only immutable base features, without fitting a
probe or model; those cached features are keyed by input pixel hashes, model
revision and the pooling/preprocessing policy. Fine-tuned features never use that
cache.

The study launcher `scripts/clip/run_clip_training.py` runs both models for seeds
42/43/44, preserving exact commands and logs. It refuses to overwrite model runs.
`--resume-preparation` only resumes before any model job exists;
`--allow-recorded-source-gaps` accepts explicitly recorded download failures,
without waiving missing unrecorded files. Its fixed study output is
`results/clip_finetuning_20261015/`; use the single-model command for a new study.

`scripts/clip/resume_clip_training.py` can adopt running jobs and schedule independent
seed replications, with up to three concurrent GPU jobs. It validates the prior
controller identity before replacement. Initial seed-42 execution used convolution
autotuning; later jobs disable it because variable frame-batch shapes otherwise
trigger repeated convolution searches. The initial executed code is preserved in
the study's code snapshots. Input labels, held-out splits and selection rules
remain fixed.

Checkpoints contain classifier parameters and genuinely updated last-stage
visual parameters, with the exact pinned base model revision. The text encoder,
early visual layers and batch-normalization statistics remain frozen. Augmentation
preserves full scene content. Sampling balances task/class, dataset domain and
source group. Distillation and parameter anchoring limit pretrained-feature drift.

The frozen-feature probe's C and the fine-tuned checkpoint are selected on
development macro task/source AP. Per-target routing among the pretrained score,
probe and fine-tuned score is also fixed on development data. Repeated seeds
measure optimization variability; they do not add independent source cameras.

## Classify images with the trained bundle

After training completes, use the primary seed's frozen per-target routing and
operating points:

```bash
scripts/evaluate_gpu.sh predict-clip --model siglip2-b16-224 \
  --images frame_0.png frame_7.png frame_13.png frame_20.png \
  --timestamps 0 7 13 20 --output results/prediction.json
```

Use one image for static evidence, four timed views for duration-dependent tasks,
and eight uniformly spaced views over a littering clip for its video-level model.
The image order must be chronological. Static objectives use the endpoint frame.
The interface returns unknown for loitering/unattended items when sufficient timed
history is absent. Scores are logits, not calibrated event probabilities. The
operating point defaults to maximum development F1; choose
`--operating-point dev_fpr_budget` to use the empirical development 5% FPR point.
That development budget does not guarantee the same FPR on a new camera.
In this study the default threshold gives nearly 100% unattended-item FPR on
some held-out cameras, despite improved ranking. It is retained to report the
development-selection protocol faithfully, not presented as a validated alert
setting. Read the source-specific operating-point tables before using decisions.
`--operating-point raw` uses the predefined zero-logit binary decision. All three
points are reported; changing the threshold after examining a test score is not
an untouched test comparison.

Final quality evaluation and inference use float32 with TF32 disabled. Initial
AMP inference showed batch-shape-dependent score differences; its original
artifacts are retained. The final evaluation keeps every checkpoint and
development-selected method fixed and recalibrates thresholds using development
scores only. It lives in each model/seed's `float32_evaluation/` directory.
`--precision bfloat16` explicitly requests the original AMP policy; float32 is the
default. The final quality pass can be reproduced with
`scripts/clip/evaluate_clip_float32.py --model MODEL` in the CUDA environment, into new
evaluation directories; completed output is deliberately not overwritten.

## Reports and checks

```bash
PYTHONPATH="$PWD" .venv/bin/python scripts/clip/report_clip_training.py
```

Read [the training report](../results/clip_finetuning_20261015/report.md) for all
candidate methods, frozen operating points, source-specific results, seed
variation and grouped uncertainty. The raw report never selects the best test
column as the deployment route. Previous evaluation artifacts remain historical.
The [findings overview](../reports/clip_finetuning_findings_20261016.md) summarizes
the completed study. The [model card](../results/clip_finetuning_20261015/model_card.md)
and [bundle index](../results/clip_finetuning_20261015/model_bundle.json) identify
the primary checkpoint paths, hashes, base revisions and frozen operating points.

Tests cover masked gradients, source/pixel leakage, task/source sampling balance,
temporal ordering, operating-point ties and cache invalidation. Real GPU
forward/backward and one-epoch pipeline checks were completed for both backbones
before the full study. Deployment generalization still needs more independently
verified target events on new cameras.
