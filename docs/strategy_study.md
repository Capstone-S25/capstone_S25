# Expanded data and strategy experiments

The [16 October study](../results/strategy_study_20261016/report.md) preserves the
previous training study and compares 33 candidates per objective/backbone. It
adds smoking photos to training, evaluates new video/photo sources, and runs a
separate thermal-trajectory loitering component benchmark. Every primary route
uses development selection. Historical holdouts and newly acquired source tests
are reported separately. No latency experiment is included.

## Data and source qualifications

Acquired data live under `data/strategy_sources_20261016/`; frozen study manifests,
decoded hashes, detections and features under `data/strategy_study_20261016/`.
The original `data/clip_training_20261015/` manifest remains the prior study's
identity. Training in this screen has 3,973 examples and development has 1,812.
The 224 new smoking test photos come from the same collection that contributes
training photos; HMDB51, MEVA and new-author Commons photos are transfer cohorts.

- [Smoking photos](https://www.kaggle.com/datasets/sujaykapadnis/smoking): 1,120
  acquired, 1,118 retained; official folders preserved and related examples
  excluded. Camera/subject identities are not supplied.
- [Author-hosted HMDB51](https://huggingface.co/datasets/Serrelab/hmdb51): pinned
  release and archive SHA-256 retained; 906 clips from smoke/drink/eat/chew/talk/
  sit/stand. This is an offline action-source transfer test, not the standard
  51-class benchmark. Other primary actions are not exhaustive smoking controls.
- [Loitering annotations](https://github.com/johnnynunez/RS-WACV24_Loitering):
  CSV files match the recorded source Git blobs. The benchmark retains 17,878
  complete human trajectories across 31 dates; early dates train, middle dates
  develop, late dates test. It uses supplied tracks on one thermal camera.
- [MEVA](https://mevadata.org/): 13 available annotated videos produce 15
  abandonment windows and 12 pre-event controls. This action-window proxy does
  not independently verify persistent unattended-object state. Contributed and
  training annotations have different audit levels.
- Commons/author-linked Flickr: 40 relevant photos downloaded, 29 provisional
  visual labels, 28 eligible new-author test examples after exclusions. Unknown
  posture/context is omitted. Labels never describe housing status.

The broad Commons search initially returned irrelevant animals/art/scene names;
those are excluded, not easy negative training examples. Focused human categories
and explicitly linked, matching openly licensed author copies provide the final
review queue. Wikimedia retry delays are respected. The remaining unlabelled or
unavailable candidates are not counted as labelled examples.

VADD's author link requests an access form. Other littering collections found in
papers did not expose an immediately downloadable matching event release. No
form, email or contact request was submitted. Existing IWDD data remain the event
source; unrelated garbage images are not relabelled as disposal events.

## Experiment design

The screen compares pretrained last-frame scores, source-balanced linear probes,
explicit ordered-view features, automatically detected contextual regions,
sparse person/item geometry, two task-specific MLP representations, and visual
weight interpolation at 0/0.25/0.5/1. Interpolation uses the original seed-42
fine-tuned visual tensors and pinned pretrained base; the classifier is fitted
again on training examples. It is not another full visual fine-tuning run.

The fixed detector is torchvision Faster R-CNN ResNet50 FPN v2, COCO weights,
confidence 0.5. It uses the highest-confidence person and bag regions with 20%
context padding; their union preserves nearby context. Missing regions fall back
to the scene. Ground-truth boxes never choose the main crops. Sparse geometry
describes detections across sampled views; it does not establish ownership or
provide a validated multi-person tracker.

Linear C values are 0.1/1/10. MLP heads have 64 hidden units, dropout 0.3,
AdamW lr 0.001/decay 0.01, maximum 80 epochs and eight-epoch development patience.
Training labels are target-specific; class/source/group balancing is retained.
Primary selection maximizes mean development-source AP within each target.
An exploratory worst-source policy is also retained. All choices are separate
from threshold calibration and new test labels.

Final feature extraction uses float32 with TF32 disabled. Linear fitting uses the
library solver; saved coefficients are float32. Thresholds are calibrated again
on development scores from the serialized head, with weights/routes fixed.
Maximum-development-F1, a development 5% FPR-budget point and zero-logit decisions
are all reported. New-camera false-positive rates can remain extremely high.

## Reproduction

These scripts run from a checkout with `PYTHONPATH="$PWD"`. Use a CUDA-enabled
PyTorch/torchvision environment for GPU stages, plus the repository requirements.
Preparation additionally requires OpenCV, SciPy and PyYAML. In this workspace,
OpenCV is isolated at `/tmp/capstone_eval_deps`; prepend that path only for CPU
video-preparation commands. Host-driver compatibility uses the same local NVIDIA
libraries as the original GPU study; no system driver was changed.

Acquisition utilities include `fetch_strategy_sources.py`,
`fetch_meva_transfer.py`, and `extract_hmdb_sources.py`. The latter uses the
system libarchive to extract only selected, path-checked RAR entries. HMDB AVI
indexed seeking proved unreliable; sequential decoding recovers all 906 clips.
The failed indexed-seek manifest is preserved as preparation diagnostics and is
never used for model selection.

After freezing inputs, the main stages are:

```bash
export PYTHONPATH="$PWD"

# New output directories are required; completed model selections are immutable.
.venv-gpu/bin/python scripts/strategy/detect_strategy_regions.py
.venv-gpu/bin/python scripts/strategy/extract_strategy_features.py --model siglip2-b16-224
.venv-gpu/bin/python scripts/strategy/extract_strategy_features.py --model siglip2-b16-224 --stage crop
.venv-gpu/bin/python scripts/strategy/run_strategy_screen.py --model siglip2-b16-224

# Once transfer representations are ready:
.venv/bin/python scripts/strategy/evaluate_strategy_transfer.py --model siglip2-b16-224
MPLCONFIGDIR=/tmp/capstone_strategy_mpl .venv/bin/python scripts/strategy/report_strategy_study.py
.venv-gpu/bin/python scripts/strategy/verify_strategy_study.py --live
```

Repeat model-specific commands for MobileCLIP2-S2. The transfer preparation,
screening and feature commands retain individual manifest/tag arguments so each
cohort has its own identity. `selection.json`, every candidate head, raw scores,
final thresholds and feature metadata remain in the study folders. Completed
manifests/runs refuse overwrite; use new output locations for a different study.
These are study-specific scripts with dated `DATA`/`RUN` locations. For a new
study, configure those locations together before running; the examples reproduce
this protocol rather than overwriting its completed results.

The separate trajectory component retains `selected_estimator.joblib` under
`results/strategy_study_20261016/thermal_trajectories/`. It consumes the 14-feature
descriptor in `benchmark_loitering_trajectories.describe`, not RGB frames. Its
export replays the previously saved held-out scores without another model or
threshold selection.

## Experimental inference

```bash
PYTHONPATH="$PWD" .venv-gpu/bin/python scripts/strategy/predict_strategy.py \
  --model siglip2-b16-224 \
  --images frame_0.png frame_7.png frame_13.png frame_20.png \
  --timestamps 0 7 13 20 --operating-point raw \
  --output results/strategy_prediction.json
```

Use one image for static targets, four distinct timed views over approximately
20 seconds for duration targets, and eight uniform complete-clip views for the
offline littering model. The interface returns unknown when required history is
missing. It loads saved heads, exact pinned base weights, interpolation parameters
and the fixed detector. Scores are logits, not calibrated event probabilities.
This research command is separate from the earlier `predict-clip` bundle.

The [protocol](../reports/strategy_protocol_20261016.md),
[complete results](../results/strategy_study_20261016/report.md), and
[verification](../results/strategy_study_20261016/verification.json) document
the practical limits. AP gains on historical cameras do not validate new-camera
alert thresholds, and the new movie/action sources expose remaining transfer
failures.
