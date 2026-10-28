# Causal video-event research pilot

The [findings](../reports/video_pilot_findings_20261016.md) and
[full run](../results/video_pilot_20261016/report.md) cover a first smoking and
unattended-item pilot. The [protocol](../reports/video_pilot_protocol_20261016.md)
records the controlled comparisons and pre-score amendments. All five targets
have a provisional error review. This does not replace the earlier evaluation
or establish deployment readiness for all five objectives.

## Inputs and source separation

`data/video_pilot_20261016/` contains the prepared recordings, 2 Hz video frames,
sixteen-view HMDB clips, canonical manifests, detections and frozen features.
`verified_windows.jsonl` completes/repairs inherited sparse-stage pixel-hash
metadata without changing labels, views, timestamps or splits. The prepared
manifest stays intact. `modeling_windows.jsonl` additionally purges ten training
clips sharing a known film-part alias with development. No test clip is removed
by that source guard. Earlier fits are retained in `prefix_split_diagnostic/`;
final selection uses the source-guarded fit directories at the run root.

HMDB now contributes 530 training clips; its 205 evaluation clips have already
been inspected in the earlier transfer study. They are retrospective, grouped
action evidence, not a new surveillance test. Source names do not guarantee
complete movie/actor identity knowledge. Static camera images remain static.

[The authors' AVSS bundle](http://www-vpu.eps.uam.es/publications/AODsurvey/)
provides three new videos and temporal annotations. EASY/MEDIUM have event ends
inside their decoded streams. HARD's annotated tail exceeds the supplied stream
and is a separately reported right-censored diagnostic. Nominal annotation FPS
is not substituted for encoded video FPS. MEDIUM's unused XML schema contains an
unclosed element; preparation parses the unchanged data block. No downloaded
source code is executed. The receipt records URLs and local content hashes.

## Model and alert design

The two pretrained CLIP encoders stay frozen. Fixed COCO detections propose up to
four persons, each with contextual whole/upper-body crops, and two item/owner
contexts. Scene features, region mean/max features and optional causal geometry
feed linear or GRU heads. A smoking multiple-instance head additionally uses
matched automatic regions labelled from **training-camera** COCO annotations;
inference never chooses a crop using source boxes. Its hard-negative variant
mines training normals within sources, preserving class/source sampling mass.

A simple Hungarian IoU/velocity tracker is a baseline, not ByteTrack. Ownership
is a proximity hypothesis requiring observed history; there is no independent
ownership ground truth. Existing low-resolution CAVIAR item boxes expose a
closed-vocabulary detector failure. The additional
[Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
geometry ablation uses revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, fixed
generic person/bag/box queries, box threshold 0.25 and text threshold 0.20. Its
geometry changes while the CLIP image representations remain fixed.

GRUs are causal over twenty-second histories, with variable-length static/clip
auxiliaries. Selection maximizes worst development-source AP, then mean AP.
All three seeds and regressions are retained. Nonlinear head training and image
feature extraction use float32 on GPU; no inference-speed comparison is run.

Event thresholds maximize development event recall subject to the specified
normal false-alert budget. Two-second persistence, one-second release and a
ten-second cooldown are fixed. Alert times are actual emission times, not
backdated. A continuing alert emitted before a labelled onset does not later
become a true positive. Unknown intervals never supply normal exposure.

AVSS's known source criterion is also evaluated separately: the same item must
be stationary for thirty seconds and its associated owner absent for at least
one sample. Both this gate alone and fixed classifier-plus-gate results are
reported. This is separate from unchanged global-rule transfer.

The historical calibration comparison uses entire disjoint normal recordings
and excludes them from both evaluation arms. It does not fit thresholds to
evaluation negatives. Fresh AVSS calibration is unavailable. Short normal
exposure cannot establish a reliable per-hour alert guarantee.

## Reproduction

Use the repository requirements plus OpenCV for video preparation. In this
workspace `.venv-gpu` supplies CUDA PyTorch, and OpenCV is isolated under
`/tmp/capstone_eval_deps`; prepend that location only for CPU preparation. GPU
commands need the existing matching NVIDIA libraries:

```bash
export PYTHONPATH="$PWD"
export LD_LIBRARY_PATH="$PWD/data/models/nvidia_runtime/extracted${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Use new output directories when changing an experiment. Previous completed
manifests and selections refuse overwrite. The existing prior-study/source
inputs referenced by preparation must be present.

```bash
DATA=data/video_pilot_20261016
RUN=results/video_pilot_20261016
.venv/bin/python scripts/video/audit_video_failures.py --output "$RUN/error_audit"

.venv/bin/python scripts/datasets/fetch_avss_pilot.py --data "$DATA"
PYTHONPATH="/tmp/capstone_eval_deps:$PWD" .venv/bin/python scripts/video/prepare_video_pilot.py --output "$DATA"
PYTHONPATH="/tmp/capstone_eval_deps:$PWD" .venv/bin/python scripts/datasets/prepare_avss_pilot.py --data "$DATA"
.venv/bin/python scripts/video/verify_video_data.py --data "$DATA"
.venv/bin/python scripts/video/guard_video_sources.py --data "$DATA"
.venv-gpu/bin/python scripts/video/detect_video_pilot.py --data "$DATA"

# Repeat extraction and fitting for each backbone.
.venv-gpu/bin/python scripts/video/extract_video_pilot.py --data "$DATA" --model siglip2-b16-224
.venv-gpu/bin/python scripts/video/train_video_pilot.py --data "$DATA" --output "$RUN" --model siglip2-b16-224
.venv-gpu/bin/python scripts/video/train_person_pilot.py --data "$DATA" --run "$RUN" --model siglip2-b16-224

.venv/bin/python scripts/video/audit_smoking_regions.py --data "$DATA" --output "$RUN/error_audit/pilot_region_coverage.json"
.venv/bin/python scripts/video/audit_video_tracks.py --data "$DATA" --output "$RUN/error_audit/track_components.json"

# Acquire/verify the pinned public detector, then audit training detections.
HF_HUB_DISABLE_XET=1 .venv-gpu/bin/python scripts/datasets/fetch_grounding_pilot.py
.venv-gpu/bin/python scripts/video/detect_grounded_pilot.py --data "$DATA" --scope train
.venv/bin/python scripts/video/audit_video_tracks.py --data "$DATA" --tracks "$DATA/grounding_tracks_train.jsonl" --output "$RUN/error_audit/grounded_track_components.json"
.venv-gpu/bin/python scripts/video/detect_grounded_pilot.py --data "$DATA" --scope all --batch-size 12
.venv-gpu/bin/python scripts/video/train_grounded_pilot.py --data "$DATA" --run "$RUN" --model siglip2-b16-224

# Once all three training commands have completed for BOTH backbones:
.venv/bin/python scripts/video/freeze_video_pilot.py --run "$RUN"
.venv-gpu/bin/python scripts/video/evaluate_video_pilot.py --data "$DATA" --run "$RUN" --model siglip2-b16-224
# Repeat evaluation for MobileCLIP, then export, report and replay checks.
.venv/bin/python scripts/video/export_video_pilot.py --data "$DATA" --run "$RUN"
MPLCONFIGDIR=/tmp/capstone_video_mpl .venv/bin/python scripts/video/report_video_pilot.py --data "$DATA" --run "$RUN" --summary reports/video_pilot_findings_20261016.md
.venv-gpu/bin/python scripts/video/verify_video_pilot.py --data "$DATA" --run "$RUN" --live
```

The error-export scripts are `audit_video_failures.py`, `audit_smoking_regions.py`
and `audit_video_tracks.py`. Review queues retain original labels and explicitly
distinguish provisional assistant notes from independent human review.

## Reviewing a saved prediction

```bash
PYTHONPATH="$PWD" .venv-gpu/bin/python scripts/video/predict_video_pilot.py \
  --data data/video_pilot_20261016 --run results/video_pilot_20261016 \
  --model siglip2-b16-224 --recording-id avss:AVSSS07_EASY \
  --output results/video_pilot_replay.json
```

This command replays cached real-video/image features through actual saved
heads. It is not a streaming application or a new-video decoder. Add
`--known-avss-rule` for the separately specified source-criteria diagnostic.
Use `--window-id` for a single static/clip case; it returns a state score rather
than claiming a continuous event alert. Scores are uncalibrated logits.
