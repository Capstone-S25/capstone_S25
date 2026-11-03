# Experiment history

Dated studies retain their original protocols, qualifications and published artifacts. Current commands and script locations are in [the workspace guide](workspace.md).

The [expanded accuracy evaluation](../reports/expanded_accuracy_review_20261014.md)
adds 55,156 GPU predictions and increases unattended-item coverage from 36 to
1,770 endpoints. It also attempts littering, loitering and rough sleeping as
explicitly qualified diagnostics. The [14 October slide deck](../reports/slides/model_evaluation_expanded_20261014.pdf)
covers the findings and methodology without an additional latency experiment.

The [supervised CLIP training workflow](../docs/training.md) adds masked five-target
heads, frozen-feature probes and actual visual fine-tuning for MobileCLIP2-S2 and
SigLIP2. Its [source-holdout protocol](../reports/clip_training_protocol_20261015.md)
uses new labelled video/photo sources and selects checkpoints on development data.
Training results are kept separately under `results/clip_finetuning_20261015/`.
The [completed training findings](../reports/clip_finetuning_findings_20261016.md)
cover all six GPU runs, stronger waste-dumping/unattended ranking, mixed smoking
transfer and unresolved loitering. The [research model card](../results/clip_finetuning_20261015/model_card.md)
links the checkpoints and records input and threshold limitations.

The [expanded strategy study](../results/strategy_study_20261016/report.md) adds new
smoking images, HMDB action clips, MEVA abandonment windows, public-sleep photos
and a separate date-held-out thermal loitering study. It compares 330 candidate
configurations and keeps genuine new-source transfer separate from historical
diagnostics. [Strategy reproduction and experimental inference](../docs/strategy_study.md)
describe the saved heads, automatic crops, ordered features and weight blending.
The [findings overview](../reports/strategy_findings_20261016.md) summarizes the
historical gains, new-source transfer failures and remaining threshold problems.

The [causal video pilot](../reports/video_pilot_findings_20261016.md) follows those
failures with a five-objective error audit, smoking/person supervision, temporal
heads, automatic tracking, an open-vocabulary item-detector ablation and fresh
AVSS source annotations. It reports event recall and actual alert exposure,
separating source criteria and disjoint-recording calibration. See the
[workflow and saved-prediction guide](../docs/video_pilot.md) for reproduction.

The [17 October improvements](../reports/event_improvements_findings_20261017.md)
compare 38 source-aware smoking policies, item-state/memory ablations and fixed
tiled item detection. Tiling raises CAVIAR reference-item recall at IoU 0.5 from
48.5% to 85.3%; the SigLIP ensemble improves retrospective HMDB AP, while CCTV
recall remains poor. Extra event memory does not improve the primary AVSS result.
New highway-debris videos are component stress tests, not verified unattended
events. The [reproduction guide](../docs/event_improvements.md) records these limits.

The [19 October SmolVLM fine-tuning study](../reports/smolvlm_finetuning_findings_20261019.md)
extends actual model adaptation to all five target definitions with language-layer
LoRA, 512-pixel ordered image inputs and source-separated supervision. It compares
the original model, a joint adapter and development-selected target routes, with
matched historical/transfer cohorts against the earlier CLIP strategies. The
[training and inference guide](../docs/smolvlm_finetuning.md) explains partial labels,
temporal input guards, checkpoints and the limits of the binary query interface.

The dedicated [training and evaluation slides](../reports/slides/model_training_evaluation_20261020.pdf)
compare MobileCLIP, SigLIP2, SmolVLM and VadCLIP across completed training stages:
original/zero-shot, linear probing, visual updates, LoRA, temporal adapters and
later source-data experiments. The [editable deck](../reports/slides/model_training_evaluation_20261020.pptx)
and [companion dataset/metric tables](../reports/slides/training_evaluation_20261020/README.md)
preserve the different test protocols and unavailable model stages explicitly.

The [20 October public-source pilot](../reports/generalization_pilot_20261020/report.md)
adds 740 scripted action clips, 36 controlled CLIP classifier fits, a frozen
SmolVLM transfer comparison and a complete-stream tiling test. Mixed SigLIP
improves proxy-source AP but regresses CCTV AP and false-positive rate; tiling
reduces primary unattended-event recall from 2/2 to 1/2. The new clips remain
weak gesture evidence, and no model is promoted. See the
[PDF addendum](../reports/generalization_pilot_20261020/runthrough.pdf),
[ten-slide PDF](../reports/slides/model_generalization_pilot_20261020.pdf),
[editable slides](../reports/slides/model_generalization_pilot_20261020.pptx), and
[reproduction guide](../docs/generalization_pilot.md).

The [unified dataset downloader](../docs/dataset_fetching.md) now includes the later
research acquisitions through XRF55, with configurable sources, concurrent
resumable transfers, integrity checks and failure recovery. Run
`.venv/bin/python -m anomaly_eval fetch-all --plan` to inspect the plan, then
remove `--plan` to fetch it. The [consolidated source/license registry](../docs/dataset_sources.md)
includes every recorded download link, version/hash and per-image Commons/Flickr
attribution. Acquisition counts are kept separate from eligible training samples.

The full-history [45-slide progress PDF](../reports/slides/model_evaluation_progress_20261019.pdf)
and [editable PowerPoint](../reports/slides/model_evaluation_progress_20261019.pptx)
cover all evaluation, data expansion, training and improvement stages, including
the three SmolVLM LoRA runs and their matched-control results. See the
[slide index](../reports/slides/README.md) for the summary, notes and source hashes.

The [complete technical runthrough](../reports/technical_runthrough_20261019/model_evaluation_runthrough_20261019.pdf)
documents the work through 19 October in 102 pages: data collection and audits,
evaluation protocols, actual training, iterative strategies, event metrics,
results and remaining generalization limits. Its [LaTeX source and build guide](../reports/technical_runthrough_20261019/README.md)
include the full candidate appendices, numeric exports and source hashes.

