# CLIP fine-tuning findings — 16 October 2026

Both **MobileCLIP2-S2 and SigLIP2-B/16-224 were trained on the GPU**, with three
seeds per backbone. All six runs completed. The study improves waste-dumping and
unattended-item ranking, but **does not establish reliable generalization for all
five objectives**. CCTV smoking remains source-dependent, loitering remains
unresolved, and rough-sleeping evidence is too small and provisional for a strong
claim.

The [full evaluation](../results/clip_finetuning_20261015/report.md) contains every
candidate, source-specific metrics, thresholds, seed replications, uncertainty
and training curves. The [model card](../results/clip_finetuning_20261015/model_card.md)
and [bundle index](../results/clip_finetuning_20261015/model_bundle.json) identify
the usable checkpoints. [Training and inference instructions](../docs/training.md)
describe the new CLI commands. Previous results and slide decks remain historical.

## What changed

The frozen dataset contains **7,498 target-specific examples** and 12,478 distinct
view images: 3,258 training, 1,633 development, 1,801 test, plus 806 additional
Stanford40 official-test photos. These units include photos, video clips and
overlapping endpoints; they are not 7,498 independent anomaly events.

New sources include [Stanford40](http://vision.stanford.edu/Datasets/40actions.html)
smoking/action photographs, [MIVIA-IWDD](https://github.com/MiviaLab/IllegalWasteDumping)
waste-dumping videos, [Street Scene](https://www.merl.com/research/?research=license-request&sw=StreetScene)
track/type annotations, and openly licensed Commons public-sleep photographs.
Existing CCTV smoking and reviewed unattended-item sources were retained with
camera/source holdouts. Acquisition metadata, excluded related sources, missing
files, pixel hashes and label evidence are recorded with the manifest.

The two unavailable IWDD files are training videos. All 100 official test videos
were acquired. Another 63 acquired training videos were excluded because their
perceptual-source groups touched the official test. The final IWDD partition is
274 training, 61 development and 100 test clips.

Each backbone was evaluated as a pretrained text score, a frozen-feature linear
probe, and an actual visual fine-tune. Fine-tuning updated 18.1 million MobileCLIP
or 14.2 million SigLIP visual parameters plus five binary heads. Unknown target
labels were masked. Task/class/source/group balancing, scene-preserving
augmentation, frozen early layers and pretrained-feature/weight regularization
were used to limit overfitting.

Checkpoints, probe regularization, learning rates and per-target routing were
selected using development data only. Seed 42 was predesignated as the primary
candidate; seeds 43/44 measure optimization variation. The routes retain the
pretrained score or probe when development evidence favors them. No lucky seed
or test-optimal threshold was substituted after seeing held-out results.

## Held-out results

Average precision (AP) measures ranking across thresholds; higher is better.
Each cell compares the **same study's pretrained baseline → development-selected
route**, for primary seed 42. Improvements combine learned supervision and, for
video tasks, temporal features; they are not isolated backbone-only effects.

| Objective / evidence | Test examples | MobileCLIP AP | SigLIP AP | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Smoking, CCTV camera c1 | 646 | 0.642 → 0.564 | 0.669 → 0.650 | Neither primary fine-tune improves CCTV ranking |
| Smoking, official photo holdout | 806 | 0.220 → 0.901 | 0.499 → 0.954 | Strong photo-domain gain; this domain also contributes training photos |
| Littering / waste dumping | 100 clips | 0.424 → 0.858 | 0.499 → 0.885 | Clear ranking improvement on the official video holdout; broader than small littering |
| Loitering | 75 windows, 11 positive | 0.320 → 0.093 | 0.097 → 0.097 | Unresolved; selected routes rank below positive prevalence 0.147 |
| Rough sleeping / visible public sleep | 18 photos, 6 positive | 0.502 → 0.976 | 0.733 → 0.915 | Promising small photo diagnostic with provisional assistant labels |
| Unattended items | 962 endpoints | 0.209 → 0.652 | 0.309 → 0.896 | Stronger ranking; provisional labels and only two held-out camera families |

SigLIP's primary routes are fine-tuning for smoking/unattended items, probing for
waste dumping/public sleep, and the pretrained score for loitering. MobileCLIP's
primary routes use its fine-tuned candidate for all five targets. All alternatives
remain in the full report, including those whose test scores exceed their
development-selected route; that does not retrospectively change selection.

## Ranking improvements do not solve threshold transfer

For SigLIP unattended items, the predefined zero-logit point gives **F1 0.749,
recall 61.2%, precision 96.7%, FPR 0.7%** on this holdout. However, the threshold
that maximizes development F1 gives **99.6% test FPR**; even the threshold meeting
the development 5% FPR budget gives **99.4% test FPR**. Thus the improved AP does
not make the default development threshold suitable for new cameras.

Smoking has the same warning: both primary maximum-development-F1 thresholds
produce more than 93% FPR on the held-out CCTV camera. High F1 alone can hide this
failure in a cohort with many positive examples. The report retains all three
predefined operating points, confusion counts, recall and FPR without test
recalibration. A zero-logit point is also not a universal deployment threshold.

## What the generalization evidence supports

Source/recording/author groups stay within one split; pixel identities are checked
before training and evaluation. Official photo/video test partitions are retained
where available, with related training sources excluded. Nevertheless:

- Smoking and unattended CCTV sources were historically inspected, so these are
  retrospective source holdouts, not a newly untouched deployment benchmark.
- Street Scene uses a supervised re-split of original test recordings from one
  camera, not its official unsupervised benchmark. Sparse global-view features
  have not learned reliable loitering under this design.
- Rough-sleeping and unattended labels remain provisional assistant review.
  Public-sleep labels describe visible posture/bedding/context, with no inference
  about housing status. Six positive test photos cannot establish robust coverage.
- Waste dumping is evaluated offline using eight uniform frames over each full
  clip. It does not measure event localization, online alert delay or all forms of
  pedestrian littering.
- Repeated seeds measure training variation on the same sources. They add no new
  cameras or independent events; the full report shows that variation rather than
  claiming a single score proves generalization.

These models classify scenes/clips and do not output bounding boxes or establish
ownership. Duration-dependent inference returns unknown if the required timed
history is missing. Independent label review, new camera development/test cohorts,
and better person/object temporal evidence remain necessary for the unresolved
objectives and threshold transfer.

## Verification and artifacts

All six checkpoints contain finite, changed visual tensors. The final audit
recomputed metrics for **46,926 held-out candidate predictions** and 29,394
development predictions, checked development selections and input identities,
and tested real checkpoint-backed inference for both backbones. The largest
observed live-versus-saved score difference in those checks was below 0.000001.

Training used float32 weights with bfloat16 autocast. A numerical audit found
batch-shape sensitivity in AMP inference. Final evaluation and the default
inference interface therefore use float32 with TF32 disabled for every model,
seed and method. Weights and routes were kept fixed; thresholds were refitted on
float32 development scores only. Original AMP results remain available. No latency
experiment was run.

Final repository validation passed: **153 unit tests**, Ruff lint and formatting
for all 53 Python files, whitespace checks, report/document link checks, and
SHA-256 verification of the four primary checkpoint files against the bundle.

- [Protocol and execution notes](clip_training_protocol_20261015.md)
- [Complete results and figures](../results/clip_finetuning_20261015/report.md)
- [Research model card](../results/clip_finetuning_20261015/model_card.md)
- [Checkpoint paths, hashes, routes and thresholds](../results/clip_finetuning_20261015/model_bundle.json)
- [Verification evidence](../results/clip_finetuning_20261015/verification.json)
- [Reproduction and inference guide](../docs/training.md)
