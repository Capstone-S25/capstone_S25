# CLIP adaptation protocol

Study started 15 October 2026. This protocol is fixed before the full training
study opens held-out results. Models: MobileCLIP2-S2 and SigLIP2-B/16-224 at the
same pinned pretrained revisions as the preceding evaluation. The VLMs and
supervised VadCLIP smoking model remain comparison evidence, without retraining.

## Objectives and labels

Retain the five definitions in `anomaly_eval/config/anomalies.json`. Each example
supervises **only its annotated target**. A normal smoking example is not a
negative example for the other four objectives. The classifier has five binary
heads, allowing multiple events rather than forcing mutually exclusive labels.

Smoking uses annotated CCTV Smoking/Not Smoking frames and Stanford40's smoking
action versus drinking, blowing bubbles, brushing teeth, phoning and texting.
Stanford40 supplies primary-action labels, not exhaustive annotation of every
person in an image. Public images are obtained from the original author's
[Stanford40 dataset](http://vision.stanford.edu/Datasets/40actions.html), using a
recorded public mirror for the image archive because the original image server
was slow.

Littering uses the [MIVIA-IWDD release](https://github.com/MiviaLab/IllegalWasteDumping):
author-labelled waste dumping and hard-negative videos. Uniformly sample eight
frames over the whole video. **The target is video-level classification**, not a
claim that every sampled frame is a dumping event. Onset annotations are retained
but do not determine which test frames the model receives. Static waste dumping
includes bulky items as well as small rubbish; it is related to, and broader than,
pedestrian littering.

Loitering uses [Street Scene](https://www.merl.com/research/?research=license-request&sw=StreetScene)
track annotations and the anomaly-type mapping in Tables 4–5 of the authors'
paper. The paper uses one-based event indices; annotation track IDs are
zero-based. Use a 15-second endpoint grid plus target-track endpoints, with
20 seconds of past context in four chronological frames. Exclude other anomalous
events as negative examples. This is a **supervised research re-split of the
original Test recordings**, not the official unsupervised benchmark. The source
has a single camera and therefore cannot establish unseen-camera generalization.

Unattended items reuse the expanded chronological review intervals, retaining
their **provisional assistant-review** status. Use four native past frames over
20 seconds, rather than a reduced contact sheet. Human verification remains
necessary. Existing local test results were historically inspected; new training
results on these sources are a retrospective source-family holdout.

Rough sleeping uses a small new photo study from openly licensed Wikimedia
Commons images, visually reviewed before training. Positive labels require
visible apparently sleeping in public, or lying with bedding there. Controls
include awake people on benches, awake people lying on grass, indoor sleepers,
and clear empty context. Ambiguous rest/sleep, unclear public/private context,
paintings and animals are omitted. Labels concern visible posture/context;
**housing status is neither labelled nor inferred**. Labels are provisional
assistant review, not independent human ground truth. The original synthetic
lying data is not promoted to rough-sleeping supervision.

## Generalization safeguards

- CCTV smoking: train camera c4, develop on c2, test on c1, keeping all dates from
  the same camera together. Historical images are not described as untouched.
- Stanford40: retain the official photo test split. Split the official training
  pool into train/development groups; conservatively connect perceptually similar
  images first. Remove related training images and previously used debug-test
  images from held-out claims. The photo domain contributes training images, so
  its official image holdout is not unseen-dataset transfer.
- IWDD: retain the official test set, excluding training videos whose first/last
  perceptual hashes connect them to test. Split remaining source groups for
  train/development. Camera identities are not supplied; perceptual clustering
  cannot prove complete camera/subject independence.
- Unattended: train CAVIAR/ABODA lobby, develop on ABODA entrance, test on ABODA
  classroom/walkway. Related views stay together.
- Street Scene: whole recordings stay together. Photo study: whole author and
  perceptual-neighbour groups stay together. Each split must contain both classes.
- Validate decoded pixel identities and view ordering; reject source/pixel leakage
  before inference. Cache frozen features by pixel hashes, model revision and
  input/pooling policy; fine-tuned features never reuse that cache.

The split search uses annotation counts only to ensure class support and balance.
It does not use model predictions. Perceptual hashes are conservative duplicate
screening, not a guarantee against all related sources.

## Training and selection

Compare a pretrained last-frame text decision, an L2-regularized frozen-feature
linear probe, and actual last-stage visual fine-tuning. The probe and fine-tuned
head use the same ordered temporal representation: concatenate the last, mean,
max and last-minus-first normalized image embeddings, then normalize the vector.
For single images, the temporal difference is zero. Temporal pooling changes the
earlier contact-sheet protocol; improvements over the last-frame baseline combine
learned classification and temporal context. Fine-tuning versus probing compares
the same input/pooling protocol.

Initialize fine-tuning from the development-selected probe. Train MobileCLIP's
last visual stage/final convolution/projection, or SigLIP's last visual transformer
block/pooling head, plus the five classifier heads. Freeze the text encoder, early
vision layers and all batch-normalization statistics. Balance sampling by task,
class, dataset source and recording/author group. Preserve scene content with
letterbox preprocessing and mild colour jitter/flip; do not randomly crop away
small cigarettes or unattended objects. Add feature distillation and anchoring to
the pretrained visual parameters to limit drift.

Pre-register probe C values 1/10/100; visual learning rates 1e-5/3e-5 for seed 42;
head learning rate 3e-4; maximum ten epochs with three-epoch development patience.
Selection uses macro task AP, averaging dataset-source AP within each task.
Freeze selected checkpoints and operating thresholds before test evaluation.
Also freeze a per-target routing choice among pretrained/probe/fine-tuned scores
using development AP only; ties prefer the simpler model. Do not force a tuned
model over a better development baseline.

Run seeds 43 and 44 using the visual learning rate selected by seed 42's
development results. This replication plan is set before examining test results.
Report all methods/seeds, their variation and source-group uncertainty; random
seeds do not create new independent cameras. Do not select a lucky seed from its
test performance. Seed 42 is the primary deployable candidate.

GPU training is unpaced, with float32 weights and bfloat16 autocast, two concurrent
model jobs, batch 32 samples, six decoding workers and eight CPU threads per job.
No latency study is included. Dataset servers' rate limits are respected during
downloads; they do not limit model training or inference.

Outputs are new checkpoints, raw held-out scores, training histories, development
selection records and a source-qualified report under
`results/clip_finetuning_20261015/`. An improved held-out score supports the tested
source conditions; it cannot guarantee deployment generalization across all five
events. The trained model is a scene/clip classifier, not an object-box localizer.

## Execution notes added after training

All six model/seed runs finished on 16 October 2026. The scheduler eventually
allowed up to three concurrent GPU jobs. Initial seed-42 jobs used convolution
autotuning; later replications disabled it to avoid repeated searches for variable
frame-batch shapes. Executed source snapshots are retained with their hashes.
Neither change alters labels, splits, the selection plan or the training budget.

An inference consistency audit found bfloat16 scores sensitive to batch shape.
The final quality evaluation therefore uses float32 operations with TF32 disabled
for **all** methods and seeds. Trained weights, hyperparameters and previously
development-selected routes remain fixed. Only operating thresholds are refitted,
using float32 **development** scores. Original AMP artifacts remain available.
This is a numerical consistency correction, not another test-driven model search.
