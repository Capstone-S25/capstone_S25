# Video-event pilot protocol — 16 October 2026

## Purpose and status

Test whether causal video features and automatic person/item tracking address
the recognition and threshold-transfer failures in the previous CLIP studies.
The first trained targets are smoking and unattended items; the error audit
covers all five objectives. Existing studies remain immutable. No latency test.

## Evidence policy

- Existing camera/source tests have already been inspected. Any reuse, including
  a newly grouped HMDB experiment, is **retrospective** and cannot establish
  untouched-source generalization.
- Group recordings by camera, source title and synchronous recording session
  before generating overlapping windows. A recording must never cross folds.
- New final recordings, if available with usable annotations, stay outside
  training, early stopping, threshold fitting and qualitative error selection.
- Source annotations retain their actual meaning: HMDB primary-action labels
  are not exhaustive smoking labels; MEVA abandonment actions are not labels of
  persistent unattended state; apparent sleep is not housing status.
- Assistant visual review is provisional and is not independent human review.
  Export a review queue for independent verification rather than claiming it.

## Controlled experiment

Use the same recordings and sampled timestamps for MobileCLIP2-S2 and
SigLIP2-B/16-224. Frozen pretrained visual encoders feed either a static pooled
linear baseline, a causal temporal head, or that temporal head with automatic
track features. Additional visual fine-tuning is deferred until this controlled
experiment identifies a representation bottleneck. Select epochs/routes using
development recordings only. Retain all seeds and regressions.

The concrete screen uses a fixed-C=1 scene linear baseline, a fixed-C=1
multiple-region linear ablation, and region GRUs without/with the 18 track
descriptors. GRUs use seeds 42/43/44, at most 60 epochs and eight stale epochs.
Epoch and primary-route selection maximize the weakest development-source AP,
with mean source AP breaking ties. This rule is frozen before pilot scores.
HMDB clips use 16 uniform views; continuous recordings use 2 Hz. The two
pre-existing ABODA entrance recordings develop the persistent-item heads.

For the optional scene-group calibration comparison, reserve the complete
`aboda_video7` recording for the classroom group and `aboda_video2` for the
walkway group. Use only their pre-existing reviewed normal intervals. Exclude
both complete recordings from **both** arms of the calibration comparison.
They remain in the separately reported global-threshold all-recording result.
Shift the development-fitted threshold by the difference between calibration
and development normal-score 95th percentiles. Report the actual short normal
exposure; this is not a guarantee of a 5% deployment false-positive rate.
AVSS has no independent normal calibration recording or verified camera IDs,
so the calibrated-new-camera result is unavailable there.

Pre-score amendment after the crop audit: add a smoking person-supervised head
using only matched automatic regions from the training-camera COCO annotations,
alongside frame/clip multiple-instance supervision. Evaluate using automatic
regions only. This is a labelled-region ablation with additional annotation
supervision, not an input-identical head ablation. Motivation: the old crop
misses more than half of every annotated smoker in 40.2% of held-camera positive
frames; 71.8% have all annotated smoker heights below 12 pixels at 224 input.
No pilot scores or fresh AVSS images have been inspected for this amendment.
The person head has baseline and hard-negative variants, each with the same
three seeds. Mining begins at epoch four and uses training scores/labels only,
upweighting difficult normals within each source while retaining its original
class/source mass. No test-derived negative is added to training.
Person regions include a contextual full-body crop and a deterministic upper
55%-body crop per person to retain small hand/mouth details. This is a geometric
heuristic, not a validated face/hand detector. Region caps are four persons and
two items per frame, with scene fallback when absent; retain region coverage.

Pre-score component amendment: the closed-vocabulary detector/association
pipeline has zero matches to 136 sampled source item boxes on the CAVIAR
training camera at IoU 0.3 or 0.5. Evaluate pinned Grounding DINO tiny with the
fixed query `person . bag . backpack . suitcase . cardboard box . package .`,
box threshold 0.25 and text threshold 0.20. First audit its training-camera
component recall; if it recovers source items, retain an additional three-seed
temporal geometry ablation using its causal tracks. Keep existing CLIP visual
features fixed to isolate the geometry change. The new detector's item label
1001 is a private package/bag class, not a COCO category. Its scores are not
calibrated probabilities. No model scores or fresh images motivate this change.

Automatic regions use the existing pinned COCO Faster R-CNN weights. A new,
explicitly documented detector resolution and low-score threshold may be used
for tracking; this is common to every pilot model. No annotation box chooses
an inference crop. A simple documented association tracker is a pilot baseline,
not ByteTrack and not ground-truth ownership. Include missed detections in
end-to-end results. All window inputs end at the decision timestamp.

## Event evaluation

Before fresh scoring, also freeze an annotation-criteria diagnostic for AVSS:
in addition to the unchanged global alert policy, report a shared Grounding
DINO state gate requiring the same observed item to be stationary for at least
30 seconds and its previously associated owner absent for at least one sampled
step (0.5 seconds). Compare the state rule alone and each fixed classifier
gated by that state. This uses the authors' published criterion, not test-score
tuning. Keep it separate from global-rule transfer; old provisional annotations
do not consistently impose AVSS's 30-second stationary criterion. Missing owner
history yields no candidate alert and remains an end-to-end miss if appropriate.

AVSS acquisition amendment, before pilot scoring: EASY and MEDIUM have event
ends within their decoded timelines and form the primary fresh event subset.
HARD has an annotation end beyond the supplied decoded file and a different
encoded frame rate; retain its observed prefix as a right-censored diagnostic.
Report both clean sequences, the censored sequence and the qualified all-three
aggregate. Do not rescale event onsets or choose this subset from model scores.

Count alerts after a development-frozen persistence/merging rule. Match alerts
one-to-one to annotated events. Count repeated alerts, early alerts and alerts
outside event intervals. Report event recall, false alerts per camera-hour of
reviewed normal exposure, and delay from annotated onset. Unknown/unreviewed
time must never contribute normal exposure or become a negative training label.
Separately report annotation coverage and all source/camera results. Confidence
intervals use recording groups; small numbers of cameras/events are explicit.

Report global-threshold unseen-camera transfer separately from calibration
using a disjoint normal calibration recording. If a camera has no separate
reviewed normal recording, its calibrated result is unavailable. Do not fit
camera thresholds from its evaluation negatives. Clip-level action experiments
remain clip experiments and cannot report false alerts per camera-hour.

## Completion evidence

Source-identity amendment before held-out scoring: normalized raw HMDB title
prefixes leave The Departed Part 1/Part 2 across training/development. Purge the
ten training clips sharing that canonical title with development, retaining
the original manifests and all earlier fits under `prefix_split_diagnostic/`.
Final heads are fitted again using `modeling_windows.jsonl`. No test clip is
excluded by this guard in this release. Canonicalization covers explicit
part/pt/clip/scene/episode suffixes and does not verify every movie/actor alias.

Save data identities, splits, raw scores, selected checkpoints, development
thresholds, error-review boards, event accounting, tests and a findings report.
Any inaccessible source or missing independent annotation is listed as an
outstanding limitation, not replaced by inferred ground truth.
