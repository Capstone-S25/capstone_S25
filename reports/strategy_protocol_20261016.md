# Expanded data and strategy study

Started 16 October 2026, before opening new source test predictions. Preserve the
previous training study and its manifests. Its test results are historical, not
fresh evidence for selecting this study's strategies.

Acquire public event-labelled smoking images, author-labelled loitering
trajectories, and abandoned-package video/annotations where accessible. Preserve
source documentation, download identities, source labels, missing-file receipts
and source qualifications. Do not map unrelated abnormal events to the five
targets or treat all unlabelled targets as normal. Keep new rough-sleeping image
candidates separate until visually reviewed; review is provisional assistant work.

Use old training/development splits for strategy development, plus new training
sources after duplicate screening. Preserve an available new official test split;
otherwise assign whole source/near-duplicate groups before fitting models. Exclude
new training examples related to any old holdout images. New-source test images
must not enter fitting, early stopping, hyperparameter or strategy selection.

For both MobileCLIP2-S2 and SigLIP2-B/16-224, compare frozen full-scene probes,
automatically detected crops plus full-scene context, explicit ordered-view
features, and pretrained/fine-tuned visual weight interpolation. The detector is
the fixed torchvision COCO Faster R-CNN ResNet50 FPN v2 model, confidence 0.5,
with contextual padding. Missing detections fall back to the complete scene;
include failures in evaluation. Ground-truth boxes are not used for the main crop
comparison. All final feature evaluation uses float32 with TF32 disabled.

Fit target-specific linear probes with C in 0.1/1/10. Fit small task-specific
regularized MLPs as another fixed candidate. Select on development AP, averaging
sources within each task. Include a source-wise selection score and retain all
candidate results. Use seed 42 for this exploratory strategy screen. It does not
replace the previous three-seed study or prove deployment readiness.

Visual interpolation coefficients are 0, 0.25, 0.5 and 1 between pretrained
parameters and the previously trained seed-42 parameters. Fit a new classifier
for the resulting representation on training labels; select coefficients and
regularization on development data only. The old trained weights used only the
old training split. Any interpolation strategy remains an experiment, not an
assumed robustness improvement.

Evaluate loitering trajectory descriptors using whole-recording/day groups from
the newly acquired thermal annotations. Keep the thermal modality and annotated
tracks explicit: this is a trajectory-component benchmark, not an end-to-end RGB
CLIP result or a test of an automatic tracker's detection accuracy.

Calibrate binary thresholds on development data only. Report AP, precision,
recall, F1, FPR, source counts, detection coverage and missing/uncertain labels.
Do not convert an offline video/whole-trajectory score into an online detection
claim. No latency experiment or artificial model pacing is included.

## Execution additions before external transfer scores

The primary strategy screen uses 33 candidates per target/backbone: endpoint,
pooled and ordered features, automatic-region/context features, sparse detector
geometry, four visual-weight interpolation levels, a zero-shot reference,
regularized linear probes and two task-specific MLP representations. Primary
routes were frozen before the new 224-photo holdout was scored.

An additional worst-development-source selection is retained as a secondary
analysis. It was specified after the primary photo results were available and
before opening HMDB/MEVA transfer scores. Thus its photo result is exploratory;
it does not replace the primary route or become a new untouched photo claim.
Transfer-family hyperparameters are fixed from saved development results.

The newly acquired HMDB51 smoking/drink/eat/chew/talk/sit/stand subset is a whole
unseen-source action-classification transfer cohort, not the official HMDB51
51-class benchmark. It uses eight uniform views per complete clip. Other action
labels are not exhaustive annotations of every person's smoking status. All 906
clips decode successfully with sequential reading; the failed indexed-seek
preparation is retained as a diagnostic and never used for model selection.

MEVA contributes 13 available source videos. Fifteen abandonment windows and
twelve matched pre-event windows form an explicitly qualified activity-transfer
cohort. Annotation-end anchoring uses event knowledge to construct this offline
diagnostic; it does not measure continuous event discovery or prove persistent
unattended-item status. One additional annotated video is unavailable in the
public prefix. Related simultaneous views remain source groups.

Classifier serialization uses float32 coefficients. Before final transfer
reporting, recompute operating thresholds from the serialized classifier's
development scores; keep every primary route and fitted weight fixed. The prior
study's bundle retains its original float32-development thresholds. This avoids
using slightly different numeric representations to calibrate and infer.

The focused public-sleep acquisition adds 40 downloaded human-context photos.
Visual review retains 17 positives and 12 normals and omits 11 ambiguous images.
Excluding previously used photographers/related images leaves 28 new-author
photos (17 positive, 11 normal, 25 author groups). These are transfer tests for
the already frozen strategies, not extra training inputs in this round. Labels
remain provisional assistant review. Flickr variants come only from explicitly
linked public author copies with their open-license metadata retained.
