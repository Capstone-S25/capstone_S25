# Expanded data and strategy findings — 16 October 2026

Both CLIP backbones completed a development-selected screen of **330 candidate
configurations** spanning automatic contextual crops, temporal pooling versus
ordered features, sparse detection geometry, linear/MLP heads and visual-weight
interpolation. This round fits new classifiers and evaluates visual blends of the
previously trained backbone; it is not 330 separate visual fine-tuning runs.

Training grows from 3,258 to **3,973 examples**, with **1,812 development examples**.
The remaining new sources provide transfer tests and a separate trajectory study.
No latency testing was performed. The [complete source-separated report](../results/strategy_study_20261016/report.md)
contains all operating points and candidate-family comparisons.

## Data acquired

- **1,120 smoking/control photos**, with 1,118 retained after screening:
  715 train, 179 development and 224 official photo holdout.
- The author-hosted **HMDB51 archive**, extracting **906 clips** across smoking
  and six confusable normal action classes. All clips decode sequentially;
  indexed seeking had produced false failures and was corrected.
- **874,936 thermal human observations / 19,737 labelled trajectories** over
  **31 dates**. The trajectory benchmark retains 17,878 eligible tracks.
- **MEVA annotations and 13 relevant videos**, yielding 15 abandonment-action
  windows and 12 pre-event controls. One further video is unavailable.
- **40 new public-sleep/control photographs**, visually reviewing 29 usable
  cases and omitting 11 ambiguous cases. Author/duplicate screening retains
  **28 new-author test photos: 17 positive, 11 normal, 25 author groups**.

Sleeping-photo labels remain provisional assistant review. Their target is
visible apparent sleep/bedding in public context; housing status is never inferred.
Existing waste-dumping data remain the littering event source. Further searched
collections require an access form or lack an immediately accessible matching
release; garbage-object images were not substituted for event annotations.

## Historical diagnostic gains and regressions

These are **previously inspected holdouts**, so improvements are retrospective
evidence. Values show previous bundle → new primary development-selected AP.

| Evidence | MobileCLIP2-S2 | SigLIP2-B/16-224 |
| --- | --- | --- |
| Smoking, CCTV | 0.564 → 0.690 | 0.650 → 0.722 |
| Littering / waste dumping | 0.858 → 0.806 | 0.885 → 0.893 |
| Loitering, Street Scene | 0.093 → 0.489 | 0.097 → 0.550 |
| Public sleep, old photos | 0.976 → 0.931 | 0.915 → 0.883 |
| Unattended, old cameras | 0.652 → 0.980 | 0.896 → 0.925 |

Crops were selected for smoking. Sparse geometry was selected for loitering.
Ordered-view probes on the previously adapted encoder were selected for
unattended items. MobileCLIP's waste-dumping route and both historical public-sleep
photo rankings regress. The report retains these outcomes instead of selecting
better test columns after evaluation.

## New image holdout and source transfer

| Evidence | MobileCLIP2-S2 | SigLIP2-B/16-224 |
| --- | --- | --- |
| Smoking: 224-photo holdout | 0.953 → 0.997 | 0.995 → 1.000 |
| Smoking: 906 HMDB clips | 0.605 → 0.576 | 0.685 → 0.664 |
| Public sleep: 28 new-author photos | 0.925 → 0.952 | 0.943 → 0.933 |
| Unattended: 27 MEVA action windows | 0.569 → 0.545 | 0.637 → 0.621 |

The 224-photo holdout shares its collection with new training photos. Its strong
scores therefore concern that image domain. HMDB is a wholly new movie/web action
source; both primary routes slightly underperform their previous bundle there.
MEVA is an abandonment-action proxy, and neither primary route improves its AP.
This round **does not establish broad video-source generalization**.

The 28-photo public-sleep study is small and provisional: MobileCLIP improves its
point estimate; SigLIP slightly regresses. Repeated views from the same photographer
remain grouped. These are not independently human-verified deployment events.

## Ranking and alert decisions remain different outcomes

On historical unattended cameras, MobileCLIP AP rises to 0.980 and SigLIP to
0.925. However, their maximum-development-F1 thresholds give **99.7% and 100%
FPR**, respectively. At the predefined zero-logit point, F1 is only **0.087 and
0.131**. The corresponding previous SigLIP bundle had zero-logit F1 0.749.
Higher AP alone therefore does not justify replacing an existing alert rule.

Historical CCTV-smoking maximum-development-F1 FPR is still **75.9% MobileCLIP /
84.6% SigLIP**. Source-specific calibration and separately labelled deployment
normal recordings remain necessary. No threshold was fitted using test labels.

The separate thermal component study selects a trajectory forest on development
AP 0.863 and obtains held-out-date AP **0.577**, compared with **0.283** for the
duration-only baseline. It uses complete supplied tracks, not an automatic tracker
or RGB CLIP. Its substantial date/season drop and single physical camera limit
the generalization claim.

## Artifacts and validation

- [Full results, figures and operating points](../results/strategy_study_20261016/report.md)
- [Research model card](../results/strategy_study_20261016/model_card.md)
- [Checkpoint and policy index](../results/strategy_study_20261016/model_bundle.json)
- [Protocol](strategy_protocol_20261016.md) and [reproduction/inference guide](../docs/strategy_study.md)
- [Verification](../results/strategy_study_20261016/verification.json)

The audit replays **117,612 predictions**, checks all candidate selections and
input identities, and tests real checkpoint-backed inference for **all five
objectives on both backbones**. Maximum observed live-versus-saved score difference
is below 0.000005. New candidates are saved separately from the previous study.

Final repository checks pass: **159 tests**, lint, formatting of all 71 Python
files, document links and checkpoint hashes. The selected trajectory estimator
is also saved, with its held-out scores replayed.
