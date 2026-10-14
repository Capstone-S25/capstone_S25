# Expanded model accuracy findings

Finalized 14 October 2026 (Asia/Jakarta). The preparation/run identifiers remain
`20261013`. All **40 model/cohort/split runs completed**, producing **55,156 new GPU
predictions**, or **13,789 input/protocol evaluations per model**. Shared endpoint
windows and repeated images queried for different definitions are not independent
events. No additional latency experiment was run.

## What changed

Unattended-item coverage increased from **36 to 1,770 endpoints**, with **962
held-out inputs** per protocol: 242 positive and 720 normal. The test split holds
out the ABODA walkway and classroom camera families, covering seven recordings
and five positive intervals. Development contains 808 inputs from three other
camera families. Related recordings stay together, and byte-identical source video
copies are not counted twice. Both single frames and four-frame past-only
20-second contact sheets were evaluated.

Labels were frozen before inference using **assistant timeline review**, with XML
checks for CAVIAR positive intervals. They are **provisional and not independently
human verified**. The old contact sheets covered six seconds; the expanded study
changes temporal context as well as coverage. Only two held-out camera families
remain, so adjacent frame counts must not be treated as independent evidence.

## Main unattended-item findings

| Raw frame decision | Precision | Recall | F1 | False-positive rate |
| --- | --- | --- | --- | --- |
| MobileCLIP2-S2 | 25.31% | 75.21% | 0.379 | 74.58% |
| SigLIP2-B/16-224 | 0.00% | 0.00% | 0.000 | 2.64% |
| SmolVLM-500M-Instruct | 18.58% | 63.64% | 0.288 | 93.75% |
| MiniCPM-V 4.0 | 77.06% | 34.71% | 0.479 | 3.47% |

MiniCPM has the highest raw frame-test F1 and no longer gives constant-normal
answers on this expanded cohort. It still misses nearly two thirds of positive
endpoints. MobileCLIP and SmolVLM produce many false alarms. SigLIP's continuous
frame scores have AUROC 0.675 and AP 0.309 despite zero categorical recall.

Twenty-second windows do not improve every model. MiniCPM's recall falls to
10.74% with 0.14% FPR. SmolVLM reaches 96.28% recall with 97.36% FPR and 14
unanswered inputs. Development-calibrated CLIP thresholds also transfer unevenly:
MobileCLIP's empirical 5% development-FPR point gives 20.28% frame-test FPR and
35.14% window-test FPR. No thresholds were refitted on test labels.

## The other three anomalies were attempted

- **Littering:** 6,667 deduplicated generic action-labelled candidates. No matched
  normal cohort, and source actions do not uniformly establish rubbish disposal.
- **Loitering:** 1,675 person-labelled images. Person presence is observable;
  reviewed duration events are absent. A separate person-presence query checks
  grounding on the same images.
- **Rough sleeping:** 111 synthetic lying/fall scenes. These support posture
  diagnostics, without verified public sleeping events.

The reported values for these three are **target-prediction rates**, not verified
event recall, precision or F1. Low rates may reject unsupported event claims;
high rates cannot establish event accuracy. These exports are useful and remain
development-only. All available eligible source candidates were included after
deduplication, rather than taking the earlier bounded review preview.

The separate source-box pose study has 121 crops, with 44 lying and four standing
crops on test. SigLIP classifies all 48 correctly; MiniCPM finds 40/44 lying and all
four standing controls (F1 0.952). Ground-truth boxes are supplied free, the data
are synthetic, and four negatives are insufficient for a stable FPR estimate.
This component result does not validate rough-sleeping detection.

## Execution and validation

All four models ran concurrently on the RTX 5000 Ada with CUDA, eight CPU threads
per worker, CLIP batch 64 and VLM batch one. CLIP uses float32 and unquantized VLMs
use bfloat16. Inference is unpaced, without a default accuracy RAM cap. Shared
letterbox inputs are bounded to 768 pixels; generation is deterministic with a
64-token budget. MiniCPM retains the one-slice configuration.

All 144 existing tests passed. Recomputing metrics from all 55,156 saved
predictions matches all 40 summaries; all runs used CUDA with zero pacing and zero
runtime errors. SmolVLM has 1,457 abstentions and MiniCPM 45, retained in the
appropriate denominators. See the [verification record](../results/expanded_accuracy_20261013/verification.json).

The prior 3,278-frame smoking development comparison and separately supervised
offline VadCLIP smoking study are retained unchanged. VadCLIP was not repurposed
for the other anomalies or pooled into an image-model ranking.

## Updated artifacts

- [Full accuracy report](../results/expanded_accuracy_20261013/report.md),
  [qualified numeric export](../results/expanded_accuracy_20261013/accuracy_comparison.csv)
  and [exact GPU commands](../results/expanded_accuracy_20261013/state.json).
- [Expanded PDF slides](slides/model_evaluation_expanded_20261014.pdf),
  [editable PowerPoint](slides/model_evaluation_expanded_20261014.pptx) and
  [one-slide summary](slides/summary_slide_expanded_20261014.pdf).
- [Source review decisions](../data/expanded_20261013/review_decisions.json),
  [cohort metadata](../data/expanded_20261013/cohorts.json),
  [speaker notes](slides/expanded_speaker_notes_20261014.md) and
  [slide source hashes](slides/expanded_source_manifest_20261014.json).

The next event-quality evaluation should independently verify the provisional
intervals and include real littering, loitering and rough-sleeping events with
matched controls on fresh cameras. Detector crops and native temporal inference
are separate development experiments. After any tuning, freeze the protocol and
use fresh source recordings rather than repeatedly selecting on this test set.
