# Inspection of evaluation_20261011T120315_062384Z

Execution completed, but this run cannot establish a deployment winner. Two evaluation issues need correction: case-sensitive parsing of SmolVLM labels and repeated CAVIAR source moments in the selected cohorts. This inspection preserves the original run files and metrics.

## Execution and verification

- Status: `complete_for_available_tasks`; all three stages exited with code 0 and all twelve model/cohort summaries report zero inference errors.
- Four image models, three cohorts, eight inputs per cohort: 96 recorded measured inference attempts, plus excluded warmups. VadCLIP was not requested.
- Quick mode, frame inputs, 16 CPU threads, duty cycle 1.0, no rate cap. Every model summary records zero pacing sleep.
- Host: AMD Ryzen Threadripper PRO 7975WX, 32 physical / 64 logical CPUs, about 251 GiB RAM. These timings do not measure a Ryzen AI Max+ 395 deployment.
- The benchmark stages total 286.43 seconds (4 minutes 46 seconds), excluding preparation and parent-side analysis/reporting.
- Each model's prediction identities and counts exactly match its planned cohort. All saved classification metrics reproduce from the original prediction rows. Experiment fingerprints verify, and the recorded source-code hashes match the inspected checkout.
- Saved group-bootstrap results and paired differences reproduce exactly using the recorded 200 draws and seed 42. Development-fitted calibration also reproduces exactly. These computations ran on a temporary copy.
- Unattended-item dev/test recording groups and recorded image byte hashes are disjoint. This verifies recorded cohort separation, not every original remote image file: the smoking corpus is unavailable here, and locally derived CAVIAR files have different byte hashes from the server's derived PNGs.

## Findings

### 1. SmolVLM's reported abstentions are a parsing defect

All eight smoking responses are ` Normal.`. All sixteen unattended-item responses are ` Unattended_items.`. The parser trims whitespace and a final period but checks the label against lowercase definition keys without normalizing capitalization. See [the parser](../../anomaly_eval/models.py).

Consequently, saved prediction coverage is 0 even though all responses contain an unambiguous configured label. This is not evidence that the model answered `unknown` or failed inference.

A diagnostic interpretation that normalizes only exact-label capitalization gives:

| Cohort | Coverage | Accuracy | Event recall | Event F1 | False-positive rate |
|---|---:|---:|---:|---:|---:|
| Smoking development | 1.00 | 0.50 | 0.00 | 0.00 | 0.00 |
| Unattended development | 1.00 | 0.25 | 1.00 | 0.40 | 1.00 |
| Unattended test | 1.00 | 0.25 | 1.00 | 0.40 | 1.00 |

These are post-hoc parsing diagnostics, not a fresh corrected inference run. They still use the duplicated selected samples described below. SmolVLM would predict normal for every smoking frame and anomaly for every unattended-item frame, matching constant-prediction baselines rather than demonstrating useful discrimination. JSON compliance remains a separate issue; accepting an exact label must not turn arbitrary prose or multiple labels into a valid prediction.

### 2. Both unattended-item positives repeat one source moment

In development, the two positive inputs are raw-derived and bundled versions of `caviar_LeftBag_0_3_3.png`. In test, they are versions of `caviar_LeftBag_AtChair_2_3_3.png`. The two positive responses have identical continuous scores within each CLIP model.

Locally reproduced derived frames and their bundled counterparts have identical decoded RGB pixels. The locally derived file bytes differ from those recorded on the server, so this is a local reproduction plus recorded source-name/score evidence, not a byte-for-byte audit of remote PNGs.

The preparation code merges by file SHA-256. Equivalent frames encoded differently can therefore survive the merge. See [dataset preparation](../../anomaly_eval/datasets/prepare.py). The selected unattended cohorts contain eight file paths but only seven distinct named source moments; positive support is one source moment from one positive recording per split, rather than two independent positive observations.

Merge raw and bundled examples by a reliable source/protocol/time identity, and use decoded-pixel checks to catch encoding-only differences. Detect conflicting labels before merging and preserve source groups across splits. Rebuild the prepared dataset before rerunning; reusing it with `--skip-prepare` would retain these selections.

### 3. The other three models detect no selected events

MobileCLIP2-S2, SigLIP2 and MiniCPM-V predict normal for every selected input: 24 normal predictions per model, with zero event recall/F1. Their smoking accuracy of 50% and unattended-item accuracy of 75% match the always-normal baseline.

Development-calibrated CLIP thresholds also have zero unattended-item test recall. The maximum-development-F1 thresholds cause 3/6 false positives for MobileCLIP and 4/6 for SigLIP on test. The development-FPR-budget thresholds cause no test false positives and still detect no test events. Thus changing from raw argmax to the saved calibrated thresholds does not rescue this run.

Only six development negatives were used: the empirical FPR resolution is 1/6, about 16.7%. This cannot establish a reliable 5% false-alert operating point. Source-group intervals are exploratory with three CAVIAR recordings and four smoking recordings.

## Runtime measurements

The following are the saved unattended-item **test** measurements; quality fields describe the original parser.

| Model | p95 inference seconds | Peak RSS GiB | Saved prediction coverage | Raw event F1 |
|---|---:|---:|---:|---:|
| MobileCLIP2-S2 | 0.054 | 0.91 | 1.00 | 0.00 |
| SigLIP2-B/16-224 | 0.056 | 1.23 | 1.00 | 0.00 |
| SmolVLM-500M | 3.840 | 3.10 | 0.00 | 0.00 |
| MiniCPM-V 4.0 | 3.244 | 18.08 | 1.00 | 0.00 |

CLIP models pass the sampled one-frame/sec latency and memory screen on this host. That indicates computational feasibility, not successful anomaly detection. MiniCPM's largest RSS across this run was about 18.45 GiB. Its logs contain a processor deprecation warning, with no recorded inference exception.

## Coverage and next evaluation

Smoking is development only. Unattended items have preliminary CAVIAR dev/test coverage. Littering, loitering and rough sleeping were requested but have no confirmed cohorts; `full_task_coverage` is false. The 7,200 proxy-development candidates are not confirmed event test labels.

Correct label parsing and duplicate selection first, then regenerate preparation and run the uncapped evaluation on confirmed cohorts. Compare frame and past-window inputs on development data, keeping prompt/preprocessing choices fixed before a fresh held-out recording evaluation. Temporal/contextual evidence and small-object visibility need review; this frame-only pilot does not establish their adequacy.

## Code corrections after this inspection

The parser now accepts unambiguous exact labels case-insensitively and separately records explicit unknowns, invalid responses and response format. Preparation merges equivalent decoded pixels and matching CAVIAR source moments, prefers bundled canonical copies and preserves connected recording groups. Saved sample identities include pixel hashes and origins; coverage includes groups per label. Prepared schema 3 prevents reuse of the older duplicated selection with `--skip-prepare`.

Regression tests now use the observed capitalized replies and alternate PNG encodings, exercise contradictory labels and cross-split duplicates, check recording aliases and historical-group protection, and validate saved response diagnostics. Original predictions, summaries and statistics in this folder remain the evidence from the old code. Corrected results require a fresh preparation/inference run; these patches do not make this pilot a comprehensive five-event or live-deployment evaluation. See [the evaluation guide](../../docs/evaluation.md).
