# Causal video-event pilot — 16 October 2026

This study audits all five objectives and trains new smoking/unattended-item heads over frozen pretrained MobileCLIP2-S2 and SigLIP2-B/16-224 encoders. It is not another full visual-backbone fine-tuning sweep. No latency tests were run. The pilot gives mixed recognition changes and does not establish reliable camera transfer. MobileCLIP improves modestly on CCTV smoking while SigLIP regresses. The most useful fresh-source result is the explicit item-state rule; its two-event result and roughly 35-second median delay require substantially more validation.

## What changed

- Prepared 936 existing video recordings and 8,931 modeling windows/static examples after excluding 10 known shared-film training clips. Counts include highly correlated overlapping windows and unknown periods, not independent events.
- HMDB now contributes 530 training clips; its grouped 205-clip evaluation is retrospective. Movie/web actions do not establish CCTV transfer, and primary action labels are non-exhaustive smoking controls.
- Added three author-hosted AVSS2007 videos and published temporal unattended-object annotations. EASY/MEDIUM form the primary fresh subset; HARD is a separately reported censored diagnostic. Camera identities are not verified, so all are conservatively one source block.
- Compared scene and multiple-region linear heads, causal GRUs, automatic track geometry, training-person supervision, training-only hard-negative mining, and open-vocabulary detector geometry. Three seeds are retained for learned nonlinear/instance heads.
- Epochs and routes maximize worst development-source AP, with mean AP breaking ties. Alert thresholds/rules freeze using development recordings before fresh scoring.

## Error audit and component findings

- Reviewed 117 deliberately selected error cases across the five objectives. The review is provisional assistant triage; it is not independent human verification or a representative estimate of error causes.
- CCTV smokers have median largest annotated person height 11.2 pixels at 224 input. The fraction of positive frames with less than half of every target in the selected crop falls from 40.2% to 0.3%. This is geometric coverage, not recognition accuracy; detector resolution and crop policy both change.
- On source CAVIAR training-camera item boxes, track recall at IoU 0.5 changes from 0.0% to 48.5% with Grounding DINO. The annotations omit other bags, so unmatched item tracks cannot be interpreted as proven false positives.
- Coughing, drinking and eating are useful smoking controls. Loitering and unattended state require identified history. Static sleeping photographs remain insufficient to validate persistent public sleep.

## Smoking — previously inspected recordings

| Backbone | Source | Cases | Pilot scene baseline AP | Selected pilot AP | Previous bundle AP on same cases | Recall / FPR at development 5% frame-FPR-budget point |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| mobileclip2-s2 | smoking_cctv | 646 | 0.626 | 0.725 | 0.690 | 16.5% / 7.9% |
| mobileclip2-s2 | hmdb51_action_transfer | 205 | 0.842 | 0.908 | 0.697 | 86.7% / 2.9% |
| siglip2-b16-224 | smoking_cctv | 646 | 0.717 | 0.688 | 0.722 | 2.3% / 1.6% |
| siglip2-b16-224 | hmdb51_action_transfer | 205 | 0.888 | 0.778 | 0.762 | 66.7% / 1.7% |

Selected mobileclip2-s2 smoking route: `person_supervised_hardneg_seed_42`.
Selected siglip2-b16-224 smoking route: `region_temporal_seed_42`.

The previous bundle used eight HMDB views and no HMDB training; the pilot uses sixteen views and grouped HMDB adaptation. This comparison combines data and method changes. The pilot scene baseline shares its training/data/views with the new heads. CCTV case IDs/labels are unchanged and that cohort remains retrospective.

## Unattended events — primary fresh subset

These are **two annotated events**, not hundreds of independent samples. Overlapping decision windows are used to compute frame-state AP separately. The global rule uses development-only thresholds, two-second persistence, one-second release, and ten-second cooldown. Alerts match only if they start inside the labelled event; an early continuing alert does not become a true positive later.

| Backbone | Fixed route | Protocol | Detected events | Normal false / repeated alerts | Reviewed normal minutes | False alerts / normal camera-hour | Median detection delay (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| mobileclip2-s2 | `region_linear` | Global rule | 2/2 | 1 / 3 | 2.76 | 21.7 | 25.8 |
| mobileclip2-s2 | `region_linear` | Known AVSS 30-second state gate | 2/2 | 0 / 0 | 2.76 | 0.0 | 36.3 |
| mobileclip2-s2 | `region_linear` | State gate alone | 2/2 | 0 / 0 | 2.76 | 0.0 | 34.8 |
| siglip2-b16-224 | `grounded_track_temporal_seed_42` | Global rule | 2/2 | 6 / 3 | 2.76 | 130.3 | 12.1 |
| siglip2-b16-224 | `grounded_track_temporal_seed_42` | Known AVSS 30-second state gate | 2/2 | 0 / 1 | 2.76 | 0.0 | 34.8 |
| siglip2-b16-224 | `grounded_track_temporal_seed_42` | State gate alone | 2/2 | 0 / 0 | 2.76 | 0.0 | 34.8 |

The known-criteria gate requires the same detected item to be stationary for 30 seconds and its previously associated owner absent for at least 0.5 seconds. It is a separately specified source-criteria diagnostic, not a test-fitted threshold. Its owner relation is a proximity heuristic and can fail. The state-only prediction is independent of the CLIP backbone; duplicate rows make the common comparison explicit.

Zero false alerts over a few minutes cannot establish a reliable one-alert-per-hour operating point. Global transfer, known source criteria, and camera calibration are separate protocols. HARD has a censored annotation tail and is not included in this primary table.
The alert-count column separates false alerts during normal footage from repeated alerts during an event. The SigLIP classifier plus state gate still repeats one alert; the state rule alone does not.

## Disjoint normal-recording calibration — historical diagnostic

The complete classroom video7 and walkway video2 recordings are reserved for calibration in this comparison and excluded from both evaluation arms. Only their existing reviewed normal periods provide offsets. These are scene groups with provisional labels, not fresh cameras or a well-powered calibration set. AVSS calibration is unavailable because there is no independent normal recording and camera identity is unverified.

| Backbone | Protocol on identical remaining recordings | Events detected | Normal false alerts | Normal minutes | False alerts / normal camera-hour |
| --- | --- | ---: | ---: | ---: | ---: |
| mobileclip2-s2 | global_same_recordings | 0/4 | 0 | 2.28 | 0.0 |
| mobileclip2-s2 | calibrated_disjoint_normal_recording | 0/4 | 0 | 2.28 | 0.0 |
| siglip2-b16-224 | global_same_recordings | 0/4 | 4 | 2.28 | 105.1 |
| siglip2-b16-224 | calibrated_disjoint_normal_recording | 1/4 | 1 | 2.28 | 26.3 |

## Limits and next evidence needed

- Historical event recall uses conservative provisional onset windows. Alerts in unreviewed transition gaps are unknown, not established false alarms; some precede a positive interval. Independent boundary review is needed before treating those cases as certain detection failures.
- Public source diversity and independent event counts remain small despite substantial downloads and many frames. Smoking still lacks a new independently labelled surveillance-video source; the CCTV evaluation is static and retrospective.
- Historical unattended intervals and public-sleep labels remain provisional assistant review. MEVA is an abandonment-action proxy; it does not certify persistent unattended state. Its metrics are kept separate.
- Known film-part aliases are purged, exact decoded-pixel overlap is checked, and title/recording groups remain separated. Actor/movie identities beyond supplied names are not independently established.
- Hard-negative runs can select a checkpoint before mining begins at epoch four; epoch metadata expose this. Mining is not claimed effective merely because its run was configured.
- Existing CLIP backbone weights remain unchanged in this pilot. New weights are region/temporal heads; detector weights are pinned public pretrained checkpoints.
- Littering, loitering and public sleep have an error audit but no newly trained temporal detector in this first pilot. Their next step is event/track annotation and a compatible detector rather than treating posture or rubbish presence as ground truth.

## Artifacts

[Full run and tables](../results/video_pilot_20261016/report.md).


- `comparison.csv`: all source-separated operating points for every retained seed/candidate.
- `events.csv`: strict event metrics, including censored/all-three and known-criteria diagnostics.
- Per-backbone `joint_selection.json`, checkpoints, development predictions, `event_policies.json`, raw `predictions.jsonl`, and `results.json`.
- `error_audit/`: source score shifts, crop coverage, component audits, original-resolution review flags and the independent-review queue.
- Data `input_identities.jsonl`, `data_verification.json`, `source_guard.json`, canonical verified/modeling manifests and AVSS source receipt.
- `verification.json` and `unit_tests.log` document the final replay/checks when complete.
