# Generalization pilot — speaker notes

## 1. Test the complete system and the source shift

Focused addendum to the 19 October paper and 45-slide history. These are actual new experiments. The data source is scripted smoking-action/gesture evidence, not independently verified CCTV smoking. No new backbone or SmolVLM training and no latency test are claimed.

## 2. Public data acquired, with explicit label limits

Author repository airslab2020/XRF55-repo; public Kaggle RGB/depth/IR release version 6. There are 886,570,077 bytes and 11,840 sixteen-view JPEG inputs. The release README records CC BY-NC 4.0. Numeric subject aliases are conservatively treated as global. Seventeen scene-subject training groups collapse to 15 numeric groups. One test scene and three groups do not represent 240 independent people. Assistant visual review used training clips only. PETS links were unavailable; synthetic and rendered method-output luggage media were rejected. Other exclusions remain in source_audit.json.

## 3. Same selection rules across old, new and mixed data

Old-only2812/new-only340/mixed3152 training cases. Frozen visual encoders, alpha0 only. Float32 CUDA features; normalized last/mean/max/delta pooling and automatic COCO crops. All36 logistic regression fits converged;0 new visual fine-tunes. Selection tie-breaks mean recall, worst AP, mean AP. 1916 held-out cases per encoder include1676 previously inspected historical cases and240 new proxy cases. Source-aware development FPR does not guarantee held-out FPR.

## 4. MobileCLIP: data addition has source-specific effects

All recipe/C choices are development-only. Historical CCTV is retrospective. XRF55 is a one-scene, three-subject action-proxy holdout. All source cohorts and raw/maxF1/budget points are in the report CSV; photo-source results must not be pooled into CCTV accuracy. The genuine pretrained text control and historical last-frame native reference are also retained. This comparison selects within each matched family, not a single fixed head.

## 5. SigLIP2: data addition has source-specific effects

All recipe/C choices are development-only. Historical CCTV is retrospective. XRF55 is a one-scene, three-subject action-proxy holdout. All source cohorts and raw/maxF1/budget points are in the report CSV; photo-source results must not be pooled into CCTV accuracy. The genuine pretrained text control and historical last-frame native reference are also retained. This comparison selects within each matched family, not a single fixed head.

## 6. SmolVLM: prior LoRA does not improve this source

Original AP is 0.3855 versus 0.3573 for LoRA. The existing smoking route is seed 42 / learning rate 0.0001, with 4,341,760 language parameters trained previously. Both development routes were calibrated before any new-source test forward. There is no scene-4 route selection or new fine-tuning. Eight uniform views are the validated prompt maximum; CLIP uses sixteen. Native-versus-cached margin errors are below 0.002. This result concerns scripted action source transfer, not true CCTV smoking accuracy.

## 7. Full-stream tiling reverses the component gain

Ten recordings are processed from time zero to EOF at 2 Hz: 2,512 images and 12,560 views across three arms. Full-frame views plus four 60% corner tiles use class-wise NMS at IoU 0.5. Item-only tiles retain full-frame person proposals. The unchanged rule requires known ownership, 30 seconds of stationarity, 0.5 seconds of owner absence and two seconds of alarm persistence. Primary normal exposure is 205.739 seconds, or 3.43 minutes; its zero-alert one-sided 95% Poisson upper bound is 52.42/hour. Saved results also align the earlier t >= 20 exposure of 165.739 seconds. Forty added seconds are the same footage. Delay measures onset to alarm, not inference latency.

## 8. Keep provisional and censored evidence separate

ABODA has five provisional events and 359 normal seconds; full-frame finds 4/5, while each tile arm finds 2/5. These intervals are not independently reviewed. Right-censored HARD remains separate from the two primary events. All three arms detect its annotated interval, but repeated alarms persist; all-tiles adds a normal false alert. MEDIUM item-track IDs increase from 284 to 497 and EASY from 162 to 371, consistent with spurious or fragmented associations. These counts are not ground-truth identity-switch rates. No rule was retuned on these test streams.

## 9. Replay and native forwards bind the results

Verification.json and native_feature_validation.json passed before report build.192 unit tests passed previously, including5newtilefusioninvariants; currentchecklog isstoredwithrun. Exact decoded-pixel disjointness cannotruleoutperceptual/correlated repetition. No latency benchmark, camera-rate simulation or CPU-only quality cap. Inference uses availableGPU. Convex fitting/metric replay usesCPU.

## 10. Next evidence needed for the five objectives

No new littering, loitering or public-sleep scores are claimed. Prior results and failed transfers remain in the 19 October report. About three normal camera-hours with zero alerts would give a 95% Poisson upper bound near one/hour, subject to that model's assumptions; this does not replace camera diversity. Documentation: reports/generalization_pilot_20261020/report.md and runthrough.pdf. Reproduction: docs/generalization_pilot.md. The existing 19 October paper and 45 slides remain preserved.
