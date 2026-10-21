# Event and small-item improvements — 17 October 2026

Completed development-only smoking operating-point/ensemble selection, causal item-state/alert-memory ablations, and a GPU full-image versus tiled small-object detector comparison. No new backbone training or latency benchmark was performed. Historical smoking, CAVIAR and AVSS sources were already inspected; all reported policy changes on those sources are retrospective.

## Smoking: source-aware low-FPR selection

Each encoder compares 14 saved heads, four equal-weight averages of sigmoid scores within three-seed families, and one equal-weight all-head score average: **19 policies per encoder / 38 total**. Sigmoid values are score transforms, not calibrated probabilities. A single cutoff must meet <=5% empirical FPR in every development source. Selection maximizes worst-source recall, then mean recall, then worst/mean AP. Both encoders’ selections freeze before loading held-out scores. The saved heads and backbone tensors remain unchanged. This reuses the same development set used for previous model selection, so it is exploratory and may overfit that set.

| Backbone / policy | CCTV AP | CCTV recall / FPR | HMDB AP | HMDB recall / FPR |
| --- | ---: | --- | ---: | --- |
| mobileclip2-s2 / previous | 0.725 | 16.5% / 7.9% | 0.908 | 86.7% / 2.9% |
| mobileclip2-s2 / new: person_supervised_hardneg_seed_42 | 0.725 | 10.9% / 5.5% | 0.908 | 86.7% / 2.3% |
| siglip2-b16-224 / previous | 0.688 | 2.3% / 1.6% | 0.778 | 66.7% / 1.7% |
| siglip2-b16-224 / new: all_head_ensemble | 0.720 | 1.8% / 1.6% | 0.964 | 76.7% / 1.7% |

SigLIP’s all-head ensemble improves retrospective HMDB ranking and recall at the same observed FPR. Its CCTV recall remains very low. MobileCLIP retains the same head with a more conservative threshold, sacrificing recall. Per-source development constraints do **not** guarantee the same budget on a new camera. HMDB movie/web clips are not CCTV generalization. The AP-selected alternative is retained in the JSON rather than silently substituted after testing.

## Stationarity smoothing and per-item event memory

The original proximity-owner tracker remains the control. The new variant uses a trailing three-center median and requires one second of sustained displacement to reset stationarity; pending movement blocks an alert. A per-item memory uses spatial one-to-one association across short track-ID gaps and emits once until sustained owner return, movement, or a 30-second absence expires memory. The memory does not invent ownership or extend stationary evidence across missing tracks. All rule settings are fixed engineering choices, not fitted on test labels. The same 30-second stationary/.5-second owner-away criterion and two-second alert persistence apply.

| Policy, AVSS primary (retrospective) | Events | Normal false / repeat alerts | Median onset delay |
| --- | ---: | ---: | ---: |
| original_tracks_item_memory | 2/2 | 0 / 0 | 34.8 s |
| robust_tracks_item_memory | 2/2 | 0 / 0 | 35.1 s |
| original_tracks_global_debounce | 2/2 | 0 / 0 | 34.8 s |
| robust_tracks_global_debounce | 2/2 | 0 / 0 | 35.1 s |

Neither smoothing nor memory improves the two-event primary result; smoothing adds about 0.25 seconds to median delay. Zero false alerts in only 165.74 seconds of labelled normal exposure gives a **one-sided 95% Poisson upper bound of 65.1 false alerts/hour**, assuming a homogeneous Poisson process; it is not a camera-transfer guarantee. At least three hours with zero false alerts would be needed just for that conditional upper bound to fall below one/hour.

On the five historical test events the fixed source-criteria state rule detects 4/5, with zero normal false alerts and one unknown-time alert. Original global debounce already has zero repeated alerts; smoothing plus item memory also has zero. On the separately censored AVSS HARD diagnostic, item memory increases repeats from one to two. It is not promoted as an accuracy improvement. Unit tests verify independent simultaneous items, track-fragment memory, causality and reset semantics; these synthetic checks are not real-world event results. Historical provisional onset windows do not share a consistent 30-second source definition, so their state-rule diagnostic must remain separate from classifier metrics.

## Small-object detection and new data

Downloaded the pinned author archive for [the abandoned-object release](https://github.com/hangsuuuu/video-of-abandoned-object-detection). Three entries explicitly alias existing ABODA videos and are excluded from new-data counts. The other three are 7.6–14.6-second highway videos with small debris boxes; they lack owner histories and 30-second event evidence. They are a component stress test, not three new verified unattended-item events. Source boxes/timestamps are unchanged, frame sizes checked, actual encoded FPS used, and ten positive annotated frames per bundled video are sampled. Out-of-stream timestamps are exposed in the preparation audit; no onset is rescaled.

A fixed full-image plus four overlapping 60%-size tiles is compared with full-image Grounding DINO using the same pinned public weights, generic query, score thresholds and NMS. Source boxes never select inference tiles. The 166 annotated frames produce 830 GPU image views in float32 with TF32 off and no pacing.

| Reference source | IoU | Full-image recall | Full + tiles recall |
| --- | ---: | ---: | ---: |
| caviar_training_component | 0.3 | 69/136 (50.7%) | 117/136 (86.0%) |
| caviar_training_component | 0.5 | 66/136 (48.5%) | 116/136 (85.3%) |
| author_highway_debris_component | 0.3 | 0/30 (0.0%) | 7/30 (23.3%) |
| author_highway_debris_component | 0.5 | 0/30 (0.0%) | 7/30 (23.3%) |

References are partial item annotations. Unmatched detector boxes cannot establish false positives, so no item-precision or event-accuracy claim is made. Adjacent frames are correlated. The CAVIAR cohort is training-source evidence; the new highway collection is an out-of-domain stress test. Tiling has not yet been validated as a full event-tracking replacement.

## Evaluation status and next action

Retain the earlier released bundles as baselines. The SigLIP ensemble is a candidate for a newly frozen camera test, not an automatic replacement. Keep the simple state rule until stronger evidence favors the added tracker complexity. Review and label event ownership/boundaries and matched normal footage from independent target cameras. Littering needs pedestrian disposal actions plus benign placing/carrying; loitering needs complete tracks and operational-duration labels; public sleep needs time persistence and rest/lying controls. More adjacent frames and more hyperparameter trials do not supply that evidence.

## Reproduction and verification

Run `scripts/run_event_improvements.py` in a new output directory to replay ensembles and state ablations. Run `scripts/prepare_item_component.py` once after the pinned author source acquisition, then `scripts/benchmark_item_tiles.py` with the existing GPU runtime and `scripts/report_event_improvements.py` for independent score/alert/component-metric replay. The completed result directory refuses overwrites for scored studies. CPU postprocessing reuses actual saved GPU predictions; only the new tiled detector requires fresh encoder inference.

Verified **5,643 smoking predictions**, all 38 development-policy constraints/selections, **6,390 causal trace decisions** and 29 emitted memory alerts. Component reference matches are recomputed from saved merged predictions. Source/code hashes, acquisition receipt, pixel inputs and raw detector views are retained. Earlier failed preparation attempts were corrected before the completed run; no labels, model weights or selected thresholds were changed in response to held-out results.

Artifacts: `smoking_selection.json`, `smoking_results.json`, per-encoder smoking predictions, `event_results.json`, `event_traces.jsonl`, `component_results.json`, `component_predictions.jsonl`, `protocol.json`, `verification.json`, and data `component_protocol.json`, `component_frames.jsonl`, `tiled_detections.jsonl`, `receipt.json`.
