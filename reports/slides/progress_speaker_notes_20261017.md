# Evaluation and training progress — speaker notes

Finalized 17 October 2026 (Asia/Jakarta). Historical protocols stay separate.

## 01. Five-anomaly model evaluation

This deck replaces neither historical results nor their selection protocols. It consolidates the four-model comparison, expanded data, six actual visual fine-tunes, 330 strategy configurations, 50 video-pilot heads and 38 new ensemble/threshold policies, plus tracker and tiled-detector ablations. Candidate/prediction counts reuse sources and are not independent events. No deployment winner across all five objectives is established. No new latency benchmark was run. Stage-specific regressions and data limits remain visible.

## 02. Five objectives require different evidence

The user accepted existing definitions and annotations. There is no newly invented universal loitering threshold. Public-sleep labels describe visible sleep/bedding/context, never housing status. AVSS specifies stationary for 30 seconds and an absent owner; historical provisional windows and MEVA abandonment actions do not uniformly implement this criterion. Grounding boxes and source annotations are separate from inferred ownership.

## 03. Four image models and a separate VadCLIP study

Original model names: MobileCLIP2-S2, SigLIP2-B/16-224, SmolVLM-500M-Instruct, MiniCPM-V 4.0 and VadCLIP. Initial comparisons use identical inputs within each image protocol. VadCLIP receives different offline sequence supervision and cannot be ranked directly against the four image models as an equal-supervision five-task benchmark. Later encoder studies do not rerun the VLMs and must not imply their current architecture cannot improve.

## 04. The work progressed from prompts to event evidence

13 Oct: 3,350 protocol inputs per model, including shared endpoint variants; 104 input aliases removed. 14 Oct: 13,789 input/protocol evaluations per model and 40 summaries. Fine-tuning: 7,498 target-specific examples and 12,478 unique view images. Strategy screen: 330 configurations across five tasks and two backbones, including pretrained references; not 330 visual fine-tunes. Video pilot: 50 saved heads, 99,528 saved predictions. Latest: 19 ensemble/threshold policies per backbone, four item-rule ablations and 166 source-annotated images × five fixed views. Counts should not be added as independent events.

## 05. Many files did not provide complete event labels

Weak collections were evaluated as target-prediction rates, not mislabeled as event accuracy. Littering exports name dropping/dumping/placing/throwing, which are not uniformly rubbish disposal. Person boxes lack duration; synthetic fallen/lying people lack verified public-sleep context. Independent human validation of historical assistant-reviewed unattended and sleeping labels remains missing. These sources are useful; their supported measurement differs from event validation.

## 06. Training covered all five targets with uneven evidence

Training totals 3,258, development 1,633, test 1,801, plus 806 Stanford40 test photographs (666 normal / 140 smoking). Classes are target-specific, with unknown labels for other objectives masked. CCTV camera c4 trains, c2 develops, c1 tests; original photos are included in training/development where partitions permit. Public sleep is provisional. Unattended endpoint counts represent only two held-out camera families, not 962 events. Later strategies expand train to 3,973 and dev to 1,812 with additional photos.

## 07. More data was acquired and screened for compatibility

1,120 new smoking photos were downloaded and 1,118 retained; 715 train,179 dev,224 test. HMDB: 109 smoking and 797 confusable action clips before grouped splits. Thermal: 19,737 track IDs, 17,878 eligible complete trajectories, 19/6/6 chronological dates. MEVA: 15 action windows plus12 pre-event controls,10 synchronization groups across7 cameras, not persistent unattended truth. Public sleep:40 downloaded,29 usable,28 author-screened cases(17+/11normal;25 author groups). AVSS HARD is censored and excluded from the primary two-event result. New author clips show highway debris, not owner histories; three other entries explicitly reference ABODA and are not new sources.

## 08. Keep related sources together and freeze decisions

The actual fine-tuning primary is seed42;43/44 show optimization variability. The later video pilot retains all seeds and uses worst dev-source AP then meanAP; final selections precede scoring new AVSS. Its source guard removes10 known film-alias training clips and no test clips. Exact pixels are verified for25,452 pilot inputs; hashes and source-name grouping do not prove all near-duplicate/actor identities are known. On17Oct, AVSS and historical scores are already inspected, so all policy comparisons there are retrospective even though choices still use development scores.

## 09. Accuracy runs used available GPU compute

Expanded image comparisons use shared letterbox preprocessing capped at768pixels, CLIP float32 and unquantized VLM automatic BF16,64-token deterministic generation, MiniCPM one-slice baseline. Those are explicit baseline settings, not a search of optimal native video methods. The fine-tuning audit found AMP inference sensitivity to batch shape; weights and routes were fixed, final evaluation rerunF32 and thresholds refitted usingF32dev only. Model bundles and feature/checkpoint replays record numerical tolerances. The isolated CUDA runtime was validated with actual computation and used for the new detector comparison.

## 10. Ranking, frame decisions and alerts are different

Pilot global rule:2s persistence,1s release,10s cooldown, reset for score gaps>1.1s. Thresholds maximize dev event recall within the stated normal false-alert budget. Strict matching counts only alert starts within positive intervals and separately reports repeated event alarms. Historical onset windows are provisional, so some unknown-time alarms may be useful early detections; do not portray those as proven normal false alarms. Overlapping windows do not create independent event observations. Confidence from a few cameras is limited.

## 11. Expanded unattended images did not yield a clear winner

These are 14 October results preceding CLIP adaptation. MiniCPM has 84 TP, 25 FP, 158 FN and 695 TN (frame F1 .479). With 20-second contact sheets recall falls to 10.74%, FPR .14% and F1 .193. SmolVLM has 1,457 abstentions over 13,789 evaluations; MiniCPM has 45; both have zero runtime errors. Hard categorical responses have no real continuous ranking scores. These models have not been rerun or fine-tuned in subsequent CLIP studies. A contact-sheet comparison is not native temporal-video inference.

## 12. Both CLIP visual backbones were updated

The study has 7,498 target-specific examples and 12,478 distinct views. Temporal features use last/mean/max/delta pooling, preserving source groups. Probe C values 1/10/100 and fine-tuning settings are selected on dev. Seed 42 was primary before held-out results. Six checkpoints contain finite changed visual tensors, checked by delta norms and hashes. Verification replays 46,926 held-out and 29,394 dev predictions. Models classify scenes/clips and do not themselves output object boxes or ownership.

## 13. Fine-tuning helped some tasks and hurt others

These are source holdouts that were historically inspected. Gains combine supervision and temporal representations; some routes retain a probe or pretrained score based on dev. On 806 photo tests SigLIP AP rises .499 to .954 and MobileCLIP .220 to .901. Loitering has 11 positives among 75 windows (prevalence .147), with both selected APs below prevalence. Sleep has six positive photos and provisional labels. Unattended labels are provisional and cover only two held-out camera families. All seeds and alternatives remain preserved.

## 14. The next screen combined data, crops and adaptation

Earlier adapted weights are reused; this is not a new visual fine-tune per configuration. Seed 42 is primary. Automatic crops and sparse geometry do not supply full MOT or ownership. Verification replays 117,612 predictions and checks 20,980 image identities and actual inference for all five tasks on both encoders.

## 15. Historical AP rose unevenly after the strategy screen

These development-selected routes concern already inspected historical holdouts. Crops help smoking and geometry helps loitering, while dumping/public-sleep regressions remain. Dev-F1 FPR is still 75.9% / 84.6% on CCTV smoking and 99.7% / 100% on unattended items. SigLIP unattended zero-logit F1 falls from .749 to .131. Preserve threshold failures alongside AP gains.

## 16. The broader transfer tests exposed remaining failures

The 224-photo holdout shares its collection with new training photos. HMDB was an unseen source here; both routes regress on 906 clips. Later HMDB adaptation uses 530 training and 205 test clips, a different protocol. Public sleep has 28 provisional new-author cases, 17 positive and 11 normal. MEVA has 15 action windows and 12 controls, not verified persistent unattended state.

## 17. Loitering and public sleep remain data-limited

Thermal data supplies 874,936 observations and 19,737 track IDs; 17,878 eligible trajectories remain. A selected 400-tree depth-eight forest has test AP .577, F1 .664 and FPR .187 versus duration-only AP .283. Development AP .863 exposes date transfer loss. These are supplied tracks from one thermal camera. RGB Street Scene is separate and has only 11 positive test windows. Photos cannot establish sleep duration.

## 18. Small people and missing items were concrete bottlenecks

At 224 input, the median largest smoker is 11.2 pixels tall; all targets are below 12 pixels in 71.8% of positive CCTV frames. Automatic multiple-person/upper-body crops reduce losing half every target from 40.2% to 1/393. Detector resolution also changes; coverage is not recognition. CAVIAR partial item-reference recall rises from 0/136 to 66/136. Unmatched tracks are not proven false positives and ownership has no independent truth.

## 19. Separate appearance from tracked event state

This pilot freezes pretrained encoders, separate from earlier visual fine-tuning. Dense 2 Hz video and 16 HMDB views provide scene, person, upper-body and item contexts. Eighteen geometry values come from causal Hungarian IoU/velocity tracking, not ByteTrack or ReID. GRUs use 128 projection and 64 hidden dimensions with past-only histories and gap timing. The cached-feature replay CLI is not a new-video streaming application.

## 20. Fifty heads tested region, sequence and person evidence

HMDB contributes 530 train, 161 dev and 205 test clips after ten known shared-film training aliases are purged. No test clip is removed. Person supervision uses 1,460 matched automatic training-camera regions with 277 positives. Training-only hard-negative mining starts at epoch four; checkpoints selected earlier do not demonstrate its benefit. Heads use float32 AdamW, learning rate .0005, at most 60 epochs and patience eight. Most of 936 prepared recordings are action clips, not cameras.

## 21. CCTV gains did not repeat consistently across backbones

The previous bundle is recomputed on the exact same 205 HMDB cases, using eight views and no HMDB training versus 16 views and 530 training clips now. Gains combine data and methods. At the old pooled dev-budget point, CCTV recall/FPR are 16.5%/7.9% for MobileCLIP and 2.3%/1.6% for SigLIP. HMDB values are 86.7%/2.9% and 66.7%/1.7%. Movie/web recognition does not establish CCTV transfer.

## 22. Ensembling improves HMDB; low-FPR CCTV remains hard

Each encoder compares 14 previous heads, four seed ensembles and one all-head ensemble: 19 policies. Equal sigmoid averages are not calibrated probabilities. Both selections freeze before held-out scores load. Every dev source must meet the FPR budget under one cutoff. SigLIP HMDB AP improves .7779 to .9645 with recall 66.7% to 76.7% at observed FPR 1.71%; CCTV recall remains 1.78%. MobileCLIP retains its head with a stricter cutoff. These are retrospective cohorts and reused development sources.

## 23. Explicit item state was more useful than a global score

Global alerts use two-second persistence, one-second release and ten-second cooldown. The one-false-alert/hour development target has only 28 seconds of dev normal footage and cannot certify that rate. Observed global normal rates are 21.7/hour and 130.3/hour. The separate AVSS source rule requires the same item stationary for 30 seconds and its associated owner absent. State-only results are independent of CLIP. The three videos appear to share one platform view; HARD remains a censored diagnostic.

## 24. A good AP or zero observed alarms is not a guarantee

Entire ABODA classroom video7 and walkway video2 provide disjoint normal calibration and are excluded from both evaluation arms. Four events and 137 normal seconds remain. MobileCLIP detects 0/4 in both arms; SigLIP changes 0/4 to 1/4, with four normal false alerts reduced to one, still 26.3/hour. AVSS lacks disjoint normal calibration. Zero alarms over 165.74 normal seconds gives a one-sided 95% Poisson upper rate of 65.07/hour, assuming a homogeneous process. It is not a camera-transfer guarantee.

## 25. More state machinery did not improve the AVSS result

The new rule uses three trailing centers and one second of sustained .015-frame displacement; pending motion blocks alerts. Item memory uses spatial one-to-one association and rearms on five seconds of owner return, three seconds of motion or 30 seconds absent. Synthetic tests cover simultaneous items and ID gaps without proving field benefits. AVSS stays 2/2 with zero false/repeated alerts; smoothing raises delay from 34.818 to 35.068 seconds. Memory raises HARD repeats from one to two. Historical intervals use different provisional definitions.

## 26. The new highway clips are component stress tests

Source: https://github.com/hangsuuuu/video-of-abandoned-object-detection at revision 966439fcac1103b392c49d4135f128860a39d1f8. The receipt records content hashes. Three bundled clips show highway debris; three entries explicitly alias ABODA. Actual FPS is 30, about 14.966 and 19.912, with 357, 219 and 151 decoded frames. Beyond-stream timestamps are exposed, not rescaled. Thirty annotated frames form a component stress test with no owner or complete event labels. Visual review is provisional assistant review.

## 27. Fixed overlapping tiles test the small-object bottleneck

Grounding DINO tiny uses the same pinned weights, generic query and thresholds, float32 GPU inference and TF32 off. Four fixed 60%-size tiles overlap by 20%; detections merge with full-image output using NMS .5. Source boxes never choose tiles. There are 136 CAVIAR training-source item references and 30 highway-debris references. Partial annotations support recall, not precision. Tiling does not overwrite pilot event tracks or heads.

## 28. VadCLIP remains a qualified smoking baseline

Original VadCLIP dual_mean gives AUROC .5874, AP .63323, F1 .56475, recall .51786 and FPR .35685 with a development-frozen threshold. Training uses 1,910 frames in nine clips; validation uses 312 frames in four clips and a 12.642M-parameter adapter. This offline supervised smoking protocol has different source groups and supervision from image models. The constant visual branch remains a recorded failure. No new VadCLIP training was done this round.

## 29. No single model is ready to cover all five objectives

SigLIP ensemble HMDB AP .9645 concerns 205 retrospective adapted movie/web clips; CCTV recall remains 1.8% at 1.6% FPR. MobileCLIP CCTV AP .725 does not ensure transferable thresholds. Dumping AP .893 uses 100 full-clip IWDD tests, not online localization. Street Scene and thermal tracks use different protocols/sensors. Public-sleep photos remain provisional; unattended evidence has two primary events and a few normal minutes. Each candidate needs target-camera event validation.

## 30. Retain demonstrated gains and preserve regressions

Preserved regressions include CCTV fine-tuning, MobileCLIP dumping, sleep photos, HMDB/MEVA transfer, SigLIP pilot CCTV, conservative-cutoff recall loss and HARD memory repeats. Hard-negative runs can select before mining begins. Repeated development searches can overfit sources even when no test selection occurs. More adjacent frames, seeds and hyperparameter trials do not add independent cameras/events. Tiling is a component finding, not an established event-pipeline improvement.

## 31. Prioritize independently labelled target-camera evidence

Acquire compatible evidence with matched controls: smoking versus drinking/eating/coughing; disposal versus benign placing/carrying; complete dwell tracks; persistent apparent sleep versus rest/lying; owner-associated items with retrieval and occlusion controls. Independent target-camera review and event diversity are still needed. Freeze a fresh camera holdout before scores are inspected. Normal-hour bounds depend on assumptions and do not certify source transfer. Missing duration, ownership or visibility should permit unknown decisions.

## 32. Every stage remains inspectable and replayable

Latest report: results/event_improvements_20261017/report.md; overview: reports/event_improvements_findings_20261017.md. Earlier expanded, fine-tuning, strategy and video-pilot reports remain preserved. New scripts replay ensemble selection, causal memory and detector merges/matches; source/code/weight hashes are retained. The slide builder reads saved results and runs no inference. Editable PPTX and vector PDF share one checked layout, with notes and source hashes. Earlier decks remain historical.
