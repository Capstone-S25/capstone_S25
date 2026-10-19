# Evaluation and adaptation progress — speaker notes

Updated 19 October 2026 (Asia/Jakarta). Historical protocols stay separate.

## 01. Five-anomaly model evaluation

The full history remains visible. Six CLIP visual fine-tunes and three actual SmolVLM language-layer LoRA runs changed model weights. The 330 strategy configurations, 50 temporal heads and 38 smoking policies are separate studies, repeatedly using some sources. SmolVLM has 3,792 unique held-out examples across ten cohorts, not 3,792 independent events. Four historical target rankings improve with primary per-task routing, but loitering regresses and unattended FPR is 70.8% at the development-budget cutoff. No five-event deployment winner is established; no new latency benchmark is run.

## 02. Five objectives require different evidence

The user accepted existing definitions and annotations. There is no newly invented universal loitering threshold. Public-sleep labels describe visible sleep/bedding/context, never housing status. AVSS specifies stationary for 30 seconds and an absent owner; historical provisional windows and MEVA abandonment actions do not uniformly implement this criterion. Grounding boxes and source annotations are separate from inferred ownership.

## 03. Adaptation now includes SmolVLM language weights

SmolVLM-500M-Instruct is adapted through q/k/v/o and gate/up/down language projections. No new MiniCPM or VadCLIP training is claimed. The older four-model common-image comparison remains historical. New SmolVLM AP uses first-token binary margins, not hard JSON categories; its original matched baseline is essential. VadCLIP remains a separately supervised 1,056-frame smoking protocol.

## 04. From common prompts to controlled model adaptation

The SmolVLM round reuses collected and previously inspected source data. Its three runs are seed42 at learning rates 3e-5 and 1e-4, plus seed43 at the primary winning rate 1e-4. All choose epoch one after three completed epochs. More configurations, crops, overlapping windows or seeds are not more independent cameras/events.

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

These are 14 October results preceding CLIP adaptation. MiniCPM has 84 TP, 25 FP, 158 FN and 695 TN (frame F1 .479). With 20-second contact sheets recall falls to 10.74%, FPR .14% and F1 .193. SmolVLM has 1,457 abstentions over 13,789 evaluations; MiniCPM has 45; both have zero runtime errors. Hard categorical responses have no real continuous ranking scores. At the 14 October cutoff these are original prompted models. A separate 19 October study later adapts SmolVLM language weights under a new matched binary protocol. A contact-sheet comparison is not native temporal-video inference.

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

## 29. Use the original model as a matched weight control

Score = logit(Yes) minus logit(No). This is a genuine continuous ranking score and is not calibrated event probability. Yes+No probability mass is already high before training: minimum98.4%, median99.9% on development. The original and all adapted paths share the frozen image encoder and connector. No other task is fabricated as negative. The new interface differs from the earlier free-generated JSON protocol.

## 30. All five tasks are trained; evidence is uneven

These are normal/positive counts, unlike the total/positive header in the Markdown summary. Smoking test comprises646 CCTV frames and224 Kaggle photos; Stanford40 adds806 external photos with140 positives. Training uses exactly the existing3973 strategy examples, not additional new downloads. Loitering has one physical camera despite multiple recording groups. Sleep labels are provisional. Unattended endpoints are correlated, with two test source families.

## 31. Ten cohorts cover 3,792 unique held-out examples

HMDB other-primary-action controls are non-exhaustive negatives. Eight-view clip smoking differs from single-image smoking training. Twenty-eight public-sleep cases have25 author groups. MEVA has15 action windows and12 pre-event controls and cannot establish persistent ownership. Existing historical tests have been inspected before. No new dataset is downloaded for this LoRA round.

## 32. Adapt 4.34 million language parameters

LoRA also freezes the vocabulary output head. Every epoch visits3973 unique training examples once, then3973 balanced resamples;7946 visits total. The resampled half balances task/class/source/group, while the combined epoch retains original imbalance. AdamW weight decay.01,5%warmup,linear decay,gradient clip1,microbatch8×accumulation4. CUDA accuracy is unpaced; there is no new latency benchmark.

## 33. Three trained candidates all select epoch one

Seed42 compares two rates. Seed43 repeats only the winning high rate; it is not a full repeated grid. Planned maximum6epochs,patience2; each completes3epochs and selects1. Routes and thresholds precede their held-out predictions. Joint means one high-rate adapter for all tasks. Routed means per-task original/adapter choices using development AP. Do not substitute a better test seed.

## 34. Training loss falls while development AP declines

All three runs select epoch one. The primary high-rate training loss is .4789,.2895,.2235, while development macro AP is .7026,.6485,.6276. More epochs fit the training corpus but worsen this source-transfer diagnostic. Nine actual epochs total across candidates yield71514visits, not71514newexamples. Frozen visual extraction avoids repeating unchanged computation.

## 35. LoRA improves selected rankings and regresses on loitering

Four of five historical target rankings improve under per-task routing, counting CCTV as the smoking target. Loitering falls .4250→.1822. Historical sleep has six positives: joint falls .7437→.6327 while routed lowLR reaches .7885. Unattended ranking rises .4762→.6933 but decisions remain problematic. Photo smoking has different prevalence and domain from CCTV. The constant-score CCTV AP is393/646=.6084.

## 36. Photo and action-source gains do not validate events

Stanford40 is a historical external photograph holdout. HMDB includes906clips with109smoking positives and797other primary actions; controls are not exhaustive negatives. New-author sleep has28cases and25author groups. MEVA27windows do not prove persistent unattended state. These sources were inspected in earlier experiments, so transfer is retrospective. No continuous event false-alert rate is measured here.

## 37. The development FPR cutoff fails on unattended items

At the development-budget point unattended recall94.6%andFPR70.8%; original20.2%and1.9%. This is not a uniformly superior decision tradeoff. Litter recall30%FPR8%;loiter9.1%/6.2%;historical sleep33.3%/8.3%. MEVA elsewhere is66.7%/58.3%. Endpoint FPR is not hourly alert rate. Cutoffs are fitted per target on pooled development normals, not every source separately.

## 38. The second seed repeats the loitering failure

Seed43 is a limited stability check and is never selected using test outcomes. Sleep joint AP.6327versus.9167is sensitive on18cases. The primary per-task sleep route is a different low-LR adapter,AP.7885. HMDB joint AP.7837versus.7862andunattended.6933versus.7052repeat broadly. Loitering worsens .1822→.1314. Existing source/annotation limits remain.

## 39. CLIP strategies lead on most historical cohorts

This compares complete pipelines, not an architecture-only ablation. Same learning/development splits are3973/1812. The matched original SmolVLM is the clean weight-adaptation control. Later SigLIP ensemble AP.9645 uses530HMDBtrainingclips,205testclipsand16views; comparing that directly to Smol906case8viewtransfer would confound supervision and examples. Historical VadCLIP also has a different split.

## 40. Require a development gain in both optimization seeds

Exploratory rule specified after primary held-out results were inspected and after the first independent-seed development epoch. Retrospective safeguard, not fresh prospective evidence. Independent optimization seeds do not provide independent cameras or annotation review. The function reads only development scores, but the decision to invent it was made after primary held-out inspection and the first secondary development epoch. Restored targets reuse original development thresholds; others retain primary routes. Seed43 repeats the high LR, so this is not exact replication of the low-LR sleep route. CLI option: --repeated-gains.

## 41. Check actual checkpoints and native image execution

Largest native-versus-saved error7.2479e-5isbelow.002tolerance. Four actual native CLI scoring cases and two missing-timestamp temporal guards pass. Some chosen positive inputs are classified absent under strict cutoffs, consistent with low recall. PEFT export changes machine-local base config to public pinned identity, leaving weight bytes unchanged. Source hashes, response mass and exact cutoffs are preserved. No new latency test is performed.

## 42. No single model reliably covers all five events

SmolVLM provides a useful language-adaptation experiment but does not supersede all CLIP strategies. Keep the loiter regression, joint-sleep regression and unattended FPR failure visible. Tiling improves component recall, not verified event performance. AVSS still has only two clean primary events and2.76normalminutes. MiniCPM and VadCLIP are historical baselines, with different scope.

## 43. Generalization needs evidence beyond another training run

The study makes failure mechanisms more concrete: few source families, tiny positive cohorts, small person/object detail, incomplete ownership/duration labels and threshold domain shift. More epochs, generic language explanations or extra correlated frames alone do not address those evidence gaps. Neither current research routes nor the optional safeguard are validated deployment choices.

## 44. Freeze new camera evidence before selecting another model

Acquire smoking controls(drinking,eating,coughing);disposal versus placing/pickup;legitimate waiting and complete dwell tracks;persistent sleep versus awake rest;owner-associated items with retrieval/occlusion. Use unseen camera/day blocks and review uncertainty. Evaluate controlled low-LR/regularized adapters or vision adaptation only after adding compatible evidence. Do not claim normal alert safety from a few zero-alert minutes.

## 45. Updated paper, editable slides and source-linked findings

Updated paper: reports/technical_runthrough_20261019/model_evaluation_runthrough_20261019.pdf. SmolVLM findings: reports/smolvlm_finetuning_findings_20261019.md; guide:docs/smolvlm_finetuning.md. Rebuild using scripts/build_technical_runthrough_20261019.py and reports/slides/build_progress_presentation_20261019.py. The paper and slides cite stage-specific sources and preserve failed interventions.
