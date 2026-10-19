# SmolVLM fine-tuning for all five anomaly targets — 19 October 2026

Actual CUDA LoRA training of the pinned SmolVLM-500M model is complete. All five objectives share adapted language weights; development-selected target routes are reported separately. Original and adapted scores use identical task queries, 512-pixel images, ordered views and frozen visual tokens. This is retrospective source-held-out evaluation, not a fresh deployment test.

## Data and supervision

| Target | Train N / positive | Dev N / positive | Test N / positive | External N / positive |
|---|---:|---:|---:|---:|
| smoking | 2812 / 1073 | 1314 / 502 | 870 / 505 | 806 / 140 |
| littering | 274 / 144 | 61 / 32 | 100 / 50 | 0 / 0 |
| loitering | 323 / 41 | 98 / 13 | 75 / 11 | 0 / 0 |
| rough_sleeping | 74 / 25 | 21 / 7 | 18 / 6 | 0 / 0 |
| unattended_items | 490 / 74 | 318 / 181 | 962 / 242 | 0 / 0 |

There are 8,616 main examples and 13,596 distinct decoded images. Additional transfer uses 906 HMDB action clips, 28 new-author public-sleep photos and 27 MEVA action windows; none enter training or selection. Photo and camera groups remain disjoint, and decoded pixels are checked. A target-negative annotation never labels the other four objectives as absent.

Unattended items have two training source groups, one development group and two historical test groups; loitering uses different recordings from a single camera. Rough-sleeping labels remain provisional visible-sleep annotations, with no inference about housing status. HMDB controls are other primary actions, not exhaustive smoking-negative verification. MEVA labels describe abandonment action windows, not persistent unattended-item state.

## Optimization, selection and reproducibility

The vision tower, multimodal connector, token embeddings and vocabulary output head stay frozen. LoRA trains 4,341,760 language-projection parameters (rank 8, alpha 16, dropout 0.05), using full-vocabulary first-answer-token cross entropy for ` Yes`/` No`. All input-token loss positions are excluded. Each epoch visits all 3,973 training examples, then an equally sized task/class/source/group-balanced resampled pass. AdamW uses weight decay 0.01, 5% warmup, linear learning-rate decay, gradient clipping 1.0, microbatch 8 and accumulation 4. Training is bfloat16; all development selection and final inference are float32 with TF32 off.

Primary seed 42 compares learning rates 0.00003 and 0.0001, with at most six epochs and patience two. The selection metric averages AP by dataset source within each target, then equally over the five targets. The best joint adapter and per-task original/adapter routes are frozen before held-out predictions. A second seed, when present, repeats only the primary winning learning rate; it never replaces the primary based on test results.

| Seed | Candidate | Best epoch | Completed epochs | Development macro source AP |
|---|---|---:|---:|---:|
| 42 | seed_42/lr_3e-05 | 1 | 3 | 0.6584 |
| 42 | seed_42/lr_0.0001 | 1 | 3 | 0.7026 |
| 43 | seed_43/lr_0.0001 | 1 | 3 | 0.6548 |

![Training loss and development checkpoint selection](../results/smolvlm_finetuning_20261019/figures/learning_curves.png)

Matched original development macro source AP: **0.5657**. Primary joint adapter: `seed_42/lr_0.0001` (**0.7026** development macro source AP).

| Target | Development-selected route | Development source AP |
|---|---|---:|
| smoking | seed_42/lr_0.0001 | 0.7709 |
| littering | seed_42/lr_0.0001 | 0.7764 |
| loitering | seed_42/lr_0.0001 | 0.3384 |
| rough_sleeping | seed_42/lr_3e-05 | 0.9341 |
| unattended_items | seed_42/lr_0.0001 | 0.7193 |

Response-format check on development data: original Yes+No token mass has minimum 98.4% and median 99.9%; selected joint LoRA has minimum 98.7% and median 100.0%. Both already concentrate probability on the required binary responses.

## Matched held-out results

AP measures ranking, not false-alert control. The original, joint adapter and per-target route receive the same inputs. Routing chooses development AP, so it can regress on held-out sources. All regressions below are retained.

| Target / source | N / positive | Original AP | Joint LoRA AP | Routed AP |
|---|---:|---:|---:|---:|
| smoking / kaggle_sujaykapadnis_smoking | 224 / 112 | 0.9670 | 0.9844 | 0.9844 |
| smoking / smoking_cctv | 646 / 393 | 0.5999 | 0.6225 | 0.6225 |
| littering / mivia_iwdd | 100 / 50 | 0.5922 | 0.7660 | 0.7660 |
| loitering / street_scene | 75 / 11 | 0.4250 | 0.1822 | 0.1822 |
| rough_sleeping / commons_visible_sleeping | 18 / 6 | 0.7437 | 0.6327 | 0.7885 |
| unattended_items / unattended_reviewed | 962 / 242 | 0.4762 | 0.6933 | 0.6933 |
| smoking / stanford40 | 806 / 140 | 0.5831 | 0.8900 | 0.8900 |
| smoking / hmdb51_action_transfer | 906 / 109 | 0.7638 | 0.7837 | 0.7837 |
| rough_sleeping / commons_new_author_transfer | 28 / 17 | 0.8062 | 0.8441 | 0.8904 |
| unattended_items / meva_abandon_action_transfer | 27 / 15 | 0.6076 | 0.6457 | 0.6457 |

The primary routed configuration does not improve every objective: loitering AP changes from 0.4250 to 0.1822. Unattended-item ranking improves, but the development-budget threshold gives 70.8% held-out FPR at 94.6% recall. These results do not justify replacing an operational alert pipeline with this research adapter.

AP is ranking performance, not classification accuracy. A constant-score baseline has AP equal to the positive fraction. For CCTV smoking that reference is 0.6084, versus 0.6225 for the routed adapter: the improvement is small and this remains a weak CCTV detector.

![Matched original and adapted AP](../results/smolvlm_finetuning_20261019/figures/matched_ap.png)

## Threshold transfer

Each method has its own development-only threshold. The following point allows at most 5% pooled development FPR for that target; the held-out FPR can be much higher. It is not a 5% deployment guarantee or an event false-alert rate.

| Target / source | Original recall | Original FPR | Routed recall | Routed FPR | Routed TP / FP |
|---|---:|---:|---:|---:|---:|
| smoking / kaggle_sujaykapadnis_smoking | 93.8% | 12.5% | 93.8% | 1.8% | 105 / 2 |
| smoking / smoking_cctv | 0.0% | 0.0% | 1.8% | 0.0% | 7 / 0 |
| littering / mivia_iwdd | 2.0% | 4.0% | 30.0% | 8.0% | 15 / 4 |
| loitering / street_scene | 0.0% | 1.6% | 9.1% | 6.2% | 1 / 4 |
| rough_sleeping / commons_visible_sleeping | 0.0% | 0.0% | 33.3% | 8.3% | 2 / 1 |
| unattended_items / unattended_reviewed | 20.2% | 1.9% | 94.6% | 70.8% | 229 / 510 |
| smoking / stanford40 | 77.1% | 25.4% | 78.6% | 4.4% | 110 / 29 |
| smoking / hmdb51_action_transfer | 63.3% | 2.9% | 69.7% | 2.8% | 76 / 22 |
| rough_sleeping / commons_new_author_transfer | 0.0% | 0.0% | 41.2% | 0.0% | 7 / 0 |
| unattended_items / meva_abandon_action_transfer | 0.0% | 0.0% | 66.7% | 58.3% | 10 / 7 |

The predeclared raw threshold is margin zero: choose Yes when its logit is at least No's logit, without a fitted cutoff.

| Target / source | Original raw recall | Original raw FPR | Routed raw recall | Routed raw FPR | Routed raw F1 |
|---|---:|---:|---:|---:|---:|
| smoking / kaggle_sujaykapadnis_smoking | 100.0% | 98.2% | 93.8% | 1.8% | 0.9589 |
| smoking / smoking_cctv | 8.9% | 9.9% | 0.0% | 0.0% | 0.0000 |
| littering / mivia_iwdd | 84.0% | 70.0% | 60.0% | 26.0% | 0.6452 |
| loitering / street_scene | 0.0% | 1.6% | 9.1% | 4.7% | 0.1333 |
| rough_sleeping / commons_visible_sleeping | 100.0% | 100.0% | 100.0% | 25.0% | 0.8000 |
| unattended_items / unattended_reviewed | 26.4% | 5.3% | 64.9% | 25.8% | 0.5368 |
| smoking / stanford40 | 100.0% | 98.5% | 78.6% | 3.2% | 0.8118 |
| smoking / hmdb51_action_transfer | 94.5% | 61.2% | 67.0% | 1.6% | 0.7487 |
| rough_sleeping / commons_new_author_transfer | 100.0% | 100.0% | 70.6% | 18.2% | 0.7742 |
| unattended_items / meva_abandon_action_transfer | 0.0% | 0.0% | 6.7% | 8.3% | 0.1176 |

## Seed stability

| Target / source | Seed 42 joint AP | Seed 43 joint AP |
|---|---:|---:|
| smoking / kaggle_sujaykapadnis_smoking | 0.9844 | 0.9848 |
| smoking / smoking_cctv | 0.6225 | 0.6295 |
| littering / mivia_iwdd | 0.7660 | 0.7656 |
| loitering / street_scene | 0.1822 | 0.1314 |
| rough_sleeping / commons_visible_sleeping | 0.6327 | 0.9167 |
| unattended_items / unattended_reviewed | 0.6933 | 0.7052 |
| smoking / stanford40 | 0.8900 | 0.8934 |
| smoking / hmdb51_action_transfer | 0.7837 | 0.7862 |
| rough_sleeping / commons_new_author_transfer | 0.8441 | 0.8498 |
| unattended_items / meva_abandon_action_transfer | 0.6457 | 0.6620 |

## Context against the previous five-target CLIP strategies

These are the existing 16 October primary development-selected CLIP routes, using identical training/development splits and exactly matched held-out IDs/labels. Crops, resolution, representation and training method differ, so this is a pipeline comparison; the original SmolVLM remains the controlled weight-adaptation baseline. Later smoking video heads use additional HMDB training and a different input protocol and are documented separately, rather than mixed into this same-cohort table.

| Target / source | MobileCLIP strategy AP | SigLIP strategy AP | SmolVLM routed AP |
|---|---:|---:|---:|
| smoking / kaggle_sujaykapadnis_smoking | 0.9975 | 0.9998 | 0.9844 |
| smoking / smoking_cctv | 0.6902 | 0.7225 | 0.6225 |
| littering / mivia_iwdd | 0.8064 | 0.8934 | 0.7660 |
| loitering / street_scene | 0.4894 | 0.5505 | 0.1822 |
| rough_sleeping / commons_visible_sleeping | 0.9306 | 0.8833 | 0.7885 |
| unattended_items / unattended_reviewed | 0.9803 | 0.9250 | 0.6933 |
| smoking / stanford40 | 0.8803 | 0.9373 | 0.8900 |
| smoking / hmdb51_action_transfer | 0.5764 | 0.6639 | 0.7837 |
| rough_sleeping / commons_new_author_transfer | 0.9521 | 0.9330 | 0.8904 |
| unattended_items / meva_abandon_action_transfer | 0.5452 | 0.6205 | 0.6457 |

## Optional routing that requires development gains in both seeds

Exploratory rule specified after primary held-out results were inspected and after the first independent-seed development epoch. Retrospective safeguard, not fresh prospective evidence. Independent optimization seeds do not provide independent cameras or annotation review.

Keep the primary target route only when its source-averaged development AP and the independent-seed joint adapter's development AP both strictly exceed the original; otherwise use the original. No held-out score enters this function.

| Target | Primary development AP gain | Second-seed development AP gain | Optional route |
|---|---:|---:|---|
| smoking | +0.0872 | +0.0828 | seed_42/lr_0.0001 |
| littering | +0.1589 | +0.1017 | seed_42/lr_0.0001 |
| loitering | +0.0308 | -0.1620 | original |
| rough_sleeping | +0.3084 | +0.3426 | seed_42/lr_3e-05 |
| unattended_items | +0.1253 | +0.0805 | seed_42/lr_0.0001 |

This option leaves the original primary results above unchanged. On a rejected target it reuses the original model scores and original development thresholds; on other targets it retains the primary route. It does not solve camera shift, temporal evidence or the unattended-item false-positive problem. Run with `predict --repeated-gains`; do not treat this retrospective rule as an independently validated deployment policy.

## Verification and limits

Verification passed: 20,971 decoded image contents, 7,248 development predictions and 15,168 held-out predictions replayed; 3 adapter checkpoints checked for finite weights and nonzero updated LoRA B matrices, and 20 native-image GPU comparisons against cached/batched scores. Selections and frozen thresholds are reconstructed from development predictions. Full optimization coverage is checked for every completed epoch.

The score is `logit( Yes) - logit( No)` at the first response position. The saved two-token probability mass helps reveal when the original vocabulary distribution prefers other answers. This forced binary interface has no trained abstention, explanations or bounding boxes. It differs from the older six-label JSON protocol, so the matched original here is the attribution baseline. Static and sparse-window AP does not prove event detection, item-owner association or continuous dwell time. No new latency testing was performed.

The primary inference configuration is seed 42, chosen before seeing its held-out results. Keep the original fallback for targets selected that way; do not choose a seed or swap a route based on the tables above. A future deployment claim needs newly held-out cameras, independent review of temporal/ownership labels, and hours of normal video for event false-alert measurement.

Artifacts: [protocol](../results/smolvlm_finetuning_20261019/protocol.json), [CSV metrics](../results/smolvlm_finetuning_20261019/metrics.csv), [model bundle](../results/smolvlm_finetuning_20261019/model_bundle.json), [verification](../results/smolvlm_finetuning_20261019/verification.json), [primary selection](../results/smolvlm_finetuning_20261019/seed_42/selection.json), and per-candidate adapter/history folders. See [the reproduction guide](../docs/smolvlm_finetuning.md) and [the publisher's model card](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) / [PEFT LoRA reference](https://huggingface.co/docs/peft/v0.17.0/package_reference/lora) for implementation context.

Software checks pass: 187 unit tests, Ruff lint and formatting, and whitespace validation. The actual inference CLI also passed 6 checkpoint/score or missing-context checks; these validate the interface, not accuracy on those individual examples.
