# Benchmarking Vision-Language Models for Public-Space Anomaly Detection on CPU

**Date:** 2026-10-11 · **Platform:** Linux x86_64, 64 logical CPUs, ~251 GiB RAM, Python 3.12.3, CPU-only inference (no GPU)
**Scope:** Five candidate models evaluated for detecting public-space anomalies — smoking, littering, loitering, rough sleeping, unattended items — with emphasis on accuracy, F1, inference speed, memory, and deployability.

## Executive Summary

This report benchmarks four frame-level vision-language models head-to-head on a fixed 80-image test set (MobileCLIP2-S2, SigLIP2-B/16-224, SmolVLM-500M, MiniCPM-V), and separately trains and ablates VadCLIP — a weakly-supervised video anomaly detector — on the smoking corpus. All inference ran on CPU in isolated worker processes; every evaluation is programmatic (no image was opened for visual inspection by an evaluator).

**Head-to-head results (80 images: 16 per class, except unattended_items = 0):**

| Model | Accuracy | Macro-F1 | Binary F1 | ROC-AUC | Latency (s/sample) | Peak RSS |
|---|---|---|---|---|---|---|
| MobileCLIP2-S2 | 0.188 [0.113, 0.275] | 0.086 | 0.460 | 0.459 | 0.109 (9.1/s) | ~1.1 GiB |
| SigLIP2-B/16-224 | 0.275 [0.175, 0.375] | 0.213 | **0.827** | **0.626** | 0.183 (5.5/s) | ~1.2 GiB |
| SmolVLM-500M | — (abstained on all 80) | — | — | — | 2.93 (0.34/s) | ~3.0 GiB |
| MiniCPM-V | **0.375** [0.263, 0.488] | **0.266** | 0.400 | — | 17.3 (0.058/s) | ~18.2 GiB |

Intervals are 95% bootstrap CIs (B=2000, seed 12345). Key findings:

1. **SigLIP2 is the best frame screener** — highest ROC-AUC (0.626 [0.486, 0.759]) at 0.18 s/sample and ~1.2 GiB RSS. However its binary F1 of 0.827 is largely a base-rate artifact: with 64/80 samples anomalous (prevalence 0.80), an always-anomaly classifier scores 0.889, so SigLIP2 reaches only ~93% of that ceiling; its AUC shows merely moderate discrimination, and it misses smoking entirely (recall 0).
2. **MiniCPM-V has the best accuracy and macro-F1** and the only usable rough_sleeping F1 (0.769), but at 17.3 s/sample (~158× slower than MobileCLIP2) and ~18 GiB RSS — suitable for asynchronous review, not real-time screening.
3. **MobileCLIP2-S2 is the fastest and smallest** (9.1 samples/s, ~1.1 GiB) but has the weakest accuracy (0.188) and a chance-level AUC (0.459 [0.321, 0.601]).
4. **SmolVLM-500M is unusable as configured**: it abstained on all 80 samples (every response was the literal string `unknown.`), giving zero prediction coverage.
5. **VadCLIP**, trained separately on the smoking corpus, reaches dual-mean frame AUPR 0.525 vs its own zero-shot (0.285) and linear-probe (0.281) baselines, with AUROC 0.624; ablations show a window-1 attention variant is best at L=256 (AUPR 0.542) and the L=512 run attains F1@opt ≈ 0.60. It is the only model that treats smoking as a temporal process, but it remains far from deployment-ready on a single-activity corpus of 16 sessions.

**Principal caveats:** n = 80; no unattended_items data exists (macro-F1 is structurally capped at 5/6 ≈ 0.833); the 0.8 base rate inflates binary F1 for all models; results are from a single CPU-only machine and may not transfer to GPU or edge deployments.

## 1. Introduction and Scope

**Task.** The goal is to determine which vision-language model (VLM) or contrastive language–image pretraining (CLIP-family) model is most suitable for detecting public-space anomalies — smoking, littering, loitering, rough sleeping, unattended items — under a strict deployment constraint: CPU-only inference on commodity hardware. Suitability is judged along five axes: detection accuracy and F1, inference speed (latency and throughput), peak memory, robustness of the prediction interface (does the model actually emit usable labels?), and overall deployability.

**Candidate models.** Five candidates were evaluated, spanning roughly three orders of magnitude in per-sample cost:

| Model | Family | Role in benchmark |
|---|---|---|
| MobileCLIP2-S2 | Compact CLIP (convolutional backbone) | Fastest / smallest reference-free baseline |
| SigLIP2-B/16-224 | Sigmoid-loss CLIP variant | Mid-size frame screener |
| SmolVLM-500M | Small open VLM (~0.5 B parameters, per its name) | Generative label output at low cost |
| MiniCPM-V (`minicpm-v4`) | Open VLM with dynamic-resolution image input | Quality-first asynchronous reviewer |
| VadCLIP | Weakly-supervised video anomaly detector (AAAI 2024) | Temporal model of smoking; trained and ablated separately |

The first four are **reference-free**: they score a single image against text definitions of each anomaly class with no task-specific training. They are compared head-to-head on the same fixed 80-image test set under identical prompts, preprocessing, and scoring code paths (Section 4). VadCLIP is fundamentally different: it consumes temporal sequences of CLIP frame features through an attention-based adapter trained for weakly supervised video anomaly detection (WSVAD), so it cannot be scored on single frames in the head-to-head. It was instead implemented from its published design, trained on the smoking corpus, and ablated as a separate study (Section 6).

**Methodological constraint.** All evaluation is programmatic. No image or frame file was opened for visual inspection by any evaluator: ground-truth labels come from dataset manifests, predictions come from model inference in isolated worker processes, and every figure in this report plots benchmark metrics only (accuracy, F1, AUC, latency, throughput, RSS) — never dataset imagery. Sample provenance is reported as paths, group IDs, camera/session identifiers, and frame indices so that any result can be audited without visual inspection.

**Report layout.** Section 2 documents the data; Section 3 the evaluation protocol; Section 4 each model's exact prompts, preprocessing, and scoring formulas; Sections 5–6 present results for the head-to-head and VadCLIP respectively; Section 7 consolidates the optimizations and experiments performed to maximize measurement quality and performance; Sections 8–9 discuss suitability and limitations; Section 10 gives full reproducibility details.

## 2. Data

**Anomaly taxonomy.** All frame-level models score images against six text definitions — five anomaly classes plus a normal class — stored in `scripts/anomalies.json` (SHA-256 `589882c0…a95038c`). The definitions, verbatim: *normal* — "A public area with ordinary activity and none of the defined anomalies visibly present."; *littering* — "A person visibly discarding rubbish on the ground instead of using a bin."; *loitering* — "A person remaining in the same public area for an extended time without an apparent activity. Duration requires evidence across time."; *rough_sleeping* — "A person apparently sleeping or lying down with bedding in a public area. Describe visible evidence only; do not infer housing status."; *smoking* — "A person visibly smoking a cigarette, cigar, or pipe, with a smoking object near the mouth or visible smoke."; *unattended_items* — "A bag, box, or other personal item apparently left behind without its owner nearby. Ownership and duration require evidence across time."

**Head-to-head test set.** Eighty images from `data/anomaly_manifest.csv` (SHA-256 `18087c4d…b2cc582`), all with split = "test": exactly 16 per class for loitering, littering, smoking, rough_sleeping, and normal; **unattended_items has zero samples**. Provenance by group:

- *loitering* — Roboflow dataset (`data/raw/datasets/loitering/Roboflow Dataset (New Workspace)/{train|valid|test}/…jpg`), one image per group `loit:<hash>`, 16 distinct groups.
- *littering* — `Littering/{train|test}/images/video_N_M_*.jpg`, grouped by source video as `litter:video_N`; 12 distinct videos across the 16 images.
- *smoking / normal* — the same smoking-video.v1i.coco corpus, in a **paired design**: each of 16 camera/date sessions (c1/c2/c4 × dates 2025-09-11 through 2025-10-29) contributes exactly one smoking frame and one normal frame — hard negatives from the same camera on the same day. Group IDs are `smoke:cX:YYYY-MM-DD`.
- *rough_sleeping* — kaggle laying_dataset (`layingN.png`), 16 distinct images, group `rough:layingN`.

**VadCLIP corpus.** The full smoking-video.v1i.coco corpus with dense per-frame smoke labels (0/1). Session-level split (seed 234, valid_frac = 0.25): train = 12 sessions / 2562 raw frames / 1097 positive; valid = 4 sessions / 716 raw frames / 325 positive.

| Split | Session (camera:date) | Raw frames | Positive frames |
|---|---|---|---|
| train | c1:2025-09-11 | 16 | 3 |
| train | c1:2025-09-18 | 28 | 7 |
| train | c1:2025-10-07 | 62 | 35 |
| train | c1:2025-10-13 | 212 | 140 |
| train | c1:2025-10-26 | 292 | 180 |
| train | c2:2025-09-11 | 94 | 31 |
| train | c2:2025-09-14 | 30 | 11 |
| train | c2:2025-09-15 | 115 | 39 |
| train | c2:2025-10-07 | 153 | 57 |
| train | c2:2025-10-14 | 175 | 110 |
| train | c2:2025-10-29 | 448 | 145 |
| train | c4:2025-10-26 | 937 | 339 |
| valid | c1:2025-09-14 | 3 | 2 |
| valid | c1:2025-09-15 | 33 | 26 |
| valid | c4:2025-10-16 | 91 | 27 |
| valid | c4:2025-10-28 | 589 | 270 |

After `build_clip` at visual length L, representation counts are: L = 256 → 1653 train reps / 383 valid reps (115 positive); L = 128 → 255 valid (60 pos); L = 512 → 639 valid (276 pos). Only the long c4:2025-10-28 session (589 raw frames) exceeds any L, so changing L alters only that session's compression.

**Overlap disclosure.** VadCLIP's four valid sessions are among the sixteen head-to-head `smoke:` groups. This is not contamination for the frozen reference-free models (no training), and VadCLIP is evaluated separately on its own valid split; the overlap is disclosed so the two studies can be read together honestly.

## 3. Evaluation Protocol

**Environment.** All runs executed on a single Linux x86_64 host (kernel 7.0.0-28-generic, glibc 2.39), Python 3.12.3, 64 logical CPUs, ~251 GiB RAM (257,066 MB). Inference is CPU-only and offline: no GPU, no network access during evaluation (`offline = true`). Each model ran in an isolated subprocess worker with `threads = 2` (capped so timing stays comparable across models) and duty cycle 1.0; a psutil monitor thread sampled the worker's RSS throughout the run. Model load time is recorded separately from per-sample latency, so cold-start cost does not contaminate throughput numbers.

**Head-to-head protocol.** Each of the four frame-level models scored all 80 test images once (`repeats = 1`, with one `warmup` pass excluded from timing; seed 42). Images were loaded with PIL, EXIF-transposed, converted to RGB, and downscaled with LANCZOS so the longest side is at most 512 px. Per-model metrics:

- **accuracy_including_failures** — abstentions or errors count as incorrect;
- **macro-F1** over all six labels (unattended_items contributes F1 = 0 because it has no support);
- **binary F1 and ROC-AUC** for anomaly-vs-normal, using each model's continuous anomaly score;
- **latency** mean/p50/p95 in seconds per sample, active throughput (samples/s during inference) and wall-clock throughput; **peak RSS**.

**Bootstrap confidence intervals.** For every metric we resampled the 80 labeled samples with replacement (B = 2000, seed 12345), re-scored each resample through the *same* `detection_metrics` code path used to build `comparison.csv`, and took percentile [2.5, 97.5] intervals. Because point estimates are recomputed through the identical scoring path, they exactly reproduce `comparison.csv` — the CIs come for free with no approximation.

**VadCLIP protocol.** The WSVAD literature reports both frame-level (AUPR, AUROC) and segment-level (mAP) detection metrics. Our corpus has dense per-frame labels but no annotated anomaly segments, so we use **frame-level AUPR and AUROC as the primary metrics** — the canonical frame-detection measures in the WSVAD literature — plus precision/recall/F1 at the max-F1 operating point. (VadCLIP's headline 84.51% on its own benchmark is segment mAP, which cannot be computed here.) All VadCLIP variants were scored on the same valid frame representations from Section 2, and both baselines (zero-shot CLIP cosine; linear probe) used identical representations so comparisons are apples-to-apples.

## 4. Models, Prompts, and Scoring

**Shared prompt.** All four frame-level models received the same six-class classification prompt built by `classification_prompt(definitions)` from `scripts/models.py`:

> "Classify the visible scene using these definitions:" followed by one line per label — `{label}: {description}` with the verbatim Section 2 texts — then: "Choose one best matching label. Use normal only when no defined anomaly is visible. Use unknown when evidence is insufficient, including missing evidence of duration or ownership. Reply only with JSON: {"label": "LABEL"}. Allowed labels: normal, littering, loitering, rough_sleeping, smoking, unattended_items, unknown."

**Shared response parsing.** VLM outputs were parsed by `parse_vlm_response`: it strips ``` fences and a trailing "." or quotes; if the text exactly matches an allowed label that label is returned; otherwise it attempts `json.loads` and accepts only a dict whose string `"label"` field is in the definitions. Anything else is an **abstention** (label = None), which counts as incorrect under accuracy_including_failures.

**Shared preprocessing.** `load_image(path, max_side=512)`: PIL open → EXIF transpose → RGB → LANCZOS thumbnail so the longest side ≤ 512 px.

### 4.1 MobileCLIP2-S2 (open_clip)

Loaded via open_clip as `hf-hub:<checkpoint>` on CPU in fp32, with `reparameterize_model(model, inplace=True)` applied **before timing** — this fuses MobileCLIP's convolutional branches for inference and changes neither weights nor scores. Text features: `encode_text(tokenizer(list(definitions.values())), normalize=True)`. Per-image scoring:

```python
features = self.model.encode_image(self.preprocess(image).unsqueeze(0), normalize=True)
logits = self.model.logit_scale.exp() * features @ self.text_features.T
weights = logits.softmax(dim=-1)[0].tolist()
scores = dict(zip(self.labels, weights))
return Prediction(max(scores, key=scores.get), 1 - scores["normal"], scores)
```

Label = argmax of the six-way softmax; anomaly score = 1 − P(normal).

### 4.2 SigLIP2-B/16-224 (Hugging Face)

Text features from `[description.lower() for description in definitions.values()]`, padding="max_length", max_length=64. Per-image scoring:

```python
logits = (self.model.logit_scale.exp() * features @ self.text_features.T + self.model.logit_bias)[0]
values = logits.tolist()
weights = logits.sigmoid().tolist()          # per-class sigmoid; does not sum to one
scores = dict(zip(self.labels, weights))
raw_scores = dict(zip(self.labels, values))
label = max(raw_scores, key=raw_scores.get)  # argmax of raw logits
margin = (max(v for n, v in raw_scores.items() if n != "normal") - raw_scores["normal"])
return Prediction(label, margin, scores)
```

Label = argmax of the raw (bias-shifted) logits; anomaly score = margin between the strongest non-normal logit and the normal logit. Because sigmoid outputs do not sum to one, SigLIP2's per-class "weights" are independent probabilities, unlike MobileCLIP2's softmax distribution.

### 4.3 SmolVLM-500M (Hugging Face)

The image processor was configured with `size = {"longest_edge": max_side}` (max_side = 512). The prompt is applied through the chat template:

```python
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
self.prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
```

Generation is **greedy** (`do_sample=False`, `max_new_tokens=32`); only the newly generated tokens are decoded and passed to `parse_vlm_response`. The number of generated tokens per sample was recorded for diagnostics.

### 4.4 MiniCPM-V (minicpm-v4)

Loaded with `trust_remote_code=True, attn_implementation="sdpa"` — the only adapter using scaled-dot-product attention; all others use eager attention for CPU determinism. Inference uses the publisher's chat interface:

```python
messages = [{"role": "user", "content": [image, self.prompt]}]
text = self.model.chat(
    msgs=messages, tokenizer=self.processor.tokenizer, processor=self.processor,
    sampling=False, num_beams=1, repetition_penalty=1.0,
    max_new_tokens=self.max_new_tokens, max_slice_nums=1,
)
return parse_vlm_response(text, self.definitions)
```

`max_slice_nums=1` caps the dynamic-resolution tiling at one slice per image; decoding is greedy (`sampling=False`, `num_beams=1`). Output parsing is identical to SmolVLM's.

### 4.5 VadCLIP (implemented from its published design)

VadCLIP ("Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection", AAAI 2024) was implemented as a clean port: a **frozen OpenAI CLIP ViT-B/16** encoder (weights SHA-256 verified against the official OpenAI hash; loaded with `weights_only=False` deliberately because that is how the official file ships) feeds an **LGT-Adapter** — a temporal attention Transformer whose self-attention uses a block-diagonal windowed mask, plus a temporal-distance GCN branch. The adapter has two heads:

- **Visual classifier** (`logits1`) → p_vis = sigmoid(logits1);
- **Language–visual alignment** (`logits2`): cosine similarity between the adapted visual features and two class-text prompts — `PROMPTS[0] = "a normal scene with no one doing anything unusual"`, `PROMPTS[1] = "a person smoking a cigarette in a public place"` (index 1 = anomaly).

**Gradient-stability decision.** The reference implementation scales logits2 by 1/0.07 before softmax; at that temperature the two-class softmax saturates and gradients vanish on our data. Both training *and* inference scoring therefore use `logits2_raw` — the unscaled cosine in [−1, 1] — with p_align = softmax(logits2_raw)[:, :, 1]. The scaled logits2 is retained only for faithful reporting of the reference architecture.

**Combined scores.** For each frame: dual_mean = 0.5·(p_vis + p_align); dual_max = max(p_vis, p_align). All four score streams (visual, align, dual_mean, dual_max) are reported per variant in Section 6 and Appendix D.

## 5. Head-to-Head Results

### 5.1 Run configuration

The head-to-head run (`results/anomaly_headtohead/run.json`, created 2026-10-10T04:58:59Z) used these options, which are authoritative for every number in this section:

| Option | Value |
|---|---|
| manifest | `data/anomaly_manifest.csv` (SHA-256 `18087c4d…b2cc582`) |
| definitions | `scripts/anomalies.json` (SHA-256 `589882c0…a95038c`) |
| models | mobileclip2-s2, siglip2-b16-224, smolvlm-500m, minicpm-v4 |
| limit / repeats / warmup | 80 images / 1 scored repeat / 1 untimed warmup pass |
| seed | 42 |
| max_side (px) | 512 |
| max_new_tokens | 32 |
| timeout_seconds | 3600.0 |
| offline | true |
| threads / duty_cycle | 2 / 1.0 |

All four models completed with `status = ok`, zero errors, and full prediction coverage except SmolVLM (Section 5.5).

### 5.2 Accuracy and macro-F1

![Accuracy and macro-F1 with bootstrap CIs](figures/fig01_accuracy_macrof1_ci.png)

| Model | Accuracy incl. failures [95% CI] | Macro-F1 [95% CI] |
|---|---|---|
| MobileCLIP2-S2 | 0.1875 [0.1125, 0.275] | 0.0855 [0.0476, 0.1338] |
| SigLIP2-B/16-224 | 0.275 [0.175, 0.375] | 0.2125 [0.1355, 0.2823] |
| SmolVLM-500M | 0.0 [0, 0] (abstained on all) | — |
| MiniCPM-V | **0.375** [0.2625, 0.4875] | **0.2660** [0.1882, 0.3894] |

MiniCPM-V is the only model whose accuracy CI excludes chance (chance = 1/6 ≈ 0.167 for six labels; its lower bound 0.2625 clears it). SigLIP2's interval overlaps chance. MobileCLIP2-S2 sits at chance level — consistent with its AUC of Section 5.3. Macro-F1 is structurally capped at 5/6 ≈ 0.833 because unattended_items has no support (Section 9), so even MiniCPM-V's best value reflects a genuinely weak multi-class signal: every model's per-class F1s are low except isolated cells (Section 5.4).

### 5.3 Binary anomaly-vs-normal scoring quality

![Binary F1 and ROC-AUC with bootstrap CIs](figures/fig02_binary_f1_auc_ci.png)

| Model | Binary F1 [95% CI] | ROC-AUC [95% CI] |
|---|---|---|
| MobileCLIP2-S2 | 0.4598 [0.325, 0.5773] | 0.4590 [0.3209, 0.6011] |
| SigLIP2-B/16-224 | **0.8271** [0.7520, 0.8936] | **0.6260** [0.4859, 0.7591] |
| SmolVLM-500M | — (no usable scores) | — |
| MiniCPM-V | 0.400 [0.2588, 0.5238] | not computed (hard labels only) |

**Base-rate caveat.** The test set is 64/80 anomalous (prevalence 0.80). A degenerate always-anomaly classifier therefore scores binary F1 = 2·0.8·1/(0.8+1) = **0.889**. SigLIP2's headline 0.827 is ≈ 93% of that ceiling — it is largely a base-rate artifact, not strong discrimination. The threshold-free ROC-AUC is the honest measure: 0.626 [0.4859, 0.7591] indicates only moderate separation, and its CI includes values near chance. MobileCLIP2-S2's AUC of 0.459 is *below* chance (anti-predictive at this operating point). MiniCPM-V emits hard labels rather than calibrated scores, so its binary F1 (0.40) reflects label assignment only and no AUC was computed for it.
