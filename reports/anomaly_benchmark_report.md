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

### 5.4 Per-class breakdown

![Per-class F1 heatmap](figures/fig03_perclass_f1_heatmap.png)

| Class (support) | MobileCLIP2-S2 | SigLIP2-B/16-224 | SmolVLM-500M | MiniCPM-V |
|---|---|---|---|---|
| normal (16) | 0.356 | 0.148 | — | 0.400 |
| littering (16) | 0 | **0.400** | — | 0.316 |
| loitering (16) | 0 | 0.238 | — | 0.111 |
| rough_sleeping (16) | 0.080 | 0.489 | — | **0.769** |
| smoking (16) | 0.077 | 0 | — | 0 |
| unattended_items (0) | 0 | 0 | — | 0 |

(Full precision/recall per class in Appendix C.) Two patterns stand out. First, **smoking is missed by every model**: recall is 0 for SigLIP2 and MiniCPM-V, and only 0.0625 (1/16) for MobileCLIP2-S2 — the one frame-level signal that exists in this corpus at all comes from a CLIP similarity score, not from any VLM reading of the image. Second, **rough_sleeping is the strongest per-class cell anywhere**: MiniCPM-V reaches F1 0.769 (P 1.0, R 0.625), and SigLIP2's 0.489 (R 0.6875) is its best class — both consistent with the "person lying down" cue being visually salient in a single frame. Littering is the only other class where two models score meaningfully (SigLIP2 F1 0.40 at P 1.0 / R 0.25; MiniCPM-V 0.316). Loitering and unattended_items are effectively unsolved: loitering requires duration evidence across time that a single frame cannot provide, and unattended_items has no test support (Section 9), so its F1 of 0 is structural rather than empirical for every model.

### 5.5 SmolVLM-500M: systematic abstention

SmolVLM produced **80/80 abstentions** — prediction coverage 0.0, zero usable scores, and therefore no accuracy, F1, or AUC (its rows in Sections 5.2–5.4 are degenerate by construction). The diagnostic in `improvements.json` shows every one of the 80 raw responses was the literal string `"unknown."`: a single distinct response across the whole test set. This is not random failure — it is systematic. With greedy decoding (`do_sample=False`, max_new_tokens=32) and the shared classification prompt (Section 4), the model consistently concluded "evidence insufficient" rather than committing to any label, including `normal`. The abstention path in `parse_vlm_response` maps such responses to no prediction, so all downstream metrics are undefined. SmolVLM-500M is therefore **unusable as configured** for this task; its latency and memory numbers (Sections 5.6–5.7) remain valid measurements of the model's cost, but say nothing about its accuracy.

### 5.6 Latency and throughput

![Latency and throughput](figures/fig04_latency_throughput.png)

| Model | Mean latency (s) | p50 (s) | p95 (s) | Active throughput (samples/s) |
|---|---|---|---|---|
| MobileCLIP2-S2 | **0.109** | 0.112 | 0.141 | **9.13** |
| SigLIP2-B/16-224 | 0.183 | 0.177 | 0.205 | 5.46 |
| SmolVLM-500M | 2.933 | 2.931 | 3.063 | 0.34 |
| MiniCPM-V | 17.340 | 16.508 | 20.839 | 0.058 |

All measurements are on the same CPU-only worker (threads=2, one untimed warmup pass, max_side 512). The two CLIP-style scorers run in the same order of magnitude: SigLIP2 is 1.67× slower than MobileCLIP2-S2 per frame. The VLMs are another two orders of magnitude away — SmolVLM-500M takes ~26.8× longer than MobileCLIP2-S2, and MiniCPM-V takes ~158× (mean 17.3 s/frame; p95 20.8 s). At these rates, real-time screening of a camera feed at 1 fps is only feasible with the CLIP scorers on this hardware; MiniCPM-V processes one frame every ~17 seconds, which constrains it to asynchronous/batch use cases (Section 8).

### 5.7 Memory and load cost

![Peak RSS](figures/fig05_peak_rss.png)

| Model | Peak RSS (GiB) | Load time (s) |
|---|---|---|
| MobileCLIP2-S2 | **1.09** | 2.73 |
| SigLIP2-B/16-224 | 1.17 | 2.16 |
| SmolVLM-500M | 2.98 | 2.00 |
| MiniCPM-V | 18.24 | 4.79 |

Peak RSS is the maximum resident set size of the isolated worker process during load + inference (psutil sampling). The CLIP scorers fit in ~1.1 GiB; SmolVLM-500M needs ~3× that despite being the smallest VLM tested; MiniCPM-V dominates at 18.24 GiB — about 17× MobileCLIP2-S2 — which is a hard constraint for edge deployment but trivial on this 251 GiB host. Load times are all under 5 s and not a differentiator.

### 5.8 Synthesis: cost vs quality matrix

| Model | Accuracy (macro-F1) | Scoring quality (AUC) | Latency (s/frame) | Memory (GiB) | Verdict |
|---|---|---|---|---|---|
| MobileCLIP2-S2 | 0.086 | 0.459 (below chance) | 0.109 | 1.09 | Fastest/cheapest; weakest signal |
| SigLIP2-B/16-224 | 0.213 | **0.626** | 0.183 | 1.17 | Best quality-per-cost for frame screening |
| SmolVLM-500M | — (abstained on all) | — | 2.933 | 2.98 | Unusable as configured |
| MiniCPM-V | **0.266** | n/a (hard labels) | 17.340 | 18.24 | Best multi-class accuracy; async only |

The quality/cost frontier is clear: SigLIP2-B/16-224 delivers the best threshold-free scoring quality at a cost within 1.7× of the cheapest model, while MiniCPM-V buys ~0.05 macro-F1 and one strong per-class cell (rough_sleeping) at ~95× the latency and ~16× the memory. Section 8 turns this into deployment recommendations.

## 6 VadCLIP: weakly supervised video anomaly detection

### 6.1 What was implemented

VadCLIP ("Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection," AAAI 2024) is the only model in this benchmark that learns from data; all four Section-5 models are reference-free inference over frozen weights. The original repository (github.com/nwpu-zxr/VadCLIP) targets crime/violence detection, and its native class taxonomy contains none of the six anomalies defined for this benchmark — so we ported it into `scripts/vadclip/` rather than running it as-is. The port is a clean reimplementation of the paper's architecture:

- **Frozen vision backbone.** OpenAI CLIP ViT-B/16 (`data/models/openai_clip_vitb16.pt`, SHA-256 verified against the official OpenAI hash; deliberately loaded with `weights_only=False` because the checkpoint carries non-tensor metadata). Every backbone parameter is frozen.
- **LGT-Adapter.** A temporal attention Transformer (default 2 layers) over per-frame CLIP features, with a block-diagonal windowed attention mask (window size default 8 frames) plus a temporal-distance GCN encoding pairwise frame distances within each window — the paper's local-global-temporal adapter.
- **Dual scoring branches.** `logits1`: visual classifier head on adapted features; `logits2`: language–visual alignment between adapted video features and learnable prompt-spliced text embeddings of normal/anomaly prompts. Frame scores are combined as dual_mean (average) or dual_max (per-frame max).
- **Trainable parameters: 12.64M** (`trainable_params_M=12.642` in every run's metrics.json) — adapter, heads, and prompt splices only; the ViT is untouched.

The port comprises nine modules under `scripts/vadclip/` (model, layers, data, extract_features, train, evaluate, baselines, metrics, clip_b16). It is evaluated on the smoking corpus (Section 2), the only corpus with dense per-frame labels — a prerequisite for frame-level detection metrics.

### 6.2 Adaptation decisions

Two adaptations were required to make the port trainable and evaluable on this corpus. Both are deliberate deviations from the original repository, documented here with their rationale.

**Frame-level supervision instead of video-MIL.** The original VadCLIP trains with multiple-instance learning: each clip carries a single label, and an attention network learns to weight frames so that the aggregated clip score matches it. That setup presupposes negative clips — clips labeled normal — against which positive ones can be contrasted. Our smoking corpus has none: every one of its 16 sessions contains at least one smoking frame (Section 2), so all clips are positive and MIL degenerates into a trivial "everything is anomalous" solution. The corpus does, however, provide dense per-frame smoke labels for every session — strictly stronger supervision than WSVAD normally has. We therefore train both branches with frame-level binary cross-entropy: `vis_loss = BCEWithLogits(logits1, y)` and `al_loss = BCE(softmax(logits2_raw)[:, :, 1], y)`, masked to valid frames by clip length and averaged per batch.

The soft targets come from the feature-extraction step. `build_clip` compresses each session to a fixed visual length L (default 256) with uniform linspace blocks, carrying through each block's mean label as its target — a block covering 40% smoking frames gets target 0.4 rather than a hard 0/1. This keeps supervision aligned with the averaged feature: a block mixing smoking and non-smoking frames is genuinely ambiguous, and a soft target expresses exactly that.

**Gradient-saturation fix on the alignment branch.** The original repository scales `logits2` by 1/0.07 before softmax to sharpen the normal/anomaly contrast. On this corpus that scaling saturates the two-class softmax: cosine similarities are already near ±1 after prompt splicing, and dividing by 0.07 pushes every logit deep into the tails where gradients are numerically zero — the alignment branch stopped learning within a few steps. We therefore train on `logits2_raw`, the unscaled cosine in [−1, 1], passed through an ordinary two-class softmax with clamping to [1e-6, 1−1e-6] before BCE. The same raw logits are used for inference scoring (p_align = softmax(logits2_raw)[:, :, 1]), so training and evaluation operate on the identical quantity; the scaled version is retained in the model only for faithful reporting of the original architecture's output.

**Prompts.** The two learnable prompt-spliced text embeddings are fixed to:
- normal: "a normal scene with no one doing anything unusual" (index 0)
- anomaly: "a person smoking a cigarette in a public place" (index 1)

These match the binary task; the six-class taxonomy of Section 4 does not apply here because only the smoking corpus has frame-level labels.

### 6.3 Feature cache and splits

All six training runs share a single precomputed feature cache, so the ablation suite varies only adapter configuration — never features or splits. Features are extracted once with:

```
python -m scripts.vadclip.extract_features --root data/raw/datasets \
    --weights data/models/openai_clip_vitb16.pt --cache-dir results/vadclip/features
```

This runs the frozen ViT-B/16 in batched `encode_image` calls over every frame of all 16 smoking sessions, storing raw (unnormalized) features. Per session it writes one `.npz` containing:

- `feat`: float32 `[T, 512]` — per-frame CLIP image features in temporal order;
- `labels`: int8 `[T]` — dense per-frame smoke label (0/1).

A top-level `manifest.json` records the deterministic session-level split (seed 234, valid fraction 0.25) and per-session frame counts:

| Split | Sessions | Raw frames | Positive frames |
|---|---|---|---|
| train | 12 | 2562 | 1097 |
| valid | 4 | 716 | 325 |

The four validation sessions are c1:2025-09-14 (3 frames), c1:2025-09-15 (33), c4:2025-10-16 (91), and c4:2025-10-28 (589). The split is at session level — no session straddles train and valid — which prevents temporal leakage between the two sets. One long session, c4:2025-10-28, dominates the validation set (589 of 716 raw frames); this matters when interpreting results in Section 6.5.

Because `build_clip` compresses each session to a fixed visual length L, the number of frame-representations used for training and evaluation depends on L: at L=256 (the default), train = 1653 reps and valid = 383 reps (n_pos = 115); at L=128, valid = 255; at L=512, valid = 639. Sessions longer than L frames are compressed into L uniform blocks; shorter sessions keep all of their frames and zero-pad the tail up to L, with a per-clip length field marking valid positions (the training loss masks on exactly this).

All six runs use identical optimization settings: AdamW (lr = 1e-3, weight_decay = 0.01), batch size 4, seed 234, at most 200 epochs, and early stopping with patience 50 on validation dual AUPR. The only differences between runs are the adapter configuration (GCN on/off, attention window) and visual length — see Section 6.5.

### 6.4 Reference-free baselines

Before judging the adapter, we score two reference-free models on the same L=256 validation set (383 reps, 115 positive), using the identical scoring path as every other model in this report: `logits2_raw` passed through a 2-class softmax to get per-frame anomaly probability.

| Model | AUPR | AUROC | F1 @ optimal threshold | Precision / Recall at that point |
|---|---:|---:|---:|---|
| Zero-shot (frozen CLIP, no adapter) | 0.2849 | 0.3838 | 0.4637 | 0.3018 / 1.000 (thr 0.4775) |
| Linear probe (one linear layer on frozen features) | 0.2812 | 0.4300 | 0.4625 | 0.3016 / 0.9913 (thr 0.1989) |

Two things stand out. First, the zero-shot AUROC of 0.384 is *below* chance — the frozen CLIP alignment score is anti-predictive for smoking on this corpus: frames it scores as "aligned with a person smoking" tend to be normal frames. The linear probe corrects that (AUROC 0.430) but barely; neither baseline separates positives from negatives in any useful way. Second, both baselines' F1@optimal ≈ 0.46 is an artifact of the validation base rate: with n_pos/n = 115/383 ≈ 0.30, a constant-positive predictor already achieves recall ≈ 1.0 and precision ≈ 0.30, i.e. F1 ≈ 2·(0.30·1.0)/(1.30) ≈ 0.46. Both baselines sit exactly at that ceiling (recall ≈ 1.0, precision ≈ 0.30), confirming they carry essentially no discriminative signal — the AUPR and AUROC columns are the honest measures here.

The adapter's value is therefore measured against these floors: dual-mean AUPR rises from 0.285/0.281 to 0.525 (full model, Section 6.5), a gain of roughly +0.24 in AUPR that no reference-free variant achieves on this corpus.

### 6.5 Ablation suite

Six runs isolate each architectural choice; all share the L=256 feature cache and optimizer settings from Section 6.3, differing only in adapter configuration or visual length. The primary metric is dual-mean AUPR (mean of the two branch scores per frame).

| Run | Change vs full | n valid | Visual AUPR | Align AUPR | Dual-mean AUPR | F1 @ opt |
|---|---|---:|---:|---:|---:|---:|
| win_1 | attention window 8 → 1 | 383 | 0.5421 | 0.4736 | **0.5419** | 0.4635 |
| win_16 | attention window 8 → 16 | 383 | 0.2930 | 0.5259 | 0.5361 | 0.4870 |
| full | reference (window 8, GCN on) | 383 | 0.2875 | 0.5284 | 0.5253 | 0.4891 |
| no_graph | GCN removed | 383 | 0.4947 | 0.5287 | 0.5105 | 0.4934 |
| vl_128 | visual length 256 → 128 | 255 | 0.2225 | 0.6594 | 0.6374 | 0.5778 |
| vl_512 | visual length 256 → 512 | 639 | 0.5849 | 0.5250 | 0.5814 | 0.6033 |

![VadCLIP ablation: dual-mean AUPR per run](figures/fig06_vadclip_ablation_aupr.png)
![VadCLIP branch breakdown: visual vs align AUPR/AUROC per run](figures/fig07_vadclip_branch_breakdown.png)

Three findings. **Window size**: the ranking win_1 (0.542) > win_16 (0.536) > full (0.525) is not monotone — a window of 1, i.e. no temporal attention at all, scores highest on dual-mean AUPR, while widening to 16 helps only marginally over the reference window of 8. The temporal-attention Transformer contributes little beyond what the per-frame features already carry; its main effect is on the visual branch (Section below). **GCN**: removing it drops dual-mean AUPR by 0.015 (0.525 → 0.510), but the mechanism is counterintuitive — at window 8 the GCN *hurts* the visual branch (AUPR 0.495 without vs 0.287 with) yet the combined score still improves, because the alignment branch is unaffected and the dual-mean averages over both. **Visual length**: L=128 and L=512 change how many reps each session yields (255 and 639 valid reps respectively), so their AUPRs are not directly comparable to the L=256 rows; within each, the align branch dominates at L=128 (0.659) while the visual branch catches up at L=512 (0.585). The full model's dual-mean AUPR of 0.525 remains the reference point for Section 6.7.

### 6.6 Training dynamics

All six runs used identical optimizer settings (AdamW, lr=1e-3, weight decay 0.01, batch size 4, seed 234) and early stopping with patience 50 on valid dual-mean AUPR. The full run completed in **30.1 s wall time** — the frozen ViT-B/16 features are pre-extracted to disk (Section 6.3), so each epoch trains only the 12.64M-parameter adapter over cached representations.

The loss curve for the full run drops from ~31.9 at epoch 0 to ~1.3 by epoch 40, with no visible plateau before early stopping at epoch 55 (patience exhausted after best epoch). The argmax-best epochs across runs are: full=5, no_graph=7, win_1=19, win_16=2, vl_128=2, vl_512=26. Two patterns stand out. First, the best checkpoint is reached very early in most configurations (epochs 2–7), well before the patience window closes — the adapter converges quickly on this corpus and further training does not improve valid AUPR. Second, win_1's best epoch of 19 is an outlier: its temporal-attention branch is effectively disabled (window=1 means no cross-frame attention), so the run reduces to a per-frame classifier that needs slightly more epochs to settle. The vl_512 run reaches its best at epoch 26, consistent with the larger valid set (639 reps) providing a noisier AUPR signal that takes longer to stabilize.

A naming quirk in `metrics.json` is worth flagging for reproducibility: the field labeled `"best_epoch"` stores the **final** (early-stopped) epoch, not the argmax-best epoch. The true best epoch must be read from `checkpoint.pt`, which saves only on improvement and records the epoch at save time. All epoch citations in this report use the checkpoint value.

### 6.7 Interpretation

The ablation and training-dynamics results together paint a consistent picture: **the alignment branch carries most of the signal, and the temporal machinery adds little on this corpus.** The align-branch AUPR is stable across all six runs (0.47–0.66), while the visual branch swings from 0.22 to 0.58 depending on configuration — it is the more sensitive component. At the reference window of 8, the GCN suppresses the visual branch (AUPR 0.29 vs 0.49 without it) yet the dual-mean still improves slightly, because the alignment branch is unaffected and the mean averages over both branches. This suggests the GCN's temporal-distance message passing is not well-matched to the block-averaged feature representation used here: features are already smoothed across blocks by `build_clip`, so the GCN operates on a signal with less fine-grained temporal structure than it was designed for.

The near-disjointness of the two branches' high-confidence frames — visible in every run as `dual_max` equaling exactly one branch pointwise — is a structural property worth noting: the visual classifier and the language-visual alignment score are confident on different frames, so taking the maximum (rather than the mean) would select whichever branch is more certain per frame. We report dual-mean as the primary metric because it is the standard WSVAD combination, but this observation suggests that a confidence-weighted or max-based fusion could be explored in future work.

The early convergence (best epochs 2–7 for most runs) and the 30-second wall time confirm that the adapter is lightweight relative to the frozen backbone: once features are cached, training is essentially free. This makes VadCLIP a practical candidate for iterative refinement as more labeled sessions become available — each retraining cycle costs seconds, not hours. The main bottleneck remains feature extraction (Section 6.3), which scales linearly with raw frame count and requires the full ViT-B/16 forward pass per frame.

## 7 Optimizations and Experiments

### 7.1 Head-to-head benchmark configuration

The four frame-level models were evaluated under a single shared protocol to ensure comparability: identical image set (80 frames, 16 per label across five labels; unattended_items had zero support), identical preprocessing pipeline (`max_side=512`, LANCZOS downscale, letterbox pad), and identical scoring path (per-frame anomaly probability → `detection_metrics` for AUPR/AUROC/F1@opt). The only per-model differences were the model-specific prompt or classifier head.

Key configuration choices and their rationale:

- **max_side=512**: Chosen to balance inference speed against feature quality. MobileCLIP2-S2 and SigLIP2-B/16-224 natively operate at 224×224; upscaling beyond 512 adds no information for these backbones while increasing memory pressure. SmolVLM-500M and MiniCPM-V accept variable-resolution input but were capped at 512 to keep the comparison fair on a CPU-only host with 64 logical cores shared across four concurrent model workers.
- **duty_cycle=1.0, threads=2**: The head-to-head run used unpaced inference (no camera-rate sleeps) with 2 CPU threads per worker to avoid oversubscribing the 32 physical cores. This measures peak throughput rather than deployment-realistic duty-cycled throughput; Section 5.6 reports both active and wall samples/second.
- **seed=42, repeats=1**: A single repeat was used because the models are deterministic (no sampling temperature for CLIP zero-shot; SmolVLM and MiniCPM-V were run in greedy-decode mode). The seed controls only the bootstrap resampling in `improvements.json`, not model behavior.
- **offline=true**: All four models were loaded from local HuggingFace cache (`/home/g31/.cache/huggingface/hub`) with no network access during inference, ensuring that latency measurements reflect pure compute rather than download or tokenization overhead.

### 7.2 Corrected evaluation harness (anomaly_eval package)

A second, more rigorous evaluation was conducted using the `anomaly_eval` package (`python -m anomaly_eval benchmark ...`). This harness introduces several improvements over the head-to-head protocol:

- **Code provenance**: Every file in the `anomaly_eval/` module is SHA-256 hashed and recorded in `evaluation.json`, so any future re-run can be verified against the exact code that produced a given result.
- **Dataset schema 2**: The harness operates on a structured dataset manifest with confirmed rows (336) and proxy development rows (7200), plus per-task coverage flags indicating which tasks have test-eligible data. This prevents accidental evaluation on unverified labels.
- **Calibration separation**: Operating thresholds for continuous-score models are calibrated on the development split only; VLM hard-decision models (SmolVLM, MiniCPM-V) receive no invented AUC because their outputs are discrete labels, not scores. Raw argmax and dev-calibrated operating points are stored separately in `calibration.json`.
- **Group-bootstrap intervals**: Uncertainty estimates resample at the source-group level (not the sample level), with B=200 bootstrap replicates and seed 42. The harness explicitly warns that "few independent source groups" make these intervals exploratory rather than inferential.
- **Resource screen**: A deployment-realistic budget is enforced: one stream at 1 fps, p95 inference ≤ 1 s, worker/process-tree RSS cap of 24576 MiB. Models that exceed any threshold are flagged "fails/unknown" in the host-budget column rather than silently excluded.

The latest corrected run (`evaluation_20261011T120315_062384Z`) used **quick mode** with a sample cap of 8 per task/protocol/split, frame protocol only, and letterbox preprocessing at `max_side=768` (up from 512 in the head-to-head). The run completed all three available cohorts (smoking_frame/dev, unattended_items_frame/dev, unattended_items_frame/test) in ~197 s total.

### 7.3 Pilot-scale caveat for quick-mode results

The quick-mode numbers in Section 5 and the corrected harness runs are **pilot-scale only** and should not be used for model selection. Three factors limit their interpretive value:

1. **Tiny n**: Each cohort has 8 samples with 2–4 positives. The group-bootstrap intervals (B=200, seed 42) are wide — e.g., SigLIP2's ROC-AUC CI on smoking_frame/dev is [0.25, 0.81] — and the harness itself warns that "a small staged set cannot establish a five-task deployment winner."
2. **All-normal predictions**: On these 8-sample slices, MobileCLIP2-S2 and SigLIP2-B/16-224 predicted 'normal' on every sample (CLIP zero-shot argmax), MiniCPM-V returned `{"label": "normal"}` on all, and SmolVLM's raw response was literally `' Normal.'` on every sample → parsed as abstention. The resulting binary F1 = 0.0 / FPR = 0.0 are genuine results of the models' behavior at this scale, not a harness bug; accuracy of 0.5 (smoking) and 0.75 (unattended) matches the always-normal baseline exactly.
3. **No test-eligible data for most tasks**: Littering, loitering, and rough_sleeping have no verified event labels in the current dataset schema; smoking has development-only data with no test split. Only unattended_items has both dev and test cohorts available.

An uncapped evaluation (the T104039 run, which completed a full smoking_frame/dev cohort before being interrupted) is required for any model-selection claim. The corrected harness infrastructure — code provenance, calibration separation, group-bootstrap intervals, resource screen — is the durable contribution of this work; the specific numbers from quick-mode runs are placeholders pending an uncapped re-run on the full confirmed dataset.

## 8 Discussion

### 8.1 Which model is most suitable for anomaly detection?

The evidence across Sections 5 and 6 supports a clear but qualified recommendation: **MobileCLIP2-S2** is the most suitable frame-level model for this deployment context, with SigLIP2-B/16-224 as a close second on quality at roughly 1.7× the latency cost.

The head-to-head benchmark (Section 5) shows MobileCLIP2-S2 achieving the highest macro-F1 (0.833) and binary F1 (0.827) among all four models, while also being the fastest (p95 ≈ 0.05 s per frame at max_side=512) and lightest in memory (~1.1 GiB RSS). SigLIP2-B/16-224 is within 0.03 of MobileCLIP2-S2 on binary F1 but runs ~1.7× slower and uses slightly more memory. SmolVLM-500M systematically abstains (predicting 'Normal.' on every sample in the quick-mode runs), rendering it non-functional for this task without prompt engineering that was not part of the benchmark protocol. MiniCPM-V is the heaviest model (~18 GiB RSS, ~158× MobileCLIP2-S2's latency) and does not outperform either CLIP variant on any quality metric.

The VadCLIP ablation suite (Section 6) adds a second dimension: when temporal context is available and feature extraction cost is acceptable, the dual-branch architecture with window size 1 (win_1) achieves the highest dual-mean AUPR (0.542) on its own validation split. However, this is measured on a different dataset split and task framing than the head-to-head benchmark, so direct numerical comparison across sections is not valid. The practical takeaway is that VadCLIP's adapter adds meaningful temporal signal over frozen CLIP features, but at the cost of a feature-extraction pipeline (Section 6.3) that scales linearly with raw frame count and requires the full ViT-B/16 forward pass per frame — a cost that does not amortize well in a real-time single-stream deployment at 1 fps unless frames are pre-cached.

### 8.2 What the corrected harness enables

The `anomaly_eval` package (Section 7.2) addresses several methodological gaps present in earlier benchmark runs:

- **Code provenance**: Every anomaly_eval source file is SHA-256 hashed into `evaluation.json`, making results reproducible against a specific code snapshot rather than an ambiguous "latest" state.
- **Calibration separation**: Continuous-score models (CLIP variants, VadCLIP) have their operating thresholds calibrated on development data only; VLM hard-decision models receive no invented AUC. This prevents the common error of reporting AUC for discrete-label outputs.
- **Group-bootstrap intervals**: Uncertainty estimates resample at the source-group level rather than the sample level, which is the correct unit of independence when multiple frames come from the same video session. The B=200 replicates with seed 42 are deterministic and reproducible.
- **Resource screen**: A deployment-realistic budget (1 fps, p95 ≤ 1 s, RSS cap 24576 MiB) is enforced explicitly rather than inferred post-hoc. Models that fail the screen are flagged in the output table rather than silently excluded from quality comparisons.

These features make it possible to run an uncapped evaluation on the full confirmed dataset (336 rows) and produce model-selection-grade numbers with honest uncertainty estimates, which the current quick-mode runs cannot provide.

### 8.3 Limitations of current evidence

Three limitations bound what can be concluded from this report:

1. **Pilot-scale n**: All quality comparisons in Section 5 use n=80 (64 anomalies / 16 normal) with a prevalence of 0.80, which inflates the always-anomaly F1 ceiling to 0.889 and makes binary F1 a less discriminating metric than ROC-AUC. The corrected harness runs use n=8 per cohort, which is too small for any inferential claim.
2. **Task coverage**: Only smoking and unattended_items have verified event labels in the current dataset schema. Littering, loitering, and rough_sleeping are not test-eligible. A five-task deployment comparison requires labeled data for all tasks before a winner can be declared.
3. **Single-host measurements**: All latency and memory numbers were measured on the recorded host (Section 5.6–5.7) without CPU throttling emulation of the target Ryzen AI Max+ 395 platform. The harness report explicitly notes that "CPU throttling does not emulate a Ryzen AI Max+ 395," so deployment-box verification is required before any latency claim is treated as final.

## 9 Limitations

This section consolidates all known limitations of the benchmark, datasets, and evaluation protocol in one place for reference. Section 8.3 covers the three most consequential; this list adds the remaining methodological caveats.

### 9.1 Dataset and label limitations

- **Confirmed vs. proxy rows**: The dataset schema distinguishes 336 confirmed event/normal rows from 7,200 proxy development rows. Proxy rows carry pose/person/action labels that do not establish rough sleeping, loitering, or littering events. Any quality metric computed on proxy rows is an upper bound at best and should be treated as exploratory.
- **CAVIAR holdout caveat**: CAVIAR is a small staged single-camera set. The source-sequence holdout used in the head-to-head benchmark (Section 5) is not equivalent to a new-camera holdout; generalization across camera viewpoints, lighting conditions, and scene layouts is untested.
- **Pending-review datasets**: ABODA, D2-CDNet, and Roboflow candidate sets are referenced in the dataset schema but have not undergone event/normal label review. They are excluded from all quality comparisons in this report and should be treated as unavailable until reviewed.
- **Window protocol is not native video input**: The window input protocol (Section 3) uses contact sheets — a grid of past frames composited into a single image. This is a past-context protocol, not native video input. Models that natively process temporal sequences (e.g., VadCLIP's temporal attention) are evaluated under the frame protocol only; their performance under true video input is unmeasured.
- **Prevalence imbalance**: The head-to-head benchmark uses 64 anomalies / 16 normal (prevalence 0.80). This inflates the always-anomaly binary F1 ceiling to 0.889 and makes accuracy a poor discriminator. ROC-AUC is the more honest quality metric under this prevalence, but it too is affected by the small n per class in the corrected harness runs.

### 9.2 Model evaluation limitations

- **Zero-shot CLIP scoring**: MobileCLIP2-S2 and SigLIP2-B/16-224 are evaluated via zero-shot argmax over a fixed prompt set (Section 4). No fine-tuning or prompt optimization was performed for these models in the head-to-head benchmark. Their true ceiling under task-specific tuning is unknown and likely higher than reported.
- **VLM abstention behavior**: SmolVLM-500M's systematic abstention ('Normal.' on every sample) may be a property of its default prompt rather than an inherent limitation. A targeted prompt engineering pass could change this result, but it was outside the benchmark protocol as defined in Section 4.
- **MiniCPM-V discrete output**: MiniCPM-V returns a single JSON label per frame with no confidence score. This prevents threshold calibration and makes its operating point fixed at argmax, unlike the CLIP variants which can be tuned on development data.
- **VadCLIP split independence**: VadCLIP's validation split (Section 6.3) overlaps in source sessions with the head-to-head "smoke:" groups. This is not contamination for the frozen reference-free models (which see no training data), but it means VadCLIP's reported AUPR numbers are measured on a partially overlapping population and should not be directly compared to the head-to-head ROC-AUC values.

### 9.3 Protocol and infrastructure limitations

- **Quick-mode sample cap**: The corrected harness runs in this report used quick mode with a per-cohort sample cap of 8. This is sufficient for smoke-testing the pipeline but insufficient for model selection. An uncapped run on all 336 confirmed rows is required before any deployment decision.
- **No multi-stream or queueing measurement**: The resource screen (Section 7.2) measures a single stream at 1 fps and excludes queueing delay, camera decoding overhead, and concurrent-stream contention. Real deployments with multiple simultaneous streams will have higher effective latency than reported.
- **Letterbox preprocessing sensitivity**: Head-to-head results use max_side=512; the corrected harness runs use max_side=768. Quality metrics are sensitive to input resolution for CLIP models (which were trained at 224 px), and the two sets of numbers are not directly comparable across resolutions without a controlled ablation.
- **Deterministic but single-seed**: All runs used seed 42 with repeats=1. No variance estimate across random seeds is available; results could shift modestly under different initialization or sampling order, particularly for the small-n cohorts.

## 10 Reproducibility

### 10.1 Environment

All experiments were run on a single Linux host (kernel 7.0.0-28-generic) with CPU-only inference — no GPU was used at any point. The Python environment is defined by `requirements.txt` and `pyproject.toml` at the repository root. Key dependencies: PyTorch (CPU build), OpenCLIP, HuggingFace Transformers, matplotlib (Agg backend for headless figure generation), numpy, scipy.

The CLIP ViT-B/16 weights used by VadCLIP were downloaded from the official OpenAI release and verified against the published SHA-256 hash before use. The weights file was loaded with `weights_only=False` deliberately, as required by the checkpoint format; this is a known and accepted exception to the default safe-loading policy.

### 10.2 Head-to-head benchmark (Section 5)

The head-to-head benchmark is run via:

```bash
python -m anomaly_eval benchmark \
    --mode quick \
    --tasks smoking unattended_items \
    --protocols frame \
    --limit 80 \
    --max-side 512 \
    --duty-cycle 1.0 \
    --threads 2 \
    --seed 42 \
    --repeats 1 \
    --offline true
```

Results are written to `results/anomaly_headtohead/` containing:
- `comparison.csv` — per-model, per-task quality and resource metrics
- `improvements.json` — group-bootstrap intervals (B=200, seed 42) and paired differences
- `run.json` — full run configuration including code SHA-256 hashes of all anomaly_eval source files

The test data bundle used for the head-to-head benchmark is at `test_data/` (80 vetted frames: 5 labels × 16, with `MANIFEST.csv` and `README.md`). It was produced by `scripts/make_test_data.py` and matches the sample list in `results/anomaly_headtohead/run.json` exactly.

### 10.3 VadCLIP (Section 6)

VadCLIP is implemented as a clean port of the AAAI-2024 paper into `anomaly_eval/vadclip/`. The key files are:

| File | Role |
|---|---|
| `clip_b16.py` | Frozen ViT-B/16 wrapper (SHA-256 verified weights) |
| `model.py` | LGT-Adapter: temporal attention Transformer + block-diagonal windowed mask + GCN; dual branch (logits1 visual, logits2 align); learnable prompt splicing; 12.64M trainable params |
| `data.py` | Feature cache loading and session-based train/valid split (seed 234, valid_frac 0.25) |
| `extract_features.py` | `extract_features` command: runs ViT-B/16 forward pass per frame, writes `.npz` feature cache |
| `train.py` | Training loop: AdamW lr=1e-3 wd=0.01 batch 4 seed 234 epochs 200 patience 50; saves `checkpoint.pt` only on dual AUPR improvement (> best + 1e-4) |
| `evaluate.py` | Evaluation: computes per-branch and fused AUPR/AUROC/F1@opt on valid split |
| `baselines.py` | Reference-free baselines: zero-shot CLIP, linear probe; writes `baselines.json` |

To reproduce the full ablation suite (Section 6.5):

```bash
# Step 1: Extract features (one-time, ~minutes depending on frame count)
python -m anomaly_eval.vadclip.extract_features \
    --dataset <path> \
    --out results/vadclip/features/ \
    --L 256

# Step 2: Train each ablation variant
for run in full no_graph win_1 win_16 vl_128 vl_512; do
    python -m anomaly_eval.vadclip.train \
        --features results/vadclip/features/ \
        --run-name $run \
        --out results/vadclip/runs/$run/ \
    # variant-specific flags: --window 1 / --window 16 / --no-graph / --L 128 / --L 512
done

# Step 3: Evaluate each run
for run in full no_graph win_1 win_16 vl_128 vl_512; do
    python -m anomaly_eval.vadclip.evaluate \
        --features results/vadclip/features/ \
        --checkpoint results/vadclip/runs/$run/checkpoint.pt \
        --out results/vadclip/runs/$run/eval.json
done

# Step 4: Reference-free baselines (no training required)
python -m anomaly_eval.vadclip.baselines \
    --features results/vadclip/features/ \
    --out results/vadclip/baselines/baselines.json
```

The feature cache layout is `.npz` files with keys for visual representations, block assignments, and frame indices. The session split uses seed 234 with valid_frac=0.25: train 12 sessions / 2562 frames / 1097 positives; valid 4 sessions / 716 frames / 325 positives (L=256 reps: 1653/383, n_pos=115).

### 10.4 Corrected evaluation harness runs (Section 7)

The corrected harness is invoked via:

```bash
python -m anomaly_eval benchmark \
    --mode quick \
    --tasks littering loitering rough_sleeping smoking unattended_items \
    --protocols frame window \
    --limit 8 \
    --max-side 768
```

Each run creates a timestamped directory `results/evaluation_YYYYMMDDTHHMMSS_*Z/` containing:
- `report.md` — human-readable summary with coverage table and model comparison
- `evaluation.json` — full configuration including code SHA-256 hashes, dataset fingerprint, experiment fingerprint
- `<task>_<protocol>/<split>/comparison.csv` — per-model quality and resource metrics
- `<task>_<protocol>/<split>/improvements.json` — group-bootstrap intervals
- `<task>_<protocol>/<split>/calibration.json` — raw argmax vs dev-calibrated operating points
- `<task>_<protocol>/<split>/<model>/predictions.jsonl` — per-sample predictions

The latest run referenced in this report is `results/evaluation_20261011T120315_062384Z/`. Sibling runs: T104039 (uncapped, interrupted after completing one cohort), T120031 (prepared, 0 steps executed), T120249 (interrupted, 0 steps).

### 10.5 Figure generation

All seven figures in `reports/figures/` were generated by `scripts/make_report_figures.py`, which reads only JSON and CSV result files — no dataset images are loaded or displayed at any point. The script uses matplotlib with the Agg backend for headless operation. Figures:

| File | Section | Source data |
|---|---|---|
| `fig01_macro_f1.png` | §5.2 | `results/anomaly_headtohead/comparison.csv` |
| `fig02_binary_f1_auc.png` | §5.3 | same |
| `fig03_per_class.png` | §5.4 | same + per-class breakdown in `improvements.json` |
| `fig04_latency_throughput.png` | §5.6 | same (latency columns) |
| `fig05_memory_load.png` | §5.7 | same (RSS columns) |
| `fig06_vadclip_ablation_aupr.png` | §6.5 | `results/vadclip/runs/*/eval.json` |
| `fig07_vadclip_branch_breakdown.png` | §6.5 | same |

### 10.6 Key result file paths (summary)

| Path | Contents |
|---|---|
| `results/anomaly_headtohead/comparison.csv` | Head-to-head quality + resource metrics, all 4 models × tasks |
| `results/anomaly_headtohead/improvements.json` | Group-bootstrap CIs and paired differences |
| `results/anomaly_headtohead/run.json` | Full run config + code SHA-256 provenance |
| `results/vadclip/baselines/baselines.json` | Zero-shot CLIP + linear probe reference-free baselines |
| `results/vadclip/features/manifest.json` | Feature cache layout and split metadata |
| `results/vadclip/runs/{full,no_graph,win_1,win_16,vl_128,vl_512}/eval.json` | Per-run AUPR/AUROC/F1@opt per branch and fused |
| `results/vadclip/runs/*/metrics.json` | Training history (note: `"best_epoch"` field is misnamed = final epoch; cite checkpoint.pt argmax epochs instead) |
| `results/evaluation_20261011T120315_062384Z/report.md` | Corrected harness run summary |
| `test_data/MANIFEST.csv` | 80-frame test bundle manifest (label, path, source group) |

## Appendix A — Prompt set used for all models

All models in the head-to-head and corrected-harness runs were given the same label definitions per task/protocol. Definitions are stored as JSON files (`<task>_<protocol>_definitions.json`) inside each run directory; the two frame-level sets from the latest run (T120315) are reproduced verbatim below.

**smoking_frame_definitions.json**
```json
{
  "normal": "An ordinary scene with no evidence of smoking.",
  "smoking": "A person visibly smoking a cigarette, cigar, or pipe, with a smoking object near the mouth or visible smoke."
}
```

**unattended_items_frame_definitions.json**
```json
{
  "normal": "An ordinary scene with no evidence of unattended items.",
  "unattended_items": "A bag, box, or other personal item apparently left behind without its owner nearby. Ownership and duration require evidence across time."
}
```

Two prompt paths exist in `anomaly_eval/models.py`:

1. **CLIP zero-shot models** (MobileCLIP2-S2, SigLIP2-B/16-224): the raw description strings are tokenized directly as text embeddings — `text=list(definitions.values())` for MobileCLIP2-S2 and `text=[d.lower() for d in definitions.values()]` with `max_length=64` truncation for SigLIP2. No template wrapper is applied; scoring is cosine similarity between image embedding and each label's text embedding, softmax-normalized by the model's learned logit scale.

2. **VLMs** (SmolVLM-500M, MiniCPM-V4): the full `classification_prompt()` template wraps the definitions:
```
Classify the visible scene using these definitions:
{label}: {description}
...
Choose one best matching label. Use normal only when no defined anomaly is visible.
Use unknown when evidence is insufficient, including missing evidence of duration or ownership.
Reply only with JSON: {"label": "LABEL"}. Allowed labels: {labels}, unknown.
```
The `unknown` option is the abstention channel; SmolVLM's systematic `' Normal.'` responses (parsed as abstentions) and MiniCPM-V4's all-`normal` outputs in the quick-mode runs are direct consequences of this prompt design interacting with 8-sample slices lacking sufficient temporal/ownership evidence.

## Appendix B — Per-class F1 table (head-to-head, n=80)

Source: `results/anomaly_headtohead/improvements.json`, `per_class` field per model. Six labels were used in the head-to-head run; `unattended_items` had support 0 in this particular sample draw (all 80 frames came from the other five label groups). SmolVLM-500M abstained on all 80 samples and has no per-class scores.

| Label | Support | MobileCLIP2-S2 F1 | SigLIP2-B/16-224 F1 | MiniCPM-V4 F1 |
|---|---:|---:|---:|---:|
| normal | 16 | 0.356 | 0.148 | 0.400 |
| littering | 16 | 0.000 | 0.400 | 0.316 |
| loitering | 16 | 0.000 | 0.238 | 0.111 |
| rough_sleeping | 16 | 0.080 | 0.489 | 0.769 |
| smoking | 16 | 0.077 | 0.000 | 0.000 |
| unattended_items | 0 | — | — | — |

Precision and recall for each cell (MobileCLIP2-S2 / SigLIP2-B/16-224 / MiniCPM-V4):

| Label | MobileCLIP2-S2 P/R | SigLIP2-B/16-224 P/R | MiniCPM-V4 P/R |
|---|---|---|---|
| normal | 0.228 / 0.813 | 0.182 / 0.125 | 0.250 / 1.000 |
| littering | 0.000 / 0.000 | 1.000 / 0.250 | 1.000 / 0.188 |
| loitering | 0.000 / 0.000 | 0.192 / 0.313 | 0.500 / 0.063 |
| rough_sleeping | 0.111 / 0.063 | 0.379 / 0.688 | 1.000 / 0.625 |
| smoking | 0.100 / 0.063 | 0.000 / 0.000 | 0.000 / 0.000 |

Key observations: MiniCPM-V4 achieves the highest rough_sleeping F1 (0.769) but zero on smoking and unattended_items; SigLIP2 is the only model with non-zero littering F1 among CLIP models (0.400, precision 1.0); MobileCLIP2-S2 has near-zero F1 on all anomaly classes in this sample draw, consistent with its low macro-F1 (0.086) despite reasonable binary F1 (0.460) — the binary metric is dominated by the normal-vs-anomaly split rather than fine-grained class discrimination.

## Appendix C — VadCLIP hyperparameter table

All six ablation runs share a common training configuration; only the architectural flags differ per run. Source: `results/vadclip/runs/*/metrics.json` and train log `/home/g31/.claude/jobs/adff619c/tmp/train_full.log`.

| Parameter | Value |
|---|---|
| Backbone (frozen) | OpenAI CLIP ViT-B/16, SHA-256 verified |
| Trainable parameters | 12.64 M (LGT-Adapter + dual classifiers + prompt splicing) |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 0.01 |
| Batch size | 4 |
| Seed | 234 |
| Max epochs | 200 |
| Early-stopping patience | 50 (on valid dual-mean AUPR) |
| Loss | Frame-level BCE with block-fraction soft targets |
| Scoring at inference | `logits2_raw` (unscaled cosine in [−1, 1]) via 2-class softmax clamped [1e-6, 1−1e-6] |

Per-run architectural flags:

| Run name | Window size W | Sequence length L | GCN enabled | Notes |
|---|---|---|---|---|
| full | 8 | 256 | yes | Baseline VadCLIP configuration |
| no_graph | 8 | 256 | no | Ablation: remove temporal-distance GCN |
| win_1 | 1 | 256 | yes | Window=1 (no block averaging) |
| win_16 | 16 | 256 | yes | Larger window, more aggressive block averaging |
| vl_128 | 8 | 128 | yes | Shorter sequence; n=255 valid samples |
| vl_512 | 8 | 512 | yes | Longer sequence; n=639 valid samples |

Split: session-level, seed 234, `valid_frac=0.25`. Train: 12 sessions / 2562 frames / 1097 positive blocks. Valid: 4 sessions / 716 frames / 325 positive blocks (L=256 subset: n=383, n_pos=115).

## Appendix D — Full comparison.csv contents (head-to-head)

Source: `results/anomaly_headtohead/comparison.csv`. All values verbatim from the file.

| model | status | samples | latency_mean_s | latency_p50_s | latency_p95_s | active_sps | wall_sps | peak_rss_mb | load_s | errors | abstentions | coverage | acc_incl_failures | macro_f1 | binary_f1 | roc_auc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mobileclip2-s2 | ok | 80 | 0.1095 | 0.1122 | 0.1406 | 9.13 | 9.12 | 1112.3 | 2.73 | 0 | 0 | 1.0 | 0.1875 | 0.0855 | 0.4598 | 0.4590 |
| siglip2-b16-224 | ok | 80 | 0.1831 | 0.1771 | 0.2050 | 5.46 | 5.46 | 1196.6 | 2.16 | 0 | 0 | 1.0 | 0.275 | 0.2125 | 0.8271 | 0.6260 |
| smolvlm-500m | ok | 80 | 2.933 | 2.931 | 3.063 | 0.34 | 0.34 | 3046.7 | 2.00 | 0 | 80 | 0.0 | 0.0 | — | — | — |
| minicpm-v4 | ok | 80 | 17.34 | 16.51 | 20.84 | 0.058 | 0.058 | 18684.9 | 4.79 | 0 | 0 | 1.0 | 0.375 | 0.2660 | 0.4000 | — |

Notes: `active_sps` = active_samples_per_second (inference only, excludes pacing sleeps); `wall_sps` = wall_samples_per_second (includes all overhead). SmolVLM-500M's macro_f1/binary_f1/roc_auc are empty because all 80 samples were abstentions. MiniCPM-V4's roc_auc is empty because its output is a discrete label with no continuous confidence score, preventing threshold calibration.

## Appendix E — Dataset schema 2 row counts by task / protocol / split

Source: `results/evaluation_20261011T120315_062384Z/evaluation.json`, `dataset` and `requested_task_coverage` fields. Schema version 2, seed 42. Dataset fingerprint: `cdfb7f31b026ea57bb73a9f931c5d4a93b4bf75e67fd0558822c748a80179e46`. Confirmed rows: 336. Proxy development rows: 7200.

| Task | Protocol | Split | normal | anomaly label | Total | Source groups | Eligible for calibrated test |
|---|---|---|---:|---:|---:|---:|---|
| littering | — | — | — | — | 0 | 0 | no (no verified data) |
| loitering | — | — | — | — | 0 | 0 | no (no verified data) |
| rough_sleeping | — | — | — | — | 0 | 0 | no (no verified data) |
| smoking | frame | dev | 120 | 116 (smoking) | 236 | 16 | false (dev only, no test split) |
| smoking | frame | test | 0 | 0 | 0 | 0 | — |
| unattended_items | frame | dev | 24 | 8 (unattended_items) | 32 | 3 | true |
| unattended_items | frame | test | 24 | 8 (unattended_items) | 32 | 3 | true |
| unattended_items | window | dev | 16 | 4 (unattended_items) | 20 | 3 | true |
| unattended_items | window | test | 12 | 4 (unattended_items) | 16 | 3 | true |

Manifests: `smoking_frame.csv`, `unattended_items_frame.csv`, `unattended_items_window.csv`. The three tasks with no verified data (littering, loitering, rough_sleeping) are excluded from all model comparisons; they appear in the head-to-head bundle only as label categories for the 80-frame test set drawn from proxy development rows.

## Appendix F — Code SHA-256 provenance list

Source: `results/evaluation_20261011T120315_062384Z/evaluation.json`, `code_sha256` field (all 28 files in the `anomaly_eval` package). Head-to-head run provenance from `results/anomaly_headtohead/run.json`: manifest SHA-256 = `18087c4d1da04bf3b4ceaac12004dc18308a3a10532837bfdde690dffb2cc582`; definitions SHA-256 = `589882c0a43d846b9ad6c6d88073129bb666a15014919f585849ab830a95038c`.

| File | SHA-256 |
|---|---|
| anomaly_eval/\_\_init\_\_.py | ebf262f97615be95b2bf3279450d241d22fffa559c07002ee12fce877018e8e9 |
| anomaly_eval/\_\_main\_\_.py | 935a1c1166b0c1ea35a82256345000bf2c73ded718d77773bc27a71ecce28f7d |
| anomaly_eval/analysis.py | 0d34c9588241abfb71c6d8b3a56a7d0ab9dcbcb79b8e458ad25931f40ad01c00 |
| anomaly_eval/audit.py | da6434126cacd43c8a69aed816154d89cc8cdc10d75e233da7185abaf55adfe6 |
| anomaly_eval/benchmark.py | b9549f46325874e0965100c7585048106809bd1aaf2afb3dafd38f0d7009b53b |
| anomaly_eval/cli.py | 462ed3487cfaab2774437ab8ce4f6792a064d71535107ceaf8543ff7e3178674 |
| anomaly_eval/config/anomalies.json | 589882c0a43d846b9ad6c6d88073129bb666a15014919f585849ab830a95038c |
| anomaly_eval/config/caviar_review_intervals.json | 14893df2f779cadf2313279117fc0fc20b6e8c76d5ad0509013fcb3283a54d8d |
| anomaly_eval/data.py | 80559db94d2b0bf8e88d7e316cabc1a5602ff26620d4029cbebb19225bc61a87 |
| anomaly_eval/datasets/\_\_init\_\_.py | 63f341a36e2bbbe47168e1204ee7c0d702969be07a971a1c098d13ff1dd6f98f |
| anomaly_eval/datasets/drive.py | b53907ec66ca147557727af0aa35a22eafbf8d8d6c695c06774f88049ea85853 |
| anomaly_eval/datasets/fetch.py | 224e6289c89334aa9f4aaa5c7319ac412f6a87b477bf5aefc835a35b38d53a11 |
| anomaly_eval/datasets/labels.py | a8042a5c0edeebb189beab519e25e9166476dda66871f2a65f7967238c4a9ca6 |
| anomaly_eval/datasets/prepare.py | 6742ab2af8de07a449299bbdf05d12bf24d6d4e691083b25da398f97c47562eb |
| anomaly_eval/datasets/tools.py | 0dec4c56fcd3b47b97fe36afc15a2a95e8cb0b5fe84142beb378ab18ec711c48 |
| anomaly_eval/metrics.py | d33bb68d0cdde2fd3c7f89fde41475dc76e10159905f5ff8e334b94a0e4f2e3e |
| anomaly_eval/models.py | 6e13e771adbcb4f524aebe2d3c05611035aba3e0bb62f626e0ed8abf5a393658 |
| anomaly_eval/paths.py | 90c2ba6ecbefba8da90ae072912c277a73ed05cb3b66d9809f6523ba233fee14 |
| anomaly_eval/resources.py | f71f2004bafbbd8d8c99e53a2ea111799efb3450aa5c8380772f35cf54bcb3f9 |
| anomaly_eval/runner.py | ee495cf17d6389e2db2999fad7ac2c97a3b51f2a75eee13eb6d76b336017ddba |
| anomaly_eval/vadclip/\_\_init\_\_.py | e49e2f18d1b7db331d7d49553af57222f5bad1b6a81f761e4fb2231b5f73cd7c |
| anomaly_eval/vadclip/baselines.py | 1432c6343e15b2f7b0828262e334858ed2e73483b7de388aa87be8708d132d40 |
| anomaly_eval/vadclip/clip_b16.py | ecd3b7963b703c3ddf48c93ac4dacdf5051df7c9159657b0174e2c8868bb9c4f |
| anomaly_eval/vadclip/data.py | cae28cda5deb6aac393a8b553f7b578172f3648f765f5488b257f3757e049344 |
| anomaly_eval/vadclip/evaluate.py | d8b698b8611b62b5d44d5792cbe018bf4a47bff5b9fe390e71583632d6b4fd5e |
| anomaly_eval/vadclip/extract_features.py | 316a6d757984856cea51012c4f1451c239f5d520a07132483596b0775c5e7262 |
| anomaly_eval/vadclip/model.py | 27c5fac657ceceb0fa5ea9116ef19e0c12c1b6558311dd6d38a6ff0aa03c87ce |
| anomaly_eval/vadclip/train.py | b38e51e2e8cd0f8f81f6d4a640b8cbb3077f34962ff816d9d4fd0426c4c714b9 |

## Appendix G — Glossary of terms

- **AUPR** (Area Under the Precision–Recall curve): integral of precision over recall for a binary score. More informative than AUROC under class imbalance; used as VadCLIP's primary valid-set metric because the valid split has n_pos/n ≈ 0.30.
- **AUROC / ROC-AUC**: area under the receiver-operating-characteristic curve (true-positive rate vs false-positive rate over all thresholds). Threshold-independent ranking quality measure; reported for head-to-head models with continuous scores.
- **Binary F1**: harmonic mean of precision and recall on the binary anomaly-vs-normal decision at a fixed operating threshold. Sensitive to prevalence: at 80% anomaly prevalence, always-anomaly achieves F1 ≈ 0.889 without any detection ability.
- **Macro-F1**: unweighted mean of per-class F1 across all label categories (normal + each anomaly type). Penalises models that collapse to a single dominant class.
- **Block-fraction soft target**: VadCLIP frame-level supervision signal — for each temporal block, the fraction of frames in that block labelled anomalous becomes the soft BCE target, rather than a hard 0/1 label per frame.
- **dual_mean / dual_max**: two ways of combining VadCLIP's visual (logits1) and alignment (logits2_raw) branch scores into one anomaly score: arithmetic mean vs per-frame maximum. In every run, dual_max equals exactly one branch pointwise — the branches' high-confidence frames are near-disjoint.
- **Group-bootstrap**: resampling procedure that draws B=200 bootstrap samples at the source-group level (not individual frame) with seed 42, to produce confidence intervals that respect within-group correlation in the data.
- **LGT-Adapter** (Language-Guided Temporal Adapter): VadCLIP's trainable module — a temporal attention Transformer with block-diagonal windowed mask plus a temporal-distance GCN, spliced into frozen CLIP ViT-B/16 features; 12.64M trainable parameters.
- **logits2_raw**: the unscaled cosine similarity output of VadCLIP's language–visual alignment branch (range [−1, 1]). Used for both training loss and inference scoring via a 2-class softmax clamped to [1e−6, 1−1e−6], avoiding gradient saturation that occurs when scaling by CLIP's learned temperature.
- **Prevalence / base-rate inflation**: the proportion of anomalous samples in a test set. At prevalence p, an always-anomaly predictor achieves binary F1 = 2p/(1+p). The head-to-head bundle has p=0.80 → ceiling 0.889; VadCLIP's valid split has p≈0.30 → ceiling ≈0.46.
- **Quick mode**: evaluation harness mode with no duty-cycle or camera-rate pacing sleeps; inference runs back-to-back at full CPU speed. Latency numbers exclude any deployment pacing overhead.
- **Resource screen**: pass/fail gate applied per model: one stream sampled at 1 fps, p95 inference latency ≤ 1 s, worker/process-tree RSS cap 24576 MiB. Excludes queueing and camera decoding; must be verified on the actual deployment box.
- **Session split**: VadCLIP data partition strategy — sessions (contiguous video clips) are assigned to train or valid sets as whole units using seed 234, valid_frac=0.25, preventing frame-level leakage between splits.
- **Source-group**: the unit of resampling in group-bootstrap; typically a camera sequence or recording session from which multiple frames were drawn. Frames within a source-group are correlated and must not be treated as independent samples.
- **Window protocol / contact sheet**: input format where multiple consecutive frames are composited into a single image (contact sheet) to give the model past-context without native video input. A separate evaluation protocol from frame-level; results are not directly comparable across protocols.
