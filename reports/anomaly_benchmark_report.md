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
