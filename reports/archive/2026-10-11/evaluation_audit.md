# Evaluation audit: CLIP/VLM suitability for the five anomaly definitions

Audit date: 2026-10-11. Deployment objective: CPU only, approximately 32 GB RAM, initially one camera stream.

## Assessment

**The current experiments do not establish which model is best for the five requested anomalies.** They establish that four configured image pipelines execute and have very different CPU costs. The 80-image bundle is useful for smoke tests and development diagnostics, but its labels, scene coverage, missing temporal evidence, and negative distribution make it unsuitable as the deciding deployment test. The separate temporal study has reproducible implementation defects and does not constitute an evaluation of the published weakly supervised VadCLIP method on all five anomalies.

This review read the scripts, both notebooks' source and saved text outputs, the report, and all 80 bundled images. Images were visually inspected using contact sheets, unlike the historical experiment's stated programmatic-only approach. The original raw annotation corpus, checkpoint/feature cache, per-image predictions, run metadata, and `results/` directories are absent from this checkout. Therefore published inference results and VadCLIP scores could not be independently recomputed. Findings below distinguish reproduced code behavior, visible dataset issues, and claims requiring those missing artifacts.

Evidence: [machine-readable checks](evaluation_audit_checks.json), [portable bundle manifest](evaluation_audit_manifest.csv), and [reproduction script](../scripts/audit_evaluations.py). The portable manifest retains the original proxy labels; it does not certify or relabel them.

## What the experiments actually do

### Image experiment in `anomaly_benchmark.ipynb`

The notebook prepares or reuses `data/anomaly_manifest.csv`, requesting 16 images each for smoking, littering, rough_sleeping, loitering, and normal. `unattended_items` remains in the six-class definitions and prediction vocabulary but has no positive test examples. All samples have split `test`; the four image models are frozen/reference-free, so using source datasets' `train` images is not by itself task-training leakage. Model selection and later prompt/threshold tuning on these images would, however, make them development data rather than a final untouched test.

[derive_labels.py](../scripts/derive_labels.py) constructs labels as follows:

| Label | Exact implemented rule | What it establishes |
| --- | --- | --- |
| smoking | Any COCO box named `Smoking`; mixed Smoking/Not Smoking frames are positive. | A smoking annotation exists somewhere in the frame. |
| normal | A `Not Smoking` box exists and no `Smoking` box exists. | Not-smoking in the smoking corpus; not verified absence of all five anomalies. |
| littering | Matching YOLO file contains any nonblank line. | An annotation in the littering action dataset, without validating class IDs or box syntax. |
| rough_sleeping | Matching YOLO class tokens are exactly `{"0"}`. | A selected pose class; not verified sleeping with bedding. Mixed classes are excluded. |
| loitering | Every image in the designated dataset is positive. | Dataset membership, not an annotated duration or loitering event. |

Sampling is seeded round-robin over `group_id`, with a cap that is relaxed if necessary. Loitering and rough_sleeping group IDs use each filename stem; they do not represent independent videos/cameras. Sampling smoking and normal separately happens to cover the same 16 sessions in this bundle, but the implementation does not explicitly construct paired positive/negative samples.

The benchmark chooses one shared shuffled sample list, then runs MobileCLIP2-S2, SigLIP2-B/16-224, SmolVLM-500M, and MiniCPM-V 4.0 in separate subprocesses. The saved notebook configuration is 80 samples, seed 42, two CPU threads, duty cycle 1, one scored repeat, one warmup inference, maximum input side 512, maximum new tokens 32, and a 3600-second per-model timeout. Warmup/load time is excluded from sample latency; image decoding, common preprocessing, native preprocessing, inference, and response parsing are timed. RSS includes loading and inference.

### Older experiment in `benchmark_models.ipynb`

This uses a **different label builder and test protocol**: `build_manifest.py`, 25 normal + 25 smoking images, a separate 12-per-class four-class set, and a 300-frame normal/smoking extension for the two CLIP models. It uses **four threads**, rather than two. Its normal pool can include smoking negatives, empty/missing littering labels, and standing poses. It omits loitering and unattended_items entirely. Its displayed speed table uses the extended CLIP run but the binary VLM run. Its saved VadCLIP status is `not_configured`.

These tables must not be merged with the newer 80-frame experiment or interpreted as the same benchmark. Differences in thread budget, image distribution, label rules, sample size, and response lengths can explain timing/accuracy changes.

### Exact image model behavior

| Model | Text and decision path | Image/runtime path |
| --- | --- | --- |
| MobileCLIP2-S2 | Encodes **description strings only**, caches normalized text features, scores normalized image features using learned logit scale, softmaxes over six classes, takes argmax; anomaly score is `1 - P(normal)`. | OpenCLIP pretrained transforms, fp32 CPU; timm convolution reparameterization before timing. |
| SigLIP2-B/16-224 | Encodes lowercased description strings, fixed 64-token padding; normalized feature dot products with learned scale and bias. Per-class scores use independent sigmoids; argmax uses logits; anomaly score is max anomalous logit minus normal logit. | Native HF image processor, fp32 CPU, eager attention. |
| SmolVLM-500M | Receives the full classification instruction, six descriptions, and an explicit `unknown` option. Greedy generation; JSON or an exact allowed label is parsed. `unknown` is an abstention. | Common thumbnail plus processor size override to longest edge 512; native processor/chat template; fp32 CPU, eager attention. |
| MiniCPM-V 4.0 | Same classification instruction as SmolVLM; publisher's chat interface; greedy decoding, one beam, at most 32 new tokens. Hard labels, without numeric anomaly confidence. | Pinned custom checkpoint revision, native processor, **one image slice**, fp32 CPU, SDPA language attention. |

See [models.py:145](../scripts/models.py#L145), [models.py:189](../scripts/models.py#L189), [models.py:254](../scripts/models.py#L254), and [models.py:301](../scripts/models.py#L301). The report's claim that all four receive an identical classification prompt and identical preprocessing is inaccurate. They share class definitions and the initial thumbnail, but CLIP models do not receive the VLM instruction and native image transforms differ. Native processing differences are legitimate; they must be recorded accurately.

### Temporal experiment

The `scripts/vadclip/` package extracts frozen CLIP ViT-B/16 features, groups smoking images by camera/date, sorts by filename frame index, and splits sessions with seed 234 and valid fraction 0.25. The report describes 12 training and four validation sessions. Each session is compressed/padded to visual length 128, 256, or 512. Training uses **dense frame/representative labels**, visual BCE and alignment BCE; it does not use the paper's weak video-level MIL objective. Alignment training/scoring uses unscaled cosine softmax, although the model also computes the paper-style `/0.07` logits. Six adapter configurations and four branch combinations are evaluated.

This is a supervised smoking-specific temporal CLIP adapter inspired by VadCLIP, not a five-anomaly zero-shot model comparison. It has no final separate test split. The top-level `scripts/vadclip.py` is a different, unfinished integration wrapper; neither notebook drives the new training/evaluation package end to end.

## Findings that affect evaluation

### F01 — High: fractional positives are silently converted to negatives

[data.py:98](../scripts/vadclip/data.py#L98) averages labels within feature blocks. [metrics.py:24](../scripts/vadclip/metrics.py#L24), [evaluate.py:70](../scripts/vadclip/evaluate.py#L70), and [baselines.py:78](../scripts/vadclip/baselines.py#L78) cast these fractions to integers. Thus every label between 0 and 1 becomes 0: a block containing both smoking and normal frames is classified as negative for evaluation. The linear probe uses `>= 0.5` for its training labels at [baselines.py:83](../scripts/vadclip/baselines.py#L83), creating an additional train/evaluation label mismatch.

Reproduction: raw labels `[0, 1, 1, 0]` compressed to two blocks yield `[0.5, 0.5]`; evaluation sees `[0, 0]`. Both positives disappear. The report's own validation counts vary from 60 positives at L=128 to 115 at L=256 and 276 at L=512. Those AUPR values therefore describe different target/prevalence distributions and cannot isolate the effect of visual length. Changing L also changes several training sessions, not only the long validation session.

**Required correction:** preserve raw frame labels and map representative scores back to their original frame intervals before evaluation. Alternatively define a segment-label rule explicitly and compare all variants on the same fixed segment targets, labeling the result as segment rather than raw-frame evaluation. Preserve timestamps/coverage when frames are sparse. Re-extract/re-score baselines and all adapter variants.

### F02 — High: reported optimal thresholds do not match their metrics

[metrics.py:34](../scripts/vadclip/metrics.py#L34) chooses `thr[i - 1]` instead of the threshold corresponding to precision/recall index `i`; when `i == 0`, it chooses infinity. [scikit-learn documents](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) that precision/recall at index `i` correspond to `thresholds[i]`, except the final endpoint without a threshold.

Reproduction: labels `[0, 1]`, scores `[0.1, 0.9]` report F1=1, P=1, R=1, threshold=0.1, and accuracy=0.5. Applying 0.1 actually produces P=0.5 and F1=2/3; the correct perfect-classification threshold is 0.9. With tied scores `[0.5, 0.5]`, the code reports the all-positive PR point but assigns an infinite threshold, yielding all-negative decisions.

**Required correction:** select from threshold-bearing PR entries, index the matching threshold, and recompute all operating-point metrics from actual thresholded predictions. This bug affects threshold/accuracy consistency; it does not itself change rank-based AUPR/AUROC or the maximum curve F1 value. Choose deployment thresholds on development data, not by maximizing F1 on the final test.

### F03 — High: the CPU best-weight snapshot changes during further training

At [train.py:174](../scripts/vadclip/train.py#L174), `v.detach().cpu()` shares storage with an already-CPU parameter. Subsequent optimizer steps change the supposed best snapshot. The final `load_state_dict(best_state)` consequently need not restore the selected epoch. `metrics.json.best_epoch` separately records the final executed epoch at line 191, not the best epoch.

Reproduction: snapshot a CPU parameter with value 1 using that expression, update the parameter to 2, and the saved in-memory value also becomes 2. **Required correction:** clone/deep-copy tensors or reload `checkpoint.pt`, and track the selected epoch explicitly. The file serialized immediately on an improvement can still contain the correct best weights; `evaluate.py` loads that disk checkpoint. Therefore this finding invalidates the claimed in-memory restoration and training summary, but does not prove that every saved `eval.json` used the wrong weights.

### F04 — High: validation is reused for model/epoch/branch/threshold selection

[train.py:163](../scripts/vadclip/train.py#L163) evaluates the same four validation sessions every epoch and chooses weights using their dual-mean AUPR. `evaluate.py` then reports that same split; the ablation study compares six configurations on it, and `frame_metrics` searches the best F1 threshold on it. There is no untouched final test set. Fixed baselines are compared against an adapter selected on the evaluation data.

**Impact:** these are development/validation results, with optimistic selection effects, not unbiased test or deployment estimates. **Correction:** group-separated train/dev/test, tune configurations/branch/threshold on dev, then evaluate once on test. With only 16 sessions, use repeated/nested group splits or report substantial uncertainty; do not shuffle individual correlated frames into splits.

### F05 — High: dataset labels do not fully match the specified events

Loitering at [derive_labels.py:198](../scripts/derive_labels.py#L198) is assigned to every dataset frame without any duration or event interval. The VLM prompt explicitly asks for `unknown` when duration evidence is missing. Penalizing that answer against a dataset-membership positive is a mismatch between the requested task and its ground truth.

Rough sleeping at [derive_labels.py:178](../scripts/derive_labels.py#L178) means a pose-class token of 0. The images show people lying/on the floor in corridors, warehouses, shops, stations, and outdoors; several look like fall/pose examples. Bedding and confirmed sleep are generally not established. Examples: [laying12](../test_data/rough_sleeping/laying12.png), [laying32](../test_data/rough_sleeping/laying32.png), and [laying48](../test_data/rough_sleeping/laying48.png). A high score here establishes recognition of this pose/source proxy, not reliable rough-sleeping detection.

The normal rule only verifies no annotated smoking in that corpus, not the absence of all target events. Missing annotations are not automatically verified negatives.

I additionally checked the public littering `data.yaml`: its five class names are **dropping, dumping, putting-down, throwing, tossing** (SHA-256 `8be63be832723eeeaff5298ccbec858d6fb8057aded3d23fc5996239b1e7bb7c`, [source file](https://drive.google.com/file/d/1N-C__He_l8AZ0DhbQKuhc85E3KKn3jaO/view)). This reduces the concern that this particular non-empty-label rule is merely picking person/bin boxes: the mapped classes are actions. However, the parser does not validate IDs/syntax or establish that each action is discarding rubbish contrary to the definition. Historical per-image labels could not be checked against the absent raw annotations.

**Correction:** obtain event-aligned annotations and temporal clips; distinguish lying/falling/resting from the operational sleeping definition; manually adjudicate ambiguous examples using context. Keep uncertainty/multiple simultaneous events explicit. Use the current pose/loitering proxies only as named auxiliary tasks.

### F06 — High: source and negative distribution confound the image comparison

All 16 normal examples and all 16 smoking examples are industrial CCTV views. Littering is a courtyard/campus corpus; rough_sleeping is another pose corpus; 11 of the 16 visually inspected loitering examples share the same ATM-like view, including mirrored views. Normal actions in those other scenes are absent. A detector can learn/source-match the scene rather than the anomaly.

Every class also has a distinctive image size except smoking/normal: loitering 640×640, littering 1920×1080, rough_sleeping 1024×1024, smoking/normal 640×480. A post-hoc rule using **only dimensions** and guessing normal for the smoking/normal size achieves 64/80 = **80% accuracy**, binary F1 0.857 and AUC 0.875. This is a dataset-design diagnostic, not a legitimate held-out model baseline; native resizing means models need not directly see those raw dimensions, but the metadata separability and visible scene differences expose the source confounding.

**Correction:** collect positive and matched normal/hard-negative examples for every source/camera and action, with source-separated final test coverage. Include sitting/resting versus sleeping, brief waiting versus loitering, owned bags versus unattended bags, legitimate placement/pick-up versus littering, and non-smoking hand-to-mouth actions.

The reported SigLIP binary F1=0.8271 and normal-class F1=0.148, combined with 64 anomalous/16 normal samples and full coverage, uniquely imply **TP=55, FP=14, FN=9, TN=2** under the implemented scorer. Thus it flags **14/16 normal frames (87.5%)** at this operating point. This reconstruction from rounded report numbers is recorded in the audit checks; it needs confirmation against the absent raw predictions. A high binary F1 here is not evidence of an acceptable alarm policy. Compare recall at a fixed, development-calibrated false-positive budget, and evaluate the event alert policy separately rather than extrapolating frame FPR directly to alarms per hour.

### F07 — High: the requested five-anomaly coverage is incomplete

`test_data/` has 16 samples each of four anomaly proxies plus normal, and **zero unattended_items**. It contains stills, not ordered clips with timestamps/duration/ownership evidence. Nothing here establishes unattended-item recall or loitering event detection. Six class strings in a prompt do not substitute for test support. The temporal adapter is trained only for normal versus smoking.

**Correction:** evaluate still-visible cues separately from temporal events, and add real positive and matched-negative unattended-item clips plus loitering duration labels. Do not assign a measured F1 of zero to an untested positive category or recommend any candidate across all five from this bundle.

### F08 — Medium: abstentions are excluded from most headline metrics

[metrics.py:41](../scripts/metrics.py#L41) removes abstentions/errors before building confusion matrices, class supports, macro-F1, binary recall/F1, and AUC. Only coverage and `accuracy_including_failures` retain the complete labeled denominator. Thus a model can abstain on difficult positives and look excellent on conditional F1/recall.

Reproduction: one correct smoking prediction, one abstained smoking sample, and one correct normal sample yield reported binary recall=1 and F1=1 at coverage=2/3. If abstention means no automatic detection, smoking recall is 1/2 and binary F1 is 2/3. This is a valid *selective* metric only if labeled clearly and accompanied by the full-data metric. It is not a comparable full-dataset detection score.

**Correction:** use all ground-truth samples for automatic-detection recall, record abstention/error as separate confusion outcomes, and report coverage/review burden. If abstentions are routed to a reviewer, evaluate that two-stage policy. The notebook's per-anomaly calculation treats abstentions differently, so its tables do not share one scoring policy.

### F09 — Medium: macro-F1 has a changing class denominator

[metrics.py:115](../scripts/metrics.py#L115) averages over classes occurring in either valid targets or predictions, rather than a fixed six-class set or a fixed supported five-class set. This denominator can vary by model and bootstrap sample. Reproduction: perfect predictions on the five supported labels give macro-F1=1.0, contradicting the report/test-data README's claimed structural maximum of 5/6. An unsupported class gets included if a model predicts it, but excluded if it never does.

**Correction:** explicitly fix the scored class set, document unsupported categories separately, and regenerate tables/CIs/figures with the same definition. Without the raw predictions, I cannot determine which published point estimates happened to include the sixth class; the contradictory implementation/documentation is confirmed.

### F10 — Medium: bootstrap ignores the group structure and does not establish a winner

[benchmark_improve.py:102](../scripts/benchmark_improve.py#L102) resamples independent rows, not sessions/videos. The bundle has 80 rows but only 60 declared groups; smoking/normal share sessions, littering repeats four source videos, and loitering's declared groups do not capture the visibly repeated camera/source. These intervals do not account for that dependence. Point estimates are not checked against `comparison.csv`, despite the script docstring claiming an assertion. The script also accepts partial prediction files without checking the planned sample count/model status.

**Correction:** paired, stratified cluster bootstrap over real source units; compare model *differences* on matching examples; preserve sample identity and assert exact coverage/point-estimate consistency. The existing reported SigLIP AUC interval includes 0.5, and the existing CLIP AUC intervals overlap. Neither proves population superiority or above-chance discrimination at 95% confidence. Overlapping individual intervals alone also do not prove that the paired difference is insignificant.

### F11 — Medium: padded frames influence real predictions

The loss masks padded positions, but [model.py:173](../scripts/vadclip/model.py#L173) passes `None` to temporal attention; the distance graph includes all padded positions; and [model.py:227](../scripts/vadclip/model.py#L227) pools all positions into the visual-conditioned text features. Learned position embeddings/biases make padding nonzero internally. This is especially important for three-, 16-, or 33-frame sessions padded to 256.

A tiny random model with two real and six padded positions changes the real visual logits by up to 2.10 and alignment logits by 0.97 when only padding changes. This proves a dependency, not the magnitude of an error in the published trained model. The upstream architecture has similar unmasked paths, so this is an adaptation/short-session issue rather than proof that the entire port differs from the paper.

**Correction:** mask attention, graph adjacency and prompt pooling consistently, verify padding invariance, and guard normalization. Keep the change documented if comparing against the upstream architecture. The additional nondivisible-window probe finds entirely masked rows for L=8/window=3; reported L/windows divide evenly, so that is a latent CLI failure, not evidence against the listed runs.

### F12 — Medium: session labels/order and feature-cache reuse need stronger provenance

[vadclip/data.py:75](../scripts/vadclip/data.py#L75) treats every non-smoking-annotated frame as negative, while the image collector excludes frames without explicit Smoking/Not Smoking annotations. The unused `nonsmoke_ids` set makes this difference clear. The fixture with only a person annotation is excluded from the image manifest but included as negative in the temporal corpus.

Camera/date grouping and frame-number ordering are not verified to be continuous video/timestamps. [extract_features.py:79](../scripts/vadclip/extract_features.py#L79) reuses an existing NPZ without checking image list/order, label digest, model-weight identity or transform settings; its manifest stores counts, not frame identity. Changed annotations/renamed/missing frames can therefore leave stale features/labels in place.

**Correction:** distinguish explicit negatives from unannotated frames; store image hashes, timestamps, raw frame order and annotation/weight/preprocessing hashes; invalidate mismatched caches. Evaluate on real time intervals and use causal/sliding windows for streaming. The current model's graph, attention and global prompt pool can use future frames, and the corpus's extracted frames are not demonstrated to provide real-time durations.

### F13 — Medium: notebook reuse can silently show stale results

`anomaly_benchmark.ipynb` cell 7 checks seed, limit, model list, and `status=ok`, but not manifest/definition hashes, thread count, processor/adapter/scorer version, model revision, token/image budget, or precision. If parsing `run.json` fails, it catches the exception and still accepts comparison rows. Its manifest reuse checks class counts only. `benchmark_models.ipynb` cell 3 checks manifest hash/model name but similarly ignores definitions/config/code changes and does not require a successful complete summary.

Both runners delete their results directory when the reuse check fails. Saved outputs in both notebooks show **reuse**, not fresh execution of inference. These outputs are useful historical records but do not prove the current source produced them.

**Correction:** fail closed; use a full experiment fingerprint including source/weights/input bytes/labels/config and a new results directory for each changed experiment. Preserve historical results, and validate successful completion plus exact sample coverage before reuse.

### F14 — Medium: the old VadCLIP wrapper is not a working upstream integration

[vadclip.py:84](../scripts/vadclip.py#L84) expects `from main import get_model` and later `model.score(...)`. The released upstream uses `src/model.py` and `xd_test.py`/`ucf_test.py`, with direct `CLIPVAD` construction and frame scores, not this API. The wrapper also builds **one scalar per video**, despite promising per-frame outputs. `find_weights` accepts unrelated PT/PTH names, so finding a CLIP backbone is not proof that a detector checkpoint exists.

**Correction:** retire/replace the wrapper or wire a genuine published/custom adapter and preserve frame scores/timestamps. Make notebook/report naming consistent with the actual `scripts/vadclip/` package. See the [official testing implementation](https://github.com/nwpu-zxr/VadCLIP/blob/main/src/xd_test.py).

### F15 — Medium: hardware/runtime tests do not select the best deployment pipeline

The head-to-head numbers describe fp32, batch size one, two threads, on a reported 64-logical-CPU/~251-GiB host. They do not predict the Ryzen-class target's throughput/RAM under its power/memory constraints. The newer notebook has no explicit memory limit in its command. The temporal train/evaluate/extract scripts do not apply the image harness's thread limits/RSS watcher, nor measure end-to-end encoder + adapter + decoding latency.

Full-scene smoking images are only 640×480 and people/cigarette cues are small; additional thumbnail/native resizing can erase evidence. MiniCPM uses one image slice. SmolVLM overrides native processor resizing. These are real pipeline tradeoffs but no native-resolution/ROI/prompt/quantization ablations establish their effect on quality. Reported zero smoking recall for SigLIP/MiniCPM and all-unknown SmolVLM cannot establish that their best deployable configuration is incapable of the task.

**Correction:** preserve an out-of-the-box baseline, tune each model on the same dev data/budget, then test frozen settings on held-out data. Compare full frames, detector-generated person/item regions, and temporal windows, including detector/decoder cost. Ground-truth box crops are permissible as an explicitly privileged diagnostic, not an end-to-end deployment result. Benchmark 4/8/16/32 thread settings on the actual CPU target, relevant quantized runtimes, sustained one-stream queue delay, p95 latency, RSS, and event false alarms per hour.

## Dataset suitability and integrity

| Check | Result |
| --- | --- |
| Files and supported classes | 80 images, 16 each for normal/smoking/littering/loitering/rough_sleeping |
| Unattended items | 0 positive examples |
| Hash and decode checks | All pass; no byte-identical duplicates |
| Declared independent groups | 60 total; 16 smoking sessions shared with normal, 12 littering videos, 16 nominal loitering and 16 pose groups |
| Near-image review | 25 low-resolution dHash candidates at distance ≤5; this is a similarity screen, not proof of duplicate/incorrect labels |
| Temporal inputs | Stills only; no complete event clips/ownership histories/duration labels |
| Raw annotations/predictions available here | No |
| Bundle manifest usable by benchmark directly | No: it has `file` rather than required `path`; the added portable manifest resolves this packaging issue |

The low-resolution candidate screen includes a same-view normal/smoking pair and repeated loitering views. Small action differences can legitimately produce different labels in similar frames, so these pairs should be reviewed/grouped, not automatically discarded. SHA-256 matches verify file integrity, not label correctness.

Use this bundle for reproducibility, parser/runtime regressions, and early visual diagnostics. Use it as **development data** for resolving the discovered failures. Build a separate, manually reviewed final test set with positives and realistic matched negatives for all five tasks, independent source sessions/videos, real clips for temporal tasks, and enough normal recording hours to estimate alert precision/false alarms. A balanced diagnostic set is useful, but also report deployment-prevalence behavior; the existing 64/80 anomaly prevalence is far from a normal-heavy camera stream.

## Report and testing discrepancies

1. **The report is unfinished.** It stops at §5.5 while citing §5.6–§10 and appendices that are absent. The executive summary's VadCLIP/ablation findings lack their promised detailed section and raw artifacts in this checkout.
2. **The published VadCLIP headline is misidentified.** The report calls 84.51% segment mAP. The [paper](https://arxiv.org/abs/2308.11681) reports 84.51% AP on XD-Violence, and the released test code computes that AP from frame-level ground truth and repeated frame scores; it computes segment mAP separately. Missing event segments does not by itself prevent reproducing the frame-AP protocol. The claim that this model uses a 3D-conv/optical-flow feature pipeline is also inaccurate for this released CLIP-feature method.
3. **Chance/baseline statements need correction.** A classifier using the five supported balanced labels has 20% uniform-random/majority-class accuracy; uniform random over all six allowed strings is a different 16.7% baseline. Moreover SigLIP's reported lower accuracy CI of 0.175 exceeds 1/6, contrary to the paragraph saying only MiniCPM's interval does. The always-anomaly F1 of 0.889 at prevalence 0.8 is a trivial baseline, not a mathematical ceiling; a perfect detector scores 1. AUC is not an operating-point metric, and the reported MobileCLIP interval includes 0.5, so “anti-predictive” is too strong.
4. **Composite fit scores are subjective.** Notebook cell 16 min-max normalizes metrics, assigns weights .35/.20/.15/.20/.10, and converts missing VLM AUC to zero. This penalizes output format and rewards binary F1 from an 80%-positive dataset. It is not evidence of universal model superiority. Prefer hard latency/RAM/false-alarm constraints and a per-task Pareto comparison.
5. **Per-anomaly zero F1 is displayed as undefined.** Notebook cell 14 uses `if (precision and recall)`; legitimate zeros become `None`. Other code reports zero. Its repeat/abstention handling also differs from the central scorer. Figure 1 computes macro-F1 CIs but does not draw them, despite the joint CI description; the notebook's “5 classes” macro-F1 label does not describe the central changing denominator.
6. **Reproduction packaging is incomplete.** `test_data/README.md` uses `$7` for the digest even though SHA-256 is CSV column 6; it also refers to one image per group despite 16 littering images spanning 12 groups. A CSV-aware verifier is preferable to `awk`. All 80 absolute source paths point to another machine; use the portable manifest and the hash-verifying audit script. No raw `run.json`, `summary.json`, `predictions.jsonl`, improvements, NPZ scores/features, or checkpoint files are present to validate the report's calculations.
7. **The label builders are inconsistent.** The old builder hardcodes smoking IDs 2/1, treats missing litter labels as normal, accepts mixed pose scenes, ignores loitering, and ignores its custom data-root argument when calling collectors. The new builder uses category names, skips missing litter labels, rejects mixed poses and adds dataset-level loitering. Results from their protocols are not interchangeable. The fetcher's Drive-ID renaming is not consumed by either label builder: renamed image/annotation IDs differ and COCO files reference original names, so future builds can silently omit files or manufacture negatives unless the rename mapping is reconciled first.
8. **Dependencies/CI do not cover this study.** `requirements.txt` lacks scikit-learn and matplotlib required by the temporal/report/notebook code. The only CI workflow filters `data-pipeline/**`, runs in a nonexistent `data-pipeline` directory, and tests `video_ingestion`; it does not run this project's experiments/tests on relevant changes.
9. **Existing tests are useful but incomplete.** All **73 existing tests passed** in a temporary CPU environment with the pinned model libraries and explicit sklearn/numpy installs. They verify manifests, fetching, leakage guards, parser behavior, worker limits and some adapter contracts. SigLIP has a real tiny native-forward equivalence test; MobileCLIP and MiniCPM rely on mocked adapters. There are no tests for the new temporal package's threshold alignment, soft-label conversion, best-checkpoint restoration, padding invariance, cache provenance, final holdout, or the actual event-label semantics. Passing these tests does not validate the published experiments.

Recommended regression checks: PR metrics must equal direct thresholded confusion counts including tied scores; CPU snapshot values must remain unchanged after an optimizer step; raw positive frames must survive score reprojection; padding cannot alter valid outputs; cache changes must invalidate reuse; all-sample recalls must include abstention outcomes; scored class support/averaging must be fixed; annotations/rename mappings must resolve before selection.

## Decision and next experiment

Retain **MobileCLIP2-S2 and SigLIP2-B/16-224** as the two inexpensive screening candidates. Retain **SmolVLM-500M and MiniCPM-V 4.0** as candidates for a separately evaluated prompted/quantized confirmation stage. The current evidence does not justify discarding SmolVLM entirely or declaring SigLIP the best five-anomaly detector. The temporal adapter needs corrected scoring, consistent masking/targets, and an untouched test before its gains can guide deployment.

Work in this order:

1. Freeze the operational event definitions, required observation window, acceptable delay, and false-alarm budget. Relabel/remove the proxy examples for final-task scoring and collect matched negatives/temporal positives for the missing tasks.
2. Correct the confirmed scoring/checkpoint bugs and choose explicit policies for abstentions, unknown labels and unsupported classes. Reconcile annotation paths after Drive renaming. Version datasets, code, model revisions and caches.
3. Create source-group-separated train/dev/test or nested group evaluations. Tune per-model prompts, class thresholds, image crops/native resolutions and runtime precision on dev only. Preserve the current settings as a baseline rather than assuming one common instruction is each model's best configuration.
4. Evaluate fixed pipelines on the same held-out inputs/ground truth, using paired source-level uncertainty, per-task precision/recall/AP, coverage and event metrics. Test the temporal adapter on the same binary task and identical raw evaluation frames as its baselines.
5. Apply the actual CPU-only 32-GB/one-stream constraints on the deployment machine. Include decoding, region detection, feature extraction, temporal buffering, generation, queue delay and peak memory. Select from the resulting quality/latency/memory tradeoff rather than the arbitrary fit score.

## Audit reproduction

No pretrained weights or model inference are needed for the bug probes. The existing project dependencies plus `numpy` and `scikit-learn` are required for the tensor/metric probes; the image integrity checks use Pillow.

```bash
python scripts/audit_evaluations.py
python -m unittest discover -s tests -v
```

The script writes `reports/evaluation_audit_checks.json` and `reports/evaluation_audit_manifest.csv`. Nonfinite values from buggy metric outputs are recorded as strings so the evidence is strict JSON. The checks include source SHA-256 values. The portable manifest can be passed to `benchmark.py` for future development reruns; its inherited labels still need the corrections described above. This audit added evidence/reporting artifacts only and left historical experiment code, notebooks, results and test images unchanged.
