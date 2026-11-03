# Gemma 4 accuracy study

This study tests the five existing anomalies across the official instruction-tuned Gemma 4 E2B, E4B, 12B, 26B-A4B and 31B vision models. It reuses the established grouped data and definitions, with no latency benchmark or Gemma weight training. Actual outputs are in `results/gemma4_evaluation_20261027/`; the portable report and CSVs are copied to `reports/gemma4_evaluation_20261027/`.

## Environment

Keep this runtime separate from the older model experiments: Gemma 4 requires the newer Transformers implementation. The installed `.venv-gemma4` uses Transformers 5.21.0 and the existing CUDA PyTorch 2.8.0 runtime. Dependencies are pinned in `requirements/gemma4.txt`.

The workspace's GPU environment can be reused without another large PyTorch installation:

```bash
python3 -m venv .venv-gemma4
.venv-gemma4/bin/python - <<'PY'
import site
from pathlib import Path
gpu_packages = next(Path('.venv-gpu/lib').glob('python*/site-packages')).resolve()
Path(site.getsitepackages()[0], 'capstone_gpu_runtime.pth').write_text(str(gpu_packages) + '\n')
PY
.venv-gemma4/bin/python -m pip install -r requirements/gemma4.txt
```

This assumes the existing `.venv-gpu` has CUDA PyTorch; for a standalone environment install the matching CUDA PyTorch wheel instead of adding the `.pth`. CPU-only PyTorch cannot run this study. On this host, GPU jobs need the local matching driver libraries supplied by `scripts/gemma4_gpu.sh`. The execution service must expose the actual GPU devices and host `/dev/shm`.

## Acquire pinned checkpoints and runtime

`anomaly_eval/config/gemma4_models.json` records repositories, immutable revisions, precision, licenses, weight sizes and publisher SHA-256 values. All five releases are public; the fetcher does not require a token. The large models use official Google QAT Q4_0 releases. Their memory footprint and precision differ from native BF16; report these as distinct pipelines, not a controlled size-only experiment.

```bash
.venv-gemma4/bin/python -m anomaly_eval.gemma4 fetch-runtime
.venv-gemma4/bin/python -m anomaly_eval.gemma4 fetch --models all --workers 2
```

Native E2B/E4B checkpoints use `data/models/gemma4_hub`. To avoid exhausting the existing disk, 12B and the two large GGUF checkpoints use `/dev/shm/capstone_gemma4_20261027`. That cache is ephemeral and must be fetched again after reboot. Override `--cache` and `--ram-cache` for persistent storage on another machine. SHA-256 and file size are checked before a model's `checkpoint.json` is issued. Interrupted HF downloads can resume. The runtime has its own pinned archive SHA and installation receipt.

The first shared Xet download stalled on 12B and restarted its unique temporary file when resumed. The fallback below isolates the transfer, reuses the largest existing contiguous prefix, and downloads bounded HTTP ranges concurrently. Completed part hashes permit resume, malformed or truncated range responses are rejected, and the final whole-file publisher hash is mandatory before publication. Stop the corresponding owned HF downloader first; do not run two writers for the same model.

```bash
.venv-gemma4/bin/python -m anomaly_eval.gemma4 fetch-parallel --model gemma4-12b --workers 8
```

This fallback supports all three single-file native checkpoints. It uses the existing repository's transfer client for connection timeouts and server retry conditions. Tests cover ignored ranges, wrong offsets, oversized bodies, interrupted bodies, neighbor preservation and successful retry. Old incomplete files are retained until the completed checkpoint is verified.

Official references:

- [Gemma model overview](https://ai.google.dev/gemma/docs/core)
- [Native Hugging Face inference](https://ai.google.dev/gemma/docs/core/huggingface_inference)
- [Transformers Gemma 4 implementation](https://huggingface.co/docs/transformers/model_doc/gemma4)
- [Google Gemma 4 model collection](https://huggingface.co/collections/google/gemma-4)
- [Pinned llama.cpp runtime release](https://github.com/ggml-org/llama.cpp/releases/tag/b11925)

## Frozen inputs and optimization

```bash
bash scripts/gemma4_gpu.sh --prepare
cp -n anomaly_eval/config/gemma4_ensembles.json results/gemma4_evaluation_20261027/ensemble_protocol.json
```

Preparation checks inherited source groups, cross-split decoded-pixel hashes, every image's current decoded identity, duplicate IDs and target validity. It includes 1,812 primary development cases, 160 XRF55 source-development cases and 4,032 held-out cases: **6,004 examples and 20,331 distinct image views**. Training examples enter leakage checks only. The inputs come from the 19 October main and additional-transfer manifests and the 20 October XRF55 manifest; no additional data were fetched for this comparison. Dataset sources and terms remain in the existing central `anomaly_eval/config/dataset_sources.json` registry and accompanying license records.

The 20,331 count is the manifest's referenced views. The fixed maximum of eight views actually selects **17,131 distinct source views** before model resizing or sheet composition. `dataset_counts.csv` records referenced images, selected unique images and selected view occurrences separately. All referenced decoded pixels were verified; repeated views and multiple policies are not counted as independent new data.

"All benchmarks" here means the applicable five-target classifier/clip and source-transfer cohorts with usable positive and negative labels. Older dense frame/window experiments are represented by the standardized reviewed temporal cohorts rather than counted again as independent evidence. `test_data/EVALUATION.csv` is a small, task-specific legacy smoke/unattended manifest, with partial labels rather than an exhaustive six-class test. Positive-only weak anomaly CSVs cannot establish specificity or AP against negatives. Person-presence/oracle-pose controls evaluate different targets; detector box recall, thermal trajectories and continuous tracker/event metrics require their own model interfaces and are not represented as Gemma classification results. Those older artifacts remain preserved in the historical report.

Four policies are declared before study scores: definition/native 280 tokens per image; evidence/native 280; evidence/native 560; evidence/contact-sheet 1,120 tokens per sheet. Up to eight chronological views are used, uniformly subsampling longer sequences. Public timing metadata is supplied where present; missing duration is never invented. Gemma receives only images, the queried objective/definition and neutral timing instructions. It does not receive source identity, class label, split or reviewer notes.

A fixed, reproducible primary-development screen (274 cases; at most 64 per task, balanced across source/class/group) chooses each objective's mean-source AP winner. Ties favor the simplest baseline. Full development calibrates four operating points: raw zero, maximum F1, pooled FPR ≤5%, and a single threshold satisfying FPR ≤5% in each primary-development source. Both baseline and optimized thresholds are saved **before either method scores any held-out example**. XRF55 development only calibrates secondary source thresholds; it does not select the primary route.

`ensemble_protocol.json` predeclares two additional comparisons: a per-task model chosen by full primary-development source AP, and an equal-weight mean of model margins standardized using primary-development means and population standard deviations. No test-dependent ensemble weights or family choices are allowed. These comparisons reuse cached model predictions; they do not create independent evidence.

`safeguard_protocol.json` separately declares a third comparison: choose the baseline or optimized route and model with best mean-source AP across the complete primary development set. This can reject a small-screen prompt winner and costs no additional GPU inference. It was declared on 27 October at 10:40 UTC, before any completed Gemma results were reviewed but after initial held-out forwards began. The report retains this timeline and identifies the addition as retrospective. The original model evaluations and the earlier two-combination protocol are unchanged.

After E2B completed, `layout_support_protocol.json` recorded a separate methodological correction. A selected contact-sheet policy exercised only on single images in primary development changes to an unvalidated layout on multi-image inference. Such requests now receive an additional diagnostic comparison using the unchanged baseline margins and its own frozen cutoff; supported requests retain the optimized route. The branch depends only on input view count and primary-development coverage, with no source identity or test-label condition. Original results remain intact. This correction was motivated by E2B's transfer regression and is explicitly retrospective, not a new untouched test. It is reported as `layout_supported`, separate from the original `optimized` method.

## Run and resume

```bash
export LD_LIBRARY_PATH="$PWD/data/models/nvidia_runtime/extracted${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
.venv-gemma4/bin/python -m anomaly_eval.gemma4 suite
```

The controller evaluates whichever verified model is ready in the declared catalog order, one GPU job at a time. Every model first checks all visual policies on development inputs. It records failed preflights/runs and continues to other ready sizes. It does not kill other GPU users. It waits for downloads, preserves per-model logs, and updates the report after completions. One controller holds `suite.lock`. For an existing separately launched study process, `--wait-pid PID` queues work after that owned process exits. Running the controller again resumes unfinished work and skips completed models.

`--wait-pid` also accepts multiple owned process IDs. E2B and 26B initially shared the GPU after successful concurrent preflights. The 26B process was then interrupted during development to release memory for E4B/31B/12B interface checks; its append-only scores were preserved, with a scheduling receipt. This changes execution order, not inputs or selection. The final controller resumes that cache and runs pending sizes sequentially. The suite reports completion only after the final report/audit succeeds; otherwise `suite_status.json` retains the failure.

One model can also be invoked directly:

```bash
bash scripts/gemma4_gpu.sh --model gemma4-e2b --checkpoint /absolute/path/from/checkpoint.json
```

`--batch-size` controls dispatch/checkpoint granularity. Actual tensor forwards use a canonical single example. Mixed-length native BF16 tensor batches changed answer margins by more than one logit in preflight, even after position-ID correction. Canonical forwards eliminate composition-dependent scores; this is a numerical correctness choice, not artificial pacing. All neural inference uses the GPU, with no simulated CPU speed benchmark. Thinking is disabled consistently across sizes.

Native scoring uses the real Yes-minus-No answer logit and retains full-vocabulary Yes+No probability mass. GGUF scoring uses the same margin derived from post-sampling probabilities, with identical positive bias on both label tokens, temperature 1, no truncation and no penalties. Equal bias cancels in the log-probability difference; preflight validates +50 versus +100 on development cases. Raw unmodified answer mass is unavailable from that biased path and is explicitly null. The server's global image-token cap is set for each policy; its microbatch capacity covers the maximum cap to prevent silent visual-budget truncation. Each server is locally owned and terminated by its own backend.

These are forced binary scores. Native models can place little probability on either literal label for some inputs, even when the conditional Yes/No margin is finite. Every case stays in the metrics; `answer_mass.csv` reports availability, minimum, percentile, median and low-mass counts by held-out source. Preflight captures top first-answer tokens for native development cases with mass below 0.5. A numerical preflight pass certifies input support and repeatability, not universal response-format compliance or accuracy. Thinking-enabled generation is outside this run.

Score caches bind the manifest, model receipt, runtime metadata, prompt code, image helpers, implementation and frozen protocol hashes. Changed settings cannot reuse old scores. Individual implementation snapshots, routes, calibrations, all margins, runtime logs and progress JSONs are retained. Do not edit the frozen scoring modules during a run; use a new output directory and protocol for a changed study.

## Reports and checks

```bash
.venv-gemma4/bin/python -m anomaly_eval.gemma4 report
.venv-gemma4/bin/python -m anomaly_eval.gemma4.verify
.venv/bin/python -m unittest tests.test_gemma4 -v
```

The report exports dataset counts, all operating-point metrics and confusion matrices, source AP/AUROC, task/source-balanced macro AP, policies and matched prior-pipeline context. Plots include source AP, precision–recall and ROC curves. Historical CSVs retain zero-shot/probe/fine-tuned stages; VADCLIP's different session/event cohort is labeled separately.

The replay audit reconstructs route selection, every calibration, both held-out prediction sets and every source-level operating metric from raw cached margins. It verifies source manifests, model receipts, implementation snapshots, prediction hashes, finite scores and native answer mass. `verification.json` states how many planned sizes have actually completed; a partial audit does not claim the five-size suite is finished. `environment.json` records the exact packages, native model implementation hashes, GPU and CUDA driver used.

The completed five-size study is summarized in [the published report](../reports/gemma4_evaluation_20261027/report.md). Its opening findings distinguish ranking improvements from operating-point failures and compare the same source identities with the existing pipelines. The published report directory includes verification and protocol receipts, environment metadata, and the family-combination selections/calibrations/results. Per-example margins, implementation snapshots and model checkpoint receipts remain in the raw results directory; no model weights are copied into the report.

`historical_cohort_verification.json` checks the same 3,792 non-XRF IDs, targets, sources and groups against the earlier trained CLIP and SmolVLM predictions. Older additional-transfer rows called `test` are now called `external`, with that naming difference recorded. `common_source_summary.csv` restricts macro AP to those same ten sources; it must not be confused with the eleven-source Gemma summary that additionally includes XRF55.

Interpret the denominators: provisional sleep annotations, limited temporal/owner evidence, correlated camera groups and scripted XRF gestures constrain conclusions. AP is not accuracy, a 5% development constraint is not a test guarantee, and clip FPR is not continuous false alerts/hour. This comparison does not validate a YOLO/ByteTrack/Gemma deployment pipeline.

## Second-stage optimization

The completed second-stage study uses a separate `results/gemma4_optimization_20261028/` directory. It preserves all first-stage scores and evaluates three new policies: native 1,120-token evidence, 560-token competing-explanation guidance, and full-scene plus automatic contextual crops at 560 tokens. The crops use existing label-free Faster R-CNN predictions for people and carried items; smoking crops emphasize the upper body. Cropped clips use four chronological time points with an optional detail crop after each. The other native policies use up to eight time points. Crop views from the same moment are explicitly identified as such.

```bash
bash scripts/gemma4_gpu.sh optimize
# Limit a run explicitly if needed; default is all five variants.
bash scripts/gemma4_gpu.sh optimize --models gemma4-12b gemma4-31b
bash scripts/gemma4_gpu.sh optimize --report-only
# Append source/task-balanced metrics at all operating points, including F1/FPR trade-offs.
bash scripts/gemma4_gpu.sh report --optimization
# After model runs complete, add paired group-bootstrap intervals against baseline.
bash scripts/gemma4_gpu.sh uncertainty
```

The original primary-development screen selects one new-policy finalist per objective. That finalist, the original baseline and the supported prior optimized route are fully compared on all 1,812 primary-development cases. Contact-sheet routes are excluded for objectives without multi-image development coverage. Three methods are reported: full-development source-AP selection; a safeguard requiring at least 0.005 mean-source AP improvement with no source losing more than 0.01 AP; and equal blending of baseline/selected margins after development-only standardization. The blend is an additional comparison rather than a test-selected fallback.

The blend's raw-zero operating point is a development-centered score control, with no direct Yes/No probability interpretation. Development-calibrated operating points and source-level AP/AUROC should be used to assess it alongside the original routes.

All route choices, blend constants, four primary operating thresholds and secondary XRF thresholds are written before new held-out inference. Every referenced image's decoded pixel hash and every model's pinned weight hash are checked. Previous margins can be imported only after a development numerical parity check with maximum allowed margin difference 0.02. Each model saves new implementation snapshots, cache identities, route/calibration files and per-case margins. The replay audit reconstructs finalists, routes, constants, thresholds, held-out scores and every source-level metric before publishing a completed result.

The initial input preflight found float32 normalized detection coordinates just above one (about 1e-7). A pre-score amendment records accepting boundary roundoff up to 1e-6 and clamping it; larger excursions, nonfinite and degenerate boxes still fail. No new neural score had been produced before that amendment. Original detector caches remain unchanged.

These are forced binary alternative scores with thinking disabled, rather than free-text explanation evaluations. Original probability-mass limitations still apply. No Gemma weights are trained and no latency experiment is run. The new methods were designed after reviewing the first-stage results on the same holdouts, so the study is explicitly retrospective. Independent new-camera/session evaluation remains necessary for a deployment claim.

All five variants completed the 4,032-case held-out comparison and passed replay. The [completed report](../reports/gemma4_optimization_20261028/report.md) and [operating summaries](../reports/gemma4_optimization_20261028/operating_summary.csv) retain every original and new method. The highest observed Gemma common-source macro AP is 0.7763 for the 31B baseline/selected-score blend, versus its original baseline of 0.7678. Full-development selection reaches 0.7752. The trained SigLIP reference remains higher at 0.7911 on the same 3,792-case common cohort. The 12B blend reaches 0.7640, compared with 0.7072 baseline and 0.7444 first-stage optimized AP. These are observed comparisons, not a new policy selected from untouched test data.

Operating metrics remain limiting. At the frozen every-source development FPR-budget point, 31B full-development selection raises all-source macro F1 from 0.4213 to 0.4670 while macro FPR rises from 0.0737 to 0.0966. Its CCTV smoking recall is 6.36%; the blend with the highest macro AP has zero CCTV recall at its corresponding threshold. Full-development loitering recall is 9.09%, with 15.63% FPR. The [500-resample paired group intervals](../reports/gemma4_optimization_20261028/ap_uncertainty.csv) include a negative HMDB smoking-transfer AP interval for 31B, as well as earlier model regressions. These results support further independent video/camera validation rather than an automatic deployment replacement.
