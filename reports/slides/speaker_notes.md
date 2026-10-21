# Presentation speaker notes

Evaluation: 13 October 2026. Sources are relative to the repository root.

## 01. CLIP & VLM anomaly detection

This briefing uses the completed 13 October 2026 runs, not the older report open in the IDE. It summarizes MobileCLIP2-S2, SigLIP2-B/16-224, SmolVLM-500M-Instruct, MiniCPM-V 4.0 and the separate supervised VadCLIP smoking adaptation. No five-task deployment winner is supported. Sources: reports/workflow_review_20261013.md and results/gpu_all_models_20261013/report.md.

## 02. What we learned — and how we tested it

Standalone summary slide. Methodology: source review, deduplication, recording-group splits, identical inputs and dev-only thresholds. GPU quality: four models × 3,350 inputs = 13,400 predictions, without pacing. VadCLIP has a different supervised offline split. CPU latency: same eight dev images, three repeats, four logical CPUs, float32, one worker, batch one. Smoking baseline recall: MobileCLIP 0.56%, SigLIP 0%, SmolVLM 5.41%, MiniCPM 0%. The three other tasks need verified event and normal labels; CAVIAR testing has only one positive source sequence per split.

## 03. Compare model families on shared inputs

CLIP models score image/text similarity, with cached native text features; MobileCLIP2-S2 is loaded through its OpenCLIP checkpoint and reparameterized for inference. SigLIP uses independent sigmoid scores and anomaly-minus-normal raw-logit margins. VLMs use native chat templates and deterministic JSON-label requests. MiniCPM uses the publisher's chat interface, one slice in this baseline. VadCLIP uses a frozen official CLIP-B/16 encoder plus a trained offline adapter. Publisher references: https://github.com/apple-aiml-research/ml-mobileclip ; https://huggingface.co/google/siglip2-base-patch16-224 ; https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct ; https://huggingface.co/openbmb/MiniCPM-V-4 ; https://github.com/nwpu-zxr/VadCLIP .

## 04. Only two tasks have usable event labels

Smoking video COCO: 3,278 confirmed frames, 16 known camera/day groups, 1,422 positives and 1,856 normals; development only. An additional 936 CCTV frames are all positive and need source genealogy and matched controls. CAVIAR supplies 20 dev and 16 test endpoints; window sheets share the same endpoints. There are three source groups and only one positive source sequence per split. Littering action boxes, loitering person boxes and rough-sleeping posture boxes are excluded from confirmed event scoring. Rough sleeping has 224 image files but only 112 unique byte/RGB hashes. Sources: reports/data_readiness_20261013.json, data/evaluation_all/dataset.json and the combined report.

## 05. Protect the data before measuring quality

The workflow validates annotations and source identity, then deduplicates exact bytes, decoded EXIF-corrected RGB pixels and checksum-qualified source moments. Conflicting labels and source leakage fail validation. Historical benchmark groups are development only. Recording groups remain together across frame/window protocols. Models see the same ordered cohort. Continuous thresholds are chosen on development and frozen for test. Metrics include failures and abstentions; recording-group bootstrap is paired across models. Model revisions, processors, definitions, code hashes, input hashes and raw responses remain in run artifacts.

## 06. Accuracy and latency are separate tests

GPU: RTX 5000 Ada Generation with 32,760 MiB VRAM; isolated CUDA Torch 2.8.0+cu128, no system driver changes. All 3,350 prepared confirmed inputs per model; zero per-group cap and inference limit. No duty-cycle/camera-rate sleeps, no default accuracy RAM cap. CLIP float32 batches of 32; VLM unquantized bf16 single-input calls. Shared letterbox max-side 768 and 64-token deterministic generation. CPU pilot: original Torch 2.14.1+cpu on Threadripper PRO 7975WX, affinity 0–3 and four threads, float32, one worker, batch one. Same eight unattended-item dev images, three measured repeats, three warmups, 24 observations/model. The host budget does not emulate a different processor.

## 07. Measure misses and false alarms together

Recall is TP/(TP+FN); precision is TP/(TP+FP); F1 summarizes precision and recall; false-positive rate is FP divided by labeled normal inputs. Unsupported labels are not zero-scored. AP and AUROC use genuine continuous scores, with noninterpolated AP and tied-score handling. VLM categorical outputs do not receive invented AP/AUROC. All failures/abstentions remain in categorical quality denominators; prediction coverage is separate. All main group analyses use 2,000 recording-group bootstrap draws, paired across models. Few CAVIAR groups make uncertainty exploratory. Raw argmax and dev-calibrated operating points are separate.

## 08. Current prompts miss most smoking positives

These are current-protocol development results, not unseen-camera tests. MobileCLIP TP=8 FP=4 FN=1,414 TN=1,852. SigLIP all normal. SmolVLM TP=77 FP=114 FN=1,345 TN=1,732, with 14 invalid responses (four positive and ten normal); those are included in quality. MiniCPM all normal. Positive prevalence is 0.433801, so constant-score AP is 0.434. MobileCLIP AP 0.424 and SigLIP AP 0.443 are near that reference. Paired AP/AUROC intervals do not distinguish the CLIP models robustly. SmolVLM-minus-MobileCLIP F1 CI crosses zero. These measure the tested prompts/preprocessing, not all native capabilities.

## 09. Constant answers do not show event detection

Frame and past-window protocols share the same endpoints; they are not independent event samples. Every split has only one positive source sequence. Raw frame F1: MobileCLIP 0, SigLIP 0, SmolVLM 0.4, MiniCPM 0. Raw window F1: 0.25, 0, 0.4, 0 respectively. SmolVLM predicts unattended items on all dev/test frame/window inputs. Its 0.4 test F1 equals the always-anomaly baseline, with FPR 1. MiniCPM predicts normal throughout. MobileCLIP window argmax finds one of four positives, with three false positives. MobileCLIP max-dev-F1 frame threshold obtains test recall 1, precision 1/3, F1 0.5, FPR 2/3. Dev FPR-budget thresholds give zero test recall for both CLIPs.

## 10. VadCLIP shows limited score discrimination

Frozen CLIP-B/16 encoder and supervised temporal adapter with future context. Train 9 sessions/1,910 frames; development 4 sessions/312 frames; test 3 sessions/1,056 frames. Test has 560 positives and 496 normals. Selected branch dual_mean, epoch one, dev-fitted threshold 0.7381770357355234. Test TP290 FP177 FN270 TN319; precision 0.620985, recall 0.517857, F1 0.564752, FPR 0.356855, AP 0.633230, AUROC 0.587403. Visual and dual_max branches are all-positive with AUROC 0.5 and F1 0.693069. Linear probe FPR 0.977823. This is not the official weakly supervised UCF-Crime/XD-Violence task, and historical smoking corpus inspection means the split is a controlled reanalysis, not a fresh external test.

## 11. CLIP is cheaper in the CPU latency pilot

The CPU latency pilot ran separately after quality jobs completed. Native decoding, preprocessing, model inference and parsing are included. Load and three warmups are excluded from the 24 measured timings per model; peak worker RSS includes load/warmup. MobileCLIP p95 0.056316s RSS993.316MiB; SigLIP 0.137628s RSS1248.602MiB; SmolVLM 7.563337s RSS3350.336MiB; MiniCPM 8.176413s RSS18850.168MiB. This is a small replay on Threadripper PRO 7975WX, not a hardware emulator or deployment queue/capture test. Output lengths and alternative image protocols can change costs. Model label abbreviations are used in the chart.

## 12. Remove limits that hid evaluation capacity

Changes include explicit runtime device/dtype resolution throughout adapters and VadCLIP; preserving GPU visibility instead of forced CUDA_VISIBLE_DEVICES=''; native CLIP batching and isolated concurrent workers; CUDA synchronization and memory peaks; removing default accuracy RAM cap; increasing timeout to four hours; removing the hidden per-recording preparation cap; recording available/retained sampling counts; indexing and invalidating download receipt lookups; AP and richer quality diagnostics; native one-input deployment timing. The old prepared cap produced 308 inputs versus 3,350 uncapped. Both environments passed 144 tests, plus lint/format/shell checks. No additional model inference is performed by this presentation generator.

## 13. No five-task deployment winner is supported

No present configuration has demonstrated useful recall at an acceptable false-alert rate across five anomalies. Next steps are to review littering/loitering/rough-sleeping events and task-specific controls with camera/source genealogy; define operational duration and ownership rules; evaluate detected-person/object crops with detector failures included; investigate native VLM image slicing and video/temporal protocols on development only; freeze prompts/thresholds and test new recordings/cameras; report event detection delay and alerts/hour, then test latency on actual deployment hardware. Historical smoking and already-inspected CAVIAR test inputs are development evidence for subsequent protocol changes. Improvements are proposed experiments, not measured gains.

## 14. Prompt sensitivity and uncertainty matter

All smoking intervals are recording-group bootstraps over 16 known groups, 2,000 draws. MobileCLIP F1 CI[0.002442,0.027028]; SigLIP[0,0]; SmolVLM[0.006328,0.172909]; MiniCPM[0,0]. SmolVLM-minus-MobileCLIP F1 CI is [-0.012861,0.166235]. The short-definition CPU float32 CLIP experiment changed only text definitions and used all 3,278 smoking dev frames: MobileCLIP recall0.533052 FPR0.493534 F10.489664 AP0.460063; SigLIP remains all normal. VadCLIP CPU-cache development-only LR sweep: 1e-5 AP0.369294, 1e-4 AP0.393464, 1e-3 AP0.424815, selected 1e-3 epoch1 without selecting on test labels. Only selected candidate tested, AP0.631507. Native image preprocessing and numerical environment remain explicit.

## 15. Development thresholds expose the tradeoff

The 5% FPR budget is an empirical development constraint. With 16 normal development endpoints, FPR increments are 6.25%, so the constraint requires zero observed dev false positives; it is not a population guarantee. Rows are test outcomes under frozen thresholds. Frame/window cohorts share endpoint labels and have only three source groups per split, with one positive sequence. Both CLIP budget thresholds yield zero test recall. All max-dev-F1 and budget operating points are shown, not only favorable ones. VLM hard label outputs have no score threshold fitting.

## 16. Every claim traces back to a saved run

Primary evidence: results/gpu_all_models_20261013/report.md and evaluation.json; per-model summary.json and predictions.jsonl; per-cohort improvements.json and calibration.json. Resource pilot: results/cpu_latency_20261013/run.json and per-model summaries. Dataset audit: reports/data_readiness_20261013.json. Interpretation and limitations: reports/workflow_review_20261013.md. Numeric export: reports/model_comparison_20261013.csv. Supplementary experiments: results/clip_prompt_dev_20261013/short_descriptions and results/vadclip_dev_sweep_20261013/selection.json. Reproduction on this workspace uses scripts/evaluate_gpu.sh run --device cuda --per-group 0 --batch-size 32 --vadclip. It creates a new results directory. Future five-task readiness uses --require-all-tasks; currently incomplete. Source hashes and the exact slide data are captured in source_manifest.json. Rendering uses python-pptx and ReportLab; the PDF is built from the same layout specification, not a PowerPoint export.
