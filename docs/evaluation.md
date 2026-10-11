# Running the corrected evaluation

Use Python 3.12, the repository requirements, an editable package installation, and `ffmpeg`/`ffprobe` on PATH. The CLI works from any directory; see [server paths and migration](server.md). CPU-only model inference is sequential and isolated; the default shortlist is MobileCLIP2-S2, SigLIP2-B/16-224, SmolVLM-500M and MiniCPM-V 4.0. VadCLIP is an optional separately supervised smoking study.

```bash
anomaly-eval run --prepare-only
anomaly-eval run --threads 4 --max-rss-mb 24576 --camera-fps 1
```

The first command validates/prepares datasets and prints a report without loading model weights. The second runs every available task/protocol through development and test inference, calibration, group bootstrap, paired differences, figures and reporting. Each invocation creates a new results directory. `--output` must name a directory that does not exist. `--offline` requires cached weights; `--skip-prepare` reuses a prepared dataset, but the benchmark still verifies every recorded image hash. The portable curated CAVIAR bundle works without downloading the source videos.

For a first pipeline check:

```bash
anomaly-eval run --models mobileclip2-s2 --tasks unattended_items \
  --protocols frame --threads 2 --bootstrap 200
```

## Fetch the sources

```bash
# New public videos/annotations, with a receipt for each source:
anomaly-eval run --fetch --prepare-only
# Original Drive data and additional public data, in one invocation:
anomaly-eval run --fetch-drive --use-cookies --fetch --prepare-only
# Optional 3.85 GB D2-CDNet auxiliary foreground-segmentation shard:
anomaly-eval fetch --sources d2_cdnet --d2-shards intermittentObjectMotion
# Bounded review candidates from downloaded ABODA videos and D2 archives:
anomaly-eval review
```

Drive cookies come from the local gdown cache. Import them using a signed-in browser as previously documented in README; cookie extraction is not performed by the runner. Drive transfers resume; the new HTTP downloader skips checksum-verified complete files but restarts incomplete transfers. Concurrent Drive filename collisions remain separate files; annotation lookup uses the rename receipt, and conflicting contents for an original name are rejected rather than paired arbitrarily.

Sources and label semantics:

| Source | Fetched data / access | Permitted evaluation evidence |
|---|---|---|
| [ABODA](https://github.com/kevinlin311tw/ABODA) | 11 public videos, pinned repository revision | No frame annotations in this repository; review event intervals and normals |
| [CAVIAR](https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/) | Five leaving/pickup clips, three walking controls, XML annotations and grammar | Review roles/groups over time; object presence alone is insufficient |
| [D2-CDNet](https://huggingface.co/datasets/tauseefakhund/D2-CDNet) | Metadata by default, selected tar shards on request | Foreground masks are auxiliary; never automatic unattended-item labels |
| [Abandoned object Detection](https://universe.roboflow.com/abandoned-object/abandoned-object-detection-9fijk) | API key plus explicit export version | Suspicious/non-suspicious object classes need event review |
| [Abandoned](https://universe.roboflow.com/getting/abandoned-w7arq) | API key plus explicit export version | Person/laptop/bag classes do not establish abandonment |

For the two Roboflow exports, set `ROBOFLOW_API_KEY` in your environment and choose the version shown by the dataset export page. Keys and signed download links are not saved in receipts or CLI arguments. Replace `VERSION` below with a real version; these sources are not silently substituted.

```bash
anomaly-eval fetch --sources roboflow_suspicious roboflow_objects \
  --roboflow-version roboflow_suspicious=VERSION \
  --roboflow-version roboflow_objects=VERSION
```

Keep source attribution/terms with redistributed data. CAVIAR requests attribution to EC CAVIAR/IST 2001 37540 and specifies CC BY-SA; Roboflow pages specify CC BY 4.0. D2's packaging license does not replace original CDnet source terms.

## Review and add events

The old 80-image bundle and known source groups are development only. Loitering membership, lying poses and generic action boxes remain separate development proxies. The main runner excludes them from confirmed task tests. Labels must describe the requested behavior: for example, putting down a bag is not necessarily littering, and lying on the floor does not establish rough sleeping.

The initial CAVIAR intervals in `anomaly_eval/config/caviar_review_intervals.json` use sequence XML plus chronological frame inspection. They cover bags left after the carrier departs, attended placement/retrieval and walking controls. They are preliminary, staged, single-camera examples with only two positive source sequences, so independent review and additional camera/domain coverage remain necessary.

For reviewed images or prebuilt chronological windows, create a CSV relative to its own directory:

```csv
path,label,task,group_id,source,evidence,review_note,protocol
images/event.png,loitering,loitering,cameraA:recording07,local_camera,human_review,Person remains in zone for reviewed interval per project definition,window
images/control.png,normal,loitering,cameraB:recording18,local_camera,human_review,Person walks through the zone without remaining,window
```

Use `normal` or the named task as the label, `human_review` or `sequence_annotation_review` as evidence, and `frame` or `window` as the protocol. A negative is task-specific: a “Not Smoking” annotation does not assert absence of every other event. Assign all frames, mirrored/augmented derivatives and windows from the same recording to one `group_id`. Do not invent separate groups for adjacent frames. Record the actual operational duration/owner-distance criteria in the review notes; the runner does not fabricate those requirements. At least two independent positive and negative source groups are needed for a preliminary dev/test comparison; substantially more are needed for reliable intervals and deployment claims.

```bash
anomaly-eval run --review-csv data/reviewed_events.csv --require-all-tasks
```

The runner merges identical image bytes, rejects contradictory labels, keeps connected source groups together, and applies per-group caps without relaxing them. Exact hashes cannot detect every transformed/near duplicate; review source genealogy and similarity candidates before importing data. `dataset.json` lists task/split support and limits. `--require-all-tasks` returns exit code 2 if requested task coverage remains incomplete. The default completes available tasks and records gaps instead of presenting missing tasks as zero performance.

## Metrics and fair comparisons

Every model receives the same ordered cohort within a task/input/split. End-to-end image latency includes file decoding, preprocessing, inference and response parsing. Cold load and warmup are separate; peak RSS is sampled during both loading and inference. Code/image/definition hashes, checkpoint revisions, native processors and resource settings are recorded. The input cap defaults to 768 pixels and generation budget to 64 tokens. The unified runner defaults to `--image-policy letterbox`, padding to square before native processing so center crops do not discard camera or contact-sheet edges; `--image-policy native` permits a separate development experiment with the original crop behavior; model-specific native preprocessing and VLM chat formatting remain explicit. Use `--repeats 3` or more for a more stable latency estimate; repeat zero supplies quality metrics while all repeats supply timing. Change these only as a development experiment, freeze the chosen configuration, then evaluate test once. Repeated test inspection makes that set development data for future model selection.

Primary categorical accuracy and class recall/F1 include failed and unanswered positive cases. Conditional accuracy is a separate diagnostic. Unknown normal cases are not true negatives; unanswered normal count and coverage are explicit. Macro classes come from ground-truth support, never from a model's predicted classes. Unsupported events are reported as untested.

For continuous scores, `calibration.json` reports thresholds fitted on development data for maximum F1 and a requested false-positive budget (`--target-fpr`, default 0.05). Test labels never tune thresholds. It includes AP/AUROC, recall, specificity and false-positive rate at those frozen thresholds. A low-FPR calibration with 16 negatives has resolution 1/16; it cannot establish a reliable 5% false-alert operating point. VLMs that only emit labels have no invented continuous-score AUC or threshold.

`improvements.json` resamples source recording groups, uses fixed macro classes and reports paired differences using shared resampling draws. It requires complete matched cohorts and metrics consistent with saved predictions. With three source groups the intervals are exploratory, not strong statistical evidence. Missing scores remain NA, not zero. No weighted fitness score converts incomparable or missing metrics into a winner.

Frame and past-window contact-sheet results are separate. Contact sheets provide four chronological frames ending at the labeled time, over six seconds, for both CLIP and VLM adapters. They are a common image-input experiment, not native video inference or proof of arbitrary-duration loitering. They may lose small-object detail through resizing. More frames, longer operational windows, person/object crops and tracking require separate development protocols; ground-truth boxes cannot be used as a free deployment detector.

A single stream sampled at one inference frame/sec is the default resource screen. The report compares p95 and sampled RSS with that budget; camera decoding/queueing are excluded and must be measured in deployment. `throttle.py` limits threads/affinity and adds pacing; it does not emulate a Ryzen CPU. Run the same frozen command on the actual deployment box.

## VadCLIP

```bash
anomaly-eval run --vadclip --vadclip-epochs 100
```

This requires the original smoking COCO frames. Encoder weights and content-validated features are reused under the configured cache root; run-specific adapter checkpoints remain under the results root. The runner fetches and verifies the official OpenAI CLIP-B/16 checkpoint if needed, extracts content-validated features, creates a frozen source-session train/dev/test split, trains/restores the best CPU adapter, calibrates development thresholds, and evaluates original binary test frames. CLIP zero-shot and fixed-C logistic regression use the same original frame features and split. Architecture knobs remain available in `anomaly-eval vadclip train --help`; tune them on development only, not by comparing final-test ablations.

Alignment training uses the author-scaled cosine temperature (0.07) with stable soft-target log-softmax cross-entropy; evaluation preserves score ordering in float64. This is supervised frame learning, rather than official VadCLIP weakly supervised video MIL. Whole sparse camera/date sessions can include future context, original timestamps may have gaps, and adapter-only evaluation timing excludes the encoder. Historical inspection of the smoking corpus means the new frozen split is a controlled reanalysis, not a genuinely untouched external test. It cannot by itself rank VadCLIP against zero-shot models on five different events.

## Outputs and checks

The command `anomaly-eval analyze --results-dir RUN --dev-dir DEV_RUN` recalculates statistics, calibration and figures from saved predictions. Each run contains `evaluation.json`, `report.md`, logs and one task/protocol directory with `dev/` and `test/` results. Each benchmark contains `run.json`, `comparison.csv`, per-model predictions/summaries, `improvements.json`, plots and test `calibration.json`. The notebooks are thin front ends to this same runner and contain no saved old scores.

```bash
python -m unittest discover -s tests -v
ruff check anomaly_eval scripts tests
ruff format --check anomaly_eval scripts tests
anomaly-eval audit
```

Exit code 1 indicates an execution/analysis failure; logs and partial outputs are preserved. Optional sources without credentials are explicitly recorded in the fetch receipt. Their absence is not a successful download. Preparation success does not imply model inference or complete event coverage.

Runtime paths can be set through global root flags, server JSON or `CAPSTONE_*` environment variables. Use `anomaly-eval paths` to inspect them. The default audit writes under `RESULTS_ROOT/audits`; archived historical audits stay under `reports/archive/2026-10-11/`.
