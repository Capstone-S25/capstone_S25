# Running the corrected evaluation

Use Python 3.12, the repository requirements, an editable package installation, and `ffmpeg`/`ffprobe` on PATH. The CLI works from any directory; see [server paths and migration](server.md). Model workers are isolated; the default shortlist is MobileCLIP2-S2, SigLIP2-B/16-224, SmolVLM-500M and MiniCPM-V 4.0. VadCLIP is an optional separately supervised smoking study.

```bash
anomaly-eval run --prepare-only
anomaly-eval run --device auto
```

The first command validates/prepares datasets and prints a report without loading model weights. The second runs every available task/protocol through development and test inference, calibration, group bootstrap, paired differences, figures and reporting. Each invocation creates a new results directory. `--output` must name a directory that does not exist. `--offline` requires cached weights; `--skip-prepare` reuses a prepared dataset, but the benchmark still verifies every recorded image hash. The portable curated CAVIAR bundle works without downloading the source videos.

For a first pipeline check:

```bash
anomaly-eval run --mode quick --models mobileclip2-s2 --tasks unattended_items --threads 4
```

## Expanded accuracy study, finalized 14 October 2026

The [expanded study](../reports/expanded_accuracy_review_20261014.md) uses separate
research manifests under `data/expanded_20261013/cohorts/`. It retains all eligible
deduplicated candidate inputs and holds related unattended-item camera families
together. The unattended frame/window cohorts each contain 808 development and
962 test inputs. Their interval labels are **assistant visual review**, not
independently human verified. The 20-second windows also change the earlier
six-second protocol.

The three other anomaly exports are positive-only weak diagnostics: 6,667
littering candidates, 1,675 loitering candidates and 111 rough-sleeping candidates.
Their report supplies target-prediction rates rather than event accuracy. Separate
person-presence and synthetic oracle posture tasks help distinguish visible
evidence from unsupported event claims. These manifests do not promote weak data
into the main runner's confirmed five-task coverage or satisfy `--require-all-tasks`.

All four image models completed 55,156 unpaced CUDA predictions in this expansion.
Smoking and the supervised offline VadCLIP study reuse prior GPU evidence. There
was no new latency experiment. The [full report](../results/expanded_accuracy_20261013/report.md)
and [updated slides](../reports/slides/model_evaluation_expanded_20261014.pdf) include
the limitations, raw decisions and development-frozen CLIP operating points.

`scripts/expand_accuracy.py` prepares review material and manifests;
`scripts/run_expanded_accuracy.py` executes this study into an exclusive output
directory. Preserve the completed output; the inference script deliberately
refuses to overwrite it. To regenerate the accuracy report from saved predictions:

```bash
PYTHONPATH="$PWD" .venv/bin/python scripts/report_expanded_accuracy.py
```

## Evaluation speed and deployment pacing

| `run --mode` | Inference pacing | Default input cap per task/protocol/split | Default protocols | Bootstrap draws |
|---|---|---:|---|---:|
| `evaluation` (default) | No deliberate sleeps | All prepared inputs | Frame and past-window sheet | 2000 |
| `quick` | No deliberate sleeps | 8 | Frame | 200 |
| `deployment` | At most `--camera-fps` inputs/sec; optional duty-cycle sleeps | 16 | Frame | 200 |

Use quick mode to check execution, then evaluate the full cohorts. Accuracy runs
default to all available CPU threads, CUDA when available, and no host RAM cap.
CUDA CLIP batches default to 32; CPU batches default to 1. Generative VLMs remain
sequential. `--device cuda` fails if CUDA is unavailable; `--device auto` falls back
to CPU and records the resolved device. `--dtype auto` keeps CPU and CLIP float32
and uses bfloat16 (or float16) for CUDA generative VLMs. Device, precision, GPU
memory, batching and actual input counts are saved with each model. Batched timings
are explicitly marked `batch_amortized` and must not be used as single-input
deployment latency. Optional thread and memory settings remain available.
`--workers N` runs up to N isolated model processes concurrently; threads and RSS
caps are per model worker. Choose counts that fit available RAM/VRAM. Concurrent
accuracy timings reflect contention and must not be used as a fair latency ranking.
Deployment requires one worker and batch size one.
`--camera-fps` is only a resource-budget comparison in those modes; it does not
slow inference. `--duty-cycle` below 1 requires deployment mode. The standalone
`benchmark` command also defaults to duty cycle 1 with no rate cap; its pacing
flags are explicit opt-ins.

```bash
# Small, unpaced first pass across the four default CLIP/VLM candidates:
anomaly-eval run --mode quick --threads 8
# Faster CLIP-only screen; all requested tasks with confirmed data remain eligible:
anomaly-eval run --mode quick --models mobileclip2-s2 siglip2-b16-224 --threads 8
# Full quality evaluation, rebuilding preparation without a per-group cap:
anomaly-eval run --device cuda --batch-size 32 --vadclip
# A CPU fallback can use several isolated workers without pacing:
anomaly-eval run --device cpu --workers 4 --threads 16 --batch-size 16
# Small paced resource check of one candidate on the actual deployment host:
anomaly-eval run --mode deployment --device cpu --models mobileclip2-s2 \
  --tasks unattended_items --threads 4 --camera-fps 1 --duty-cycle 0.5
```

Override presets with `--limit N` (0 means all), `--protocols frame window` or
`--bootstrap N`. The cap applies separately to each task, input protocol and
dev/test split, before repeating the chosen inputs. It limits model inference,
not downloaded files or preparation of the full dataset. Selection approximates
the original label proportions, retains both available event/normal labels,
and cycles through source groups with a fixed seed. Every model receives the
same subset; `run.json`, `evaluation.json` and the report record selected versus
available counts. Quick/subset results are pilot diagnostics, not a final model
ranking. A tiny negative sample also makes false-positive calibration coarse.

Preparation defaults to `--per-group 0` in the runner: all confirmed examples.
The older default of eight examples per label/recording hid a substantial cap even
when inference used `--limit 0`. `--per-group N` remains useful for an explicit
recording-balanced experiment, and `dataset.json` records available unique versus
retained counts. `--skip-prepare` retains the existing preparation cap; rebuild to
evaluate all annotated frames. The standalone `prepare` command retains its small
default; use `prepare --per-group 0` for all.

Prepared datasets now use schema 3. Preparation merges duplicate file bytes,
identical decoded EXIF-corrected RGB pixels and matching checksum-qualified
CAVIAR source moments. Frame and fixed past-window protocols remain distinct.
Conflicting labels fail validation, merged recording aliases stay in one split,
and bundled images take precedence over equivalent newly decoded copies.
`dataset.json` records removed aliases. Regenerate older prepared datasets by
running without `--skip-prepare`; they cannot be reused under the corrected
pipeline. JPEG/PNG decoding changes can also change pixel identity, so rebuild
preparation when changing image-decoder versions.

Limiting inputs, models and protocols is the main way to shorten a run. More CPU
threads can help some models but are not guaranteed to help every CPU/runtime;
compare settings on development data. Cached weights avoid repeated downloads,
and `--skip-prepare` avoids rebuilding prepared data. Models still load in
isolated processes per cohort, and large generative VLMs remain expensive on CPU.

Deployment mode defaults to CPU, four threads, 24 GiB RSS and batch size one. It
rejects larger batches so amortized batch time cannot pass as single-input latency.
Deployment mode paces sequential prepared inputs, including contact sheets if
explicitly requested. It is not a live camera/decoder/queue simulation and cannot
emulate another processor. Active inference latency excludes pacing sleeps;
wall throughput and pacing sleep seconds are separate measurements. A slow
model is not sped up to the requested camera rate. `--limit` applies to image
cohorts only; optional offline VadCLIP extraction/training uses its own complete
source-session split and is not supported in deployment mode.

## Which anomalies are actually tested?

All five task names are requested by default: `smoking`, `loitering`, `littering`,
`unattended_items` and `rough_sleeping`. Only confirmed task-specific data is run.
With the current curated bundle:

| Anomaly | Confirmed evaluation support |
|---|---|
| Smoking | Development frames; historical source groups are not untouched test data |
| Unattended items | Preliminary dev/test endpoint frames and past-window sheets from CAVIAR |
| Loitering | Needs reviewed duration/event and matched normal labels |
| Littering | Needs review of the action against the defined rubbish-disposal event and matched normals |
| Rough sleeping | Needs reviewed event and matched normals; lying posture alone is a proxy |

The runner prints each requested task's coverage and records missing tasks in the
report. Specifying `--tasks` does not create missing labels. Use
`anomaly-eval run --prepare-only --require-all-tasks` to check coverage; when a
requested task lacks dev/test support in the requested protocols, it returns
exit code 2 before loading any model. Add reviewed positives and task-specific
normal cases through the CSV workflow below to enable the missing evaluations.

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

The runner merges identical image bytes, rejects contradictory labels, keeps connected source groups together, and applies per-group caps without relaxing them. Exact hashes cannot detect every transformed/near duplicate; review source genealogy and similarity candidates before importing data. `dataset.json` lists task/split support and limits. `--require-all-tasks` returns exit code 2 before inference if requested task coverage remains incomplete. The default completes available tasks and records gaps instead of presenting missing tasks as zero performance.

## Metrics and fair comparisons

Every model receives the same ordered cohort within a task/input/split. End-to-end image latency includes file decoding, preprocessing, inference and response parsing. Cold load and warmup are separate; peak RSS is sampled during both loading and inference. Code/image/definition hashes, checkpoint revisions, native processors and resource settings are recorded. The input cap defaults to 768 pixels and generation budget to 64 tokens. The unified runner defaults to `--image-policy letterbox`, padding to square before native processing so center crops do not discard camera or contact-sheet edges; `--image-policy native` permits a separate development experiment with the original crop behavior; model-specific native preprocessing and VLM chat formatting remain explicit. Use `--repeats 3` or more for a more stable latency estimate; repeat zero supplies quality metrics while all repeats supply timing. Change these only as a development experiment, freeze the chosen configuration, then evaluate test once. Repeated test inspection makes that set development data for future model selection.

Primary categorical accuracy and class recall/F1 include failed and unanswered positive cases. Conditional accuracy is a separate diagnostic. Unknown normal cases are not true negatives; unanswered normal count and coverage are explicit. Macro classes come from ground-truth support, never from a model's predicted classes. Unsupported events are reported as untested.

VLM parsing accepts case-insensitive exact labels, including `Normal.` and
`Unattended_items.`, and preserves the raw response. It never finds a label by
searching arbitrary prose. Explicit `unknown`, invalid responses and ambiguous
labels have separate abstention reasons. `response_diagnostics` in model summaries
records JSON-label versus plain-label formats and their counts; `comparison.csv`
also records JSON-label fraction, explicit unknowns and invalid responses.
Formatting compliance is separate from detection quality. The report shows
categorical prediction counts so an always-normal or always-anomaly response
pattern is visible.

New manifests and saved sample identities include decoded pixel hashes and source
origins. Selection and coverage records include recording-group counts per label,
so three source groups cannot be mistaken for three independent positive events.
Manifest loading rejects repeated decoded pixels/source moments, including
re-encoded copies crossing dev/test splits, before any model weights are loaded.

## What a comprehensive model decision still requires

Unit tests validate code behavior; they do not establish the accuracy or suitability
of pretrained models. A quick run checks execution, output behavior and rough
resource cost. Its per-frame metrics and small group-bootstrap intervals do not
replace an event-level deployment evaluation.

| Requirement | Current limitation / necessary next evidence |
|---|---|
| All five defined events | Add reviewed positive events and task-specific normals for missing tasks, plus untouched smoking recordings |
| Independent held-out data | Use more recordings/cameras, conditions and people; keep related frames, views and augmentations in the same split |
| Useful operating point | Fit thresholds/prompts on development only; measure recall and false alerts with enough independent positives and negatives |
| Temporal behavior | Evaluate past-context windows and operational duration/ownership criteria; measure event recall and detection delay |
| Native model grounding | Use reviewed sanity examples and inspect raw responses/native preprocessing; successful execution alone does not prove the model uses the needed visual evidence |
| Deployment hardware | Repeat on the actual CPU/RAM configuration; include camera decoding, queueing, sustained throughput, alert aggregation and memory over a longer run |
| VadCLIP comparison | Run the separately trained smoking study if desired; account for feature extraction and its offline/future-context limitation |

The current runner records per-image quality and resource metrics. It does not yet
measure event detection delay, alerts per hour or a live camera capture/queue
pipeline. Deployment mode is a paced replay of prepared inputs. Human review and
additional data are required to fill the task-coverage gaps; relabeling proxies or
reshuffling previously inspected recordings cannot create an untouched test set.

For continuous scores, `calibration.json` reports thresholds fitted on development data for maximum F1 and a requested false-positive budget (`--target-fpr`, default 0.05). Test labels never tune thresholds. It includes AP/AUROC, recall, specificity and false-positive rate at those frozen thresholds. A low-FPR calibration with 16 negatives has resolution 1/16; it cannot establish a reliable 5% false-alert operating point. VLMs that only emit labels have no invented continuous-score AUC or threshold.

`improvements.json` resamples source recording groups, uses fixed macro classes and reports paired differences using shared resampling draws. It requires complete matched cohorts and metrics consistent with saved predictions. With three source groups the intervals are exploratory, not strong statistical evidence. Missing scores remain NA, not zero. No weighted fitness score converts incomparable or missing metrics into a winner.

Frame and past-window contact-sheet results are separate. Contact sheets provide four chronological frames ending at the labeled time, over six seconds, for both CLIP and VLM adapters. They are a common image-input experiment, not native video inference or proof of arbitrary-duration loitering. They may lose small-object detail through resizing. More frames, longer operational windows, person/object crops and tracking require separate development protocols; ground-truth boxes cannot be used as a free deployment detector.

A single stream sampled at one inference frame/sec is the default resource screen. The report compares p95 and sampled RSS with that budget; deployment mode also accounts for the selected duty-cycle overhead. Camera decoding/queueing are excluded and must be measured in deployment. Resource controls in `anomaly_eval/resources.py` limit threads/affinity and add pacing only when requested; they do not emulate a Ryzen CPU. Run the same frozen command on the actual deployment box.

## VadCLIP

Validate the original smoking corpus on the computation server first:

```bash
anomaly-eval vadclip extract --validate-only \
  --validation-report reports/vadclip_source_validation.json
```

This uses the configured data root (`raw/datasets` beneath it); global workspace
flags work as usual. It reads annotations and filenames without loading encoder
weights or computing features, and checks the source-session split. Source
identity includes camera, date, frame index and full capture timestamp. The frame
counter can restart during a day; timestamps determine chronological order,
including AM/PM and milliseconds. Camera/day groups stay together across
train/dev/test. Legacy sessions without timestamps retain index order; mixed or
malformed timestamps fail validation instead of inventing a chronology.

Duplicate exports with matching labels contribute one source frame. Opposing
labels at the same full source moment still fail, with both filenames, annotation
files and COCO image IDs in the JSON report. Review those records against the
original frame and annotations; do not automatically overwrite labels or discard
conflicts. Missing labeled images also fail validation. Feature cache fingerprints
include the new identity policy and ordered source records, so extraction rebuilds
old caches automatically. Unified runs save `vadclip/source_validation.json` even
when source validation fails.

```bash
anomaly-eval run --vadclip --vadclip-epochs 100
```

This requires the original smoking COCO frames. Encoder weights and content-validated features are reused under the configured cache root; run-specific adapter checkpoints remain under the results root. The runner fetches and verifies the official OpenAI CLIP-B/16 checkpoint if needed, extracts content-validated features, creates a frozen source-session train/dev/test split, trains/restores the best adapter on the selected device, calibrates development thresholds, and evaluates original binary test frames. CLIP zero-shot and fixed-C logistic regression use the same original frame features and split. Architecture knobs remain available in `anomaly-eval vadclip train --help`; tune them on development only, not by comparing final-test ablations.

Alignment training uses the author-scaled cosine temperature (0.07) with stable soft-target log-softmax cross-entropy; evaluation preserves score ordering in float64. This is supervised frame learning, rather than official VadCLIP weakly supervised video MIL. Whole sparse camera/date sessions can include future context, original timestamps may have gaps, and adapter-only evaluation timing excludes the encoder. Historical inspection of the smoking corpus means the new frozen split is a controlled reanalysis, not a genuinely untouched external test. It cannot by itself rank VadCLIP against zero-shot models on five different events.

## Outputs and checks

The command `anomaly-eval analyze --results-dir RUN --dev-dir DEV_RUN` recalculates statistics, calibration and figures from saved predictions. Each run contains `evaluation.json`, `report.md`, logs and one task/protocol directory with `dev/` and `test/` results. Each benchmark contains `run.json`, `comparison.csv`, per-model predictions/summaries, `improvements.json`, plots and test `calibration.json`. The notebooks are thin front ends to this same runner and contain no saved old scores.

### Collecting evidence for a report or paper

Use the corrected code on the computation server before preparing a new dataset.
Preparation alone produces coverage and provenance, not new model results:

```bash
# Fetch additional sources and rebuild/validate the prepared data:
anomaly-eval run --fetch --prepare-only
# Run the four default models on all prepared confirmed cohorts, without pacing:
anomaly-eval run --mode evaluation --skip-prepare --threads 8 --repeats 3
```

Use the same configured data, results, cache and bundle roots for both commands.
Fetching does not automatically review candidates or fill missing event labels.
"All" means all prepared confirmed inputs, subject to the preparation caps; it
does not mean every downloaded image. Each command creates its own run folder.
Share the complete **inference** folder as well as the preparation folder; a
status of `prepared` does not indicate an inference experiment. Reanalyzing an
old run recomputes its saved predictions; it does not apply the new parser or
rerun deduplication.

Keep these materials with the experiment:

- Both complete run folders, including raw predictions, response text, summaries,
  calibration, bootstrap results, figures and execution logs where generated.
- `DATA_ROOT/evaluation/dataset.json` and its prepared CSV manifests. Their paths
  remain relative to the original prepared-data directory or use workspace
  aliases; copying them elsewhere does not automatically make them runnable.
- The source-fetch receipts under `DATA_ROOT/additional`, any human-review CSV,
  anomaly definitions and CAVIAR review-interval configuration used for the run.
- The exact source-code revision, including any local changes, and notes on
  manual label review or development experiments that are not recorded in code.

These artifacts support a detailed methods and exploratory-results paper:
checkpoint choices, prompts, preprocessing, source selection, split rules,
calibration, uncertainty and CPU resource measurements can be traced to the
implementation. They do not supply evidence for untested events or explain
undocumented decisions. State task-coverage gaps and the distinction between
development, previously inspected test data and untouched test data explicitly.
After inspecting test results to change models or protocols, use fresh recordings
for subsequent final claims.

If the paper includes VadCLIP, add `--vadclip` to the inference command and retain
its adapter and baseline artifacts. Its supervised smoking adaptation and
offline temporal context must be reported separately from the four pretrained
image models. Measure the frozen selected configuration on the actual deployment
host before making hardware or live-stream claims.

```bash
python -m unittest discover -s tests -v
ruff check anomaly_eval scripts tests
ruff format --check anomaly_eval scripts tests
anomaly-eval audit
```

Exit code 1 indicates an execution/analysis failure; logs and partial outputs are preserved. Optional sources without credentials are explicitly recorded in the fetch receipt. Their absence is not a successful download. Preparation success does not imply model inference or complete event coverage.

Runtime paths can be set through global root flags, server JSON or `CAPSTONE_*` environment variables. Use `anomaly-eval paths` to inspect them. The default audit writes under `RESULTS_ROOT/audits`; archived historical audits stay under `reports/archive/2026-10-11/`.
