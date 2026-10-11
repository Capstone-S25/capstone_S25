# CPU anomaly-model evaluation: corrected status

The former 80-image comparison and VadCLIP results do **not** establish a deployment winner. The [original report](archive/anomaly_benchmark_report_legacy.md) is retained as historical evidence, and the [audit](archive/2026-10-11/evaluation_audit.md) explains the defects. Its numeric tables are not fresh results from the corrected code.

## Implemented corrections

- Classification recall/F1 includes positive abstentions and errors. Conditional accuracy, coverage, errors and unanswered normal cases are separate. Macro-F1 uses a fixed ground-truth-supported class set; absent unattended support never implied a 5/6 ceiling.
- Temporal alignment restores the author's temperature with stable soft-target log-softmax loss. Training keeps soft block targets, while evaluation restores scores to original binary frames. No fractional target is cast to zero. Precision, recall, F1 and accuracy use the same threshold. Thresholds and checkpoints are selected on development data; final test groups are separate.
- CPU best-state snapshots clone tensors and record the actual best epoch. Temporal attention, graph inputs and prompt pooling mask padding; partial attention windows remain valid. Cache provenance includes source image content, labels, preprocessing, source split and encoder weights.
- Notebook workflows call the same unified runner, clear stale outputs, preserve old result directories and remove arbitrary weighted fitness rankings. Source-group bootstrap and paired model differences require complete identical prediction cohorts.
- Three shortlisted Hugging Face checkpoints are pinned to commits; MiniCPM retains its existing pinned revision. MobileCLIP loads a pinned local OpenCLIP snapshot. Square letterboxing preserves the field of view through native center crops (with an explicit native-policy alternative). Native processors, capped input resolution, generation settings, code hashes and sample identities are recorded.

## Dataset changes

The historical 80-image bundle is development only. Lying posture, person boxes and dataset membership remain proxies for rough sleeping and loitering; the littering action annotations also need review against the actual rubbish-disposal definition. Those proxies are excluded from confirmed held-out task scoring.

Additional data was fetched from [ABODA](https://github.com/kevinlin311tw/ABODA), [CAVIAR](https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/), and [D2-CDNet](https://huggingface.co/datasets/tauseefakhund/D2-CDNet). CAVIAR contributes preliminary sequence-reviewed unattended-item positives plus attended placement/retrieval and walking controls. D2-CDNet foreground masks are auxiliary evidence, not abandonment labels. The two requested Roboflow exports need an API key and explicit dataset version; their class names are not automatically treated as unattended events.

The portable task bundle has **104 inputs**: 32 historical smoking/normal development frames and 72 new CAVIAR inputs, comprising 36 endpoint frames and 36 past-frame contact sheets. Each CAVIAR protocol has 20 development inputs (4 positive, 16 normal) and 16 test inputs (4 positive, 12 normal), with three source sequences per split. The same groups stay together across frame/window views. This is a small staged single-camera study, not a new-camera or deployment-domain validation.

## Validation and limits

Regression checks cover threshold indexing/ties, soft-target rejection, original-frame positive counts, CPU checkpoint restoration, padding invariance and finite gradients, split leakage, Drive filename collisions, cache integrity, incomplete prediction cohorts and development-only calibration. A real pinned MobileCLIP CPU smoke run completed using the unified pipeline on the new CAVIAR cohort. This validates execution; it is not a full four-model rerun or a five-task ranking.

VadCLIP remains a separately supervised **offline smoking adaptation**, with zero-shot CLIP and linear-probe controls on the same cached original frames. Its session-wide context includes future information, and adapter-only timing excludes feature extraction. It must not be described as streaming-ready or compared directly with end-to-end VLM FPS. The official VadCLIP XD-Violence headline AP is a frame metric, not segment mAP. [Official implementation](https://github.com/nwpu-zxr/VadCLIP/blob/main/src/xd_test.py).

There is still no adequate untouched, verified test set for all five events. Additional reviewed event labels and matched normal cases are necessary for littering, loitering and rough sleeping; historical smoking groups cannot become unseen evidence by reshuffling them. The runner records missing coverage and supports `--require-all-tasks` to enforce completeness. See [rerun instructions](../docs/evaluation.md).
