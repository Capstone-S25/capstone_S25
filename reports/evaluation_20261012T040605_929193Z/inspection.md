# Assessment of evaluation_20261012T040605_929193Z

The results support an extensive exploratory methods/results report, but do not
establish a final detector for all five anomalies or the deployment hardware.
The compiled report is [paper/paper.pdf](paper/paper.pdf); its editable source is
[paper/paper.tex](paper/paper.tex).

- The run uses evaluation mode, all **prepared** inputs, frame/window protocols,
  32 CPU threads, three repeats, seed 42, 2,000 bootstrap draws and no pacing.
- Prepared schema 3 contains 308 image inputs: 236 historical smoking development
  frames and 72 CAVIAR frame/window representations of 36 endpoint moments.
  Preparation removed 104 aliases before its per-group/label caps. All selected
  CAVIAR pixel/source-moment identities are unique within each protocol/split.
- Nineteen of twenty model/cohort jobs complete. MiniCPM smoking exceeds its
  3,600-second timeout after 696/708 measured rows; its first two passes are
  complete. Total saved rows are 3,684/3,696. Original status `error` is retained.
- All 28 recorded source/configuration hashes match the inspected checkout.
  Completed metric dictionaries and p95 values reproduce. All four saved
  bootstrap/paired outputs and both development-calibration files reproduce
  exactly on temporary copies. Original result files were not edited.
- All 72 CAVIAR inputs and twelve smoking inputs verify against their recorded
  byte and decoded-pixel hashes locally. Another 224 smoking files are absent
  here; this limits visual/source validation, not raw prediction-count checks.
- Smoking has no untouched test set. Littering, loitering and rough sleeping have
  no confirmed scored cohorts. VadCLIP was not requested. Proxy candidates are
  not scored event examples.
- Each CAVIAR split has three recordings and only one positive recording. Four
  positive moments are not four independent positive events. The frame/window
  views share those moments. Dev/test recording and image identities are disjoint.
- On test endpoints, MobileCLIP, SigLIP and MiniCPM always predict normal;
  SmolVLM always predicts unattended items. These match constant baselines.
  MobileCLIP window F1 is 0.250 with FPR 0.250; SmolVLM window F1 is 0.353 with
  FPR 0.833. Neither establishes useful operating performance.
- MobileCLIP's development-max-F1 endpoint threshold detects all four test
  positives but flags 8/12 normals (FPR 0.667). Window max-F1 thresholds for both
  CLIP models flag every test input. FPR-budget thresholds have zero event recall.
- Each development cohort has sixteen negatives, giving 6.25% FPR resolution.
  Test has twelve negatives, giving 8.33% per false positive. An empirical 5%
  development budget is not a demonstrated population false-alert guarantee.
- The bootstrap omits the single positive recording in 571/2,000 draws. Undefined
  F1/AUC values are omitted, and finite-only paired intervals can condition on
  different usable draws. These ranges are exploratory, not broad superiority
  or significance evidence.
- The host is a Threadripper PRO 7975WX with approximately 251 GiB RAM. CLIP
  test p95 is 0.044–0.065 s; generative VLM p95 is 2.73–5.37 s. These measurements
  do not emulate the intended Ryzen AI Max+ 395-class, approximately 32 GB host.
  No camera queues, alerts/hour, event delay or sustained live service are tested.

The [machine-readable audit](paper/audit.json) records 139 checks and an original
artifact SHA-256 inventory. Its one expected failed check is the incomplete
MiniCPM smoking cohort. The paper presents these limits alongside the full data,
methods, rationale and a concrete completion protocol; missing evidence is not
filled with assumed experiments or a fabricated winner.
