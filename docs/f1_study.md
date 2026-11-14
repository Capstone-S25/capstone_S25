# November F1 study and the ensemble classifier — 11 to 14 November 2026

> Update 14 November: `ensemble-20261114` is the default. See [Round 3](#round-3--14-november-2026-ensemble-20261114-now-the-default) for the new-site test, what was adopted and what was not, and per-camera calibration. [Round 2](#round-2--13-november-2026-ensemble-20261113) covers `ensemble-20261113` and the Gemma video test.

The screening classifier is now a **two-backbone ensemble** (SigLIP2-B/16 and
MobileCLIP2-B) with heads cross-fitted on camera-held-out folds, plus a small
YOLO26s smoking detector for CCTV frames. On the same held-out sources as the
October strategy study, macro F1 rises from **0.555 to 0.723** and macro AP from
**0.757 to 0.866**. It runs on CPU in about 8 s per five-task window (8 threads)
with 2.2 GB peak memory.

The research code is in the main checkout under `scripts/f1_study/` (see its
README), with results in `results/f1_study_20261111/`. The runtime is
`vision_runtime/ensemble.py`.

## What was wrong with the October route

| Problem | Effect | Change |
|---|---|---|
| Thresholds set on one development camera (smoking c2, unattended "entrance") | Thresholds did not transfer: unattended items fired on 99.7% of normal held-out windows; loitering never fired (F1 0.00) | Pool train+dev, split into folds that keep each camera/recording/author whole, and set the threshold on the out-of-fold scores (each fold is scored by heads that never saw that camera) |
| Heads fitted on train only | Development data unused for fitting | Final score is the mean of 15 fold heads (5 folds × 3 repeats) over train+dev |
| Fine-tuned backbone (alpha 1) trained on the training split | Its features look easier on training rows than they are; unusable for honest cross-validation | Frozen pretrained backbones only |
| Scene features carry each camera's appearance | Every camera has its own score offset | Loitering/unattended windows are compared with the camera's reference frame (first frame of the session) |
| Global 224-px embeddings cannot see a cigarette in 640×480 CCTV (head boxes are ~30×37 px) | CCTV smoking ranked barely above its 61% base rate | YOLO26s fine-tuned on the dataset's own smoking/not-smoking head boxes, cameras c4/c2 only |
| One backbone and one recipe chosen per task | Recipe choice is noisy with this little data | Average the top 3 recipes per backbone and two backbones |
| Long view assembled from a 2 fps buffer | The first frame in [t−20, t] could be 0.5 s late, so most periodic windows failed the 19.5 s history check and loitering/unattended were silently `unknown` | The long view reaches back one buffer step (≤ 0.5 s); regression test added |

## Held-out results

F1 uses each method's own threshold, chosen without test labels. "Flag
everything" is the F1 of predicting every window positive: on CCTV smoking,
new-author sleeping photos and MEVA it is already high, so F1 there says little.
AP measures ranking; FPR is the share of normal windows flagged.

| Held-out evidence | n (positive) | Flag-everything F1 | Deployed F1 | New F1 | Deployed AP | New AP | Deployed FPR | New FPR |
|---|---|---|---|---|---|---|---|---|
| Smoking: photo holdout | 224 (112) | 0.67 | 0.95 | 0.98 | 1.00 | 1.00 | 12% | 4% |
| Smoking: HMDB action clips | 906 (109) | 0.21 | 0.50 | 0.54 | 0.58 | 0.66 | 11% | 12% |
| Smoking: CCTV camera c1 | 646 (393) | 0.76 | 0.73 | 0.69 | 0.69 | 0.79 | 76% | 27% |
| Smoking: Stanford40 photos | 806 (140) | 0.30 | 0.71 | 0.80 | 0.88 | 0.95 | 16% | 10% |
| Littering: IWDD official test | 100 (50) | 0.67 | 0.83 | 0.87 | 0.81 | 0.92 | 24% | 24% |
| Loitering: Street Scene | 75 (11) | 0.26 | 0.00 | 0.64 | 0.49 | 0.75 | 0% | 12% |
| Sleeping: old photos | 18 (6) | 0.50 | 0.73 | 0.91 | 0.93 | 1.00 | 8% | 0% |
| Sleeping: new-author photos | 28 (17) | 0.76 | 0.64 | 0.64 | 0.95 | 0.92 | 0% | 0% |
| Unattended: held-out cameras | 962 (242) | 0.40 | 0.40 | 0.81 | 0.98 | 0.91 | 100% | 0% |
| Unattended: MEVA windows | 27 (15) | 0.71 | 0.68 | 0.33 | 0.55 | 0.79 | 100% | 0% |
| **Macro (sources, then tasks)** | | | **0.555** | **0.723** | **0.757** | **0.866** | | |

A paired group bootstrap (2,000 resamples) gives a macro F1 gain of +0.129,
95% interval +0.011 to +0.239. Clear gains: unattended held-out cameras +0.41,
Stanford40 photos, HMDB clips and the photo holdout. **MEVA regresses**: the old
route reached 0.68 only by flagging every window; the new route ranks MEVA better
(AP 0.55 → 0.79), but its threshold, learnt on ABODA/CAVIAR cameras, is too
strict there (27 windows).

**Smoking detector trade-off.** Cross-validation selects the detector features
(smoking out-of-fold AP 0.807 → 0.829). On held-out CCTV camera c1 it raises AP
0.72 → 0.79 and **halves false alarms (58% → 27%)**, with recall 0.77 → 0.62, so
F1 moves 0.72 → 0.69; HMDB and photos change by ≤0.03. Without it, macro F1 is
0.727 and macro AP 0.864 — the same within noise.

### What each change contributed

| Step | Macro F1 | Macro AP |
|---|---|---|
| Deployed October route (MobileCLIP2-S2, fine-tuned blends) | 0.555 | 0.757 |
| Same frozen MobileCLIP2-S2 features, October protocol | 0.580 | 0.744 |
| … with camera-held-out folds and fold-mean heads | 0.701 | 0.806 |
| Two backbones, top-3 recipes, reference-frame offsets | 0.727 | 0.864 |
| … plus the CCTV smoking detector (this release) | 0.723 | 0.866 |

The best threshold for each source would give about 0.82 macro F1; the remaining
gap is threshold transfer between very different sources (photos, movie clips,
CCTV), not the threshold rule: Youden, prevalence-balanced and bagged max-F1
rules, chosen on inner folds, were all within ±0.02.

### Measured and not adopted

- **MobileCLIP2-L/14** (304 M parameters, 213 ms/image vs ~52 ms for each chosen
  backbone): no gain in cross-validation or held-out macro F1.
- **YOLO26s boxes for every task**, removing Faster R-CNN: equal for smoking,
  littering and sleeping (adopted there), but unattended out-of-fold AP falls
  0.75 → 0.54–0.59. Faster R-CNN stays for the long-view frames of loitering and
  unattended windows, as before.
- **Stay-in-place box features alone** (no CLIP): useful as recipe parts only.

## End-to-end check on a held-out video

ABODA video9 (classroom camera family, never used for fitting; labels: no item
until 29 s, unattended item from 44 s) through the upload path on CPU:

| Window end | 20 s | 30 s | 40 s | 50–83 s |
|---|---|---|---|---|
| Label | normal | normal | (unlabelled) | unattended |
| October classifier | anomaly | anomaly | anomaly | anomaly |
| Ensemble | normal | normal | anomaly | anomaly |

Before the long-view fix, every periodic window except the final one returned
`unknown` for loitering and unattended items. The 83 s video took 73 s on 8 CPU
threads.

## Runtime

- `vision_runtime/ensemble.py` — `EnsembleStrategy`, the same `classify(candidate)`
  interface as `ClipStrategy`. It reproduces the study features (black letterbox,
  1,024-px thumbnail, each model's transform, embeddings rounded through float16
  as in the study caches) and caches per-frame results by file content.
- `vision_runtime/classifier.py` — `load_classifier(model)`; `ensemble-20261111`
  is the default for uploads (`Reviewer`, override with `VISION_CLASSIFIER`) and
  `data-pipeline/main.py --model`. The October route remains as `mobileclip2-s2`.
- `data-pipeline/.../stage1.py` — saves the first frame of each session/continuity
  epoch as the camera reference, attaches `reference_frame_path` to candidates,
  and extends the long view by one buffer step.
- `tools/ensemble_bundle.py build|verify` — assembles `models/ensemble-20261111`
  (heads, thresholds, smoking detector and the two pinned encoders, about 2.1 GB)
  with a SHA-256 manifest. `--link` symlinks the encoders for a development
  checkout; a transfer bundle needs the copies. `deploy.py smoke` uses the
  ensemble when it is installed.

CPU parity against the exported study scores (16 held-out windows per task):
standardized-score differences ≤0.0001 for loitering/unattended and ≤0.055
(mean 0.007) for the YOLO-box tasks, on a scale of about ±3. Seven real demo
windows took 42.9 s on 8 threads (October route after the earlier speed-ups:
26 s). Faster R-CNN on the long-view frames is still the largest part.

## Round 2 — 13 November 2026 (`ensemble-20261113`)

Further experiments did **not** raise benchmark F1: every variant tried lands
between 0.70 and 0.75 macro F1, inside the bootstrap noise of this benchmark
(about ±0.1). What changed is robustness to footage of *other* tasks, which the
per-task benchmark does not measure.

**Cross-task false alarms.** In deployment every head scores every window, so
a loitering head also sees littering and unattended-item footage. Applying each
`ensemble-20261111` head to the other tasks' held-out windows:

| Head | Other-task windows flagged (held out) | Worst case |
|---|---|---|
| Smoking | 2.0% | 20% of sleeping photos |
| Littering | 1.4% | 12% of loitering windows |
| Loitering | 8.1% (25.6% out-of-fold) | 30% of littering clips |
| Rough sleeping | 0.4% | 7% of littering clips |
| Unattended items | 4.0% (11.7% out-of-fold) | 6% of littering clips |

`ensemble-20261113` keeps the same encoders, recipes and CPU cost and changes
two things, both chosen on out-of-fold data only:

1. **Threshold cap.** A task's threshold is the out-of-fold max-F1 threshold,
   raised if needed so that at most 5% of other tasks' out-of-fold windows are
   flagged. It binds for unattended items (0.82 → 1.05).
2. **Other tasks' windows as negatives**, as a candidate per task (10%, 25% or
   40% of the negative weight, whole cameras still held out). Under the cap it
   is selected for loitering only (40%); the other heads are unchanged.

| | Macro F1 | Macro AP | Other-task windows flagged |
|---|---|---|---|
| `ensemble-20261111` | 0.723 | 0.866 | 3.2% |
| `ensemble-20261113` | 0.716 | 0.863 | 0.9% |

The F1 difference is the unattended-items cap: held-out-camera recall 0.66 →
0.59 at 0% false alarms (F1 0.81 → 0.74). Cap sensitivity: 10% gives macro F1
0.722 with 1.4% flagged; 2% gives 0.703 with 0.8%. Against the October route
the gain is +0.127 macro F1 (95% interval +0.004 to +0.229). On ABODA video 9
the earlier release opened a spurious loitering incident for two windows; this
one opens only the unattended-item incident. Predictions also carry a
`probability` (logistic calibration of the out-of-fold score, balanced classes)
for display. Littering positives and unattended-item positives are never used
as each other's negatives, because leaving a rubbish bag and leaving a bag look
the same.

### Measured in round 2 and not adopted

| Idea | Cross-validation | Held out | Decision |
|---|---|---|---|
| Smoking detector retrained with a second CCTV site (936 images, ~40 cameras) | Smoking out-of-fold AP 0.829 → 0.827 | CCTV AP 0.79 → 0.80 (one backbone) | No gain in cross-validation; not adopted |
| SigLIP2-B/16 at 384 px as a third backbone (2.9× the CPU time per image: 149 ms against 51 ms on 8 threads) | Out-of-fold AP: smoking 0.834 → 0.847, unattended 0.746 → 0.796 | Smoking improves (Stanford40 AP 0.95 → 0.97, HMDB 0.66 → 0.71) but loitering AP 0.75 → 0.65 and MEVA 0.79 → 0.74; macro AP 0.866 → 0.845 | Gain did not transfer and it costs CPU; not adopted |
| Other tasks' negatives for every head | Own-task out-of-fold AP falls slightly for 4 of 5 tasks | HMDB smoking AP 0.66 → 0.70–0.73; macro F1 0.73–0.74 | Kept as a per-task candidate; selected for loitering |
| Synthetic "person lying down" images for sleeping | — | — | Generated images of people lying on roads/floors, not sleeping with bedding; not used |

### Gemma: selected frames or continuous video?

827 held-out windows whose source video is on disk (HMDB smoking, IWDD
littering, ABODA and MEVA unattended items), Gemma 4 E4B, Yes/No answer margin,
same prompt. "Dense" samples the same time span evenly from the video.

| Input | Frames × tokens | Visual tokens | Seconds per window (GPU) | Macro AP | Change (95% interval) |
|---|---|---|---|---|---|
| Selected frames (current) | 4–8 × 280 | 1,797 | 0.58 | 0.772 | — |
| Dense, same budget | 8 × 280 | 2,240 | 1.03 | 0.787 | +0.015 (−0.007, +0.045) |
| Dense, same budget | 16 × 140 | 2,240 | 0.86 | 0.779 | +0.002 (−0.047, +0.052) |
| Dense, same budget | 32 × 70 | 2,240 | 0.91 | 0.782 | +0.002 (−0.059, +0.058) |
| Dense, 2× budget | 16 × 280 | 4,480 | 2.22 | 0.794 | +0.026 (−0.007, +0.073) |
| Dense, 4× budget | 32 × 280 | 8,960 | 2.99 | 0.800 | +0.026 (−0.007, +0.063) |

**Answer: not reliably.** At the same token budget the change is within noise;
five times the tokens buys about +0.03 AP with an interval that includes zero
and five times the latency. The exception is unattended items on nearby
cameras: 32 small frames reach AP 0.99 against 0.78 for 4 large ones, while the
same setting falls on distant 1080p MEVA footage (0.63 → 0.53). More frames
help when the event is "something stays"; more pixels help when the object is
small. Only E4B was available locally; the deployed 26B-A4B model was not
re-measured.

**Gemma as a veto would cost recall.** At its own Yes/No point E4B is liberal
for smoking (recall 0.98, 17% false alarms on HMDB) and very conservative
elsewhere (recall 0.04 on littering, 0.29 and 0.00 on unattended items).
Requiring agreement with the screening classifier lifts HMDB smoking F1 0.53 →
0.68 but drops littering F1 0.87 → 0.07 and held-out-camera unattended F1
0.80 → 0.24. The dashboard integration therefore opens an incident on the
screening result and attaches Gemma's verdict as a hint. Gemma ranks HMDB
smoking better than the CPU classifier (AP 0.78 against 0.67 at full-set
prevalence; 0.81 fused), so its score is useful evidence for that task.

### Gemma on the CPU machine: verdicts and descriptions (13 November, later)

The transfer bundle carries Gemma 4 E4B (Google's 4-bit QAT build, 5.7 GB) and
a CPU build of the llama.cpp server. Measured through the application's own
client on this workstation limited to 8 threads: about 8.5 GB of memory and
13–14 s per four-frame review (8.5 s to read the frames, 5 s to write the
answer). To make that affordable, Gemma reviews only windows the screening
classifier flagged, one behaviour per camera at most every 30 s of video and
never while its previous review is still running. Each answer now includes a
one- or two-sentence scene description that is stored with the incident.

Two faults found on the way:

- With the JSON grammar on, llama.cpp let Gemma reason silently first and
  spend the whole answer budget on it, so many answers were cut off and came
  back `unknown` (all 150 single-image smoking windows in the first run). The
  request now sets `thinking_budget_tokens: 0`. This affected the earlier
  confirmation path too.
- The original prompt was written for a gatekeeper ("an isolated bag does not
  establish abandonment") and made Gemma answer normal for most real
  incidents.

Three prompt wordings were compared on 341 development windows and then on 592
held-out windows (up to 40 per class and source; `gemma_confirm_eval.py`).
"Balanced" states what to look for per behaviour and asks for `unknown` only
when the scene cannot be seen; "describe first" also writes the scene summary
before the verdicts.

| Held-out source | Strict (old): P / R / F1 | Balanced | Describe first |
|---|---|---|---|
| Smoking: photo holdout | 0.97 / 0.90 / 0.94 | 0.90 / 0.93 / 0.91 | 0.86 / 0.93 / 0.89 |
| Smoking: Stanford40 | 0.80 / 0.60 / 0.69 | 0.57 / 0.78 / 0.66 | 0.63 / 0.80 / 0.70 |
| Smoking: HMDB clips | 0.88 / 0.57 / 0.70 | 0.81 / 0.72 / 0.76 | 0.81 / 0.55 / 0.66 |
| Smoking: CCTV camera c1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Littering: IWDD | 0.33 / 0.03 / 0.05 | 0.62 / 0.20 / 0.30 | 0.80 / 0.20 / 0.32 |
| Loitering: Street Scene | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Sleeping: old photos | 1.00 / 1.00 / 1.00 | 0.75 / 1.00 / 0.86 | 0.86 / 1.00 / 0.92 |
| Sleeping: new authors | 0.88 / 0.82 / 0.85 | 0.88 / 0.88 / 0.88 | 0.88 / 0.88 / 0.88 |
| Unattended: held-out cameras | 0 / 0 / 0 | 1.00 / 0.03 / 0.05 | 1.00 / 0.33 / 0.49 |
| Unattended: MEVA | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| **Macro F1** | **0.310** | **0.356** | **0.406** |

On development data describe-first won overall (0.393, balanced 0.386, strict
0.320) but littering did better with the verdict first (recall 0.24 against
0.04), so the runtime uses describe-first for the long-view tasks and
balanced for littering's short view. Held out, the two were equal for
littering.

How to read Gemma's verdict, then: for smoking in close view and for sleeping
it is a useful second opinion in both directions. For unattended items and
littering, a confirmation is trustworthy (precision 0.8–1.0) but "not
confirmed" mostly means it did not see the event. For loitering, smoking at
CCTV distance and small distant objects it never confirms, so "not confirmed"
carries no information there. The description is useful to the reviewer in
every case.

## Round 3 — 14 November 2026 (`ensemble-20261114`, now the default)

A field test of `ensemble-20261113` on a new camera flagged a man standing by
a car as littering and missed that he was smoking. Round 3 therefore measures
what the benchmark could not, **behaviour on sites no head has seen**, and
keeps only changes that hold up there.

| | Released (`20261113`) | `20261114`, balanced (default) | `20261114`, low false alarm |
|---|---|---|---|
| Benchmark macro F1 | 0.716 | **0.732** | 0.694 |
| Benchmark macro AP | 0.863 | 0.861 | 0.861 |
| Other behaviours' held-out windows flagged | 0.9% | **0.2%** | 0.2% |
| Ordinary footage flagged, cameras outside the fit | 8.6% | **0.6%** | 0.3% |
| Smokers found, unseen cameras of a second CCTV estate | 37% (with 38–41% of ordinary walkway and MEVA frames flagged) | 43% (with 0% flagged) | 43% |

Paired bootstraps against the released model: ordinary footage flagged −8.0
points (95% interval −10.6 to −4.3, cameras resampled); benchmark macro F1
+0.02 (−0.01 to +0.10), which is not a reliable gain on its own. CPU cost is
unchanged (no new encoder passes; the 83 s test video still takes 72 s on 8
threads).

**It does not fix the field case.** The frame from that report still scores
above the littering threshold with the new heads (margin +0.94; with extra
negatives +0.55 to +0.79) and smoking is still not seen. The littering head
knows two sites, and a person standing on a paved area resembles their
positives. The remedy for a camera-specific false alarm is the per-camera
calibration below, or labelled footage from the site.

### The new-site test

`round3_sources.py` builds it from footage on disk that no head had used.
Whole cameras are assigned to a **fit** side (may be used for fitting and for
thresholds) or a **shift** side (only ever scored).

| Source | Used as | Fit side | Shift side |
|---|---|---|---|
| Second CCTV estate, alert snapshots (every image has a smoker) | Smoking positives | 492 images, 206 camera clusters | 444 images, 199 clusters |
| Campus littering clips (30-frame clips rebuilt from a frame dump) | Littering positives | 96 clips | 138 clips from other views |
| VIRAT car parks | Ordinary windows, all five heads | 191 windows, 1 scene | 96 windows, 2 scenes |
| MEVA cameras | Ordinary windows: smoking, sleeping, littering | 174 windows, 3 cameras | 168 windows, 4 cameras |
| Avenue walkway (normal training videos) | Ordinary windows, all five heads | — | 41 windows |
| XRF55 indoor clips: phoning, drinking, eating | Hand-to-face negatives for smoking | 375 clips, 17 subjects | 180 clips, 3 subjects |

Ordinary windows follow the runtime (frame at *t*; eight frames over 7 s;
four frames over 20 s plus the first frame as reference), one every 10 s.
XRF55's own "smoking" clips are mimed, with no cigarette in view, and are not
used as positives. VIRAT has no labels here: it counts as ordinary for
unattended items after looking at every window the released head flagged
(groups standing, traffic, a patio; no left item), and stays unlabelled for
loitering and littering.

**Baseline.** The released heads flag 41% of Avenue and 38% of MEVA frames as
smoking, find none of the campus littering clips, and flag 12% of VIRAT
windows as unattended items. Thresholds and heads fitted on a handful of
sources do not carry over, which is what the field test showed.

### What was tried

| Idea | Result | Decision |
|---|---|---|
| **Zero-shot text member**: similarity of the SigLIP2 frame (and crop) embedding to descriptions of the behaviour minus descriptions of ordinary scenes. Nothing is fitted; the text vectors are stored in the bundle, so there is no run-time cost | Alone, smoking AP on film clips 0.81–0.84 (fitted ensemble 0.66) and sleeping AP 0.95/1.00; useless on distant CCTV smoking (0.62 at a 0.61 base rate), loitering and unattended items | **Adopted** for smoking's view channel and for sleeping |
| …for littering | Trained on one littering site and tested on the other, it lifts the ranking (campus against ordinary footage AUC 0.80 → 0.95) but not recall at the chosen threshold (0% → 5%), and lowers IWDD AP 0.92 → 0.87 | Not adopted |
| **Ordinary footage as negatives** (VIRAT, MEVA, XRF55 fit side) and a cap on how much of it a threshold may flag | Smoking: Avenue 41% → 0%, MEVA 38% → 0% flagged. Unattended items: VIRAT 12% → 4%, held-out camera AP 0.908 → 0.919 | **Adopted** |
| **A positive-only site in the heads** (second-estate smokers) | Shift recall 97%, but the head stops finding smokers on the first site (camera c1 F1 0.69 → 0.01): it learnt the site, not the cigarette | Rejected; positive-only sources go to the detector, with background frames |
| **Smoking detector retrained**: first site + second-estate fit cameras + ordinary frames as label-free background | Without background it flags 17–26% of ordinary frames; with it, at confidence 0.4: 41% of shift smokers, 0% of Avenue/MEVA/VIRAT frames, but 17% of XRF55 hand-to-face clips | Superseded by the next row |
| …plus **hand-to-face clips as hard negatives** (`allg`) | Same recall (42%), XRF55 17% → 4% | **Adopted** as `smoking_detector.pt` |
| **Does the detector transfer between CCTV sites?** | Trained on site 1 only: 1–3% of site-2 smokers. Trained on site 2 only: camera c1 AP 0.58 at a 0.61 base rate. It does not, in either direction | A detector needs footage of the site it watches |
| **Smoking as two channels** instead of one averaged score: views (heads without detector features + zero-shot) and detector, each with its own threshold, flagged when either passes | With the old detector already: Stanford40 F1 0.80 → 0.88, film clips 0.54 → 0.64, false alarms 10–12% → 0.3–2%. Averaging had let the photo-trained members outvote the detector on CCTV | **Adopted** |
| Detector input 960 or 1280 px | No gain (site-2 recall 1–3% for the old detector at every size) | Not adopted |
| Ignore detector boxes taller than 20% of the frame | Changes under 1% of images: its boxes are small even on close-ups | Not adopted |
| **Person-level head**: one SigLIP2 embedding per detected person, highest person score | Cross-camera CCTV AP 0.51 at a 0.39 base rate; as a member, film clips AP 0.66 → 0.70 (zero-shot gives 0.78 for free) | Not adopted |
| Littering trained on one site, tested on the other | IWDD → campus: 0% found. Campus → IWDD: AP 0.51 (chance). With both: every held-out campus view found, IWDD F1 0.87 → 0.89, false alarms 24% → 10% | Both sites are in the fit; expect a third site to need its own footage |
| Other behaviours' footage as littering negatives (10–40%) | Fitting-side F1 0.848 → 0.866, cross-task 0.7% → 0.2%, field frame margin +0.94 → +0.55..+0.79 (still flagged) | Not in this release; small gain, measured after the selection |

### Held-out results

F1 and false alarms at each model's own thresholds; thresholds use fitting-side
scores only.

| Held-out evidence | n (positive) | Released: F1 / false alarms | New, balanced | New, low false alarm | Released AP | New AP |
|---|---|---|---|---|---|---|
| Smoking: photo holdout | 224 (112) | 0.98 / 4% | 0.97 / 4% | 0.97 / 4% | 1.00 | 1.00 |
| Smoking: HMDB action clips | 906 (109) | 0.54 / 12% | 0.55 / 3% | 0.55 / 3% | 0.66 | 0.56 |
| Smoking: CCTV camera c1 | 646 (393) | 0.69 / 27% | 0.53 / 9% | 0.53 / 9% | 0.79 | 0.81 |
| Smoking: Stanford40 photos | 806 (140) | 0.80 / 10% | 0.86 / 3% | 0.86 / 3% | 0.95 | 0.94 |
| Littering: IWDD official test | 100 (50) | 0.87 / 24% | 0.89 / 10% | 0.89 / 10% | 0.92 | 0.92 |
| Loitering: Street Scene | 75 (11) | 0.64 / 12% | 0.64 / 12% | 0.64 / 12% | 0.74 | 0.74 |
| Sleeping: old photos | 18 (6) | 0.91 / 0% | 1.00 / 0% | 1.00 / 0% | 1.00 | 1.00 |
| Sleeping: new-author photos | 28 (17) | 0.64 / 0% | 0.74 / 0% | 0.74 / 0% | 0.92 | 0.92 |
| Unattended: held-out cameras | 962 (242) | 0.74 / 0% | 0.73 / 0% | 0.68 / 0% | 0.91 | 0.92 |
| Unattended: MEVA windows | 27 (15) | 0.33 / 0% | 0.33 / 0% | 0.00 / 0% | 0.79 | 0.79 |
| **Macro** | | **0.716** | **0.732** | **0.694** | **0.863** | **0.861** |

Two results go the wrong way and are kept visible. **CCTV camera c1** F1 falls
0.69 → 0.53 (−0.16, interval −0.22 to −0.11): the released threshold there
flagged 27% of non-smokers, close to flagging everything (which scores 0.76);
the new one flags 9% and finds 38% of smokers instead of 62%. **Film-clip
ranking** falls (AP 0.66 → 0.56) because the detector fires at chance on
close-up film frames; F1 there is unchanged and false alarms drop 12% → 3%.

| New-site evidence (shift side) | Windows | Released | New, balanced | New, low false alarm |
|---|---|---|---|---|
| Smoking: second CCTV estate, unseen cameras — smokers found | 444 | 37% | 43% | 43% |
| Smoking: indoor phoning/drinking/eating clips — flagged | 180 | 0% | 3% | 3% |
| Smoking: Avenue walkway — flagged | 41 | 41% | 0% | 0% |
| Smoking: MEVA cameras — flagged | 168 | 38% | 0% | 0% |
| Smoking: VIRAT car parks — flagged | 96 | 0% | 0% | 0% |
| Littering: campus clips, unseen views — found | 138 | 0% | 100% | 100% |
| Littering: Avenue, MEVA, VIRAT — flagged | 305 | 0% | 0% | 0% |
| Loitering: Avenue walkway — flagged | 41 | 0% | 0% | 0% |
| Loitering: VIRAT car parks (unlabelled) — flagged | 96 | 9% | 9% | 9% |
| Sleeping: Avenue, MEVA, VIRAT — flagged | 305 | 0% | 0% | 0% |
| Unattended: Avenue walkway — flagged | 41 | 2% | 0% | 0% |
| Unattended: VIRAT car parks — flagged | 96 | 12% | 4% | 1% |

Read the two "found" rows with care. The campus clips are other views of a
campus whose remaining clips were fitted, so 100% is a same-site result. The
second estate's shift cameras are unseen, but come from the same operator as
the fit cameras; a third, unrelated site should expect the transfer result
(1–3%) until it contributes footage. Loitering and VIRAT's nine loitering
flags are unchanged: no new loitering evidence was available.

### What is in `ensemble-20261114`

- **Smoking**: two channels. *Scene and person views*: the released recipes
  without detector features (SigLIP2-B and MobileCLIP2-B) plus a zero-shot
  text member, fitted with ordinary and hand-to-face footage as negatives.
  *Smoking detector*: the retrained YOLO26s head detector. The window is
  flagged when either passes its own threshold; the prediction carries
  `evidence_channel` and the incident description names it.
- **Rough sleeping**: released recipes plus a zero-shot text member, with
  ordinary footage as negatives.
- **Littering**: released recipes, fitted on IWDD and the campus clips with
  ordinary footage as negatives.
- **Unattended items**: released recipes, with VIRAT windows as negatives.
- **Loitering**: unchanged.
- **Thresholds**: out-of-fold max-F1 over the fitting sources (single-class
  sources pooled into one), raised until at most 5% of other behaviours'
  windows are flagged, then until at most a share of the fit-side ordinary
  footage is flagged: 5% (`balanced`, default) or 2% (`low_false_alarm`).
  `VISION_OPERATING_POINT` selects one; only unattended items differs between
  them in this release (recall on held-out cameras 57% against 51%, VIRAT
  flagged 4% against 1%). A channel only counts fitting sources it can
  separate (AUC ≥ 0.75), and the detector channel may flag none of the
  fit-side ordinary footage.

Runtime parity on CPU against the exported study scores (24 held-out windows
per task): largest difference 0.06 for smoking, 0.10 littering, 0.0001
loitering, 0.006 unattended items on a scale of about ±3; one sleeping window
differs by 0.30 where YOLO's person box differs between GPU and CPU (mean
0.02). The detector now pads every frame to a square: with the default, a
frame's confidence depended on whether its batch neighbours had the same size
(up to 0.14 on off-domain images).

### Calibrating a camera on its own ordinary footage

`python3.12 deploy.py calibrate --source-id camera_01` reads the scores the
camera backend already journaled and raises that camera's threshold, per
behaviour, until at most 2% of those windows would be flagged
(`vision_runtime/site.py`, `tools/calibrate_site.py`). It needs no labels,
only footage in which the behaviour did not happen; thresholds are never
lowered, and a calibration applies only to the model and operating point it
was measured with. `--clear` removes it.

Simulated on every held-out fixed camera with at least 40 ordinary windows
(half of them as calibration footage, the rest and all positives as the
check):

| Model | Ordinary windows flagged (16 task–camera pairs) | Recall (3 pairs with positives) |
|---|---|---|
| Released | 6.8% → 1.1% | 42% → 25% |
| `20261114` | 0.8% → 0.3% | 33% → 26% |

It removes what remains of a camera's own false alarms and costs recall where
a head is weak (CCTV camera c1: non-smokers flagged 10% → 2%, smokers found
38% → 16%). Through the real upload path, a VIRAT clip with two false
unattended-item flags in 30 windows calibrated to none (threshold 1.70 →
1.77).

### Limits of round 3

- The detector channel's calibration rule (no fit-side ordinary window
  flagged) was introduced after the shift test showed the first rule flagging
  30% of unseen MEVA frames, so the shift numbers for that channel are not an
  untouched confirmation.
- Selection between variants used fitting-side F1 under the ordinary-footage
  cap, but several designs were first compared on held-out sources during the
  round. Treat the benchmark gain as small and the false-alarm reduction as
  the result.
- Positive-only and negative-only sources come from different sites, so
  "found" and "flagged" percentages cannot be combined into one precision.
- The new-site ordinary footage has distant people. The field frame, with a
  person at mid distance, behaves differently from all of it.
- Gemma was not changed in this round. Asking it about every behaviour on a
  flagged window, so that it can relabel an incident, was not built.

## Limits

- The held-out sets were inspected during the October work and during this
  study. The backbone pair was chosen on CPU cost once cross-validation could not
  separate it from larger ensembles. Treat results as retrospective.
- Several sources are tiny (loitering: 11 positives, nine from one recording;
  sleeping: 6 and 17 positives; MEVA: 27 windows); sleeping and unattended labels
  are provisional assistant review. The smoking detector has seen only two
  cameras of one site.
- The live reference frame is the first frame of a session. A long-running camera
  whose lighting changes will drift from it; a refresh rule (for example, when no
  person or bag has been detected for several minutes) is not implemented.
- Scores remain scene/window evidence for review, not verified alerts.
