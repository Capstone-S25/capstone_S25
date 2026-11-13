# November F1 study and the ensemble classifier — 11 November 2026

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
