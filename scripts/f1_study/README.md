# November F1 study (11 November 2026)

Scripts behind the ensemble classifier in `feat/vision-app-integration`
(`docs/f1_study.md` there has the findings). Outputs go to
`results/f1_study_20261111/` (ignored). Run from this folder with the CUDA
environment and the matching driver libraries:

```bash
export LD_LIBRARY_PATH=$PWD/../../data/models/nvidia_runtime/extracted
P=../../.venv-gpu/bin/python
$P baseline.py                                    # October routes, same harness
$P extract.py --model siglip:google/siglip2-base-patch16-224 --name siglip2-b
$P extract.py --model open_clip:timm/MobileCLIP2-B-OpenCLIP --name mobileclip2-b
$P frcnn_references.py                            # boxes for reference frames
../../.venv-stage1/bin/python yolo_boxes.py       # tracker-detector boxes
../../.venv-stage1/bin/python smoking_detector.py --fold c4   # also c2, all
../../.venv-stage1/bin/python smoking_scores.py
$P run_heads.py --backbones siglip2-b --tag v2_siglip2-b --frozen-only --protocols november
$P ensemble.py --members v8_siglip2-b v8_mobileclip2-b --top 3 --tag e8_sb_top3
$P export.py --members v8_siglip2-b v8_mobileclip2-b --top 3 --smoke-detector --output ../../results/f1_study_20261111/bundle_v8
$P bootstrap.py baseline:x e8_sb_top3:november
```

The released members merge runs per task: smoking from `v7_*` (YOLO boxes plus
detector), littering and sleeping from `v5y_*` (YOLO boxes), loitering and
unattended items from `v3_*` (both box sources, reference offsets).

| File | Purpose |
|---|---|
| `common.py` | Rows from the October manifests, held-out sources, F1/AP metrics, reference frames |
| `representations.py` | Feature caches and recipes (pool, crop, geometry, stay-in-place, reference, offset, smoking detector) |
| `heads.py` | October protocol and the camera-held-out cross-fitted protocol |
| `run_heads.py`, `ensemble.py`, `compare.py` | Fit, combine and compare |
| `threshold_rules.py` | Threshold rules chosen on inner folds |
| `bootstrap.py` | Paired group bootstrap |
| `export.py` | Runtime bundle with parity references |

`smoking_detector.py` keeps camera c1 (the test camera) out of every detector.
The `alpha 1` October features are excluded from cross-validation because that
backbone was fine-tuned on the training split.

## Round 2 (13 November 2026) — `ensemble-20261113`

```bash
../../.venv-stage1/bin/python smoking_scores.py --other-tasks      # detector on other tasks' images
$P round2.py --tag r2c_base  --aux-share 0                        # released recipes + cross-task false alarms
$P round2.py --tag r2c_aux10 --aux-share 0.1                      # other tasks' windows as negatives (also 0.25, 0.4)
$P select_round2.py --runs r2c_base r2c_aux10 r2c_aux25 r2c_aux40 --cap 0.05 --tag s2_cap05
$P export_round2.py --selection s2_cap05 --shares r2c_base=0 r2c_aux10=0.1 r2c_aux25=0.25 r2c_aux40=0.4 \
   --output ../../results/f1_study_20261111/bundle_v9
$P bootstrap.py e8_sb_top3:november s2_cap05:november

# Gemma: selected frames or denser video (needs .venv-gemma4 and the E4B weights)
../../.venv-stage1/bin/python gemma_video_frames.py
../../.venv-gemma4/bin/python gemma_video.py
$P gemma_video_report.py
$P cascade.py --run r2c_base                                      # screening, Gemma, agreement
```

| File | Purpose |
|---|---|
| `round2.py` | Refit the released recipes, optionally with other tasks' windows as negatives; measure cross-task false alarms |
| `select_round2.py` | Per-task choice on out-of-fold F1 under a cross-task false-alarm cap |
| `export_round2.py` | Runtime bundle for that choice, with the capped thresholds and a probability calibration |
| `gemma_video_frames.py`, `gemma_video.py`, `gemma_video_report.py` | Selected frames against evenly sampled video for Gemma 4 E4B |
| `cascade.py` | Screening classifier, Gemma, and requiring both, on the same windows |

Measured and not adopted (see `docs/f1_study.md` on the integration branch):
`smoking_detector.py --extra` (second CCTV site) with `--detector x`, and a
384 px SigLIP2 member (`extract.py --model siglip:google/siglip2-base-patch16-384
--name siglip2-b384`, then `round2.py --members ... v9_siglip2-b384`).

## Round 3 (14 November 2026) — `ensemble-20261114`

Round 3 adds a **new-site test**: footage from cameras and sites no head had
seen (`round3_sources.py`), split by camera into a fitting side and a shift
side that is only ever scored. Set `F1_ROUND3=1` (the round-3 scripts do) and
`load_rows()` also returns these rows with split `site_fit` / `site_shift`.

```bash
S1=../../.venv-stage1/bin/python
$S1 round3_sources.py                      # rows + frames of the new-site footage
F1_ROUND3=1 $S1 yolo_boxes.py              # boxes, features, detector scores for them
$P round3_boxes.py
F1_ROUND3=1 $P extract.py --model siglip:google/siglip2-base-patch16-224 --name siglip2-b
F1_ROUND3=1 $P extract.py --model open_clip:timm/MobileCLIP2-B-OpenCLIP --name mobileclip2-b
$P zero_shot.py                            # text-described scores, nothing fitted
$S1 smoking_detector.py --fold all --site-fit --hard-negatives --epochs 40   # also c4, c2 (halves)
F1_ROUND3=1 $S1 smoking_scores.py --suffix g
$P round3.py --tag r3_release              # the round-2 release, scored on the new sites
$P round3.py --tag r3_final --site-fit --without smoking_site2 --variant nodet \
   --extra smoking=zeroshot:full+crop rough_sleeping=zeroshot:full
$P round3.py --tag r3_detg --site-fit --tasks smoking --no-fitted smoking --extra smoking=detector:g
$P round3_report.py r3_release "r3_final|r3_detg@n05d0" "r3_final|r3_detg@n02d0"
$P round3_bootstrap.py r3_release "r3_final+r3_detg_n05d0"
$P export_round3.py --run r3_final --smoking-channels r3_detg --output ../../results/f1_study_20261111/bundle_v10
```

| File | Purpose |
|---|---|
| `round3_sources.py` | New-site rows: second smoking site, campus littering clips, ordinary windows from VIRAT, Avenue and MEVA, hand-to-face negatives from XRF55 |
| `round3_boxes.py` | Faster R-CNN boxes for the new loitering/unattended windows |
| `zero_shot.py` | SigLIP2 text scores per row and task (behaviour descriptions minus ordinary-scene descriptions) |
| `persons.py`, `person_head.py` | One embedding per detected person and a person-level smoking head (measured, not adopted) |
| `round3.py`, `round3_recipes.py` | Fit with or without fit-side new-site rows, extra members (zero-shot, detector, persons), leave-a-source-out |
| `round3_report.py` | Thresholds (fitting side only), benchmark table, new-site table, channel fusion for smoking |
| `round3_bootstrap.py` | Paired camera bootstrap of new-site recall and ordinary footage flagged |
| `round3_detector_scale.py` | The smoking detector on new-site frames at several input sizes |
| `round3_site_calibration.py` | Simulates calibrating each held-out camera on half of its own ordinary windows |
| `round3_tables.py` | Markdown tables for the study document from the report outputs |
| `export_round3.py` | Runtime bundle with two operating points |

`smoking_detector.py --site-fit` adds the second site's fit-side smokers and
ordinary frames as label-free background; `--hard-negatives` (suffix `g`, the
released detector) also adds phoning/drinking/eating clips. The c4 and c2 fold
detectors take half of those rows each (`common.detector_half`), whole cameras
per half for `g`, so that every fit-side row has a detector that never saw
its camera. The first family (`f`) split by recording and is kept for the
comparison only; `--no-site1` trains without site 1 for the transfer check.

## Round 4 (15 November 2026) — the Drive's new labelled videos

The Drive gained curated video sets with per-video JSON labels (event times,
per-frame boxes): ICVL littering (503 videos, 4 cameras), smoking (43, 9
cameras) and loitering (75, 10 cameras), IITB Corridor loitering (8), ABODA
(10) and D2-CDNet (4) unattended items, and CHAD lying-on-the-ground clips (4).
The Roboflow littering clips are the existing `littering/Littering` images
rebuilt as video, so only their labels were fetched. Older sets moved to
"Reject" folders on the Drive and were already on disk.

```bash
.venv/bin/python scripts/datasets/fetch_drive_videos.py --list-only   # what is new
.venv/bin/python scripts/datasets/fetch_drive_videos.py --workers 4 --min-free-gb 1.5
# from scripts/f1_study:
PYTHONPATH=../../.local/worktrees/vision-app-integration \
VISION_MODELS_DIR=$PWD/../../dist/capstone-cpu-deployment-20261114/models \
  ../../.venv-stage1/bin/python round4_drive.py     # released ensemble on every new window
../../.venv-stage1/bin/python round4_report.py      # per set: AUC, AP, recall, false alarms
bash round4_features.sh                             # rows + cached features (incremental)
bash round4_fit.sh                                  # r4_base (= release) and r4_icvl
export F1_ROUND3=1 F1_ROUND4=1 F1_THRESHOLD_WITHOUT="icvl_smoking icvl_littering icvl_loitering"
$P round3_report.py "r4_mixS|r4_detg_icvl@n05d0"    # r4_mixS: smoking heads from r4_icvl
$P bootstrap.py "r4_base+r4_detg_base_n05d0:november" "r4_mixS+r4_detg_icvl_n05d0:november"
$P round3_bootstrap.py "r4_base+r4_detg_base_n05d0" "r4_mixS+r4_detg_icvl_n05d0"
$P round4_compare.py "r4_base|r4_detg_base@n05d0" "r4_mixS|r4_detg_icvl@n05d0"
$P round4_mix.py r4_mixS r4_icvl smoking            # also r4_mixSL: smoking littering
$P export_round3.py --run r4_mixSL --smoking-channels r4_detg_icvl \
   --output ../../results/f1_study_20261111/bundle_v11   # ensemble-20261116
```

| File | Purpose |
|---|---|
| `round4_mix.py` | Takes named tasks from one run and the rest from `r4_base` (`r4_mixS`, `r4_mixSL`) |
| `../datasets/fetch_drive_videos.py` | Lists the Drive (unexportable Google Docs are reported, not fatal), skips files on disk, downloads with curl, re-encodes videos to 640 px at their own frame rate |
| `round4_drive.py` | Runtime windows (2 fps buffer, 4 s / 20 s views, reference frame) labelled from each set's JSON, scored by the released `EnsembleStrategy` |
| `round4_report.py` | Released model per set, plus a threshold-only refit on the fitting-side ICVL cameras |
| `round4_sources.py` | The same windows as study rows; ICVL cameras 02, 05, 08, 11 are the shift side for every behaviour |
| `round4_features.sh`, `round4_fit.sh` | Boxes, embeddings, zero-shot and detector scores; the fits |
| `round4_compare.py` | Held-out new sources: AUC with camera intervals, recall, negatives flagged, precision |

`round3.py --no-other-site-negatives` reproduces the release: the other-site
negatives were added to `round3.py` after `r3_final` was fitted, and `r4_base`
matches the released benchmark scores to 1e-7 only with this flag.
`F1_THRESHOLD_WITHOUT` leaves sources out of threshold selection while they
are fitted: the ICVL sets are 60-85% positive, and in the macro-source F1
rule they pulled every threshold down.
