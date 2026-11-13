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
