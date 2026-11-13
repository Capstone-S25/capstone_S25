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
