#!/usr/bin/env bash
# Round 4: rows and every cached feature the fitting needs, for the Drive's new
# videos. Each step only processes frames it has not cached, so it can be rerun
# as more videos are scored. Run from scripts/f1_study.
set -euo pipefail
export LD_LIBRARY_PATH=$PWD/../../data/models/nvidia_runtime/extracted
export F1_ROUND3=1 F1_ROUND4=1
S1=../../.venv-stage1/bin/python
P=../../.venv-gpu/bin/python
$S1 round4_sources.py
$S1 yolo_boxes.py
$P round3_boxes.py
$P extract.py --model siglip:google/siglip2-base-patch16-224 --name siglip2-b
$P extract.py --model open_clip:timm/MobileCLIP2-B-OpenCLIP --name mobileclip2-b
$P zero_shot.py
$S1 smoking_scores.py --suffix g
echo FEATURES_DONE
