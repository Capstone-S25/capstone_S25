#!/usr/bin/env bash
# ensemble-20261119b: ensemble-20261119 (round5_fit.sh) with its unattended
# head refitted. With every ABODA camera that has a test video held out, the
# 20261119 head had two fit cameras left (CAVIAR, ABODA lobby) and fired on
# nothing. common.py now also fits the split's fit-side D2-CDNet scene "sofa".
# F1_FIT_ENTRANCE=1 additionally fits ABODA's entrance camera (videos 5-6),
# which is only leak-free if video5 leaves the test side of the team split.
# Exports all three operating points; the export stops if any threshold would
# flag under 5% of its own fitting positives.
# Run from scripts/f1_study after round5_fit.sh.
set -euo pipefail
export LD_LIBRARY_PATH=$PWD/../../data/models/nvidia_runtime/extracted
export F1_ROUND3=1 F1_ROUND4=1 F1_SPLIT=ian
export F1_THRESHOLD_WITHOUT="icvl_smoking icvl_littering icvl_loitering"
P=../../.venv-gpu/bin/python
ICVL="icvl_smoking icvl_littering icvl_loitering"
$P round3.py --tag r6u_sofa --tasks unattended_items --no-other-site-negatives --site-fit --without smoking_site2 $ICVL --variant nodet
F1_MIX_BASE=r5_mixSL $P round4_mix.py r6_mix r6u_sofa unattended_items
$P export_round3.py --run r6_mix --smoking-channels r5_detg_icvl --output ../../results/f1_study_20261111/bundle_v13
echo FIT_DONE
