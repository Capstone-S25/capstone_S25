#!/usr/bin/env bash
# Round 5: the round-4 release recipe refitted on the team's fit / test split
# (F1_SPLIT=ian, notebooks/exp/eval/splits.json on feat/vision-app-integration).
# ICVL cameras 02 and 10 and every ABODA camera with a test video are never fitted.
# Same recipe, same thresholds rule as ensemble-20261116; only the rows move.
# Run from scripts/f1_study after round4_features.sh.
set -euo pipefail
export LD_LIBRARY_PATH=$PWD/../../data/models/nvidia_runtime/extracted
export F1_ROUND3=1 F1_ROUND4=1 F1_SPLIT=ian
export F1_THRESHOLD_WITHOUT="icvl_smoking icvl_littering icvl_loitering"
P=../../.venv-gpu/bin/python
ICVL="icvl_smoking icvl_littering icvl_loitering"
EXTRA="smoking=zeroshot:full+crop rough_sleeping=zeroshot:full"
$P round3.py --tag r5_base --no-other-site-negatives --site-fit --without smoking_site2 $ICVL --variant nodet --extra $EXTRA
$P round3.py --tag r5_icvl --no-other-site-negatives --site-fit --without smoking_site2 --variant nodet --extra $EXTRA
$P round3.py --tag r5_detg_icvl --site-fit --tasks smoking --no-fitted smoking --variant nodet --extra smoking=detector:g --no-other-site-negatives
F1_MIX_BASE=r5_base $P round4_mix.py r5_mixSL r5_icvl smoking littering
$P round3_report.py "r5_mixSL|r5_detg_icvl@n05d0" "r5_mixSL|r5_detg_icvl@n02d0"
echo FIT_DONE
