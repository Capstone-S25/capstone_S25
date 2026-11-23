#!/usr/bin/env bash
# Round 4: the released round-3 recipe with and without the fitting-side ICVL
# cameras, scored on the held-out ICVL cameras, the other new sets, the
# benchmark and round 3's new sites. Run after round4_features.sh.
set -euo pipefail
export LD_LIBRARY_PATH=$PWD/../../data/models/nvidia_runtime/extracted
export F1_ROUND3=1 F1_ROUND4=1
P=../../.venv-gpu/bin/python
ICVL="icvl_smoking icvl_littering icvl_loitering"
EXTRA="smoking=zeroshot:full+crop rough_sleeping=zeroshot:full"
# r4_base fits exactly the rows r3_final fitted, so it must reproduce the release.
$P round3.py --tag r4_base --no-other-site-negatives --site-fit --without smoking_site2 $ICVL --variant nodet --extra $EXTRA
$P round3.py --tag r4_icvl --no-other-site-negatives --site-fit --without smoking_site2 --variant nodet --extra $EXTRA
$P round3.py --tag r4_detg_base --site-fit --tasks smoking --no-fitted smoking --variant nodet --extra smoking=detector:g --without icvl_smoking --no-other-site-negatives
$P round3.py --tag r4_detg_icvl --site-fit --tasks smoking --no-fitted smoking --variant nodet --extra smoking=detector:g --no-other-site-negatives
for op in n05d0 n02d0; do
  $P round3_report.py "r4_base|r4_detg_base@$op" "r4_icvl|r4_detg_icvl@$op"
done
echo FIT_DONE
