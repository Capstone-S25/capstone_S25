#!/usr/bin/env bash
# Isolated PEFT dependencies and the matching local CUDA driver libraries.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${CAPSTONE_GPU_PYTHON:-$task_root/.venv-gpu/bin/python}"
task_library_dir="${CAPSTONE_GPU_DRIVER_LIBS:-$task_root/data/models/nvidia_runtime/extracted}"
task_driver_version=""
if [[ -r /proc/driver/nvidia/version ]]; then
    task_driver_version="$(sed -n 's/^NVRM version:.*  \([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)  .*/\1/p' /proc/driver/nvidia/version)"
fi
if [[ -n "$task_driver_version" && -f "$task_library_dir/libcuda.so.$task_driver_version" ]]; then
    export LD_LIBRARY_PATH="$task_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$task_root/data/tools/smol_training_deps:$task_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
case "${1:-train}" in
    train) task_script="run_smolvlm_finetuning.py" ;;
    predict) task_script="predict_smolvlm.py" ;;
    verify) task_script="verify_smolvlm_finetuning.py" ;;
    transfer) task_script="evaluate_smolvlm_transfer.py" ;;
    report) task_script="report_smolvlm_finetuning.py" ;;
    export) task_script="export_smolvlm_adapters.py" ;;
    repeated-gains) task_script="select_smolvlm_repeated_gains.py" ;;
    *) echo "Usage: $0 {train|predict|verify|transfer|report|export|repeated-gains} [arguments]" >&2; exit 2 ;;
esac
if [[ $# -gt 0 ]]; then shift; fi
exec "$task_python" -m "scripts.smolvlm.${task_script%.py}" "$@"
