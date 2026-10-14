#!/usr/bin/env bash
# Use a separate CUDA environment and, when needed, matching local driver libraries.
set -euo pipefail

task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${CAPSTONE_GPU_PYTHON:-$task_root/.venv-gpu/bin/python}"
task_library_dir="${CAPSTONE_GPU_DRIVER_LIBS:-$task_root/data/models/nvidia_runtime/extracted}"

if [[ ! -x "$task_python" ]]; then
    echo "CUDA Python environment missing: $task_python" >&2
    echo "Create .venv-gpu with CUDA-enabled torch/torchvision and the project dependencies." >&2
    exit 1
fi

# A host driver update can make the old extracted libraries inappropriate.
# Select them only when their version matches the currently loaded kernel module.
task_driver_version=""
if [[ -r /proc/driver/nvidia/version ]]; then
    task_driver_version="$(sed -n 's/^NVRM version:.*  \([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\)  .*/\1/p' /proc/driver/nvidia/version)"
fi
if [[ -n "$task_driver_version" && -f "$task_library_dir/libcuda.so.$task_driver_version" ]]; then
    export LD_LIBRARY_PATH="$task_library_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$task_root${PYTHONPATH:+:$PYTHONPATH}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$task_root/data/models/hf_modules}"

if [[ $# -eq 0 ]]; then
    set -- run --device cuda
fi
exec "$task_python" -m anomaly_eval "$@"
