#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_library_paths="$task_root/data/models/nvidia_runtime/extracted"
for task_lib in "$task_root"/.venv-gpu/lib/python*/site-packages/nvidia/*/lib; do
    task_library_paths="$task_library_paths:$task_lib"
done
export LD_LIBRARY_PATH="$task_library_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$task_root/data/tools/llama_gemma4/b11925/llama-b11925/llama-server" "$@"
