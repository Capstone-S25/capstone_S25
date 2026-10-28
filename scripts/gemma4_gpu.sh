#!/usr/bin/env bash
# Separate Transformers runtime; existing experiments retain their pinned libraries.
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$task_root/data/models/nvidia_runtime/extracted${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$task_root${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
# Preserve flag-only evaluation commands; help describes the unified dispatcher.
if [[ $# -eq 0 ]]; then set -- --help; fi
if [[ "$1" == --* && "$1" != --help ]]; then set -- evaluate "$@"; fi
exec "$task_root/.venv-gemma4/bin/python" -m anomaly_eval.gemma4 "$@"
