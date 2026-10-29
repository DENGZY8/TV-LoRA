#!/usr/bin/env bash
set -euo pipefail
log="$1"
gpu="$2"
python_bin="$3"
config="$4"
shift 4
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu"
export MPLBACKEND=Agg
set +e
"$python_bin" "$repo/run.py" --config "$config" --device cuda:0 "$@" 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
set -e
printf '\nEXIT_CODE=%s\n' "$status" | tee -a "$log"
exit "$status"
