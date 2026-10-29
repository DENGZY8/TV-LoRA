#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: bash scripts/tmux_run.sh CONFIG GPU [run.py options]" >&2
  exit 2
fi
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="$1"
gpu="$2"
shift 2
python_bin="$(command -v python)"
session="tvlora_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$repo/logs"
log="$repo/logs/$session.log"
printf -v command '%q ' bash "$repo/scripts/run_job.sh" "$log" "$gpu" "$python_bin" "$config" "$@"
tmux new-session -d -s "$session" -c "$repo" "$command"
printf 'SESSION=%s\nLOG=%s\n' "$session" "$log"
printf 'Attach: tmux attach -t %s\n' "$session"
printf 'Watch: tail -f %q\n' "$log"
echo 'Detach: Ctrl+b, then d. Finished sessions close; the log remains.'
