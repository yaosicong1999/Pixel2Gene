#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   run_with_metrics.sh <metrics_prefix> <cmd...>

ENV_NAME="${ENV_NAME:-pixel2gene}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"

prefix="$1"
shift

# Make prefix absolute
if [[ "$prefix" != /* ]]; then
  prefix="$(pwd)/$prefix"
fi
mkdir -p "$(dirname "$prefix")"

metrics_txt="${prefix}.metrics.txt"
gpu_csv="${prefix}.gpu.csv"
time_txt="${prefix}.time.txt"
stdout_txt="${prefix}.stdout.txt"
stderr_txt="${prefix}.stderr.txt"

log() { echo "$*" | tee -a "$metrics_txt"; }

log "== START $(date -Is) =="
log "Host: $(hostname)"
log "PWD: $(pwd)"
log "LSB_JOBID=${LSB_JOBID:-}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
log "Command: $*"

# ---- activate conda env (hard fail if cannot) ----
if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_SH"
else
  echo "[ERROR] conda.sh not found at: $CONDA_SH" | tee -a "$metrics_txt" >&2
  exit 2
fi

conda activate "$ENV_NAME" || { echo "[ERROR] failed to activate conda env: $ENV_NAME" | tee -a "$metrics_txt" >&2; exit 2; }

log "which python: $(which python)"
python -c "import sys; print('sys.executable:', sys.executable)" | tee -a "$metrics_txt"

# (optional) torch check: uncomment if you want hard guarantee
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'cuda_available:', torch.cuda.is_available())" | tee -a "$metrics_txt"

# GPU sampling
gpu_sampler_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "timestamp,index,uuid,name,util.gpu,util.mem,mem.used,mem.total" > "$gpu_csv"
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader,nounits
      sleep 10
    done
  ) >> "$gpu_csv" &
  gpu_sampler_pid=$!
fi

# Run command
set +e
/usr/bin/time -v -o "$time_txt" "$@" \
  1>>"$stdout_txt" \
  2> >(tee -a "$stderr_txt" >&2)
rc=$?
set -e

if [[ -n "$gpu_sampler_pid" ]]; then
  kill "$gpu_sampler_pid" >/dev/null 2>&1 || true
fi

log "ExitCode: $rc"
log "== END $(date -Is) =="
exit $rc