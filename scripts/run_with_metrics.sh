#!/bin/bash
set -euo pipefail

# 用法：
#   run_with_metrics.sh <metrics_prefix> <cmd...>
# 例：
#   run_with_metrics.sh logs/foo_preprocess bash run_preprocessing.sh ...

prefix="$1"
shift

mkdir -p "$(dirname "$prefix")"

metrics_txt="${prefix}.metrics.txt"
gpu_csv="${prefix}.gpu.csv"
time_txt="${prefix}.time.txt"

echo "== START $(date -Is) ==" | tee -a "$metrics_txt"
echo "Host: $(hostname)" | tee -a "$metrics_txt"
echo "LSB_JOBID=${LSB_JOBID:-}" | tee -a "$metrics_txt"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" | tee -a "$metrics_txt"
echo "Command: $*" | tee -a "$metrics_txt"

# GPU 采样（如果有 nvidia-smi）
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

# 运行命令，并记录 time -v
# 注意：/usr/bin/time -v 的输出在 stderr，所以我们重定向到 time_txt
set +e
/usr/bin/time -v "$@" 1>>"${prefix}.stdout.txt" 2>>"$time_txt"
rc=$?
set -e

# 停掉 GPU 采样
if [[ -n "$gpu_sampler_pid" ]]; then
  kill "$gpu_sampler_pid" >/dev/null 2>&1 || true
fi

echo "ExitCode: $rc" | tee -a "$metrics_txt"
echo "== END $(date -Is) ==" | tee -a "$metrics_txt"
exit $rc