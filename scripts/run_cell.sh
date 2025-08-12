#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate istar
set -e
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_WORKSPACE_LIMIT_IN_MB=4096

pref=$1
echo "pref: $pref"

python plot_cell_correlation.py --pref=${pref}