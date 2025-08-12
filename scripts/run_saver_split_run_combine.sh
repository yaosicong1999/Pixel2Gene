#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate istar
set -e
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_WORKSPACE_LIMIT_IN_MB=4096

data_pref=$1
echo "data_pref: $data_pref"

module load R


Rscript saver_read_and_split.R --pref ${data_pref} --num 4

Rscript saver_run_split.R --pref ${data_pref} --split_idx 1
Rscript saver_run_split.R --pref ${data_pref} --split_idx 2
Rscript saver_run_split.R --pref ${data_pref} --split_idx 3
Rscript saver_run_split.R --pref ${data_pref} --split_idx 4

python saver_combine_imputed.py --pref ${data_pref} --num 4






