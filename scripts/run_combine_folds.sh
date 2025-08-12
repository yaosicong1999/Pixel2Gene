#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate istar
set -e
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_WORKSPACE_LIMIT_IN_MB=4096

data_pref=$1
output_dir=$2
mask_pref=$3
echo "data_pref: $data_pref"
echo "output_dir: $output_dir"
echo "mask_pref: $mask_pref"

python combine_kfolds.py --data_pref=${data_pref} --output=${output_dir} --mask_pref=${mask_pref} 