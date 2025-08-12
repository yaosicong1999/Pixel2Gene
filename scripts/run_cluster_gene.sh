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
mask=${data_pref}mask-small-${mask_pref}.png
echo "mask: $mask"


python cluster_gene.py ${data_pref} --output=${output_dir} --mask=${mask} --overlay 
