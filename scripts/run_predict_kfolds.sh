#!/bin/bash
set -e
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_WORKSPACE_LIMIT_IN_MB=4096
device="cuda"  # "cuda" or "cpu"
pref_predict=$1
output_train=$2
output_predict=$3
mask_pref=$4
fold_idx=$5
emb_type=$6
echo "pref_predict: $pref_predict"
echo "output_train: $output_train"
echo "output_predict: $output_predict"
echo "mask_pref: $mask_pref"
echo "fold_idx: $fold_idx"
echo "emb_type: $emb_type"

CUDA_VISIBLE_DEVICES=1 python impute_filter_predict.py ${pref_predict} --output_train=${output_train}${mask_pref}_fold_${fold_idx}/ --output_predict=${output_predict}${mask_pref}_fold_${fold_idx}/ --device=${device} --emb_type=${emb_type}