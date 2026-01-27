#!/bin/bash
set -e
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_WORKSPACE_LIMIT_IN_MB=4096
device="cuda"  # "cuda" or "cpu"
pref_train=$1
output_train=$2
mask_pref=$3
fold_idx=$4
epochs=$5
emb_type=$6
echo "pref_train: $pref_train"
echo "output_train: $output_train"
echo "mask_pref: $mask_pref"
echo "fold_idx: $fold_idx"
echo "epochs: $epochs"
echo "emb_type: $emb_type"

CUDA_VISIBLE_DEVICES=0 python impute_filter_train.py ${pref_train} --output_train=${output_train}${mask_pref}_fold_${fold_idx}/ --device=${device} --emb_type=${emb_type} --epochs=${epochs} --load_saved --mask_train=${pref_train}mask-small-${mask_pref}_train_fold_${fold_idx}.png --emb_type=${emb_type}