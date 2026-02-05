# -----------------------------
# Config
# -----------------------------
pref='xenium_CRC-P2_train_hipt_raw'
pref_train='../data/xenium/CRC-P2/CRC-P2-' ## training cnts, locs files with this prefix
output_train='../data/xenium/CRC-P2/CRC-P2_train_hipt_raw/' ## training output folder
mask_pref='filter_he_qc' ## this is the mask for filtering out training samples, i.e. only superpixels within this mask will be kept. by default we use this one after quality-control
epochs=600 ## you can modify the epochs here
emb_type='hipt_raw' ## if you want to use smoothened-version of HIPT, please specify "hipt"

## this is the folder that we will store the logs in
logdir="logs/${pref}"
mkdir -p "$logdir"


# -----------------------------
# Step 0: preprocess
# -----------------------------
## this run_preprocessing.sh assume that this node is with CUDA on (for feature extraction)
bash run_with_metrics.sh "${logdir}/preprocess" bash run_preprocessing.sh "${pref_train}"

# -----------------------------
# Step 1: k-fold training
# -----------------------------
# this run_predict_kfolds.sh assume that this node is with CUDA on (for training)
bash run_with_metrics.sh "${logdir}/fold_0" bash run_train_kfolds.sh ${pref_train} ${output_train} ${mask_pref} 0 ${epochs} ${emb_type}
bash run_with_metrics.sh "${logdir}/fold_1" bash run_train_kfolds.sh ${pref_train} ${output_train} ${mask_pref} 1 ${epochs} ${emb_type}
bash run_with_metrics.sh "${logdir}/fold_2" bash run_train_kfolds.sh ${pref_train} ${output_train} ${mask_pref} 2 ${epochs} ${emb_type}
bash run_with_metrics.sh "${logdir}/fold_3" bash run_train_kfolds.sh ${pref_train} ${output_train} ${mask_pref} 3 ${epochs} ${emb_type}
bash run_with_metrics.sh "${logdir}/fold_4" bash run_train_kfolds.sh ${pref_train} ${output_train} ${mask_pref} 4 ${epochs} ${emb_type}
bash run_with_metrics.sh "${logdir}/fold_5" bash run_train_kfolds.sh ${pref_train} ${output_train} ${mask_pref} 5 ${epochs} ${emb_type}
bash run_with_metrics.sh "${logdir}/fold_6" bash run_train_kfolds.sh ${pref_train} ${output_train} ${mask_pref} 6 ${epochs} ${emb_type}
bash run_with_metrics.sh "${logdir}/fold_7" bash run_train_kfolds.sh ${pref_train} ${output_train} ${mask_pref} 7 ${epochs} ${emb_type}
bash run_with_metrics.sh "${logdir}/fold_8" bash run_train_kfolds.sh ${pref_train} ${output_train} ${mask_pref} 8 ${epochs} ${emb_type}
bash run_with_metrics.sh "${logdir}/fold_9" bash run_train_kfolds.sh ${pref_train} ${output_train} ${mask_pref} 9 ${epochs} ${emb_type}