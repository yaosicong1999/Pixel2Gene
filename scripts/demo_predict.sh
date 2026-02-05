source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixel2gene

# -----------------------------
# Config
# -----------------------------
pref='xenium_CRC-P2_train_hipt_raw'
pref_train='../data/xenium/CRC-P2/CRC-P2-' ## training cnts, locs files with this prefix
output_train='../data/xenium/CRC-P2/CRC-P2_train_hipt_raw/' ## training output folder
mask_pref='filter_he_qc' ## this is the mask for filtering out training samples, i.e. only superpixels within this mask will be kept. by default we use this one after quality-control
epochs=600 ## you can modify the epochs here
emb_type='hipt_raw' ## if you want to use smoothened-version of HIPT, please specify "hipt"
pref_predict='../data/xenium/CRC-P2/CRC-P2-' ## prediction cnts, locs files with this prefix
output_predict='../data/xenium/CRC-P2/CRC-P2_self_predict_hipt_raw/' ## prediction output folder

logdir="logs/${pref}"
mkdir -p "$logdir"

# -----------------------------
# Step 0: preprocess
# -----------------------------
## this run_preprocessing.sh assume that this node is with CUDA on (for feature extraction)
bash run_with_metrics.sh "${logdir}/preprocess" bash run_preprocessing.sh "${pref_predict}"

# -----------------------------
# Step 2: k-fold prediction
# # -----------------------------
## this run_predict_kfolds.sh assume that this node is with CUDA on (for predicting)
bash run_with_metrics.sh "${logdir}/fold_0_predict" bash run_predict_kfolds.sh "${pref_predict}" "${output_train}" "${output_predict}" "${mask_pref}" 0 "${emb_type}"
bash run_with_metrics.sh "${logdir}/fold_1_predict" bash run_predict_kfolds.sh "${pref_predict}" "${output_train}" "${output_predict}" "${mask_pref}" 1 "${emb_type}"
bash run_with_metrics.sh "${logdir}/fold_2_predict" bash run_predict_kfolds.sh "${pref_predict}" "${output_train}" "${output_predict}" "${mask_pref}" 2 "${emb_type}"
bash run_with_metrics.sh "${logdir}/fold_3_predict" bash run_predict_kfolds.sh "${pref_predict}" "${output_train}" "${output_predict}" "${mask_pref}" 3 "${emb_type}"
bash run_with_metrics.sh "${logdir}/fold_4_predict" bash run_predict_kfolds.sh "${pref_predict}" "${output_train}" "${output_predict}" "${mask_pref}" 4 "${emb_type}"
bash run_with_metrics.sh "${logdir}/fold_5_predict" bash run_predict_kfolds.sh "${pref_predict}" "${output_train}" "${output_predict}" "${mask_pref}" 5 "${emb_type}"
bash run_with_metrics.sh "${logdir}/fold_6_predict" bash run_predict_kfolds.sh "${pref_predict}" "${output_train}" "${output_predict}" "${mask_pref}" 6 "${emb_type}"
bash run_with_metrics.sh "${logdir}/fold_7_predict" bash run_predict_kfolds.sh "${pref_predict}" "${output_train}" "${output_predict}" "${mask_pref}" 7 "${emb_type}"
bash run_with_metrics.sh "${logdir}/fold_8_predict" bash run_predict_kfolds.sh "${pref_predict}" "${output_train}" "${output_predict}" "${mask_pref}" 8 "${emb_type}"
bash run_with_metrics.sh "${logdir}/fold_9_predict" bash run_predict_kfolds.sh "${pref_predict}" "${output_train}" "${output_predict}" "${mask_pref}" 9 "${emb_type}"

# -----------------------------
# Step 2: combine folds after all predictions are done
# -----------------------------
## this run_combine_folds.sh does not need CUDA node
bash run_with_metrics.sh "${logdir}/combine" bash run_combine_folds.sh "${pref_predict}" "${output_predict}" "${mask_pref}"