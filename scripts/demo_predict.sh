#!/bin/bash
#BSUB -J xCRCP2P2PRED           # LSF job name
#BSUB -o xCRCP2P2PRED.%J.out     # Name of the job output file
#BSUB -e xCRCP2P2PRED.%J.error   # Name of the job error file

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixel2gene

# -----------------------------
# Config
# -----------------------------
pref='xenium_CRC-P2_self_predict_hipt_raw'
pref_train='../data/xenium/CRC-P2/CRC-P2-'
mask_pref='filter_he_qc'
output_train='../data/xenium/CRC-P2/CRC-P2_train_hipt_raw/'
emb_type='hipt_raw'
pref_predict='../data/xenium/CRC-P2/CRC-P2-'
output_predict='../data/xenium/CRC-P2/CRC-P2_self_predict_hipt_raw/'

logdir="logs/${pref}"
mkdir -p "$logdir"

# -----------------------------
# Step 0: preprocess the testing data
# -----------------------------
jid_preprocess=$(bsub -m "transgene2" -q mingyaogpu -gpu "num=2" -n 2 \
    -oo "${logdir}/preprocess.lsf.out" \
    -eo "${logdir}/preprocess.lsf.err" \
    bash run_with_metrics.sh "${logdir}/preprocess" \
      bash run_preprocessing.sh "${pref_predict}" \
    | awk '{print $2}' | tr -d '<>')

# -----------------------------
# Step 1: predict per fold
# -----------------------------
job_ids=()
for fold in {0..9}; do
    jid=$(bsub -w "done(${jid_preprocess})" -m "transgene2" -q mingyaogpu -gpu "num=2" -n 4 \
        -oo "${logdir}/fold_${fold}_predict.lsf.out" \
        -eo "${logdir}/fold_${fold}_predict.lsf.err" \
        bash run_with_metrics.sh "${logdir}/fold_${fold}_predict" \
          bash run_predict_kfolds.sh "${pref_predict}" \
            "${output_train}" "${output_predict}" "${mask_pref}" "${fold}" "${emb_type}" \
        | awk '{print $2}' | tr -d '<>')
    job_ids+=("$jid")
done

dep_str=$(printf "&&done(%s)" "${job_ids[@]}")
dep_str=${dep_str:2}

# -----------------------------
# Step 2: combine folds after all predictions are done
# -----------------------------
jid_combine=$(bsub -w "$dep_str" -q mingyao_normal -n 16 \
    -oo "${logdir}/combine.lsf.out" \
    -eo "${logdir}/combine.lsf.err" \
    bash run_with_metrics.sh "${logdir}/combine" \
      bash run_combine_folds.sh "${pref_predict}" "${output_predict}" "${mask_pref}" \
    | awk '{print $2}' | tr -d '<>')


# -----------------------------
# Step 3: run the clustering on the combined data
# -----------------------------
# jid_cluster_pickle1=$( \
#     bsub -w "done(${jid_combine})" -q mingyao_normal -n 16 \
#         -e logs/${pref}cluster_pickle.err \
#         bash run_cluster_pickle.sh ${pref_predict} ${output_predict}${mask_pref}/ 'hs' \
#     | awk '{print $2}' | tr -d '<>' )

# jid_cluster_pickle2=$( \
#     bsub -w "done(${jid_combine})" -q mingyao_normal -n 16 \
#         -e logs/${pref}cluster_pickle.err \
#         bash run_cluster_pickle.sh ${pref_predict} ${output_predict}${mask_pref}/ ${mask_pref} \
#     | awk '{print $2}' | tr -d '<>' )