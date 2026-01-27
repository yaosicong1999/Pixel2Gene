#!/bin/bash
#BSUB -J xCRCP2Train
#BSUB -o xCRCP2Train.%J.out
#BSUB -e xCRCP2Train.%J.error

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixel2gene

# -----------------------------
# Config
# -----------------------------
pref='xenium_CRC-P2_train_hipt_raw'
pref_train='../data/xenium/CRC-P2/CRC-P2-'
mask_pref='filter_he_qc'
output_train='../data/xenium/CRC-P2/CRC-P2_train_hipt_raw/'
epochs=600
emb_type='hipt_raw'

logdir="logs/${pref}"
mkdir -p "$logdir"


# -----------------------------
# Step 0: preprocess
# -----------------------------
jid_preprocess=$(bsub -m "transgene2" -q mingyaogpu -gpu "num=2" -n 2 \
    -oo "${logdir}/preprocess.lsf.out" \
    -eo "${logdir}/preprocess.lsf.err" \
    bash run_with_metrics.sh "${logdir}/preprocess" \
      bash run_preprocessing.sh "${pref_train}" "${pref_train}" \
    | awk '{print $2}' | tr -d '<>')


# -----------------------------
# Step 1: truth clustering 
# -----------------------------
# jid_cluster_truth0=$( \
#     bsub -w "done(${jid_preprocess})" -q mingyao_normal -n 16 \
#         -e logs/${pref}cluster_truth.err \
#         bash run_cluster_truth.sh ${pref_train} ${output_train}/ 'hs' \
#     | awk '{print $2}' | tr -d '<>' )

# jid_cluster_truth1=$( \
#     bsub -w "done(${jid_preprocess})" -q mingyao_normal -n 16 \
#         -e logs/${pref}cluster_truth.err \
#         bash run_cluster_truth.sh ${pref_train} ${output_train}/ ${mask_pref} \
#     | awk '{print $2}' | tr -d '<>' )


# -----------------------------
# Step 2: k-fold training
# -----------------------------

# Part 0: folds 0 -> 3
first_fold_part_0=0
jid_part_0=$(bsub -w "done(${jid_preprocess})" -m "transgene2" -q mingyaogpu -gpu "num=2" -n 4 \
    -oo "${logdir}/fold_${first_fold_part_0}.lsf.out" \
    -eo "${logdir}/fold_${first_fold_part_0}.lsf.err" \
    bash run_with_metrics.sh "${logdir}/fold_${first_fold_part_0}" \
      bash run_train_kfolds.sh ${pref_train} \
        ${output_train} ${mask_pref} ${first_fold_part_0} ${epochs} ${emb_type} \
    | awk '{print $2}' | tr -d '<>')

for fold in {1..3}; do
    jid_part_0=$(bsub -w "done(${jid_part_0})" -m "transgene2" -q mingyaogpu -gpu "num=2" -n 4 \
        -oo "${logdir}/fold_${fold}.lsf.out" \
        -eo "${logdir}/fold_${fold}.lsf.err" \
        bash run_with_metrics.sh "${logdir}/fold_${fold}" \
          bash run_train_kfolds.sh ${pref_train} \
            ${output_train} ${mask_pref} ${fold} ${epochs} ${emb_type} \
        | awk '{print $2}' | tr -d '<>')
done

# Part 1: folds 4 -> 6
first_fold_part_1=4
jid_part_1=$(bsub -w "done(${jid_preprocess})" -m "transgene2" -q mingyaogpu -gpu "num=2" -n 4 \
    -oo "${logdir}/fold_${first_fold_part_1}.lsf.out" \
    -eo "${logdir}/fold_${first_fold_part_1}.lsf.err" \
    bash run_with_metrics.sh "${logdir}/fold_${first_fold_part_1}" \
      bash run_train_kfolds.sh ${pref_train} \
        ${output_train} ${mask_pref} ${first_fold_part_1} ${epochs} ${emb_type} \
    | awk '{print $2}' | tr -d '<>')

for fold in {5..6}; do
    jid_part_1=$(bsub -w "done(${jid_part_1})" -m "transgene2" -q mingyaogpu -gpu "num=2" -n 4 \
        -oo "${logdir}/fold_${fold}.lsf.out" \
        -eo "${logdir}/fold_${fold}.lsf.err" \
        bash run_with_metrics.sh "${logdir}/fold_${fold}" \
          bash run_train_kfolds.sh ${pref_train} \
            ${output_train} ${mask_pref} ${fold} ${epochs} ${emb_type} \
        | awk '{print $2}' | tr -d '<>')
done

# Part 2: folds 7 -> 9
first_fold_part_2=7
jid_part_2=$(bsub -w "done(${jid_preprocess})" -m "transgene2" -q mingyaogpu -gpu "num=2" -n 4 \
    -oo "${logdir}/fold_${first_fold_part_2}.lsf.out" \
    -eo "${logdir}/fold_${first_fold_part_2}.lsf.err" \
    bash run_with_metrics.sh "${logdir}/fold_${first_fold_part_2}" \
      bash run_train_kfolds.sh ${pref_train} \
        ${output_train} ${mask_pref} ${first_fold_part_2} ${epochs} ${emb_type} \
    | awk '{print $2}' | tr -d '<>')

for fold in {8..9}; do
    jid_part_2=$(bsub -w "done(${jid_part_2})" -m "transgene2" -q mingyaogpu -gpu "num=2" -n 4 \
        -oo "${logdir}/fold_${fold}.lsf.out" \
        -eo "${logdir}/fold_${fold}.lsf.err" \
        bash run_with_metrics.sh "${logdir}/fold_${fold}" \
          bash run_train_kfolds.sh ${pref_train} \
            ${output_train} ${mask_pref} ${fold} ${epochs} ${emb_type} \
        | awk '{print $2}' | tr -d '<>')
done