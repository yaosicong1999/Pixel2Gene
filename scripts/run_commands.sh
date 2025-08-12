bsub -Is -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 'bash'
conda activate istar

bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P1_train_uni_full.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P1_train_uni_pca.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P1_train_uni_pca_gene_pca.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P1_train_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2_train_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2-downsampled_train_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2-downsampled_train_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_MB-downsampled_train_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium_CRC-P1_train_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium_CRC-P2_train_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium_CRC-P2_train_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium_Gastric_train_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium_BRCA_train_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium5k_HBC_train_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium5k_HBC_train_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_cosmx_HK2844_train_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_cosmx_HK3039_train_hipt_raw.sh


bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2_self_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2_self_predict_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2_P2full_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2_P1full_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2_P5full_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2_P2full_predict_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2_P1full_predict_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2_P5full_predict_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2_P1_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2_P1_predict_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2_P5_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2_P5_predict_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2-downsampled_self_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_CRC-P2-downsampled_self_predict_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_MB-downsampled_self_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_visiumhd_MB-downsampled_self_predict_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium_CRC-P1_P2_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium_CRC-P2_self_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium_CRC-P2_self_predict_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium_CRC-P2_P1_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium_CRC-P2_P5_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium_CRC-P2_P5_predict_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium_Gastric_self_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium5k_HBC_self_predict_hipt.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_xenium5k_HBC_self_predict_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_cosmx_HK3039_self_predict_hipt_raw.sh
bsub -m "transgene2" -q mingyaogpu -gpu "num=4" -n 16 < job_cosmx_HK2844_self_predict_hipt_raw.sh


for fold in {0..9}; do
    bsub -m "transgene2" -q mingyaogpu -gpu "num=2" -n 2 \
            -o "logs/fold_${fold}.out" -e "logs/fold_${fold}.err" \
            bash job_visiumhd_P1_self_kfolds.sh $fold
done


bsub -n 16 < job_plot_visiumhd_downsampled_1.sh
bsub -n 16 < job_plot_visiumhd_downsampled_1_saver.sh
bsub -n 16 < job_plot_visiumhd_downsampled_1.sh
bsub -n 16 < job_cluster_visiumhd_P1_downsampled_1.sh
bsub -n 16 < job_cluster_visiumhd_P1_downsampled_6.sh 
bsub -n 16 < job_cluster_visiumhd_P1_downsampled_6_saver.sh   
bsub -n 16 < job_cluster_visiumhd_P1_downsampled_6_comparison.sh
bsub -n 16 < job_cluster_visiumhd_P1_downsampled_7_saver.sh   
bsub -n 16 < job_cluster_visiumhd_P1_downsampled_1_saver.sh   
bsub -n 16 < job_cluster_visiumhd_P1_downsampled_7.sh
bsub -n 16 < job_cluster_visiumhd_P1.sh
bsub -n 16 < job_saver.sh
bsub -n 16 < job_temp.sh
bsub -n 16 < job_plot_visiumhd_good.sh
bsub -n 16 < job_plot_visiumhd_bad.sh
bsub -q mingyao_normal -n 16 < job_ROI_visiumhd_P1.sh
bsub -q mingyao_normal -n 16 < job_kfolds_combine.sh

bsub -q mingyao_normal -n 16 < job_visiumhd_MBd_saver.sh
bsub -q mingyao_normal -n 16 < job_xenium_CRC-P1_cell.sh
bsub -q mingyao_normal -n 16 < job_xenium_CRC-P2_cell.sh


## cluster the downsampled
prefix="../visiumhd_heg/CRC-P1-downsampled6-"
output="../visiumhd_heg/P1_downsampled_full_0.1_0.03/filter_locs_he_qc/"
mask="../visiumhd_heg/CRC-P1-downsampled6-mask-small-filter_locs_he_qc.png"
python cluster_pickle.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=0 --pend=1
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=0 --pend=1
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=30 --pend=50
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=40 --pend=50
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=40 --pend=60
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=50 --pend=60
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=50 --pend=75



## cluster the saver-imputed
prefix="../visiumhd_heg/CRC-P1-downsampled6-saver-imputed-"
output="../visiumhd_heg/P1_downsampled_full_0.1_0.03_saver-imputed/filter_locs_he_qc/"
mask="../visiumhd_heg/CRC-P1-downsampled6-mask-small-filter_locs_he_qc.png"
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=0 --pend=100
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=25 --pend=75
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=30 --pend=50
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=40 --pend=50
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=40 --pend=60
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=50 --pend=60
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=50 --pend=75


prefix="../visiumhd_heg/CRC-P1-downsampled7-saver-imputed-"
output="../visiumhd_heg/P1_downsampled_full_0.2_0.05_saver-imputed/filter_locs_he_qc/"
mask="../visiumhd_heg/CRC-P1-downsampled7-mask-small-filter_locs_he_qc.png"
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=0 --pend=100
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=25 --pend=75
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=30 --pend=50
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=40 --pend=50
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=40 --pend=60
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=50 --pend=60
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=50 --pend=75




prefix="../visiumhd_heg/CRC-P1-downsampled7-saver-imputed-"
output="../visiumhd_heg/P1_downsampled_full_0.2_0.05/saver_imputed/"
mask="../visiumhd_heg/CRC-P1-downsampled7-mask-small-filter_locs_he_qc.png"
python cluster_cnts.py ${prefix} --output=${output} --mask=${mask} --overlay --pstart=0 --pend=1










python downsampling.py --data_pref="../visiumhd_heg/CRC-P1-" --out_pref="../visiumhd_heg/CRC-P1-downsampled1-" --mean=0.3 --sd=0.05



dsampled_pref="../visiumhd_heg/CRC-P1-downsampled6-"
dimputed_pref="../visiumhd_heg/P1_downsampled_full_0.1_0.03/filter_locs_he_qc/"
original_pref="../visiumhd_heg/CRC-P1-"
oimputed_pref="../visiumhd_heg/P1_self_1e-5/filter_locs_he_qc/"
mask="../visiumhd_heg/CRC-P1-downsampled6-mask-small-filter_locs_he_qc.png"
output="../visiumhd_heg/P1_downsampled_full_0.1_0.03/filter_locs_he_qc/"
python plot_comparison_downsampled.py --dsampled_pref=${dsampled_pref} --dimputed_pref=${dimputed_pref} --original_pref=${original_pref} --oimputed_pref=${oimputed_pref} --output=${output} --mask=${mask} --overlay


original_dir="../visiumhd_heg/P1_self_1e-5/filter_locs_he_qc/"
downsampled_dir="../visiumhd_heg/P1_downsampled_full_0.1_0.03/filter_locs_he_qc/"
data_pref="../visiumhd_heg/CRC-P1-"
downsampled_pref="../visiumhd_heg/CRC-P1-downsampled6-"
python plot_comparison_cluster.py --original_dir=${original_dir} --downsampled_dir=${downsampled_dir} --data_pref=${data_pref}


original_dir="../visiumhd_heg/P1_self_1e-5/filter_locs_he_qc/"
downsampled_dir="../visiumhd_heg/P1_downsampled_full_0.1_0.03/filter_locs_he_qc/"
data_pref="../visiumhd_heg/CRC-P1-"
downsampled_pref="../visiumhd_heg/CRC-P1-downsampled6-"
ground_truth_dir="../visiumhd_heg/P1_self_1e-5/filter_locs_he_qc/clusters_truth/CRC-P1-downsampled6-mask-small-filter_locs_he_qc/0-100_filtersizeNone_minclsizeNone_nclusters015/"
python plot_comparison_cluster.py --original_dir=${original_dir} --downsampled_dir=${downsampled_dir} --data_pref=${data_pref} --downsampled_pref=${downsampled_pref} --ground_truth_dir=${ground_truth_dir}


original_dir="../visiumhd_heg/P1_self_1e-5/filter_locs_he_qc/"
downsampled_dir="../visiumhd_heg/P1_downsampled_full_0.1_0.03/filter_locs_he_qc/"
data_pref="../visiumhd_heg/CRC-P1-"
downsampled_pref="../visiumhd_heg/CRC-P1-downsampled6-"
ground_truth_dir="../visiumhd_heg/P1_self_1e-5/filter_locs_he_qc/clusters_truth/CRC-P1-downsampled6-mask-small-filter_locs_he_qc/0-100_filtersizeNone_minclsizeNone_nclusters015/"
python plot_comparison_cluster_saver.py --original_dir=${original_dir} --downsampled_dir=${downsampled_dir} --data_pref=${data_pref} --downsampled_pref=${downsampled_pref} --ground_truth_dir=${ground_truth_dir}



cp "../visiumhd_heg/CRC-P1-pixel-size-raw.txt" "../visiumhd_heg/CRC-P1-downsampled6-pixel-size-raw.txt" 
cp "../visiumhd_heg/CRC-P1-pixel-size.txt" "../visiumhd_heg/CRC-P1-downsampled6-pixel-size-raw.txt" 
cp "../visiumhd_heg/CRC-P1-downsampled5-radius.txt" "../visiumhd_heg/CRC-P1-downsampled6-radius.txt" 


Rscript saver_read_and_split.R --pref ../visiumhd_heg/CRC-P1-downsampled6- --num 4
Rscript saver_run_split.R --pref ../visiumhd_heg/CRC-P1-downsampled6- --split_idx 1
Rscript saver_run_split.R --pref ../visiumhd_heg/CRC-P1-downsampled6- --split_idx 2
Rscript saver_run_split.R --pref ../visiumhd_heg/CRC-P1-downsampled6- --split_idx 3
Rscript saver_run_split.R --pref ../visiumhd_heg/CRC-P1-downsampled6- --split_idx 4


Rscript saver_read_and_split.R --pref ../visiumhd_heg/CRC-P1-downsampled7- --num 4
Rscript saver_run_split.R --pref ../visiumhd_heg/CRC-P1-downsampled7- --split_idx 1
Rscript saver_run_split.R --pref ../visiumhd_heg/CRC-P1-downsampled7- --split_idx 2
Rscript saver_run_split.R --pref ../visiumhd_heg/CRC-P1-downsampled7- --split_idx 3
Rscript saver_run_split.R --pref ../visiumhd_heg/CRC-P1-downsampled7- --split_idx 4








impute_pref="../visiumhd_heg/P1_downsampled_full_0.1_0.03/filter_locs_he_qc/"
saver_pref="../visiumhd_heg/P1_downsampled_full_0.1_0.03_saver-imputed/filter_locs_he_qc/"
mask="../visiumhd_heg/CRC-P1-downsampled6-mask-small-filter_locs_he_qc.png"
python plot_comparison_downsampled_impute_vs_saver.py --impute_pref=${impute_pref} --saver_pref=${saver_pref} --mask=${mask}  --n_top=100


embs='../xenium/CRC-P1/CRC-P1-embeddings-hipt-smooth.pickle'
output='../xenium/CRC-P1/clusters_hipt/'
n_clusters=15
mask='../xenium/CRC-P1/CRC-P1-mask-small-hs.png'
python cluster_hist.py --embeddings=${embs} --output=${output} --n_clusters=${n_clusters} --mask=${mask} 

embs='../xenium/CRC-P2/CRC-P2-embeddings-hipt-smooth.pickle'
output='../xenium/CRC-P2/clusters_hipt/'
n_clusters=15
mask='../xenium/CRC-P2/CRC-P2-mask-small-hs.png'
python cluster_hist.py --embeddings=${embs} --output=${output} --n_clusters=${n_clusters} --mask=${mask} 





conda activate uni
prefix1=../visiumhd_heg/CRC-P1/CRC-P1-
prefix2=../visiumhd_heg/CRC-P1/CRC-P1-
python uni_feature_extraction_uni_fast_v2.py --read_path ${prefix1} --save_dir ${prefix2} --batch_size=128


embs='../visiumhd_heg/CRC-P1full/CRC-P1full-embedding_uni.npy'
output='../visiumhd_heg/CRC-P1full/clusters_uni/'
mask='../visiumhd_heg/CRC-P1full/CRC-P1full-mask-small-hs.png'
python cluster_uni.py --embeddings=${embs} --output=${output} --mask=${mask} 


conda deactivate
conda activate istar 
embs='../visiumhd_heg/CRC-P1/CRC-P1-embedding_uni.npy'
output='../visiumhd_heg/CRC-P1/clusters_uni/'
mask='../visiumhd_heg/CRC-P1/CRC-P1-mask-small-hs.png'
python cluster_uni.py --embeddings=${embs} --output=${output} --mask=${mask} 



embs='../visiumhd_heg/CRC-P1/CRC-P1-embeddings-hipt-smooth.pickle'
output='../visiumhd_heg/CRC-P1/clusters_hipt/'
mask='../visiumhd_heg/CRC-P1/CRC-P1-mask-small-hs.png'
python cluster_hist.py --embeddings=${embs} --output=${output} --mask=${mask} 


embs='../visiumhd_heg/CRC-P1/CRC-P1-embeddings-hipt-raw.pickle'
output='../visiumhd_heg/CRC-P1/clusters_hipt_raw/'
mask='../visiumhd_heg/CRC-P1/CRC-P1-mask-small-hs.png'
python cluster_hist.py --embeddings=${embs} --output=${output} --mask=${mask} 



conda activate uni
prefix1=../xenium/CRC-P1/CRC-P1-
prefix2=../xenium/CRC-P1/CRC-P1-
python uni_feature_extraction_uni_fast_v2.py --read_path ${prefix1} --save_dir ${prefix2} --batch_size=128

conda deactivate
conda activate istar 
embs='../xenium/CRC-P1/CRC-P1-embedding.npy'
output='../xenium/CRC-P1/clusters_uni/'
n_clusters=15
mask='../xenium/CRC-P1/CRC-P1-mask-small-hs.png'
python cluster_uni.py --embeddings=${embs} --output=${output} --n_clusters=${n_clusters} --mask=${mask} 

prefix='../xenium/CRC-P1/CRC-P1-'
python extract_features.py "${prefix}" --device="cuda:0"





embs='../xenium/CRC-P1/CRC-P1-embeddings-hipt-smooth.pickle'
output='../xenium/CRC-P1/clusters_hipt/'
n_clusters=15
mask='../xenium/CRC-P1/CRC-P1-mask-small-hs.png'
python cluster_hist.py --embeddings=${embs} --output=${output} --n_clusters=${n_clusters} --mask=${mask} 

embs='../xenium/CRC-P1/CRC-P1-embeddings-hipt-raw.pickle'
output='../xenium/CRC-P1/clusters_hipt_raw/'
n_clusters=15
mask='../xenium/CRC-P1/CRC-P1-mask-small-hs.png'
python cluster_hist.py --embeddings=${embs} --output=${output} --n_clusters=${n_clusters} --mask=${mask} 



pref_train='../visiumhd_heg/CRC-P1/CRC-P1-'
train_output='../visiumhd_heg/CRC-P1/CRC-P1_self_uni_full/training_filter_he_qc_fold_0/'
train_mask='../visiumhd_heg/CRC-P1/CRC-P1-mask-small-filter_he_qc_train_fold_0.png'
device='cuda'
epochs=2
emb_type='uni'
python impute_filter_training.py ${pref_train} --train_output=${train_output} --device=${device} --emb_type=${emb_type} --epochs=${epochs} --train_mask=${train_mask}





pref_train='../visiumhd_heg/CRC-P1/CRC-P1-'
pref_predict='../visiumhd_heg/CRC-P1/CRC-P1-'
train_output='../visiumhd_heg/CRC-P1/CRC-P1_self_uni_full/training_filter_he_qc_fold_0/'
predict_output='../visiumhd_heg/CRC-P1/CRC-P1_self_uni_full/predict_filter_he_qc_fold_0/'
device='cuda'
emb_type='uni'
python impute_filter_predicting.py ${pref_train} ${pref_predict} --train_output=${train_output} --predict_output=${predict_output} --device=${device} --emb_type=${emb_type}


conda activate uni
export LD_LIBRARY_PATH=""
python UNI_V5/UNI_16_16_pkl.py --read_path ../visiumhd_heg/CRC-P1full/CRC-P1full- --save_dir ../visiumhd_heg/CRC-P1full/CRC-P1full- --device cuda:0

python UNI_V5/UNI_16_16_pkl.py --read_path ../visiumhd_heg/CRC-P2/CRC-P2- --save_dir ../visiumhd_heg/CRC-P2/CRC-P2- --device cuda:0

conda activate istar 
embs='../visiumhd_heg/CRC-P2/CRC-P2-embedding_uni.npy'
output='../visiumhd_heg/CRC-P2/clusters_uni/'
mask='../visiumhd_heg/CRC-P2/CRC-P2-mask-small-hs.png'
python cluster_uni.py --embeddings=${embs} --output=${output} --mask=${mask} --n_clusters=20

pref='CRC-P1full-predict_uni_full_'
mask_pref='filter_he_qc_fold_1'
emb_type='uni'
pref_predict='../visiumhd_heg/CRC-P1full/CRC-P1full-'
output_predict='../visiumhd_heg/CRC-P1full/CRC-P1full_uni_full_predict/'

jid_cluster_pickle1=$( \
    bsub -q mingyao_normal -n 16 \
        -e logs/${pref}cluster_pickle.err \
        bash run_cluster_pickle.sh ${pref_predict} ${output_predict}${mask_pref}/ 'hs' \
    | awk '{print $2}' | tr -d '<>' )



pref='CRC-P1-predict_uni_pca__gene_pca_'
mask_pref='filter_he_qc'
emb_type='uni_pca_gene_pca'
pref_predict='../visiumhd_heg/CRC-P1/CRC-P1-'
output_predict='../visiumhd_heg/CRC-P1/CRC-P1_self_uni_pca_gene_pca_predict/'

jid_cluster_pickle1=$( \
    bsub -q mingyao_normal -n 16 \
        -e logs/${pref}cluster_pickle.err \
        bash run_cluster_pickle.sh ${pref_predict} ${output_predict}${mask_pref}/ 'hs' \
    | awk '{print $2}' | tr -d '<>' )

jid_cluster_pickle2=$( \
    bsub -q mingyao_normal -n 16 \
        -e logs/${pref}cluster_pickle.err \
        bash run_cluster_pickle.sh ${pref_predict} ${output_predict}${mask_pref}/ ${mask_pref} \
    | awk '{print $2}' | tr -d '<>' )


oTrue_pref="../visiumhd_heg/CRC-P1/CRC-P1-"
oPred_pref1="../visiumhd_heg/CRC-P1/CRC-P2_P1_predict_hipt_raw/filter_he_qc/"
oPred_method1="Imputation"
mask="../visiumhd_heg/CRC-P1/CRC-P1-mask-small-hs.png"
output="../visiumhd_heg/CRC-P1/CRC-P2_P1_predict_hipt_raw/filter_he_qc2/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref ${oPred_pref1} --oPred_method ${oPred_method1}  --output=${output} --mask=${mask} --cell_type="test" --gene_names CEACAM5





oTrue_pref="../visiumhd_heg/CRC-P2/CRC-P2-"
dTrue_pref="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-"
dPred_pref1="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled_self_predict_hipt/filter_he_qc/"
dPred_method1="Downsampled-Imputation"
mask="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-mask-small-hs.png"
output="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled_self_predict_hipt/filter_he_qc3/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=100 --skip_gene_plot
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --cell_type="test" --gene_names KRT8 SPARC PLA2G2A SELENOP REG1A
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100


oTrue_pref="../visiumhd_heg/CRC-P2/CRC-P2-"
dTrue_pref="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-"
dPred_pref1="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled_self_predict_hipt_raw/filter_he_qc/"
dPred_method1="Downsampled-Imputation"
mask="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-mask-small-hs.png"
output="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled_self_predict_hipt_raw/filter_he_qc4/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=100 --skip_gene_plot
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --cell_type="test" --gene_names KRT8 SPARC PLA2G2A SELENOP REG1A
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
python plot_violin_from_pickle_correlation.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
python plot_violin_from_pickle_ssim.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
python plot_violin_from_pickle_rmse.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
python plot_violin_from_pickle_moran.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100




oTrue_pref="../visiumhd_heg/MB/MB-"
dTrue_pref="../visiumhd_heg/MB-downsampled/MB-downsampled-"
dPred_pref1="../visiumhd_heg/MB-downsampled/MB-downsampled_self_predict_hipt/filter_he_qc/"
dPred_method1="Downsampled-Imputation"
mask="../visiumhd_heg/MB-downsampled/MB-downsampled-mask-small-hs.png"
output="../visiumhd_heg/MB-downsampled/MB-downsampled_self_predict_hipt/filter_he_qc3/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=300 
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --start_idx=1000 --end_idx=1100 --skip_gene_plot
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --cell_type="test4" --gene_names Slc17a7 Qk Bcl11b Kcne2 Neurod1
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --cell_type="test3" --gene_names Slc17a7 
python plot_violin_from_pickle_correlation.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=100 --end_idx=200


oTrue_pref="../visiumhd_heg/MB/MB-"
dTrue_pref="../visiumhd_heg/MB-downsampled/MB-downsampled-"
dPred_pref1="../visiumhd_heg/MB-downsampled/MB-downsampled_self_predict_hipt_raw/filter_he_qc/"
dPred_method1="Downsampled-Imputation"
mask="../visiumhd_heg/MB-downsampled/MB-downsampled-mask-small-hs.png"
output="../visiumhd_heg/MB-downsampled/MB-downsampled_self_predict_hipt_raw/filter_he_qc5/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=100 --skip_gene_plot
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --start_idx=1000 --end_idx=1100 --skip_gene_plot
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --cell_type="test" --gene_names Slc17a7 Qk Bcl11b Kcne2 Neurod1
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --cell_type="test3" --gene_names Kcne2 
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=100 --end_idx=200
python plot_violin_from_pickle_correlation.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=300
python plot_violin_from_pickle_ssim.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=300
python plot_violin_from_pickle_rmse.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=300
python plot_violin_from_pickle_moran.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=300



oTrue_pref="../visiumhd_heg/CRC-P2/CRC-P2-"
dTrue_pref="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-"
dPred_pref1="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled_self_predict_hipt/filter_he_qc/"
dPred_method1="Downsampled-Imputation"
mask="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-mask-small-hs.png"
output="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled_self_predict_hipt/filter_he_qc2/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=100 
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --cell_type="test" --gene_names KRT8 SPARC PLA2G2A SELENOP TAGLN
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100

oTrue_pref="../visiumhd_heg/CRC-P2/CRC-P2-"
dTrue_pref="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-"
dPred_pref1="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled_self_predict_hipt_raw/filter_he_qc/"
dPred_method1="Downsampled-Imputation"
mask="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-mask-small-hs.png"
output="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled_self_predict_hipt_raw/filter_he_qc3/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=300 --skip_gene_plot 
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --cell_type="test" --gene_names KRT8 SPARC PLA2G2A SELENOP TAGLN
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --dTrue_pref=${dTrue_pref} --dPred_pref ${dPred_pref1} --dPred_method ${dPred_method1}  --output=${output} --mask=${mask} --cell_type="test" --gene_names KRT8 SPARC PLA2G2A SELENOP TAGLN

python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100

oTrue_pref='../visiumhd_heg/CRC-P2/CRC-P2-'
oPred_pref='../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/'
python evaluate_ICC.py --data_pref=${oTrue_pref} --output=${oPred_pref} --n_cluster=18 --top_n=300








data_pref='../visiumhd_heg/MB-downsampled/MB-downsampled-'
output='../visiumhd_heg/MB-downsampled/clusters_hipt_raw/'
mask_pref='hs'
bsub -q mingyao_normal -n 16 \
    -e logs/cluster_hist_raw.err \
    bash run_cluster_hist_raw.sh ${data_pref} ${output} ${mask_pref} \
for n_clusters in 015 018 020 025 030; do
    hipt_dir='../visiumhd_heg/MB-downsampled/clusters_hipt_raw/nclusters'${n_clusters}'/'
    o_dir='../visiumhd_heg/MB/MB_train_hipt/clusters_truth/MB-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    d_dir='../visiumhd_heg/MB-downsampled/MB-downsampled_train_hipt_raw/clusters_truth/MB-downsampled-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    dp_dir='../visiumhd_heg/MB-downsampled/MB-downsampled_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/MB-downsampled-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../visiumhd_heg/MB-downsampled/plot_cluster_aligned_raw_0_100/'
    python plot_cluster_aligned.py --hipt_dir=${hipt_dir} --o_dir=${o_dir} --d_dir=${d_dir} --dp_dir=${dp_dir} --output=${output} --n_cl=${n_clusters}
done
for n_clusters in 015 018 020 025 030; do
    hipt_dir='../visiumhd_heg/MB-downsampled/clusters_hipt_raw/nclusters'${n_clusters}'/'
    o_dir='../visiumhd_heg/MB/MB_train_hipt/clusters_truth/MB-mask-small-hs/20-40_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    d_dir='../visiumhd_heg/MB-downsampled/MB-downsampled_train_hipt_raw/clusters_truth/MB-downsampled-mask-small-hs/20-40_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    dp_dir='../visiumhd_heg/MB-downsampled/MB-downsampled_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/MB-downsampled-mask-small-hs/20-40_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../visiumhd_heg/MB-downsampled/plot_cluster_aligned_raw_20_40/'
    python plot_cluster_aligned.py --hipt_dir=${hipt_dir} --o_dir=${o_dir} --d_dir=${d_dir} --dp_dir=${dp_dir} --output=${output} --n_cl=${n_clusters}
done





data_pref='../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-'
output='../visiumhd_heg/CRC-P2-downsampled/clusters_hipt_raw/'
mask_pref='hs'
bsub -q mingyao_normal -n 16 \
    -e logs/cluster_hist_raw.err \
    bash run_cluster_hist_raw.sh ${data_pref} ${output} ${mask_pref} \

for n_clusters in 010 015 018 020 025 030; do
    hipt_dir='../visiumhd_heg/CRC-P2/clusters_hipt_raw/nclusters'${n_clusters}
    o_dir='../visiumhd_heg/CRC-P2/clusters_truth/CRC-P2-mask-small-hs/top300_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    d_dir='../visiumhd_heg/CRC-P2-downsampled/clusters_truth/CRC-P2-downsampled-mask-small-hs/top300_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    dp_dir='../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/CRC-P2-downsampled-mask-small-hs/top300_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../visiumhd_heg/CRC-P2-downsampled/plot_cluster_aligned_raw_top300/'
    python plot_cluster_aligned.py --hipt_dir=${hipt_dir} --o_dir=${o_dir} --d_dir=${d_dir} --dp_dir=${dp_dir} --output=${output} --n_cl=${n_clusters}
done
for n_clusters in 010 012 015 018 020 025 030; do
    hipt_dir='../visiumhd_heg/CRC-P2/clusters_hipt_raw/nclusters'${n_clusters}
    o_dir='../visiumhd_heg/CRC-P2/clusters_truth_tree/CRC-P2-mask-small-hs/top300_filtersizeNone_minclsizeNone_ward_nclusters'${n_clusters}'/'
    d_dir='../visiumhd_heg/CRC-P2-downsampled/clusters_truth_tree/CRC-P2-downsampled-mask-small-hs/top300_filtersizeNone_minclsizeNone_ward_nclusters'${n_clusters}'/'
    dp_dir='../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle_tree/CRC-P2-downsampled-mask-small-hs/top300_filtersizeNone_minclsizeNone_ward_nclusters'${n_clusters}'/'
    output='../visiumhd_heg/CRC-P2-downsampled/plot_cluster_tree_ward_aligned_raw_top300/'
    python plot_cluster_aligned.py --hipt_dir=${o_dir} --o_dir=${o_dir} --d_dir=${d_dir} --dp_dir=${dp_dir} --output=${output} --n_cl=${n_clusters}
done




n_clusters=20
oTrue_pref='../visiumhd_heg/CRC-P2/CRC-P2-' 
dTrue_pref='../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-' 
dPred_pref='../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled_self_predict_hipt_raw/filter_he_qc/'
oTrue_label_dir="../visiumhd_heg/CRC-P2/clusters_truth/CRC-P2-mask-small-hs/top300_filtersizeNone_minclsizeNone_nclusters$(printf "%03d" $n_clusters)/"
dTrue_label_dir="../visiumhd_heg/CRC-P2-downsampled/clusters_truth/CRC-P2-downsampled-mask-small-hs/top300_filtersizeNone_minclsizeNone_nclusters$(printf "%03d" $n_clusters)/"
dPred_label_dir="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/CRC-P2-downsampled-mask-small-hs/top300_filtersizeNone_minclsizeNone_nclusters$(printf "%03d" $n_clusters)/"
python evaluate_ICC2.py --data_pref=${oTrue_pref} --pred_dir=${dPred_pref} --obs_label_dir=${oTrue_label_dir} --pred_label_dir=${dPred_label_dir} --n_cluster=18 --top_n=300
python evaluate_ICC2.py --data_pref=${dTrue_pref} --pred_dir=${dPred_pref} --obs_label_dir=${dTrue_label_dir} --pred_label_dir=${dPred_label_dir} --n_cluster=18 --top_n=300
oTrue_label_dir="../visiumhd_heg/CRC-P2/clusters_truth/CRC-P2-mask-small-hs/top300_filtersizeNone_minclsizeNone_nclusters030/"
dTrue_label_dir="../visiumhd_heg/CRC-P2-downsampled/clusters_truth/CRC-P2-downsampled-mask-small-hs/top300_filtersizeNone_minclsizeNone_nclusters030/"
dPred_label_dir="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/CRC-P2-downsampled-mask-small-hs/top300_filtersizeNone_minclsizeNone_nclusters018/"
python calculate_ARI.py   --oTrue=$oTrue_label_dir   --dTrue=$dTrue_label_dir   --dPred=$dPred_label_dir





data_pref='../visiumhd_heg/CRC-P2/CRC-P2-'
output='../visiumhd_heg/CRC-P2/clusters_hipt_raw/'
mask_pref='hs'
bsub -q mingyao_normal -n 16 \
    -e logs/cluster_hist.err \
    bash run_cluster_hist_raw.sh ${data_pref} ${output} ${mask_pref} \
oTrue_pref="../visiumhd_heg/CRC-P2/CRC-P2-"
oPred_pref="../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/"
oPred_method="Imputation"
mask="../visiumhd_heg/CRC-P2/CRC-P2-mask-small-hs.png"
output="../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc3/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test1" --gene_names MYH11
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test2" --gene_names GATA3 ESR1 FOXA1 KRT8 KRT18 KRT19 KRT5 KRT14 KRT17 MKI67 TOP2A CD44 CD24 PROM1 MYH9 LGALS3 DSTN NEAT1 MALAT1 COL1A1 FN1 ATF3 JUN FOS STAT1 PECAM1 VWF CDH5 CD14 CD68 CD163 CSF1R AIF1 LUM DCN CTSW S100B CST1 EMID1 HPX SCUBE2 ERBB2 CCL5 CLEC3A CXCL9 LDHB CST7 PLAT SERPINA1
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test0707" --gene_names CEACAM6 CD24 GPX2 CEACAM5 LAMP1 PRSS23 MYC ATP5F1E AXIN2 FABP1 DAB2 FERMT1 THBS1 LRATD1 HLA-DRA IGFBP7 TUBB ACTA2 PPDPF C1QBP AREG TGFBI CEACAM1 TAGLN TKT IMPDH2 PIGR CDCA7 CTSB KRT23 CXCL3 IFITM1 LCN2 TIMP3 MMP2 GREM1 LGR5 HMGB2 SELENBP1 MYH14 PRDX4 NXPE4 RNF43 STMN1 ASCL2 MUC5B SLC7A5 DES TK1 HHLA2 SOX9 IFI27 RRM2 IGHG3 PPP1R1B ID2 CES2 SLC12A2 TYMS CCL20 SOCS3 SOD3 CLCA1 PERP SCNN1A PI3 IFI6 SEC11C PECAM1 MMP12 MKI67 C3 SLC26A2 IRF8 ODF2L LEFTY1 EGFR GNA11 FKBP11 SERPINA1 RGS5 ANXA1 TRIM29 MUC12 OLFM4 EBPL CKAP4 ALDH1B1 IFITM2 DPYSL3 CMBL MAF PROX1 ETS1 HLA-DPB1 CA2 MMP11 SELENOK SPP1 MZB1 MS4A7 CYBB UBE2C KRTCAP3 REG4 S100P CD68 IL7R MUC17 ITLN1 CREB3L1 FBP1 INHBA RNASE1 CTHRC1 PCLAF KIF23 COL11A1 IL17RB TNFAIP3 VCAN DMBT1 UBD ADH1C SMIM14 MS4A12 SFXN1 LAMC2 TBC1D4 VWF CD14 PLVAP DUOX2 NOTCH3 STAB1 CXCL10 GALNT5 LEF1 SLC6A8 PDGFRA CD2 IER5 GPRIN3 PLXND1 CDK6 LEPROTL1 BMP4 PLPP2 AQP1 CDH5 STXBP6 SLC26A3 ACP5 MRC1 COL17A1 CDHR5 FZD7 ITGAX FOXA3 PTTG1 FCGR3A SDCBP2 GPR183 C1QC TNFRSF25 AZGP1 GPRC5C LOX PSTPIP2 PLCE1 CXCR4 FYB1 RPS4Y1 HDC CFTR APOE CD3E AKR1C3 SULT1B1 COMP REG1A TRAC CEACAM7 GIMAP7 CRYAB TRBC2 CD163 CLU CXCL9 APOC1 RGMB CXCL12 CCL4 PTGER4 SPOCK2 EPHB3 RORA IL1B CD3G CD3D CADPS GZMA MMP9 C2orf88 C3AR1 CA4 CCL5 IL33 IL1RN CD8A KLRB1 IDO1 RUNX1T1 ADRA2A CLCA4 CD83 CCR1 FABP2 FAP SMOC2 BRCA2 DNASE1L3 CPE FRZB RAP1GAP HAVCR2 CD79A MFAP5 DNAJC12 SLPI CHP2 UGT2B17 KRT80 CDKN2B NOSIP C1QA MMP7 AKR7A3 MMP1 AVIL CA1 CST7 LGALS2 CHI3L1 IL2RA ROBO1 BCAS1 MLPH CD177 CD6 PRF1 ARHGAP24 TNFSF13B CTLA4 XCL2 C1QB MMP3 KLK1 RORC REP15 SELENOM DEPP1 RETNLB CD69 TNFRSF17 IL32 LRMP THBS4 RSPO3 RHOV ANPEP RARRES2 CES1 CD5 HEPACAM2 CXCL11 UGT2A3 LYVE1 CD8B GALNT8 S100B KLRC2 KRT86 ANO7 CLEC10A TFF1 DERL3 HES6 IL6 ICOS PBK IL1RL1 FFAR4 ITK ETV1 AFAP1L2 CD274 CEP126 GZMK INSM1 LTB MB LAG3 TIGIT KIT CHGB CCR7 BEST2 TRGV4 RAB3B CPA3 WFDC2 GNLY CD27 MEIS2 FOXP3 L1TD1 ROBO2 TRAT1 NKG7 NEUROD1 SMIM6 SELL TMIGD1 MS4A1 CXCL13 FCN1 ATOH1 PDZK1IP1 REG3A TRBC1 BANK1 MS4A2 ANK2 CD79B ANXA13 MS4A8 NOVA1 SLC29A4 HPGDS RBP4 SCG3 SIT1 SPDEF PDE4C HRCT1 SCG5 S100A12 MAOB CD40LG SLC18A2 CHI3L2 COL19A1 BEST4 SPIB KLRD1 CDK15 GCG CA7 KIF5C B3GNT6 RIIAD1 CHGA SCGN RGS13 DKK1 RAB26 CLEC9A CALB2 GATA2 CXCR3 MCEMP1 BMX CXCR5 ISL1 AQP8 GUCA2B BATF IL22 RFX6 MSLN AGTR1 SH2D7 HTR3E SCG2 PRPH FCRL1 INSL5 FCER2 SVOPL CTSG KRT1 TRPM5 FCRLA IL4I1 PAX5 TCL1A PAX4 KLK7 MT1A VWA5B2 GUCA2A CCL19 OTOP2 CRYBA2 CTSE IL1RAPL1 LILRA4 UCN3 NKX2-2 TTR TRDV1 ARX ABCC8 CMA1 FEV TMEM61 Clonotype1_TRA1 SH2D6 CD7 Clonotype1_TRA2 Clonotype1_TRB
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test0707_2" --gene_names C1QBP CEACAM1 IMPDH2 IFITM1 GREM1 NXPE4
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
for n_clusters in 015 018 020 025 030; do
    hipt_dir='../visiumhd_heg/CRC-P2/clusters_hipt_raw/nclusters'${n_clusters}'/'
    o_dir='../visiumhd_heg/CRC-P2/CRC-P2_train_hipt/clusters_truth/CRC-P2-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/CRC-P2-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../visiumhd_heg/CRC-P2/plot_cluster_aligned_raw_0_100/'
    python plot_cluster_aligned.py --hipt_dir=${hipt_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done


pref="../visiumhd_heg/CRC-P2/CRC-P2-"
output="../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/"
mask=${pref}"mask-small-hs.png"
mask1=${pref}"mask-small-filter_he_qc.png"
python plot_truth.py --pref=${pref} --output=${output} --mask=${mask} 
python plot_truth.py --pref=${pref} --output=${output} --n_top=30 --mask=${mask1} 
python plot_truth.py --pref=${pref} --output=${output} --n_top=30 
python plot_imputed.py --pref=${pref} --output=${output} --n_top=30  --mask=${mask} 
pref="../visiumhd_heg/CRC-P2/CRC-P2-"
output="../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/"
mask=${pref}"mask-small-hs.png"
mask1=${pref}"mask-small-filter_he_qc.png"
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} --gene_names CEACAM6 CD24 GPX2 CEACAM5 LAMP1 PRSS23 MYC ATP5F1E AXIN2 FABP1 DAB2 FERMT1 THBS1 LRATD1 HLA-DRA IGFBP7 TUBB ACTA2 PPDPF C1QBP AREG TGFBI CEACAM1 TAGLN TKT IMPDH2 PIGR CDCA7 CTSB KRT23 CXCL3 IFITM1 LCN2 TIMP3 MMP2 GREM1 LGR5 HMGB2 SELENBP1 MYH14 PRDX4 NXPE4 RNF43 STMN1 ASCL2 MUC5B SLC7A5 DES TK1 HHLA2 SOX9 IFI27 RRM2 IGHG3 PPP1R1B ID2 CES2 SLC12A2 TYMS CCL20 SOCS3 SOD3 CLCA1 PERP SCNN1A PI3 IFI6 SEC11C PECAM1 MMP12 MKI67 C3 SLC26A2 IRF8 ODF2L LEFTY1 EGFR GNA11 FKBP11 SERPINA1 RGS5 ANXA1 TRIM29 MUC12 OLFM4 EBPL CKAP4 ALDH1B1 IFITM2 DPYSL3 CMBL MAF PROX1 ETS1 HLA-DPB1 CA2 MMP11 SELENOK SPP1 MZB1 MS4A7 CYBB UBE2C KRTCAP3 REG4 S100P CD68 IL7R MUC17 ITLN1 CREB3L1 FBP1 INHBA RNASE1 CTHRC1 PCLAF KIF23 COL11A1 IL17RB TNFAIP3 VCAN DMBT1 UBD ADH1C SMIM14 MS4A12 SFXN1 LAMC2 TBC1D4 VWF CD14 PLVAP DUOX2 NOTCH3 STAB1 CXCL10 GALNT5 LEF1 SLC6A8 PDGFRA CD2 IER5 GPRIN3 PLXND1 CDK6 LEPROTL1 BMP4 PLPP2 AQP1 CDH5 STXBP6 SLC26A3 ACP5 MRC1 COL17A1 CDHR5 FZD7 ITGAX FOXA3 PTTG1 FCGR3A SDCBP2 GPR183 C1QC TNFRSF25 AZGP1 GPRC5C LOX PSTPIP2 PLCE1 CXCR4 FYB1 RPS4Y1 HDC CFTR APOE CD3E AKR1C3 SULT1B1 COMP REG1A TRAC CEACAM7 GIMAP7 CRYAB TRBC2 CD163 CLU CXCL9 APOC1 RGMB CXCL12 CCL4 PTGER4 SPOCK2 EPHB3 RORA IL1B CD3G CD3D CADPS GZMA MMP9 C2orf88 C3AR1 CA4 CCL5 IL33 IL1RN CD8A KLRB1 IDO1 RUNX1T1 ADRA2A CLCA4 CD83 CCR1 FABP2 FAP SMOC2 BRCA2 DNASE1L3 CPE FRZB RAP1GAP HAVCR2 CD79A MFAP5 DNAJC12 SLPI CHP2 UGT2B17 KRT80 CDKN2B NOSIP C1QA MMP7 AKR7A3 MMP1 AVIL CA1 CST7 LGALS2 CHI3L1 IL2RA ROBO1 BCAS1 MLPH CD177 CD6 PRF1 ARHGAP24 TNFSF13B CTLA4 XCL2 C1QB MMP3 KLK1 RORC REP15 SELENOM DEPP1 RETNLB CD69 TNFRSF17 IL32 LRMP THBS4 RSPO3 RHOV ANPEP RARRES2 CES1 CD5 HEPACAM2 CXCL11 UGT2A3 LYVE1 CD8B GALNT8 S100B KLRC2 KRT86 ANO7 CLEC10A TFF1 DERL3 HES6 IL6 ICOS PBK IL1RL1 FFAR4 ITK ETV1 AFAP1L2 CD274 CEP126 GZMK INSM1 LTB MB LAG3 TIGIT KIT CHGB CCR7 BEST2 TRGV4 RAB3B CPA3 WFDC2 GNLY CD27 MEIS2 FOXP3 L1TD1 ROBO2 TRAT1 NKG7 NEUROD1 SMIM6 SELL TMIGD1 MS4A1 CXCL13 FCN1 ATOH1 PDZK1IP1 REG3A TRBC1 BANK1 MS4A2 ANK2 CD79B ANXA13 MS4A8 NOVA1 SLC29A4 HPGDS RBP4 SCG3 SIT1 SPDEF PDE4C HRCT1 SCG5 S100A12 MAOB CD40LG SLC18A2 CHI3L2 COL19A1 BEST4 SPIB KLRD1 CDK15 GCG CA7 KIF5C B3GNT6 RIIAD1 CHGA SCGN RGS13 DKK1 RAB26 CLEC9A CALB2 GATA2 CXCR3 MCEMP1 BMX CXCR5 ISL1 AQP8 GUCA2B BATF IL22 RFX6 MSLN AGTR1 SH2D7 HTR3E SCG2 PRPH FCRL1 INSL5 FCER2 SVOPL CTSG KRT1 TRPM5 FCRLA IL4I1 PAX5 TCL1A PAX4 KLK7 MT1A VWA5B2 GUCA2A CCL19 OTOP2 CRYBA2 CTSE IL1RAPL1 LILRA4 UCN3 NKX2-2 TTR TRDV1 ARX ABCC8 CMA1 FEV TMEM61 Clonotype1_TRA1 SH2D6 CD7 Clonotype1_TRA2 Clonotype1_TRB
python plot_truth.py --pref=${pref} --output=${output} --gene_names CEACAM6 CD24 GPX2 CEACAM5 LAMP1 PRSS23 MYC ATP5F1E AXIN2 FABP1 DAB2 FERMT1 THBS1 LRATD1 HLA-DRA IGFBP7 TUBB ACTA2 PPDPF C1QBP AREG TGFBI CEACAM1 TAGLN TKT IMPDH2 PIGR CDCA7 CTSB KRT23 CXCL3 IFITM1 LCN2 TIMP3 MMP2 GREM1 LGR5 HMGB2 SELENBP1 MYH14 PRDX4 NXPE4 RNF43 STMN1 ASCL2 MUC5B SLC7A5 DES TK1 HHLA2 SOX9 IFI27 RRM2 IGHG3 PPP1R1B ID2 CES2 SLC12A2 TYMS CCL20 SOCS3 SOD3 CLCA1 PERP SCNN1A PI3 IFI6 SEC11C PECAM1 MMP12 MKI67 C3 SLC26A2 IRF8 ODF2L LEFTY1 EGFR GNA11 FKBP11 SERPINA1 RGS5 ANXA1 TRIM29 MUC12 OLFM4 EBPL CKAP4 ALDH1B1 IFITM2 DPYSL3 CMBL MAF PROX1 ETS1 HLA-DPB1 CA2 MMP11 SELENOK SPP1 MZB1 MS4A7 CYBB UBE2C KRTCAP3 REG4 S100P CD68 IL7R MUC17 ITLN1 CREB3L1 FBP1 INHBA RNASE1 CTHRC1 PCLAF KIF23 COL11A1 IL17RB TNFAIP3 VCAN DMBT1 UBD ADH1C SMIM14 MS4A12 SFXN1 LAMC2 TBC1D4 VWF CD14 PLVAP DUOX2 NOTCH3 STAB1 CXCL10 GALNT5 LEF1 SLC6A8 PDGFRA CD2 IER5 GPRIN3 PLXND1 CDK6 LEPROTL1 BMP4 PLPP2 AQP1 CDH5 STXBP6 SLC26A3 ACP5 MRC1 COL17A1 CDHR5 FZD7 ITGAX FOXA3 PTTG1 FCGR3A SDCBP2 GPR183 C1QC TNFRSF25 AZGP1 GPRC5C LOX PSTPIP2 PLCE1 CXCR4 FYB1 RPS4Y1 HDC CFTR APOE CD3E AKR1C3 SULT1B1 COMP REG1A TRAC CEACAM7 GIMAP7 CRYAB TRBC2 CD163 CLU CXCL9 APOC1 RGMB CXCL12 CCL4 PTGER4 SPOCK2 EPHB3 RORA IL1B CD3G CD3D CADPS GZMA MMP9 C2orf88 C3AR1 CA4 CCL5 IL33 IL1RN CD8A KLRB1 IDO1 RUNX1T1 ADRA2A CLCA4 CD83 CCR1 FABP2 FAP SMOC2 BRCA2 DNASE1L3 CPE FRZB RAP1GAP HAVCR2 CD79A MFAP5 DNAJC12 SLPI CHP2 UGT2B17 KRT80 CDKN2B NOSIP C1QA MMP7 AKR7A3 MMP1 AVIL CA1 CST7 LGALS2 CHI3L1 IL2RA ROBO1 BCAS1 MLPH CD177 CD6 PRF1 ARHGAP24 TNFSF13B CTLA4 XCL2 C1QB MMP3 KLK1 RORC REP15 SELENOM DEPP1 RETNLB CD69 TNFRSF17 IL32 LRMP THBS4 RSPO3 RHOV ANPEP RARRES2 CES1 CD5 HEPACAM2 CXCL11 UGT2A3 LYVE1 CD8B GALNT8 S100B KLRC2 KRT86 ANO7 CLEC10A TFF1 DERL3 HES6 IL6 ICOS PBK IL1RL1 FFAR4 ITK ETV1 AFAP1L2 CD274 CEP126 GZMK INSM1 LTB MB LAG3 TIGIT KIT CHGB CCR7 BEST2 TRGV4 RAB3B CPA3 WFDC2 GNLY CD27 MEIS2 FOXP3 L1TD1 ROBO2 TRAT1 NKG7 NEUROD1 SMIM6 SELL TMIGD1 MS4A1 CXCL13 FCN1 ATOH1 PDZK1IP1 REG3A TRBC1 BANK1 MS4A2 ANK2 CD79B ANXA13 MS4A8 NOVA1 SLC29A4 HPGDS RBP4 SCG3 SIT1 SPDEF PDE4C HRCT1 SCG5 S100A12 MAOB CD40LG SLC18A2 CHI3L2 COL19A1 BEST4 SPIB KLRD1 CDK15 GCG CA7 KIF5C B3GNT6 RIIAD1 CHGA SCGN RGS13 DKK1 RAB26 CLEC9A CALB2 GATA2 CXCR3 MCEMP1 BMX CXCR5 ISL1 AQP8 GUCA2B BATF IL22 RFX6 MSLN AGTR1 SH2D7 HTR3E SCG2 PRPH FCRL1 INSL5 FCER2 SVOPL CTSG KRT1 TRPM5 FCRLA IL4I1 PAX5 TCL1A PAX4 KLK7 MT1A VWA5B2 GUCA2A CCL19 OTOP2 CRYBA2 CTSE IL1RAPL1 LILRA4 UCN3 NKX2-2 TTR TRDV1 ARX ABCC8 CMA1 FEV TMEM61 Clonotype1_TRA1 SH2D6 CD7 Clonotype1_TRA2 Clonotype1_TRB
for n_clusters in 015 018 020 025 030; do
    hipt_dir='../visiumhd_heg/CRC-P2/clusters_hipt_raw/nclusters'${n_clusters}'/'
    o_dir='../visiumhd_heg/CRC-P2/CRC-P2_train_hipt/clusters_truth/CRC-P2-mask-small-hs/0-20_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/CRC-P2-mask-small-hs/0-20_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../visiumhd_heg/CRC-P2/plot_cluster_aligned_raw_0_20/'
    python plot_cluster_aligned.py --hipt_dir=${p_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done
for n_clusters in 015 018 020 025 030; do
    hipt_dir='../visiumhd_heg/CRC-P2/clusters_hipt_raw/nclusters'${n_clusters}'/'
    o_dir='../visiumhd_heg/CRC-P2/CRC-P2_train_hipt/clusters_truth/CRC-P2-mask-small-hs/20-40_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/CRC-P2-mask-small-hs/20-40_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../visiumhd_heg/CRC-P2/plot_cluster_aligned_raw_20-40/'
    python plot_cluster_aligned.py --hipt_dir=${p_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done
for n_clusters in 015 018 020 025 030; do
    hipt_dir='../visiumhd_heg/CRC-P2/clusters_hipt_raw/nclusters'${n_clusters}'/'
    o_dir='../visiumhd_heg/CRC-P2/CRC-P2_train_hipt/clusters_truth/CRC-P2-mask-small-hs/40-60_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/CRC-P2-mask-small-hs/40-60_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../visiumhd_heg/CRC-P2/plot_cluster_aligned_raw_40-60/'
    python plot_cluster_aligned.py --hipt_dir=${p_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done





pref="../visiumhd_heg/CRC-P1/CRC-P1-"
output="../visiumhd_heg/CRC-P1/CRC-P2_P1_predict_hipt_raw/filter_he_qc/"
mask=${pref}"mask-small-hs.png"
python plot_truth.py --pref=${pref} --output=${output} --mask=${mask} 

pref="../visiumhd_heg/CRC-P1full/CRC-P1full-"
output="../visiumhd_heg/CRC-P1full/CRC-P2_P1full_hipt_predict_raw/filter_he_qc/"
mask=${pref}"mask-small-hs.png"
python plot_truth.py --pref=${pref} --output=${output} --mask=${mask} --n_top=30 --overlay
python plot_truth.py --pref=${pref} --output=${output} --gene_names COL1A1 --overlay
python add_border.py "../visiumhd_heg/CRC-P1full/CRC-P2_P1full_hipt_predict_raw/filter_he_qc/gene_expression_plot_truth/no_mask/overlay/top_25/COL1A1.png" --border_size=7
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} --n_top=30 --overlay
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} --gene_names COL1A1 --overlay
python add_border.py "../visiumhd_heg/CRC-P1full/CRC-P2_P1full_hipt_predict_raw/filter_he_qc/gene_expression_plot_imputed/CRC-P1full-mask-small-hs/overlay/top_25/COL1A1.png" --border_size=7


pref="../visiumhd_heg/CRC-P2full/CRC-P2full-"
output="../visiumhd_heg/CRC-P2full/CRC-P2_P2full_hipt_predict_raw/filter_he_qc/"
mask=${pref}"mask-small-hs.png"
python plot_truth.py --pref=${pref} --output=${output} --mask=${mask} --n_top=3000 --overlay
python plot_truth.py --pref=${pref} --output=${output} --gene_names COL1A1 PIGR --overlay
python add_border.py "../visiumhd_heg/CRC-P2full/CRC-P2_P2full_hipt_predict_raw/filter_he_qc/gene_expression_plot_truth/no_mask/overlay/top_25/COL1A1.png" --border_size=7
python add_border.py "../visiumhd_heg/CRC-P2full/CRC-P2_P2full_hipt_predict_raw/filter_he_qc/gene_expression_plot_truth/no_mask/overlay/top_25/PIGR.png" --border_size=7
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} 
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} --gene_names COL1A1 PIGR --overlay
python add_border.py "../visiumhd_heg/CRC-P2full/CRC-P2_P2full_hipt_predict_raw/filter_he_qc/gene_expression_plot_imputed/CRC-P2full-mask-small-hs/overlay/top_25/COL1A1.png" --border_size=7
python add_border.py "../visiumhd_heg/CRC-P2full/CRC-P2_P2full_hipt_predict_raw/filter_he_qc/gene_expression_plot_imputed/CRC-P2full-mask-small-hs/overlay/top_25/PIGR.png" --border_size=7



pref="../visiumhd_heg/CRC-P5full/CRC-P5full-"
output="../visiumhd_heg/CRC-P5full/CRC-P2_P5full_hipt_predict_raw/filter_he_qc/"
mask=${pref}"mask-small-hs.png"
python plot_truth.py --pref=${pref} --output=${output} --mask=${mask} --n_top=30 --overlay
python plot_truth.py --pref=${pref} --output=${output} --gene_names COL1A1 --overlay
python add_border.py "../visiumhd_heg/CRC-P5full/CRC-P2_P5full_hipt_predict_raw/filter_he_qc/gene_expression_plot_truth/no_mask/overlay/top_25/COL1A1.png" --border_size=7
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} --n_top=30 --overlay
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} --gene_names COL1A1 --overlay
python add_border.py "../visiumhd_heg/CRC-P5full/CRC-P2_P5full_hipt_predict_raw/filter_he_qc/gene_expression_plot_imputed/CRC-P5full-mask-small-hs/overlay/top_25/COL1A1.png" --border_size=7




data_pref='../xenium/Gastric/Gastric-'
output='../xenium/Gastric/clusters_hipt_raw/'
mask_pref='hs'
bsub -q mingyao_normal -n 16 \
    -e logs/cluster_hist.err \
    bash run_cluster_hist.sh ${data_pref} ${output} ${mask_pref} \
for n_clusters in 015 018 020 025 030; do
    hipt_dir='../xenium/Gastric/clusters_hipt/nclusters'${n_clusters}
    o_dir='../xenium/Gastric/Gastric_train_hipt/clusters_truth/Gastric-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../xenium/Gastric/Gastric_self_predict_hipt/filter_he_qc/clusters_pred_gene_pickle/Gastric-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../xenium/Gastric/plot_cluster_aligned_0_100/'
    python plot_cluster_aligned.py --hipt_dir=${hipt_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done
mask_dir='../xenium/Gastric/Gastric_self_predict_hipt/filter_he_qc/clusters_pred_gene_pickle/Gastric-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters018/masks/'
python add_border.py ${mask_dir} --border_size=7



data_pref='../xenium/CRC-P1/CRC-P1-'
output='../xenium/CRC-P1/clusters_hipt/'
mask_pref='hs'
bsub -q mingyao_normal -n 16 \
    -e logs/cluster_hist.err \
    bash run_cluster_hist.sh ${data_pref} ${output} ${mask_pref} \
for n_clusters in 015 018 020 025 030; do
    hipt_dir='../xenium/CRC-P1/clusters_hipt/nclusters'${n_clusters}
    o_dir='../xenium/CRC-P1/CRC-P1_train_hipt/clusters_truth/CRC-P1-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../xenium/CRC-P1/CRC-P1_self_predict_hipt/filter_he_qc/clusters_pred_gene_pickle/CRC-P1-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../xenium/CRC-P1/plot_cluster_aligned_0_100/'
    python plot_cluster_aligned.py --hipt_dir=${hipt_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done
oTrue_pref="../xenium/CRC-P1/CRC-P1-"
oPred_pref="../xenium/CRC-P1/CRC-P2_P1_predict_hipt_raw/filter_he_qc/"
oPred_method="Imputation"
mask="../xenium/CRC-P1/CRC-P1-mask-small-hs.png"
output="../xenium/CRC-P1/CRC-P2_P1_predict_hipt_raw/filter_he_qc3/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=500
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test1" --gene_names MYH11
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test1" --gene_names PIGR
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test2" --gene_names GATA3 ESR1 FOXA1 KRT8 KRT18 KRT19 KRT5 KRT14 KRT17 MKI67 TOP2A CD44 CD24 PROM1 MYH9 LGALS3 DSTN NEAT1 MALAT1 COL1A1 FN1 ATF3 JUN FOS STAT1 PECAM1 VWF CDH5 CD14 CD68 CD163 CSF1R AIF1 LUM DCN CTSW S100B CST1 EMID1 HPX SCUBE2 ERBB2 CCL5 CLEC3A CXCL9 LDHB CST7 PLAT SERPINA1
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
pref="../xenium/CRC-P1/CRC-P1-"
output="../xenium/CRC-P1/CRC-P2_P1_predict_hipt_raw/filter_he_qc/"
mask=${pref}"mask-small-hs.png"
python plot_truth.py --pref=${pref} --output=${output} --mask=${mask} 
python plot_truth.py --pref=${pref} --output=${output}  --gene_names CEACAM1 --overlay
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} 
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} --gene_names CEACAM1 --overlay



data_pref='../xenium/CRC-P2/CRC-P2-'
output='../xenium/CRC-P2/clusters_hipt/'
mask_pref='hs'
bsub -q mingyao_normal -n 16 \
    -e logs/cluster_hist.err \
    bash run_cluster_hist.sh ${data_pref} ${output} ${mask_pref} \
oTrue_pref="../xenium/CRC-P2/CRC-P2-"
oPred_pref="../xenium/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/"
oPred_method="Imputation"
mask="../xenium/CRC-P2/CRC-P2-mask-small-hs.png"
output="../xenium/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc2/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=500
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test1" --gene_names MYH11
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test2" --gene_names GATA3 ESR1 FOXA1 KRT8 KRT18 KRT19 KRT5 KRT14 KRT17 MKI67 TOP2A CD44 CD24 PROM1 MYH9 LGALS3 DSTN NEAT1 MALAT1 COL1A1 FN1 ATF3 JUN FOS STAT1 PECAM1 VWF CDH5 CD14 CD68 CD163 CSF1R AIF1 LUM DCN CTSW S100B CST1 EMID1 HPX SCUBE2 ERBB2 CCL5 CLEC3A CXCL9 LDHB CST7 PLAT SERPINA1
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test0707" --gene_names C1QBP CEACAM1 IMPDH2 IFITM1 GREM1 NXPE4
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
for n_clusters in 010 012 015; do
    hipt_dir='../xenium/CRC-P2/clusters_hipt/nclusters'${n_clusters}
    o_dir='../xenium/CRC-P2/CRC-P2_train_hipt/clusters_truth/CRC-P2-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../xenium/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/CRC-P2-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../xenium/CRC-P2/plot_cluster_aligned_raw_0_100/'
    python plot_cluster_aligned.py --hipt_dir=${hipt_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done
pref="../xenium/CRC-P2/CRC-P2-"
output="../xenium/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/"
mask=${pref}"mask-small-hs.png"
python plot_truth.py --pref=${pref} --output=${output} --overlay
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} --overlay


for n_clusters in 015 018 020 025 030; do
    hipt_dir='../xenium/CRC-P5/clusters_hipt/nclusters'${n_clusters}
    o_dir='../xenium/CRC-P5/CRC-P5_train_hipt/clusters_truth/CRC-P1-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../xenium/CRC-P5/CRC-P2_P5_predict_hipt/filter_he_qc/clusters_pred_gene_pickle/CRC-P1-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../xenium/CRC-P5/plot_cluster_aligned_0_100/'
    python plot_cluster_aligned.py --hipt_dir=${hipt_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done
for n_clusters in 010 012 015; do
    hipt_dir='../xenium/CRC-P5/clusters_hipt/nclusters'${n_clusters}
    o_dir='../xenium/CRC-P5/CRC-P5_train_hipt/clusters_truth/CRC-P1-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../xenium/CRC-P5/CRC-P5_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/CRC-P5-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../xenium/CRC-P5/plot_cluster_aligned_raw_0_100/'
    python plot_cluster_aligned.py --hipt_dir=${hipt_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done


oTrue_pref="../xenium/CRC-P1/CRC-P1-"
oPred_pref="../xenium/CRC-P1/CRC-P2_P1_predict_hipt_raw/filter_he_qc/"
oPred_method="Imputation"
mask="../xenium/CRC-P1/CRC-P1-mask-small-hs.png"
output="../xenium/CRC-P1/CRC-P2_P1_predict_hipt_raw/filter_he_qc2/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=500
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test1" --gene_names MYH11
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test2" --gene_names GATA3 ESR1 FOXA1 KRT8 KRT18 KRT19 KRT5 KRT14 KRT17 MKI67 TOP2A CD44 CD24 PROM1 MYH9 LGALS3 DSTN NEAT1 MALAT1 COL1A1 FN1 ATF3 JUN FOS STAT1 PECAM1 VWF CDH5 CD14 CD68 CD163 CSF1R AIF1 LUM DCN CTSW S100B CST1 EMID1 HPX SCUBE2 ERBB2 CCL5 CLEC3A CXCL9 LDHB CST7 PLAT SERPINA1
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
for n_clusters in 010 012 015; do
    hipt_dir='../xenium/CRC-P1/clusters_hipt/nclusters'${n_clusters}
    o_dir='../xenium/CRC-P1/CRC-P1_train_hipt/clusters_truth/CRC-P1-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../xenium/CRC-P1/CRC-P1_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/CRC-P1-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../xenium/CRC-P1/plot_cluster_aligned_raw_0_100/'
    python plot_cluster_aligned.py --hipt_dir=${hipt_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done



oTrue_pref="../xenium/CRC-P5/CRC-P5-"
oPred_pref="../xenium/CRC-P5/CRC-P2_P5_predict_hipt_raw/filter_he_qc/"
oPred_method="Imputation"
mask="../xenium/CRC-P5/CRC-P5-mask-small-hs.png"
output="../xenium/CRC-P5/CRC-P2_P5_predict_hipt_raw/filter_he_qc3/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test1" --gene_names PIGR
pref="../xenium/CRC-P5/CRC-P5-"
output="../xenium/CRC-P5/CRC-P2_P5_predict_hipt_raw/filter_he_qc/"
mask=${pref}"mask-small-hs.png"
python plot_truth.py --pref=${pref} --output=${output} --overlay
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} --overlay



oTrue_pref="../xenium/Gastric/Gastric-"
oPred_pref="../xenium/Gastric/Gastric_self_predict_hipt/filter_he_qc/"
oPred_method="Imputation"
mask="../xenium/Gastric/Gastric-mask-small-hs.png"
output="../xenium/Gastric/Gastric_self_predict_hipt/filter_he_qc3/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=1000
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test1" --gene_names ACTA2 CD3D EPCAM KRT20
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
conda activate cuml39
export LD_LIBRARY_PATH=""
python evaluate_ICC.py --data_pref=${oTrue_pref} --output=${oPred_pref} --n_cluster=18





data_pref='../xenium5k/HBC/HBC-'
output='../xenium5k/HBC/clusters_hipt/'
mask_pref='hs'
bsub -q mingyao_normal -n 16 \
    -e logs/cluster_hist.err \
    bash run_cluster_hist_smooth.sh ${data_pref} ${output} ${mask_pref} \
for n_clusters in 005 006 007 008 009 010 012 015 018 020 025 030; do
    hipt_dir='../xenium5k/HBC/clusters_hipt/nclusters'${n_clusters}
    o_dir='../xenium5k/HBC/HBC_train_hipt/clusters_truth/HBC-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../xenium5k/HBC/HBC_self_predict_hipt/filter_he_qc/clusters_pred_gene_pickle/HBC-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../xenium5k/HBC/plot_cluster_aligned_0_100/'
    python plot_cluster_aligned.py --hipt_dir=${hipt_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done
oTrue_pref="../xenium5k/HBC/HBC-"
oPred_pref="../xenium5k/HBC/HBC_self_predict_hipt_raw/filter_he_qc/"
oPred_method="Imputation"
mask="../xenium5k/HBC/HBC-mask-small-hs.png"
output="../xenium5k/HBC/HBC_self_predict_hipt_raw/filter_he_qc2/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=1000
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test1" --gene_names MYH11
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test2" --gene_names GATA3 ESR1 FOXA1 KRT8 KRT18 KRT19 KRT5 KRT14 KRT17 MKI67 TOP2A CD44 CD24 PROM1 MYH9 LGALS3 DSTN NEAT1 MALAT1 COL1A1 FN1 ATF3 JUN FOS STAT1 PECAM1 VWF CDH5 CD14 CD68 CD163 CSF1R AIF1 LUM DCN CTSW S100B CST1 EMID1 HPX SCUBE2 ERBB2 CCL5 CLEC3A CXCL9 LDHB CST7 PLAT SERPINA1
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100

pref="../xenium5k/HBC/HBC-"
output="../xenium5k/HBC/HBC_self_predict_hipt_raw/filter_he_qc/"
mask=${pref}"mask-small-hs.png"
python plot_truth.py --pref=${pref} --output=${output} --n_top=30 --mask=${mask}
python plot_truth.py --pref=${pref} --output=${output} --n_top=300 
python plot_imputed.py --pref=${pref} --output=${output} --n_top=300  --mask=${mask} 
python plot_imputed.py --pref=${pref} --output=${output} --gene_names KRT8 KRT18 KRT5 KRT14 EPCAM ESR1 ERBB2 MKI67 FOXA1 GATA3 TP63 CD3D CD3E CD8A CD8B CD4 CD79A MS4A1 NKG7 GNLY PRF1 CD68 CD163 CSF1R ACTA2 FAP PDGFRA PDGFRB COL1A1 DCN LUM PECAM1 VWF ENG CD34 VEGFA CA9 HIF1A LDHA ENO1 PGK1 --mask=${mask} 
python plot_truth.py --pref=${pref} --output=${output} --gene_names KRT8 KRT18 KRT5 KRT14 EPCAM ESR1 ERBB2 MKI67 FOXA1 GATA3 TP63 CD3D CD3E CD8A CD8B CD4 CD79A MS4A1 NKG7 GNLY PRF1 CD68 CD163 CSF1R ACTA2 FAP PDGFRA PDGFRB COL1A1 DCN LUM PECAM1 VWF ENG CD34 VEGFA CA9 HIF1A LDHA ENO1 PGK1








data_pref='../cosmx/HK3039/HK3039-'
output='../cosmx/HK3039/clusters_hipt_raw/'
mask_pref='hs'
bsub -q mingyao_normal -n 16 \
    -e logs/cluster_hist_raw.err \
    bash run_cluster_hist_raw.sh ${data_pref} ${output} ${mask_pref} \
data_pref='../cosmx/HK3039/HK3039-'
output='../cosmx/HK3039/clusters_hipt/'
bsub -q mingyao_normal -n 16 \
    -e logs/cluster_hist_smooth.err \
    bash run_cluster_hist_smooth.sh ${data_pref} ${output} ${mask_pref} \

oTrue_pref="../cosmx/HK3039/HK3039-"
oPred_pref="../cosmx/HK3039/HK3039_self_predict_hipt_raw/filter_he_qc/"
oPred_method="Imputation"
mask="../cosmx/HK3039/HK3039-mask-small-hs.png"
output="../cosmx/HK3039/HK3039_self_predict_hipt_raw/filter_he_qc2/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=1000
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test1" --gene_names MYH11
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test2" --gene_names GATA3 ESR1 FOXA1 KRT8 KRT18 KRT19 KRT5 KRT14 KRT17 MKI67 TOP2A CD44 CD24 PROM1 MYH9 LGALS3 DSTN NEAT1 MALAT1 COL1A1 FN1 ATF3 JUN FOS STAT1 PECAM1 VWF CDH5 CD14 CD68 CD163 CSF1R AIF1 LUM DCN CTSW S100B CST1 EMID1 HPX SCUBE2 ERBB2 CCL5 CLEC3A CXCL9 LDHB CST7 PLAT SERPINA1
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
pref="../cosmx/HK3039/HK3039-"
output="../cosmx/HK3039/HK3039_self_predict_hipt_raw/filter_he_qc/"
mask=${pref}"mask-small-hs.png"
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} --overlay
python plot_truth.py --pref=${pref} --output=${output} --overlay
for n_clusters in 010 011 012 013 014 015 018 020 025 030; do
    hipt_dir='../cosmx/HK3039/clusters_hipt_smooth/nclusters'${n_clusters}
    o_dir='../cosmx/HK3039/HK3039_train_hipt_raw/clusters_truth/HK3039-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../cosmx/HK3039/HK3039_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/HK3039-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../cosmx/HK3039/plot_cluster_aligned_0_100_ignore_hipt/'
    python plot_cluster_aligned.py --hipt_dir=${p_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done
for n_clusters in 010 011 012 013 014 015 018 020 025 030; do
    hipt_dir='../cosmx/HK3039/clusters_hipt_smooth/nclusters'${n_clusters}
    o_dir='../cosmx/HK3039/HK3039_train_hipt_raw/clusters_truth/HK3039-mask-small-hs/0-50_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../cosmx/HK3039/HK3039_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/HK3039-mask-small-hs/0-50_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../cosmx/HK3039/plot_cluster_aligned_0_50/'
    python plot_cluster_aligned.py --hipt_dir=${hipt_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done


data_pref='../cosmx/HK2844/HK2844-'
output='../cosmx/HK2844/clusters_hipt_raw/'
mask_pref='hs'
bsub -q mingyao_normal -n 16 \
    -e logs/cluster_hist_raw.err \
    bash run_cluster_hist_raw.sh ${data_pref} ${output} ${mask_pref} \
oTrue_pref="../cosmx/HK2844/HK2844-"
oPred_pref="../cosmx/HK2844/HK2844_self_predict_hipt_raw/filter_he_qc/"
oPred_method="Imputation"
mask="../cosmx/HK2844/HK2844-mask-small-hs.png"
output="../cosmx/HK2844/HK2844_self_predict_hipt_raw/filter_he_qc2/"
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --start_idx=0 --end_idx=1000
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test1" --gene_names MYH11
python plot_comparison_downsampled_xenium.py --oTrue_pref=${oTrue_pref} --oPred_pref=${oPred_pref} --oPred_method=${oPred_method}  --output=${output} --mask=${mask} --cell_type="test2" --gene_names GATA3 ESR1 FOXA1 KRT8 KRT18 KRT19 KRT5 KRT14 KRT17 MKI67 TOP2A CD44 CD24 PROM1 MYH9 LGALS3 DSTN NEAT1 MALAT1 COL1A1 FN1 ATF3 JUN FOS STAT1 PECAM1 VWF CDH5 CD14 CD68 CD163 CSF1R AIF1 LUM DCN CTSW S100B CST1 EMID1 HPX SCUBE2 ERBB2 CCL5 CLEC3A CXCL9 LDHB CST7 PLAT SERPINA1
python plot_violin_from_pickle.py --output=${output} --mask=${mask} --start_idx=0 --end_idx=100
pref="../cosmx/HK2844/HK2844-"
output="../cosmx/HK2844/HK2844_self_predict_hipt_raw/filter_he_qc/"
mask=${pref}"mask-small-hs.png"
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} --overlay
python plot_truth.py --pref=${pref} --output=${output} --overlay
for n_clusters in  015 018 020 025 030; do
    hipt_dir='../cosmx/HK2844/clusters_hipt_raw/nclusters'${n_clusters}
    o_dir='../cosmx/HK2844/HK2844_train_hipt_raw/clusters_truth/HK2844-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    p_dir='../cosmx/HK2844/HK2844_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle/HK2844-mask-small-hs/0-100_filtersizeNone_minclsizeNone_nclusters'${n_clusters}'/'
    output='../cosmx/HK2844/plot_cluster_aligned_raw_0_100/'
    python plot_cluster_aligned.py --hipt_dir=${hipt_dir} --o_dir=${o_dir} --p_dir=${p_dir} --output=${output} --n_cl=${n_clusters}
done







## cluster image embeddings, align the labels, for cluster=018, 020
embs='../xenium/Gastric/Gastric-embeddings-hipt-smooth.pickle'
output='../xenium/Gastric/clusters_hipt/'
mask='../xenium/Gastric/Gastric-mask-small-hs.png'
python cluster_hist.py --embeddings=${embs} --output=${output} --n_clusters=18 --mask=${mask} 
python cluster_hist.py --embeddings=${embs} --output=${output} --n_clusters=20 --mask=${mask} 

conda activate uni
export LD_LIBRARY_PATH=""
python UNI_V5/UNI_16_16_pkl.py --read_path ../xenium/Gastric/Gastric- --save_dir ../xenium/Gastric/Gastric- --device cuda:0


python downsampling.py --data_pref="../visiumhd_heg/CRC-P2/CRC-P2-" --out_pref="../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-" --mean=0.1 --sd=0.03
cp ../visiumhd_heg/CRC-P2/CRC-P2-he.jpg ../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-he.jpg
cp ../visiumhd_heg/CRC-P2/CRC-P2-cnts.parquet ../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-cnts.parquet
cp ../visiumhd_heg/CRC-P2/CRC-P2-locs.parquet ../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-locs.parquet
cp ../visiumhd_heg/CRC-P2/CRC-P2-radius.txt ../visiumhd_heg/CRC-P2-downsampled/CRC-P2-downsampled-radius.txt

















from scipy.cluster.hierarchy import fcluster
import numpy as np

def find_distance_threshold_for_n_clusters(Z, target_clusters=20, 
                                            t_min=0.1, t_max=100.0, 
                                            max_iter=50, tol=1e-2, verbose=False):
    """
    Binary search to find distance threshold that gives exactly `target_clusters` using fcluster.
    
    Parameters:
        Z (ndarray): linkage matrix
        target_clusters (int): desired number of clusters
        t_min, t_max (float): range of distance thresholds to search
        max_iter (int): max binary search iterations
        tol (float): minimum step size before stopping
        verbose (bool): print search progress

    Returns:
        labels (ndarray): cluster labels for each point
        best_t (float): the distance threshold used
    """
    for i in range(max_iter):
        t = (t_min + t_max) / 2
        labels = fcluster(Z, t=t, criterion='distance')
        n_clusters = len(np.unique(labels))
        if verbose:
            print(f"Iter {i}: t={t:.4f}, clusters={n_clusters}")
        if n_clusters == target_clusters:
            return labels, t
        elif n_clusters < target_clusters:
            t_max = t
        else:
            t_min = t
        if abs(t_max - t_min) < tol:
            break
    # Final attempt in case exact match not found
    labels = fcluster(Z, t=t, criterion='distance')
    return labels, t



from utils import load_mask
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances_argmin
import numpy as np
from scipy.cluster.hierarchy import fcluster
from cluster_basic import plot_label_masks, plot_labels


pref = '../cosmx/HK3039/HK3039-'
dir = '../cosmx/HK3039/HK3039_self_predict_hipt_raw/filter_he_qc/'


pref = "../visiumhd_heg/CRC-P2/CRC-P2-"
dir = "../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/"

pref = "../xenium/Gastric/Gastric-"
dir = "../xenium/Gastric/Gastric_self_predict_hipt/filter_he_qc/"


x = np.load(dir + 'combined_full_gene_pickles.npy')
y = x[:,:,:]
mask = load_mask(f'{pref}mask-small-hs.png')
mask_vec = mask.reshape(-1)
y_vec = y.reshape(-1, y.shape[2])
y_vec2 = y_vec[mask_vec]
y_vec2 = y_vec2.astype(np.float16)


# Step 1: KMeans to reduce data size
kmeans = MiniBatchKMeans(n_clusters=50, random_state=0).fit(y_vec2)
# kmeans = KMeans(n_clusters=50, random_state=0).fit(y_vec2)
centroids = kmeans.cluster_centers_

# Step 2: Hierarchical clustering on centroids
Z = linkage(centroids, method='ward')
def find_distance_threshold_with_merge_iterative(
    Z,
    kmeans_labels,
    y,                      # (n_samples, n_features)
    target_clusters=20,
    min_fraction=0.005,     # 0.5% of total samples
    t_min=0.1,
    t_max=100.0,
    max_iter=50,
    tol=1e-2,
    verbose=False
):
    n_samples = y.shape[0]
    best_labels = None
    best_t = None
    for i in range(max_iter):
        t = (t_min + t_max) / 2
        labels_centroids = fcluster(Z, t=t, criterion='distance')  # hierarchical on centroids
        labels_full = labels_centroids[kmeans_labels]              # project to full data
        labels_full = LabelEncoder().fit_transform(labels_full)
        # Merge small clusters
        unique, counts = np.unique(labels_full, return_counts=True)
        min_count = int(min_fraction * n_samples)
        print(f"the min_count is {min_count}")
        small_clusters = unique[counts < min_count]
        large_clusters = unique[counts >= min_count]
        if verbose:
            print(f"[{i}] t={t:.4f} → {len(unique)} raw clusters, "
                    f"{len(large_clusters)} after merging")
        if len(small_clusters) > 0:
            # Compute mean expression of large clusters
            large_centroids = np.array([y[labels_full == c].mean(axis=0) for c in large_clusters])
            # Reassign small cluster samples
            idx_small = np.isin(labels_full, small_clusters)
            reassigned = pairwise_distances_argmin(y[idx_small], large_centroids)
            labels_full[idx_small] = large_clusters[reassigned]
            # Re-encode final labels to 0-based
            labels_full = LabelEncoder().fit_transform(labels_full)
        final_clusters = len(np.unique(labels_full))
        if final_clusters == target_clusters:
            best_labels = labels_full
            best_t = t
            break
        elif final_clusters < target_clusters:
            t_max = t
        else:
            t_min = t
        if abs(t_max - t_min) < tol:
            best_labels = labels_full
            best_t = t
            break
    # Print final cluster table
    unique, counts = np.unique(best_labels, return_counts=True)
    print(f"\n✅ Final cluster count = {len(unique)}")
    print(f"🔢 Cluster size table:")
    for u, c in zip(unique, counts):
        print(f"Cluster {u:>2}: {c} samples")
    return best_labels, best_t

# After KMeans and linkage
Z = linkage(centroids, method='averge')
labels_final, t_used = find_distance_threshold_with_merge_iterative(
    Z,
    kmeans.labels_,
    y_vec2,
    target_clusters=15,
    min_fraction=0.005,
    verbose=True
)
print("Final # clusters:", len(np.unique(labels_final)))
labels_vec = np.full_like(mask_vec, -1, dtype=int)
labels_vec[mask_vec] = labels_final
labels_refactored = labels_vec.reshape(mask.shape[0], mask.shape[1])
plot_labels(labels_refactored, 'test_labels.png', white_background=True)
plot_label_masks(labels_refactored, 'test_masks/')




prefix="../cosmx/HK3039/HK3039-"
output="../cosmx/HK3039/HK3039_self_predict_hipt_raw/filter_he_qc/"
mask="../cosmx/HK3039/HK3039-mask-small-hs.png"
CUDA_VISIBLE_DEVICES=0 python cluster_cnts_tree.py ${prefix} --output=${output} --mask=${mask}
CUDA_VISIBLE_DEVICES=0 python cluster_pickle_tree.py ${prefix} --output=${output} --mask=${mask}

prefix="../cosmx/HK2844/HK2844-"
output="../cosmx/HK2844/HK2844_self_predict_hipt_raw/filter_he_qc/"
mask="../cosmx/HK2844/HK2844-mask-small-hs.png"
CUDA_VISIBLE_DEVICES=0 python cluster_cnts_tree.py ${prefix} --output=${output} --mask=${mask}
CUDA_VISIBLE_DEVICES=0 python cluster_pickle_tree.py ${prefix} --output=${output} --mask=${mask}


prefix="../visiumhd_heg/CRC-P2/CRC-P2-"
output="../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/"
mask="../visiumhd_heg/CRC-P2/CRC-P2-mask-small-hs.png"
CUDA_VISIBLE_DEVICES=0 python cluster_pickle_tree.py ${prefix} --output=${output} --mask=${mask} --pstart=0 --pend=20
CUDA_VISIBLE_DEVICES=0 python cluster_pickle_tree.py ${prefix} --output=${output} --mask=${mask} --pstart=20 --pend=40
CUDA_VISIBLE_DEVICES=0 python cluster_pickle_tree.py ${prefix} --output=${output} --mask=${mask} --pstart=40 --pend=60

prefix="../visiumhd_heg/CRC-P2/CRC-P2-"
output="../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/"
mask="../visiumhd_heg/CRC-P2/CRC-P2-mask-small-hs.png"
CUDA_VISIBLE_DEVICES=0 python cluster_cnts_tree.py ${prefix} --output=${output} --mask=${mask} --pstart=0 --pend=20
CUDA_VISIBLE_DEVICES=0 python cluster_cnts_tree.py ${prefix} --output=${output} --mask=${mask} --pstart=20 --pend=40
CUDA_VISIBLE_DEVICES=0 python cluster_cnts_tree.py ${prefix} --output=${output} --mask=${mask} --pstart=40 --pend=60







from utils import save_image, load_pickle
from visual import plot_labels
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

p_dir = "../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle_tree/CRC-P2-mask-small-hs/"
o_dir = "../visiumhd_heg/CRC-P2/CRC-P2_self_predict_hipt_raw/filter_he_qc/clusters_truth_tree/CRC-P2-mask-small-hs/"

mode = 'ward'
n_clusters = '015'

p_labels_1_dir = p_dir + f"0-20_filtersizeNone_minclsizeNone_{mode}_nclusters{n_clusters}/"
p_labels_1 = load_pickle(p_labels_1_dir + "labels.pickle")
p_labels_2_dir = p_dir + f"20-40_filtersizeNone_minclsizeNone_{mode}_nclusters{n_clusters}/"
p_labels_2 = load_pickle(p_labels_2_dir + "labels.pickle")
p_labels_3_dir = p_dir + f"40-60_filtersizeNone_minclsizeNone_{mode}_nclusters{n_clusters}/"
p_labels_3 = load_pickle(p_labels_3_dir + "labels.pickle")

o_labels_1_dir = o_dir + f"0-20_filtersizeNone_minclsizeNone_{mode}_nclusters{n_clusters}/"
o_labels_1 = load_pickle(o_labels_1_dir + "labels.pickle")
o_labels_2_dir = o_dir + f"20-40_filtersizeNone_minclsizeNone_{mode}_nclusters{n_clusters}/"
o_labels_2 = load_pickle(o_labels_2_dir + "labels.pickle")
o_labels_3_dir = o_dir + f"40-60_filtersizeNone_minclsizeNone_{mode}_nclusters{n_clusters}/"
o_labels_3 = load_pickle(o_labels_3_dir + "labels.pickle")


p_labels_1_vec = p_labels_1.flatten()
p_mask = (p_labels_1_vec == -1)

p_labels_1_vec2 = p_labels_1_vec[~p_mask]
label_mapping = {old_label: new_label for new_label, old_label in enumerate(pd.Series(p_labels_1_vec2).value_counts().index)}
label_mapping[-1] = -1
p_labels_1_vec2 = pd.Series(p_labels_1_vec2).map(label_mapping).to_numpy()
p_labels_1_out = np.full_like(p_mask, -1, dtype=int)
p_labels_1_out[~p_mask] = p_labels_1_vec2
p_labels_1_out = p_labels_1_out.reshape(p_labels_1.shape[0], p_labels_1.shape[1])
plot_labels(p_labels_1_out, p_labels_1_dir + 'labels_aligned.png')


def get_aligned_labels(cl0, cl1):
    # Compute confusion matrix between cl0 and cl1
    matrix = pd.crosstab(cl0, cl1)
    cost_matrix = -matrix.values  # Hungarian algorithm minimizes cost
    # Find optimal one-to-one mapping
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Create mapping from cl1 to cl0
    cl0_labels = matrix.index.values
    cl1_labels = matrix.columns.values
    mapping = {cl1_labels[col_ind[i]]: cl0_labels[row_ind[i]] for i in range(len(row_ind))}
    # Apply mapping
    new_cl1 = np.array([mapping[label] for label in cl1])
    return new_cl1


p_labels_2_vec = p_labels_2.flatten()
p_labels_2_vec2 = p_labels_2_vec[~p_mask]
p_labels_2_vec2 = get_aligned_labels(cl0=p_labels_1_vec2, cl1=p_labels_2_vec2)
p_labels_2_out = np.full_like(p_mask, -1, dtype=int)
p_labels_2_out[~p_mask] = p_labels_2_vec2
p_labels_2_out = p_labels_2_out.reshape(p_labels_2.shape[0], p_labels_2.shape[1])
plot_labels(p_labels_2_out, p_labels_2_dir + 'labels_aligned.png')

p_labels_3_vec = p_labels_3.flatten()
p_labels_3_vec2 = p_labels_3_vec[~p_mask]
p_labels_3_vec2 = get_aligned_labels(cl0=p_labels_1_vec2, cl1=p_labels_3_vec2)
p_labels_3_out = np.full_like(p_mask, -1, dtype=int)
p_labels_3_out[~p_mask] = p_labels_3_vec2
p_labels_3_out = p_labels_3_out.reshape(p_labels_3.shape[0], p_labels_3.shape[1])
plot_labels(p_labels_3_out, p_labels_3_dir + 'labels_aligned.png')



o_labels_1_vec = o_labels_1.flatten()
o_mask = (p_labels_1_vec == -1) | (o_labels_1_vec == -1)

p_labels_1_vec3 = p_labels_1_out.flatten()[~o_mask]
o_labels_1_vec3 = o_labels_1_vec[~o_mask]
o_labels_1_vec3 = get_aligned_labels(cl0=p_labels_1_vec3, cl1=o_labels_1_vec3)
o_labels_1_out = np.full_like(o_mask, -1, dtype=int)
o_labels_1_out[~o_mask] = o_labels_1_vec3
o_labels_1_out = o_labels_1_out.reshape(o_labels_1.shape[0], o_labels_1.shape[1])
plot_labels(o_labels_1_out, o_labels_1_dir + 'labels_aligned.png')

o_labels_2_vec = o_labels_2.flatten()
o_labels_2_vec3 = o_labels_2_vec[~o_mask]
o_labels_2_vec3 = get_aligned_labels(cl0=p_labels_1_vec3, cl1=o_labels_2_vec3)
o_labels_2_out = np.full_like(o_mask, -1, dtype=int)
o_labels_2_out[~o_mask] = o_labels_2_vec3
o_labels_2_out = o_labels_2_out.reshape(o_labels_2.shape[0], o_labels_2.shape[1])
plot_labels(o_labels_2_out, o_labels_2_dir + 'labels_aligned.png')



o_labels_3_vec = o_labels_3.flatten()
o_labels_3_vec3 = o_labels_3_vec[~o_mask]
o_labels_3_vec3 = get_aligned_labels(cl0=p_labels_1_vec3, cl1=o_labels_3_vec3)
o_labels_3_out = np.full_like(o_mask, -1, dtype=int)
o_labels_3_out[~o_mask] = o_labels_3_vec3
o_labels_3_out = o_labels_3_out.reshape(o_labels_3.shape[0], o_labels_3.shape[1])
plot_labels(o_labels_3_out, o_labels_3_dir + 'labels_aligned.png')









from utils import save_image, load_pickle
from visual import plot_labels
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

p_dir = "../cosmx/HK3039/HK3039_self_predict_hipt_raw/filter_he_qc/clusters_pred_gene_pickle_tree/HK3039-mask-small-hs/"
o_dir = "../cosmx/HK3039/HK3039_self_predict_hipt_raw/filter_he_qc/clusters_truth_tree/HK3039-mask-small-hs/"

mode = 'ward'
n_clusters = '015'

p_labels_1_dir = p_dir + f"0-100_filtersizeNone_minclsizeNone_{mode}_nclusters{n_clusters}/"
p_labels_1 = load_pickle(p_labels_1_dir + "labels.pickle")

o_labels_1_dir = o_dir + f"0-100_filtersizeNone_minclsizeNone_{mode}_nclusters{n_clusters}/"
o_labels_1 = load_pickle(o_labels_1_dir + "labels.pickle")

p_labels_1_vec = p_labels_1.flatten()
p_mask = (p_labels_1_vec == -1)

p_labels_1_vec2 = p_labels_1_vec[~p_mask]
# label_mapping = {old_label: new_label for new_label, old_label in enumerate(pd.Series(p_labels_1_vec2).value_counts().index)}
# label_mapping[-1] = -1
# p_labels_1_vec2 = pd.Series(p_labels_1_vec2).map(label_mapping).to_numpy()
p_labels_1_out = np.full_like(p_mask, -1, dtype=int)
p_labels_1_out[~p_mask] = p_labels_1_vec2
p_labels_1_out = p_labels_1_out.reshape(p_labels_1.shape[0], p_labels_1.shape[1])
plot_labels(p_labels_1_out, p_labels_1_dir + 'labels_aligned.png')
for cl in range(15):
    p_labels_1_rep = p_labels_1_out.copy()
    mask = (p_labels_1_rep != cl)
    p_labels_1_rep[mask] = -1
    plot_labels(p_labels_1_rep, p_labels_1_dir + f'labels_aligned_{cl}.png')



def get_aligned_labels(cl0, cl1):
    # Compute confusion matrix between cl0 and cl1
    matrix = pd.crosstab(cl0, cl1)
    cost_matrix = -matrix.values  # Hungarian algorithm minimizes cost
    # Find optimal one-to-one mapping
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Create mapping from cl1 to cl0
    cl0_labels = matrix.index.values
    cl1_labels = matrix.columns.values
    mapping = {cl1_labels[col_ind[i]]: cl0_labels[row_ind[i]] for i in range(len(row_ind))}
    # Apply mapping
    new_cl1 = np.array([mapping[label] for label in cl1])
    return new_cl1


o_labels_1_vec = o_labels_1.flatten()
o_mask = (p_labels_1_vec == -1) | (o_labels_1_vec == -1)

p_labels_1_vec3 = p_labels_1_out.flatten()[~o_mask]
o_labels_1_vec3 = o_labels_1_vec[~o_mask]
o_labels_1_vec3 = get_aligned_labels(cl0=p_labels_1_vec3, cl1=o_labels_1_vec3)
o_labels_1_out = np.full_like(o_mask, -1, dtype=int)
o_labels_1_out[~o_mask] = o_labels_1_vec3
o_labels_1_out = o_labels_1_out.reshape(o_labels_1.shape[0], o_labels_1.shape[1])
plot_labels(o_labels_1_out, o_labels_1_dir + 'labels_aligned.png')

import pyvips
file_in = "../xenium/Gastric/Gastric-he.jpg"
file_out= "../xenium/Gastric/Gastric-he.tif"
image = pyvips.Image.new_from_file(file_in, access="sequential")
image.tiffsave(file_out, tile=True, pyramid=True, compression="jpeg", bigtiff=True)


