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


python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=0 --pend=100 --overlay 

python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=0 --pend=1 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=0 --pend=5 --overlay 

python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=0 --pend=50 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=50 --pend=100 --overlay 

python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=0 --pend=20 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=20 --pend=40 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=40 --pend=60 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=60 --pend=80 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=80 --pend=100 --overlay 

python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=0 --pend=10 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=10 --pend=20 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=20 --pend=30 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=30 --pend=40 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=40 --pend=50 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=50 --pend=60 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=60 --pend=70 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=70 --pend=80 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=80 --pend=90 --overlay 
python cluster_cnts.py ${data_pref} --output=${output_dir} --mask=${mask} --pstart=90 --pend=100 --overlay 

