#!/bin/bash
set -e
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDNN_WORKSPACE_LIMIT_IN_MB=4096
device="cuda"  # "cuda" or "cpu"
prefix=$1
echo "prefix: $prefix"

if [ ! -f "${prefix}radius.txt" ]; then
    python rescale.py ${prefix} --radius
fi
if [ -f "${prefix}locs-raw.tsv" ]  && [ ! -f "${prefix}locs.tsv" ]; then
    python rescale.py "${prefix}" --locs
fi
if [ -f "${prefix}locs-raw.parquet" ] && [ ! -f "${prefix}locs.parquet" ]; then
    python rescale.py "${prefix}" --locs
fi
if [ ! -f "${prefix}he-scaled.jpg" ] && [ ! -f "${prefix}he.jpg" ]; then
    python rescale.py "${prefix}" --image
fi
if [ -f "${prefix}he-scaled.jpg" ] && [ ! -f "${prefix}he.jpg" ]; then
    python preprocess.py "${prefix}" --image
fi
if [ ! -f "${prefix}mask-small-hs.png" ]; then
    python get_mask_hs.py ${prefix}
fi
if [ ! -f "${prefix}embeddings-hipt-smooth.pickle" ]; then
    python extract_features.py "${prefix}" --device=${device}
fi
if [ ! -f "${prefix}mask-small-filter_he_qc.png" ]; then
    python cluster_hist.py --embeddings="${prefix}embeddings-hipt-smooth.pickle" --output="$(dirname "${prefix}embeddings-hipt-smooth.pickle")/clusters_hipt/" --n_clusters=15 --mask="${prefix}mask-small-hs.png"
    python dist_superpixel.py ${prefix} --n_clusters=15
fi
if [ ! -f "${prefix}gene-names.txt" ] && { [ -f "${prefix}cnts.tsv" ] || [ -f "${prefix}cnts.parquet" ]; }; then
    python select_genes_heg.py "${prefix}"
fi
