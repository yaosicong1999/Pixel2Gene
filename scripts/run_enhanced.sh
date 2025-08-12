#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/

device="cuda"  # "cuda" or "cpu"
n_genes=3000  # number of most variable genes to impute

python rescale.py ${prefix} --image
python preprocess.py ${prefix} --image
cp "${prefix}he.jpg" "${prefix}ref-he.jpg"

python extract_features.py ${prefix} --device=${device}
cp "${prefix}embeddings-hist.pickle" "${prefix}ref-embeddings-hist.pickle"
if [ ! -f ${prefix}mask-small.png ]; then
    python get_mask.py ${prefix}embeddings-hist.pickle ${prefix}mask-small.png
fi
cp "${prefix}mask-small.png" "${prefix}ref-mask-small.png"

python select_genes.py --n-top=${n_genes} "${prefix}cnts.tsv" "${prefix}gene-names.txt"

python rescale.py ${prefix} --locs --radius
python rescale.py "${prefix}ref-" --locs --radius

python impute.py ${prefix} --n-states=5 --n-jobs=5 --epochs=600 --device=${device} --reference --ref-epochs=600 --ref-n-states=5 --ref-n-jobs=5
python plot_superpixel_ssim.py ${prefix} --mask=${prefix}test-mask-small.png --clip0=95 --clip1=99



