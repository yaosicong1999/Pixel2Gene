import argparse
from time import time
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
# from sklearn.neighbors import kneighbors_graph
# from hdbscan import HDBSCAN
# from einops import reduce
import matplotlib.pyplot as plt
from utils import load_pickle, save_pickle, sort_labels, load_mask, load_tsv, read_lines, load_image
from plot_labels_overlay import plot_overlay
from cluster_basic import preprocess_and_cluster


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--overlay', action='store_true')
    parser.add_argument('--method', type=str, default='km')
    parser.add_argument('--n-clusters', type=int, default=None)
    parser.add_argument('--n-components', type=float, default=None)
    parser.add_argument('--filter-size', type=int, default=None)
    parser.add_argument('--min-cluster-size', type=int, default=None)
    # parser.add_argument('--stride', type=int, default=4)
    # parser.add_argument('--location-weight', type=float, default=None)
    parser.add_argument('--pstart', type=int, default=0, help="Start percentile (0-100)")
    parser.add_argument('--pend', type=int, default=100, help="End percentile (0-100)")
    parser.add_argument('--top_n', type=int, default=None, help="Number of genes to use")
    return parser.parse_args()

def main():
    args = get_args()
    pref = args.prefix
    overlay = args.overlay
    output=  args.output
    mask_name = args.mask
    pstart = args.pstart
    pend = args.pend
    top_n = args.top_n
    filter_size = args.filter_size
    n_clusters = args.n_clusters
    min_cluster_size = args.min_cluster_size
    method = args.method
    
    if os.path.isfile(f"{pref}locs.parquet"):
        print(f"The locs file ends with .parquet")
        locs = pd.read_parquet(f"{pref}locs.parquet")
    elif os.path.isfile(f"{pref}locs.tsv"):
        print(f"The locs file ends with .tsv")
        locs = pd.read_csv(f"{pref}locs.tsv", sep='\t', index_col=0)
    else:
        raise ValueError("The locs file is not found")

    factor = 16
    locs = pd.DataFrame(locs)
    locs = locs.astype(float)
    locs = np.stack([locs['y'], locs['x']], -1)
    locs //= factor
    locs = locs.round().astype(int)
    print(f"The locs shape before dropping duplicate is {locs.shape}")
    unique_rows, indices, counts = np.unique(locs, axis=0, return_index=True, return_counts=True)
    unique_row_indices = indices[counts == 1]
    print(f"The locs shape after dropping duplicate will be ({unique_row_indices.__len__()}, {locs.shape[1]})")
    idx = np.sort(unique_row_indices)

    batch_size = 50000
    max_value = np.max(idx)
    bins = np.arange(0, max_value + batch_size, batch_size)  
    bin_indices = np.digitize(idx, bins) - 1  
    split_bins = {i: idx[bin_indices == i] for i in range(len(bins) - 1)}

    if os.path.isfile(f"{pref}cnts.parquet"):
        print(f"The cnts file ends with .parquet")
        cnts_prqt = pq.ParquetFile(f"{pref}cnts.parquet")
        batch_counter = 0
        cnts = pd.DataFrame().astype(pd.SparseDtype("int", fill_value=0))  
        for batch in cnts_prqt.iter_batches(batch_size=batch_size):
            df = batch.to_pandas() 
            df = df.iloc[split_bins[batch_counter]-batch_counter*batch_size, :]
            df = df.astype(pd.SparseDtype("int", fill_value=0))  
            print("now reading batch_counter", batch_counter, "the shape of df being read is", df.shape)
            cnts = pd.concat([cnts, df], ignore_index=True)
            batch_counter = batch_counter + 1
        print(f"The shape of the raw cnts file is {cnts.shape}")
    elif os.path.isfile(f"{pref}cnts.tsv"):
        print(f"The cnts file ends with .tsv")
        cnts = pd.read_csv(f"{pref}cnts.tsv", sep='\t', index_col=0)
        cnts = cnts.iloc[idx, :]
        print(f"The shape of the raw cnts file is {cnts.shape}")
    else:
        raise ValueError("The cnts file is not found")

    if mask_name is None:
        mask_name = f"{pref}mask-small-hs.png"
    mask_input = load_mask(mask_name)

    out_dir = f"{output}clusters_truth/{os.path.splitext(os.path.basename(mask_name))[0]}/"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving to: {out_dir}")

    gene_names = read_lines(f"{pref}gene-names.txt")
    if top_n is not None:
        print(f"Selecting top {top_n} genes based on their names")
        if top_n > len(gene_names):
            raise ValueError(f"top_n ({top_n}) cannot be greater than the number of genes ({len(gene_names)})")
        gene_names = gene_names[:top_n]
        cnts = cnts.loc[:, gene_names]
        print(f"The shape of the cnts file after selecting the genes is {cnts.shape}")
        out_pref = f"{out_dir}top{top_n}_filtersize{filter_size}_minclsize{min_cluster_size}_"
    elif pend != 100 or pstart != 0:
        start = int((pstart / 100) * len(gene_names))
        end = int((pend / 100) * len(gene_names))    
        print(f"Selected {pstart}% to {pend}% of the genes (No. {start} to No. {end})")
        gene_names = gene_names[max(0, start):min(len(gene_names), end)]
        cnts = cnts.loc[:, gene_names]
        print(f"The shape of the cnts file after selecting the genes is {cnts.shape}")
        out_pref = f"{out_dir}{pstart}-{pend}_filtersize{filter_size}_minclsize{min_cluster_size}_"
    else:
        print(f"Using all {len(gene_names)} genes")
        print(f"The shape of the cnts file after selecting the genes is {cnts.shape}")
        out_pref = f"{out_dir}0-100_filtersize{filter_size}_minclsize{min_cluster_size}_"
    

    x = np.full((mask_input.shape[0], mask_input.shape[1], cnts.shape[1]), np.nan, dtype=np.float32)
    cnts = cnts.astype(np.int32) 
    x[locs[idx][:,0], locs[idx][:,1], :] = cnts.to_numpy()
    print(f"The shape of the cnts matrix after reshaping into 3D is {x.shape}")

    x[~mask_input, :] = np.nan
    x = x.transpose(2, 0, 1)
    print(f"The shape of the cnts matrix after transpoising is {x.shape}")
    assert x.shape[1] == mask_input.shape[0]
    assert x.shape[2] == mask_input.shape[1]
    assert x.shape[0] == cnts.shape[1]

    n_comp = None if x.shape[0] < 500 else 500
    print("the input x for clustering has a shape of", x.shape)    

    preprocess_and_cluster(
        x,
        n_components=n_comp,
        filter_size=filter_size,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        method=method,
        prefix=out_pref,
        mask=None)

    print(f'the clustering is done for {out_pref}...')
    if overlay:
        print("now doing the overlay of the clustered labels on the H&E...")
        parent_dir = os.path.abspath(os.path.dirname(out_pref)) + '/'
        clustered_folders = [entry.path + '/' for entry in os.scandir(parent_dir) if entry.is_dir()]
        plot_overlay(data_pref=pref, label_pref=clustered_folders, save=True)


if __name__ == '__main__':
    main()