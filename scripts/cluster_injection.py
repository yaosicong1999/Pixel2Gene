import argparse
from time import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
# from sklearn.neighbors import kneighbors_graph
# from hdbscan import HDBSCAN
# from einops import reduce
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from utils import load_pickle, save_pickle, sort_labels, load_mask, load_tsv, read_lines, load_image
from image import smoothen, upscale
from einops import reduce
from visual import plot_labels, plot_label_masks
from connected_components import (
        relabel_small_connected, cluster_connected)
from reduce_dim import reduce_dim

from plot_labels_overlay import plot_overlay
from cluster_basic import preprocess_and_cluster


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--overlay', type=str, default="True")
    parser.add_argument('--method', type=str, default='km')
    parser.add_argument('--n-clusters', type=int, default=None)
    parser.add_argument('--n-components', type=float, default=None)
    parser.add_argument('--filter-size', type=int, default=None)
    parser.add_argument('--min-cluster-size', type=int, default=None)
    # parser.add_argument('--stride', type=int, default=4)
    # parser.add_argument('--location-weight', type=float, default=None)
    parser.add_argument('--pstart', type=float, default=0, help="Start percentile (0-100)")
    parser.add_argument('--pend', type=float, default=100, help="End percentile (0-100)")
    return parser.parse_args()

def main():
    args = get_args()
    pref = args.prefix
    overlay = args.overlay
    output=  args.output
    pstart = args.pstart
    pend = args.pend
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

    gene_names = read_lines(f"{pref}gene-names.txt")
    start = int((pstart / 100) * len(gene_names))
    end = int((pend / 100) * len(gene_names))    
    gene_names = gene_names[max(0, start):min(len(gene_names), end)]
    cnts = cnts.loc[:, gene_names]
    print(f"The shape of the cnts file after selecting the genes is {cnts.shape}")

    ## this only defaults filtered out duplicated cell locations, not ANY additional mask filtering  
    ## however, we need to load the mask-small-RGB.png to create the matrix of cnts
    mask_rgb = load_mask(f"{pref}mask-small-filter_locs_he_qc.png")
    x = np.full((mask_rgb.shape[0], mask_rgb.shape[1], cnts.shape[1]), np.nan, dtype=np.float32)
    cnts = cnts.astype(np.int32) 
    x[locs[idx][:,0], locs[idx][:,1], :] = cnts.to_numpy()
    print(f"The shape of the cnts matrix after reshaping into 3D is {x.shape}")

    ## injection part:
    print("now performing injection of the imputed values...")
    mask_inject = load_mask(f"{pref}mask-small-pred_he_qc.png")
    for i in tqdm(range(len(gene_names))):
        gene_name = gene_names[i]
        p = load_pickle(f"{output}cnts-super/{gene_name}.pickle")
        x[mask_inject, i] = p[mask_inject]
    
    x = x.transpose(2, 0, 1)
    print(f"The shape of the cnts matrix after transpoising is {x.shape}")
    assert x.shape[1] == mask_inject.shape[0]
    assert x.shape[2] == mask_inject.shape[1]
    assert x.shape[0] == cnts.shape[1]

    n_comp = None if x.shape[0] < 500 else 500
    print("the input x for clustering has a shape of", x.shape)  
    mask_filename = os.path.splitext(os.path.basename(f"{pref}mask-small-RGB.png"))[0]
    out_dir = f"{output}clusters-cnts-injection/{pstart}-{pend}/{mask_filename}/"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving to: {out_dir}")
    print(f"Selected {pstart}% to {pend}% of the genes (No. {start} to No. {end})")
    out_pref = out_dir + "filtersize" + str(filter_size) + '_minclsize' + str(min_cluster_size) + '_'
    print(f"the output directory will be {out_pref}") 
    preprocess_and_cluster(
        x,
        n_components=n_comp,
        filter_size=args.filter_size,
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