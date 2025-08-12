import argparse
from time import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import load_pickle, save_pickle, sort_labels, load_mask, load_tsv, read_lines, load_image
from plot_labels_overlay import plot_overlay
from cluster_basic import preprocess_and_cluster
from utils import load_mask
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances_argmin
import numpy as np
from scipy.cluster.hierarchy import fcluster
from cluster_basic import plot_label_masks, plot_labels
import pyarrow.parquet as pq


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--overlay', action='store_true')
    parser.add_argument('--n-components', type=float, default=None)
    parser.add_argument('--filter-size', type=int, default=None)
    parser.add_argument('--min-cluster-size', type=int, default=None)
    parser.add_argument('--pstart', type=int, default=0, help="Start percentile (0-100)")
    parser.add_argument('--pend', type=int, default=100, help="End percentile (0-100)")
    parser.add_argument('--top_n', type=int, default=None, help="Number of genes to use")
    return parser.parse_args()

def main():
    args = get_args()
    pref = args.prefix
    output = args.output
    mask_name = args.mask
    pstart = args.pstart
    pend = args.pend
    top_n = args.top_n
    filter_size = args.filter_size
    min_cluster_size = args.min_cluster_size

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

    out_dir = f"{output}clusters_truth_tree/{os.path.splitext(os.path.basename(mask_name))[0]}/"
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

    mask = ~np.isnan(x).any(axis=2)
    mask_vec = mask.reshape(-1)
    x_vec = x.reshape(-1, x.shape[2])
    x_vec2 = x_vec[mask_vec]
    contains_nan = np.isnan(x_vec2).any()
    print(contains_nan)
    x_vec2 = x_vec2.astype(np.float16)

    # Step 1: KMeans to reduce data size
    kmeans = MiniBatchKMeans(n_clusters=50, random_state=0).fit(x_vec2)
    # kmeans = KMeans(n_clusters=50, random_state=0).fit(x_vec2)
    centroids = kmeans.cluster_centers_

    # Step 2: Hierarchical clustering on centroids
    def find_distance_threshold_with_merge_iterative(
        Z,
        kmeans_labels,
        y,                      # (n_samples, n_features)
        target_clusters=20,
        min_fraction=0.005,     # 0.5% of total samples
        t_min=0.001,
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


    for method in ['average', 'ward']:
        for n_clusters in [10, 11, 12, 15, 18, 20, 25, 30]:
            Z = linkage(centroids, method=method)
            labels_final, t_used = find_distance_threshold_with_merge_iterative(
                Z,
            kmeans.labels_,
            x_vec2,
            target_clusters=n_clusters,
            min_fraction=0.01,
            verbose=True)
            print("Final # clusters:", len(np.unique(labels_final)))
            labels_vec = np.full_like(mask_vec, -1, dtype=int)
            labels_vec[mask_vec] = labels_final
            labels_refactored = labels_vec.reshape(mask.shape[0], mask.shape[1])
            os.makedirs(f'{out_pref}{method}_nclusters{n_clusters:03}', exist_ok=True)
            save_pickle(labels_refactored, f'{out_pref}{method}_nclusters{n_clusters:03}/labels.pickle')
            plot_labels(labels_refactored, f'{out_pref}{method}_nclusters{n_clusters:03}/labels.png', white_background=True)
            plot_label_masks(labels_refactored, f'{out_pref}{method}_nclusters{n_clusters:03}/masks/')

if __name__ == '__main__':
    main()