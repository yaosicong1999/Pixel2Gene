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

    gene_names = read_lines(f"{output}predict-gene-names.txt")
    if mask_name is None:
        mask = load_mask(f"{pref}mask-small-hs.png")
    else:
        mask = load_mask(mask_name)  

    combined_path = f"{output}combined_full_gene_pickles.npy"
    if os.path.exists(combined_path):
        print(f"Loading existing combined gene pickles from {combined_path}")
        x = np.load(combined_path)
    else:
        print(f"Creating combined gene pickles and saving to {combined_path}")
        x = np.full((mask.shape[0], mask.shape[1], len(gene_names)), np.nan, dtype=np.float16)    
        for i in tqdm(range(len(gene_names))):
            gene_name = gene_names[i]
            x[:, :, i] = load_pickle(f"{output}cnts-super/{gene_name}.pickle")
        np.save(combined_path, x)  

    out_dir = f"{output}clusters_pred_gene_pickle_tree/{os.path.splitext(os.path.basename(mask_name))[0]}/"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving to: {out_dir}")

    if top_n is not None:
        print(f"Selecting top {top_n} genes based on their names")
        if top_n > len(gene_names):
            raise ValueError(f"top_n ({top_n}) cannot be greater than the number of genes ({len(gene_names)})")
        x = x[:, :, :top_n]
        out_pref = f"{out_dir}top{top_n}_filtersize{filter_size}_minclsize{min_cluster_size}_"
    elif pend != 100 or pstart != 0:
        print(f"Selecting genes from {pstart}% to {pend}% of the total number of genes")
        start = int((pstart / 100) * len(gene_names))
        end = int((pend / 100) * len(gene_names)) 
        print(f"Selected {pstart}% to {pend}% of the genes (No. {start} to No. {end})")        
        x = x[:, :, max(0, start):min(len(gene_names), end)]
        out_pref = f"{out_dir}{pstart}-{pend}_filtersize{filter_size}_minclsize{min_cluster_size}_"
    else:
        print(f"Using all {len(gene_names)} genes")
        out_pref = f"{out_dir}0-100_filtersize{filter_size}_minclsize{min_cluster_size}_"

    x[~mask, :] = np.nan 

    mask_vec = mask.reshape(-1)
    x_vec = x.reshape(-1, x.shape[2])
    x_vec2 = x_vec[mask_vec]
    x_vec2 = x_vec2.astype(np.float16)

    # Step 1: KMeans to reduce data size
    # kmeans = MiniBatchKMeans(n_clusters=100, random_state=0).fit(x_vec2)
    kmeans = KMeans(n_clusters=50, random_state=0).fit(x_vec2)
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
        tol=1e-5,
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
        for n_clusters in [ 8, 10, 12, 15, 18, 20, 25, 30]:
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