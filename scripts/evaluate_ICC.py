import argparse
from time import time
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from utils import load_pickle, save_pickle, sort_labels, load_mask, load_tsv, read_lines, load_image
import numpy as np
import cupy as cp
import time
from cuml.decomposition import IncrementalPCA
from tqdm import tqdm
import numpy as np
import pandas as pd


def cuml_pca_array(arr: np.ndarray, n_components: int = 50, batch_size: int = 512) -> np.ndarray:
    """
    Perform PCA on a numpy array using cuML's IncrementalPCA with progress bars.

    Args:
        arr: np.ndarray of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        batch_size: Batch size for incremental fitting

    Returns:
        np.ndarray of shape (n_samples, n_components) with PCA-transformed data
    """
    n_samples, n_features = arr.shape
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    # First pass: partial_fit with progress bar
    for start in tqdm(range(0, n_samples, batch_size), desc="Fitting PCA"):
        end = min(start + batch_size, n_samples)
        if end - start >= n_components:
            X_batch = cp.array(arr[start:end], dtype=cp.float32)
            ipca.partial_fit(X_batch)
        
    # Second pass: transform with progress bar
    transformed = np.empty((n_samples, n_components), dtype=np.float32)
    for start in tqdm(range(0, n_samples, batch_size), desc="Transforming data"):
        end = min(start + batch_size, n_samples)
        X_batch = cp.array(arr[start:end], dtype=cp.float32)
        transformed[start:end] = ipca.transform(X_batch).get()
    return transformed


def calculate_per_pc_iccs(pcs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Compute ICC values per cluster per PC.
    For each cluster, returns 50 ICCs (one per PC).
    Final output: (n_clusters * n_pcs,) = 1250 values if 25 clusters and 50 PCs.
    Args:
        pcs: np.ndarray of shape (n_samples, 50) - top PCs of one modality
        labels: np.ndarray of shape (n_samples,) - cluster labels
    Returns:
        icc_values: np.ndarray of shape (n_clusters * 50,)
    """
    # Remove unassigned (-1)
    mask = labels != -1
    pcs = pcs[mask]
    labels = labels[mask]
    unique_clusters = np.unique(labels)
    # Compute per-PC cluster means (shape: n_clusters, 50)
    cluster_means = np.array([pcs[labels == c].mean(axis=0) for c in unique_clusters])
    # Between-cluster variance per PC
    between_var = np.var(cluster_means, axis=0, ddof=1)  # shape: (50,)
    icc_values = []
    for c in unique_clusters:
        cluster_data = pcs[labels == c]  # shape: (n_samples_in_cluster, 50)
        if cluster_data.shape[0] < 2:
            continue  # skip too-small clusters
        # Within-cluster variance per PC
        within_var = np.var(cluster_data, axis=0, ddof=1)  # shape: (50,)
        # Per-PC ICCs for this cluster
        icc = (between_var / (between_var + within_var))
        icc[np.isnan(icc) | np.isinf(icc)] = 0.0  # guard against division by 0
        icc_values.extend(icc.tolist())  # 50 ICCs per cluster
    return np.array(icc_values)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ICC for clusters")
    parser.add_argument("--data_pref", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--top_n", type=int, default=None, help="Number of top genes to use")
    parser.add_argument("--n_cluster", type=int)
    parser.add_argument("--n_components", type=int, default=50, help="Number of PCA components to keep")
    return parser.parse_args()

def main():
    args = parse_args()
    data_pref = args.data_pref
    output = args.output
    n_cluster = args.n_cluster
    top_n = args.top_n

    # Load truth data
    gene_names = read_lines(f"{output}/predict-gene-names.txt")

    if os.path.isfile(f"{data_pref}locs.parquet"):
        print(f"The locs file ends with .parquet")
        locs = pd.read_parquet(f"{data_pref}locs.parquet")
    elif os.path.isfile(f"{data_pref}locs.tsv"):
        print(f"The locs file ends with .tsv")
        locs = pd.read_csv(f"{data_pref}locs.tsv", sep='\t', index_col=0)
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
    locs = locs[unique_row_indices, :]

    batch_size = 50000
    max_value = np.max(idx)
    bins = np.arange(0, max_value + batch_size, batch_size)  
    bin_indices = np.digitize(idx, bins) - 1  
    split_bins = {i: idx[bin_indices == i] for i in range(len(bins) - 1)}

    if os.path.isfile(f"{data_pref}cnts.parquet"):
        print(f"The cnts file ends with .parquet")
        cnts_prqt = pq.ParquetFile(f"{data_pref}cnts.parquet")
        batch_counter = 0
        cnts = pd.DataFrame().astype(pd.SparseDtype("int16", fill_value=0))  
        for batch in cnts_prqt.iter_batches(batch_size=batch_size):
            df = batch.to_pandas() 
            df = df.iloc[split_bins[batch_counter]-batch_counter*batch_size, :]
            df = df.astype(pd.SparseDtype("int", fill_value=0))  
            print("now reading batch_counter", batch_counter, "the shape of df being read is", df.shape)
            cnts = pd.concat([cnts, df], ignore_index=True)
            batch_counter = batch_counter + 1
        print(f"The shape of the raw cnts file is {cnts.shape}")
    elif os.path.isfile(f"{data_pref}cnts.tsv"):
        print(f"The cnts file ends with .tsv")
        cnts = pd.read_csv(f"{data_pref}cnts.tsv", sep='\t', index_col=0)
        cnts = cnts.iloc[idx, :]
        print(f"The shape of the raw cnts file is {cnts.shape}")
    else:
        raise ValueError("The cnts file is not found")

    mask = load_mask(f"{data_pref}mask-small-hs.png")

    # Load labels 
    labels_path = os.path.dirname(data_pref) + '/clusters_hipt_raw/nclusters' + f"{n_cluster:03d}"
    labels = load_pickle(labels_path + '/labels.pickle')
    labels = labels[mask]  # Filter by mask

    # Load predicted data
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

    if top_n is not None:
        gene_names = gene_names[:top_n]
        cnts = cnts[gene_names]
        x = x[:, :, :top_n]

    # Handle predicted and truth data
    truth = np.full((mask.shape[0], mask.shape[1], len(gene_names)), 0, dtype=np.float16)
    truth[locs[:,0], locs[:,1]] = cnts.values  # Assuming locs has 'y' and 'x' columns for coordinates
    truth = truth.reshape(-1, truth.shape[-1])  # Reshape to (n_samples, n_genes)
    truth = truth[mask.flatten()]  # Filter by mask 

    predicted = x.reshape(-1, x.shape[-1])  # Reshape to (n_samples, n_genes)
    predicted = predicted[mask.flatten()]  # Filter by mask

        
    if top_n < 50:
        print(f"Warning: Using only the top {top_n} genes for ICC evaluation. This may not be sufficient for robust results.")
        print(f"Skipping PCA as it is not meaningful with fewer than 50 components.")
    else:
        print(f"Using top {top_n} genes for ICC evaluation. Performing PCA on the data towr.")

    # Perform PCA on predicted and truth data
    truth = cuml_pca_array(truth, n_components=50, batch_size=512)
    predicted = cuml_pca_array(predicted, n_components=50, batch_size=512)


    # Calculate ICCs
    icc_scores = {'Observed': calculate_per_pc_iccs(truth, labels),
                    'Imputed': calculate_per_pc_iccs(predicted, labels)}
    if top_n is not None:
        np.save(f"{output}icc_scores_nclusters_{n_cluster}_top{top_n}.npy", icc_scores)
    else:
        np.save(f"{output}icc_scores_nclusters_{n_cluster}.npy", icc_scores)



if __name__ == "__main__":
    print("Starting ICC evaluation...")
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"ICC evaluation completed in {end_time - start_time:.2f} seconds.")
    print("Done.")