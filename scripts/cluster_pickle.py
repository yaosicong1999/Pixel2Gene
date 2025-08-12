import argparse
from time import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
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
    output = args.output
    mask_name = args.mask
    pstart = args.pstart
    pend = args.pend
    top_n = args.top_n
    filter_size = args.filter_size
    n_clusters = args.n_clusters
    min_cluster_size = args.min_cluster_size
    method = args.method
    
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
    
    out_dir = f"{output}clusters_pred_gene_pickle/{os.path.splitext(os.path.basename(mask_name))[0]}/"
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

    x = x.transpose(2, 0, 1)
    print(f"The shape of the cnts matrix after transpoising is {x.shape}")
    assert x.shape[1] == mask.shape[0]
    assert x.shape[2] == mask.shape[1]

    n_comp = None if x.shape[0] < 500 else 500
    print("the input x for clustering has a shape of", x.shape)   
    
    preprocess_and_cluster(
        x,
        n_components=n_comp,
        filter_size=filter_size,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        method=method,
        prefix=out_pref) 

    print(f'the clustering is done for {out_pref}...')
    if overlay:
        print("now doing the overlay of the clustered labels on the H&E...")
        parent_dir = os.path.abspath(os.path.dirname(out_pref)) + '/'
        clustered_folders = [entry.path + '/' for entry in os.scandir(parent_dir) if entry.is_dir()]
        plot_overlay(data_pref=pref, label_pref=clustered_folders, save=True)

if __name__ == '__main__':
    main()
    
# import numpy as np
# from sklearn.decomposition import IncrementalPCA
# from tqdm import tqdm
# import os

# batch_size = 100         # Number of genes per batch
# n_components = 100       # PCA output dimension
# ipca = IncrementalPCA(n_components=n_components)

# flat_mask = mask.flatten()                # (H, W) -> (H*W,)
# n_spots = np.sum(flat_mask)               # Total valid spots
# spot_sample_rate = 1.0                    # Optional: subsample spots if needed
# spot_indices = np.where(flat_mask)[0]

# # Optional: pick a subset of spots to fit PCA
# if spot_sample_rate < 1.0:
#     rng = np.random.default_rng(42)
#     spot_indices = rng.choice(spot_indices, size=int(spot_sample_rate * len(spot_indices)), replace=False)

# # === FITTING STAGE ===
# print("Fitting Incremental PCA on gene batches...")
# for i in tqdm(range(0, len(gene_names), batch_size)):
#     batch_genes = gene_names[i:i+batch_size]
#     batch_data = []
#     for g in batch_genes:
#         arr = load_pickle(f"{output}cnts-super/{g}.pickle", verbose=False)  # shape (H, W)
#         arr_flat = arr.flatten()[spot_indices]               # shape (n_spots,)
#         batch_data.append(arr_flat.astype(np.float32))       # cast to stable dtype
#     batch_matrix = np.stack(batch_data, axis=1)              # shape: (n_spots, batch_size)
#     ipca.partial_fit(batch_matrix)

# # === TRANSFORMATION STAGE ===
# print("Transforming full gene matrix to PCA space...")
# X_pca = []

# for i in tqdm(range(0, len(gene_names), batch_size)):
#     batch_genes = gene_names[i:i+batch_size]
#     batch_data = []
#     for g in batch_genes:
#         arr = load_pickle(f"{output}cnts-super/{g}.pickle", verbose=False)
#         arr_flat = arr.flatten()[flat_mask]
#         batch_data.append(arr_flat.astype(np.float32))
#     batch_matrix = np.stack(batch_data, axis=1)              # shape: (n_spots, batch_size)
#     X_pca.append(ipca.transform(batch_matrix))               # shape: (n_spots, n_components)

# X_pca = np.concatenate(X_pca, axis=1)  # Final shape: (n_spots, total_components)

# X_pca_2d = np.full(mask.shape + (X_pca.shape[1],), np.nan, dtype=np.float32)  # (H, W, n_components)
# X_pca_2d[mask] = X_pca

# np.save(f"{output}X_pca.npy", X_pca_2d)









# import numpy as np
# import cupy as cp
# from cuml.decomposition import IncrementalPCA
# from tqdm import tqdm
# import os

# batch_size = 100         # Number of genes per batch
# n_components = 100       # PCA output dimension
# ipca = IncrementalPCA(n_components=n_components)

# flat_mask = mask.flatten()
# n_spots = np.sum(flat_mask)
# spot_sample_rate = 1.0
# spot_indices = np.where(flat_mask)[0]

# # Optional: subsample spots
# if spot_sample_rate < 1.0:
#     rng = np.random.default_rng(42)
#     spot_indices = rng.choice(spot_indices, size=int(spot_sample_rate * len(spot_indices)), replace=False)

# # === FITTING STAGE ===
# print("Fitting cuML Incremental PCA on gene batches...")
# for i in tqdm(range(0, len(gene_names), batch_size)):
#     batch_genes = gene_names[i:i+batch_size]
    
#     if len(batch_genes) < n_components:
#         continue  # skip last incomplete batch
    
#     batch_data = []
#     for g in batch_genes:
#         arr = load_pickle(f"{output}cnts-super/{g}.pickle", verbose=False)  # shape (H, W)
#         arr_flat = arr.flatten()[spot_indices]               # shape (n_spots,)
#         batch_data.append(arr_flat.astype(np.float32))
#     batch_matrix = np.stack(batch_data, axis=1)              # (n_spots, batch_size)
#     batch_matrix_gpu = cp.asarray(batch_matrix)
#     ipca.partial_fit(batch_matrix_gpu)

# # === TRANSFORMATION STAGE ===
# print("Transforming full gene matrix to PCA space (cuML)...")
# X_pca_parts = []

# for i in tqdm(range(0, len(gene_names), batch_size)):
#     batch_genes = gene_names[i:i+batch_size]
    
#     # Skip last incomplete batch
#     if len(batch_genes) < ipca.n_components:
#         print(f"Skipping transform batch {i}: only {len(batch_genes)} genes")
#         continue
    
#     batch_data = []
#     for g in batch_genes:
#         arr = load_pickle(f"{output}cnts-super/{g}.pickle", verbose=False)
#         arr_flat = arr.flatten()[flat_mask]
#         batch_data.append(arr_flat.astype(np.float32))
    
#     batch_matrix = np.stack(batch_data, axis=1)              # (n_spots, batch_size)
#     batch_matrix_gpu = cp.asarray(batch_matrix)
    
#     # cuML expects n_samples > n_components
#     if batch_matrix_gpu.shape[0] <= ipca.n_components:
#         print(f"Skipping transform batch {i}: not enough spots")
#         continue
    
#     transformed_gpu = ipca.transform(batch_matrix_gpu)       # (n_spots, n_components)
#     X_pca_parts.append(cp.asnumpy(transformed_gpu))          # move back to CPU

# X_pca = np.concatenate(X_pca_parts, axis=1)  # Final shape: (n_spots, total_components)


# # === RECONSTRUCT TO 2D MAP ===
# X_pca_2d = np.full(mask.shape + (X_pca.shape[1],), np.nan, dtype=np.float32)
# X_pca_2d[mask] = X_pca

# np.save(f"{output}X_pca.npy", X_pca_2d)











# n_components = 100
# flat_mask = mask.flatten()
# spot_indices = np.where(flat_mask)[0]
# n_spots = len(spot_indices)
# n_genes = len(gene_names)
# spot_batch_size = 5000

# # === Accumulate full data in spot-wise batches ===
# print("Loading data in spot batches...")

# X_cpu = np.memmap("/tmp/pca_cpu_data.dat", mode="w+", shape=(n_spots, n_genes), dtype=np.float32)

# for i in tqdm(range(0, n_spots, spot_batch_size)):
#     spot_batch_idx = spot_indices[i:i + spot_batch_size]
#     batch_size_actual = len(spot_batch_idx)
#     batch_matrix = np.empty((batch_size_actual, n_genes), dtype=np.float32)
    
#     for j, g in enumerate(gene_names):
#         arr = load_pickle(f"{output}/cnts-super/{g}.pickle", verbose=False)
#         arr_flat = arr.flatten()[spot_batch_idx]
#         batch_matrix[:, j] = arr_flat
        
#     X_cpu[i:i+batch_size_actual] = batch_matrix
    
    
# import numpy as np
# import zarr
# import os

# # Target shape (very large)
# shape = (100, 2000, 17000)
# dtype = np.float16

# # Chunk size (tune to your memory budget)
# chunks = (100, 100, 100)  # ~2 MB per chunk (100*100*100*2 bytes)

# compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=2)

# zarr_path = "large_data.zarr"
# zarr_array = zarr.open(zarr_path, mode='w', shape=shape, dtype=dtype, chunks=chunks, compressor=compressor)

# # Simulate writing chunk by chunk
# for i0 in range(0, shape[0], chunks[0]):
#     i0_end = min(i0 + chunks[0], shape[0])
#     for i1 in range(0, shape[1], chunks[1]):
#         i1_end = min(i1 + chunks[1], shape[1])
#         for i2 in range(0, shape[2], chunks[2]):
#             i2_end = min(i2 + chunks[2], shape[2])
#             # Create random data chunk of appropriate shape
#             chunk_shape = (i0_end - i0, i1_end - i1, i2_end - i2)
#             data_chunk = np.random.rand(*chunk_shape).astype(dtype)
#             # Write chunk to zarr store
#             zarr_array[i0:i0_end, i1:i1_end, i2:i2_end] = data_chunk
#             print(f"Wrote chunk: [{i0}:{i0_end}, {i1}:{i1_end}, {i2}:{i2_end}]")

# print("Done writing large zarr dataset.")



# import zarr
# import numpy as np

# # Open the Zarr array (read-only mode)
# z = zarr.open("large_data.zarr", mode='r')  # or mode='r+' if you want to write

# print("Shape:", z.shape)
# print("Dtype:", z.dtype)
# print("Chunks:", z.chunks)

# # Example: read full array into memory (only if you have enough RAM)
# full_array = z[:]

# # Example: read a small region (e.g., one gene)
# gene_index = 1000  # e.g., 1000th gene
# gene_slice = z[:, :, gene_index]  # shape: (2000, 2000)

# # Example: iterate over small blocks (e.g., per 500x500 tile)
# for i in range(0, 2000, 500):
#     for j in range(0, 2000, 500):
#         block = z[i:i+500, j:j+500, :]  # shape: (500, 500, 17000)
#         # Process block