import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
from utils import load_tsv, load_parquet, read_lines, save_image, load_mask, load_image
from my_utils import img_reduce
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import numpy as np
import shutil

def gamma_multinom_downsample_mat(mat, locs, alpha, beta):
    """
    Down-sample gene expression based on input matrix.
    :param mat: Input matrix (numpy array or sparse matrix)
    :param mean: Mean parameter for down-sampling
    :param sd: Standard deviation parameter for down-sampling
    :return: Down-sampled matrix
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be positive.")
    
    shrinkages = np.random.gamma(alpha, 1 / beta, mat.shape[0])  # Vectorized shrinkage computation
    np.save(f"shrinkages_{alpha}_{beta}.npy", shrinkages)
    # Generate a histogram for the shrinkages
    plt.hist(shrinkages, bins=50, color='blue', alpha=0.7)
    plt.title("Histogram of Shrinkages")
    plt.xlabel("Shrinkage Values")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    plt.savefig(f"shrinkages_{alpha}_{beta}.png")
        
    n_rows, n_cols = mat.shape
    
    # Step 1: Compute row sums
    row_sums = np.sum(mat, axis=1)
    # Step 2: Skip zero rows
    non_zero_rows = row_sums > 0
    mat_non_zero = mat[non_zero_rows, :]
    locs_non_zero= locs.iloc[non_zero_rows, :]
    row_sums_non_zero = row_sums[non_zero_rows]
    shrinkages_non_zero = shrinkages[non_zero_rows]
    sampled_mat = np.zeros((row_sums_non_zero.__len__(), n_cols))
    # Step 3: Compute new library sizes
    lib_sizes2 = np.round(row_sums_non_zero * shrinkages_non_zero).astype(int)
    # Step 4: Compute proportions
    proportions = mat_non_zero / row_sums_non_zero[:, np.newaxis]
    # Step 5: Normalize proportions to ensure they sum to 1
    proportions = proportions.astype(np.float64)
    proportions /= np.sum(proportions, axis=1)[:, np.newaxis]
    # Step 6: Downsample rows using np.random.multinomial
    for i in tqdm(range(lib_sizes2.__len__())):  # Only loop over non-zero rows
        lib_size = lib_sizes2[i]
        prop = proportions[i, :]
        sampled_mat[i, :] = np.random.multinomial(lib_size, prop)
    
    return sampled_mat, locs_non_zero
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pref', type=str)
    parser.add_argument('--out_pref', type=str)
    parser.add_argument('--mean', type=float)
    parser.add_argument('--sd', type=float)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    data_pref = args.data_pref
    out_pref = args.out_pref
    mean = args.mean
    sd= args.sd
    
    if os.path.isfile(f'{data_pref}locs.parquet'):
        locs_test = load_parquet(f'{data_pref}locs.parquet')
    elif os.path.isfile(f'{data_pref}locs.tsv'):
        locs_test = load_tsv(f'{data_pref}locs.tsv')

    unique_rows, indices, counts = np.unique(locs_test, axis=0, return_index=True, return_counts=True)
    unique_row_indices = indices[counts == 1]
    locs_test = locs_test.iloc[unique_row_indices,:]

    if os.path.isfile(f'{data_pref}cnts.tsv'):
        cnts_test = load_tsv(f'{data_pref}cnts.tsv')
    elif os.path.isfile(f'{data_pref}cnts.parquet'):
        cnts_test = pd.read_parquet(f'{data_pref}cnts.parquet')
        cnts_test = cnts_test.iloc[unique_row_indices, :]

    gene_names = read_lines(f"{data_pref}gene-names.txt")
    topn = gene_names.__len__()

    mat = cnts_test[gene_names[:topn]].values

    beta = mean / sd**2
    alpha = beta * mean
    
    downsampled_mat, downsampled_locs = gamma_multinom_downsample_mat(mat, locs_test, alpha=alpha, beta=beta)        
    downsampled_mat = downsampled_mat.astype(np.float16)

    downsampled_cnts_sparse = pd.DataFrame.sparse.from_spmatrix(sp.csr_matrix(downsampled_mat), index=downsampled_locs.index, columns=gene_names[:topn])
    downsampled_cnts_dense = downsampled_cnts_sparse.sparse.to_dense()
    downsampled_cnts_dense.to_parquet(f'{out_pref}cnts.parquet', compression='brotli')
    downsampled_locs.to_csv(f'{out_pref}locs.tsv', sep='\t')

    # Copy the gene-names.txt file to the output directory
    shutil.copy(f"{data_pref}he.jpg", f"{out_pref}he.jpg")
    shutil.copy(f"{data_pref}radius.txt", f"{out_pref}radius.txt")
    shutil.copy(f"{data_pref}embeddings-hipt-smooth.pickle", f"{out_pref}embeddings-hipt-smooth.pickle")


if __name__ == "__main__":
    main()