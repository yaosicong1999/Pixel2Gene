import scanpy as sc
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--locs', action='store_true')
    parser.add_argument('--radius', action='store_true')
    args = parser.parse_args()
    return args

def main():
    cnts = pd.
    comb_adata = dst_adata.copy()
for i in range(src_adata_list.__len__()):
    comb_adata = comb_adata.concatenate(src_adata_list[i])
sc.tl.pca(comb_adata, n_comps=100)
sc.pp.neighbors(comb_adata, n_neighbors=8, n_pcs=100)
sc.tl.umap(comb_adata)
sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.tl.louvain(
    comb_adata,
    resolution=0.25
)
n_cluster = comb_adata.obs['louvain'].unique().__len__()
    
    
    