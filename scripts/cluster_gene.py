import argparse
from time import time
import os 
import numpy as np
import pandas as pd
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
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--overlay', action='store_true')
    parser.add_argument('--method', type=str, default='km')
    parser.add_argument('--n-clusters', type=int, default=None)
    parser.add_argument('--n-components', type=float, default=None)
    parser.add_argument('--filter-size', type=int, default=None)
    parser.add_argument('--min-cluster-size', type=int, default=None)
    # parser.add_argument('--stride', type=int, default=4)
    # parser.add_argument('--location-weight', type=float, default=None)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    pref = args.prefix
    overlay = args.overlay
    output=  args.output
    mask_name = args.mask
    n_components=args.n_components
    filter_size=args.filter_size
    n_clusters=args.n_clusters
    min_cluster_size=args.min_cluster_size
    method=args.method
    

    mask_name = args.mask
    if mask_name is None:
        print(f"mask is not provided, using the default mask from {pref}mask-small-RGB.png")
        mask_name = f"{pref}mask-small-RGB.png"

    embs = load_pickle(args.output + "embeddings-gene.pickle")
    if isinstance(embs, dict):
        if 'cls' in embs.keys():
            x = embs['cls']
        else:
            x = embs['sub']
        x = np.array(x)
    else:
        x = embs

    ## for this predicted embedding results, clustering should be done on the mask-small-RGB masking
    ## then other masking should be performed after clustering
    mask = load_mask(f"{pref}mask-small-RGB.png")    
    x[:, ~mask] = np.nan
    print(f"the input x for clustering after RGB masking has a shape of {x.shape}")            
    
    if mask_name is None:
        mask_name = f"{pref}mask-small-RGB.png"
        mask = None
    else:
        mask = load_mask(mask_name)
    
    out_dir = f"{output}clusters_gene_embedding/{os.path.splitext(os.path.basename(mask_name))[0]}/"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving to: {out_dir}")        
    out_pref = out_dir + "filtersize" + str(filter_size) + '_minclsize' + str(min_cluster_size) + '_' 
    print(f"the output directory will be {out_pref}")
    preprocess_and_cluster(
        x,
        n_components=n_components,
        filter_size=filter_size,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        method=method,
        prefix=out_pref,
        mask=mask)

    print(f'the clustering is done for {out_pref}...')
    if overlay:
        print("now doing the overlay of the clustered labels on the H&E...")
        parent_dir = os.path.abspath(os.path.dirname(out_pref)) + '/'
        clustered_folders = [entry.path + '/' for entry in os.scandir(parent_dir) if entry.is_dir()]
        plot_overlay(data_pref=pref, label_pref=clustered_folders, save=True)

if __name__ == '__main__':
    main()