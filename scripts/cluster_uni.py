import argparse
from time import time
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
# from sklearn.neighbors import kneighbors_graph
# from hdbscan import HDBSCAN
# from einops import reduce
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from utils import load_pickle, save_pickle, sort_labels, load_mask
from image import smoothen, upscale
from visual import plot_labels, plot_label_masks
from connected_components import (
        relabel_small_connected, cluster_connected)
from reduce_dim import reduce_dim
from cluster_basic import preprocess_and_cluster

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--method', type=str, default='km')
    parser.add_argument('--n_clusters', type=int, default=None)
    parser.add_argument('--n_components', type=float, default=None)
    parser.add_argument('--filter-size', type=int, default=None)
    parser.add_argument('--min-cluster-size', type=int, default=None)
    # parser.add_argument('--stride', type=int, default=4)
    # parser.add_argument('--location-weight', type=float, default=None)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    x = np.load(args.embeddings)
    x = np.transpose(x, (2, 0, 1))

    if args.mask is not None:
        mask = load_mask(args.mask)
        x[:, ~mask] = np.nan

    preprocess_and_cluster(
            x,
            n_components=args.n_components,
            filter_size=args.filter_size,
            n_clusters=args.n_clusters,
            min_cluster_size=args.min_cluster_size,
            method=args.method,
            prefix=f"{args.output}/")


if __name__ == '__main__':
    main()
