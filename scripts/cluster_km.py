import sys
import os
import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from utils import save_image, load_pickle, load_mask, save_pickle


def plot_super(x, outfile, truncate=None, he=None, locs=None):
    x = x.copy()
    mask = np.isfinite(x)
    if truncate is not None:
        x = np.clip(x, truncate[0], truncate[1])
    # col = cmapFader(cmap_name='turbo', start_val=0, stop_val=1)
    # img = col.get_rgb(x)[:, :, :3]
    cmap = plt.get_cmap('turbo')
    img = cmap(x)[..., :3]
    img[~mask] = 1.0
    img = (img * 255).astype(np.uint8)
    if locs is not None:
        if he is not None:
            out = he.copy()
        else:
            out = np.full((x.shape[0], x.shape[1], 3), 255)
        out[locs[:, 0], locs[:, 1], :] = img[locs[:, 0], locs[:, 1], :]
        img = out
    filter = np.isnan(x)
    if he is not None:
        img[filter] = he[filter]
    img = img.astype(np.uint8)
    img[~mask] = 0
    save_image(img, outfile)


def kmeans_cluster_multi(embs_list, k, filename_list, mask_list=None):
    assert embs_list.__len__() == filename_list.__len__()
    if mask_list is not None:
        assert embs_list.__len__() == mask_list.__len__()
    pca = PCA(n_components=0.95)
    len_list = []
    embs_comb = np.empty((0, embs_list[0].shape[2]))
    for i in range(embs_list.__len__()):
        data = embs_list[i].reshape(embs_list[i].shape[0] * embs_list[i].shape[1], embs_list[i].shape[2])
        if mask_list is not None:
            mask_list[i] = mask_list[i].reshape(mask_list[i].shape[0] * mask_list[i].shape[1], 1)
            data = data[mask_list[i][:, 0], :]
        embs_comb = np.vstack((embs_comb, data))
        len_list.append(data.shape[0])
    data_pca = pca.fit_transform(embs_comb)
    print(f"Original dimensions: {embs_comb.shape}")
    print(f"Reduced dimensions after PCA: {data_pca.shape}")
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(data_pca)
    print("Cluster Centers in PCA space:")
    print(kmeans.cluster_centers_)
    print("Cluster Labels for each sample:")
    print(clusters)
    for i in range(embs_list.__len__()):
        v_cluster = np.full(embs_list[i].shape[0] * embs_list[i].shape[1], np.nan)
        if mask_list is not None:
            if i > 0:
                v_cluster[mask_list[i][:, 0]] = clusters[np.cumsum(len_list)[i-1]:np.cumsum(len_list)[i]]
            else:
                v_cluster[mask_list[i][:, 0]] = clusters[:np.cumsum(len_list)[i]]
        v_cluster = v_cluster.reshape(embs_list[i].shape[0], embs_list[i].shape[1])
        v_cluster = v_cluster/np.nanmax(v_cluster)
        plot_super(v_cluster, outfile=filename_list[i])
        save_pickle(v_cluster, filename_list[i].replace('.png', '.pickle'))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embs",
        nargs="+",
        type=str,
        help="A list of embedding pickles to be clustered together "
    )
    parser.add_argument(
        "--mask",
        nargs="+",
        type=str,
        help="A list of corresponding mask images"
    )
    parser.add_argument('--n-clusters', type=int)
    parser.add_argument(
        "--output",
        type=str,
        nargs="+",
        help="A list of output plot filenames"
    )
    args = parser.parse_args()
    return args

def main():
    embs_list = []
    args = get_args()
    dir_list = args.embs
    filename_list = args.output
    # print(filename_list)
    k = args.n_clusters
    
    mask_list = []
    if args.mask is not None:
        # print(args.mask)
        for i in range(args.mask.__len__()):
            m = load_mask(args.mask[i])
            mask_list.append(m)
    else:
        mask_list = None
    
    # print(dir_list)
    for ind in range(dir_list.__len__()):
        d = dir_list[ind]
        ## if the embs is a image feature embedding, load the pickle file
        if d.endswith('.pickle'):
            x = load_pickle(d)
            ## if the embs is in the format of a dictionary, concatenate the cls, sub, and rgb channels and transpose the array
            if isinstance(x, dict):
                x = np.concatenate([x['cls'], x['sub'], x['rgb']])
                x = x.transpose(1, 2, 0)
        ## if the embs is a gene expression cnts matrix, load the file as a pandas dataframe
        elif d.endswith('cnts.tsv'):
            if mask_list is not None and os.path.exists(d.replace("cnts.tsv", "locs.tsv")):
                factor = 16
                cnts = pd.read_csv(d, sep='\t', index_col=0)
                locs = pd.read_csv(d.replace("cnts.tsv", "locs.tsv"), sep='\t', index_col=0)
                assert cnts.shape[0] == locs.shape[0]
                locs = locs.astype(float)
                locs = np.stack([locs['y'], locs['x']], -1)
                locs //= factor
                locs = locs.round().astype(int)
                x = np.full((mask_list[ind].shape[0], mask_list[ind].shape[1], cnts.shape[1]), 0)
                x[locs[:, 0], locs[:, 1], :] = cnts.values
            else:
                raise ValueError('Mask must be provided for gene expression data')
        else:
            raise ValueError('Unrecognized embedding format (must be .pickle or .tsv)')
        embs_list.append(x)

    kmeans_cluster_multi(embs_list, k=k, filename_list=filename_list, mask_list=mask_list)

if __name__ == '__main__':
    main()

