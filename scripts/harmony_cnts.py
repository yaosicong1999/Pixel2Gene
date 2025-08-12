import sys
import os
import numpy as np
import pandas as pd
import pickle
import argparse
import matplotlib.pyplot as plt
from utils import save_image, save_pickle, load_pickle, load_mask
import harmonypy as hm
from sklearn.decomposition import PCA

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embs",
        nargs="+",
        type=str,
        help="A list of cnts.tsv to be harmony-ed"
    )
    parser.add_argument(
        "--output",
        nargs="+",
        type=str,
        help="A list of embedding pickles after harmony-ed"
    )
    parser.add_argument('--pca', action='store_true')
    args = parser.parse_args()
    return args

def main():
    cnts_list = []
    args = get_args()
    dir_list = args.embs
    print(dir_list)
    
    for d in dir_list:
        x = pd.read_csv(d, sep='\t', index_col=0)
        cnts_list.append(x)
    data_mat = np.vstack(cnts_list).astype(np.float32)
    print(data_mat.shape)
    
    meta_data = pd.DataFrame(columns=['batch', 'row', 'col'])
    df_list = []
    for i in range(dir_list.__len__()):
        x = cnts_list[i]
        d = dir_list[i]
        df = pd.DataFrame({'batch': [d] * x.shape[0]})
        df_list.append(df)
    
    meta_data = pd.concat(df_list)
    print(meta_data.shape)
    assert meta_data.shape[0] == data_mat.shape[0]
    vars_use = ['batch']

    if args.pca:
        pca = PCA(n_components=500)
        pca.fit(data_mat)
    data_mat = np.vstack([pca.transform(i) for i in cnts_list]).astype(np.float32)

    ho = hm.run_harmony(data_mat, meta_data, vars_use)
    h = np.array(ho.Z_corr).T

    output_list = args.output
    assert output_list.__len__() == dir_list.__len__()

    idx = [x.shape[0]for x in cnts_list]
    cum_idx = np.cumsum([0] + idx)
    out = [h[cum_idx[i]:cum_idx[i + 1]] for i in range(len(idx))]
    for i in range(dir_list.__len__()):
        x = cnts_list[i]
        save_pickle(out[i], output_list[i])

if __name__ == '__main__':
    main()
