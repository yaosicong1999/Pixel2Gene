import pandas as pd
import numpy as np
import os
import seaborn as sns
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utils import load_image, save_image, load_mask, load_pickle
import argparse


def select_by_umi_percentile(umi, lower_bound=0, upper_bound=100):
    upper = np.percentile(umi, upper_bound)
    lower = np.percentile(umi, lower_bound)
    selected_umi = (umi >= lower) & (umi < upper)
    return np.where(selected_umi)[0]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pref', type=str)
    parser.add_argument('--cluster_labels', type=str, default=None)
    parser.add_argument('--upper', type=int, default=100)
    parser.add_argument('--lower', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    pref = args.pref
    cluster_labels = load_pickle(args.cluster_labels) if args.cluster_labels is not None else None
    upper = args.upper
    lower = args.lower
    
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
    print(f"The locs shape after dropping duplicate will be {unique_row_indices.__len__()}")

    mask_rgb = load_mask(f"{pref}mask-small-RGB.png")

    masked_none = np.full(mask_rgb.shape, False, dtype=bool)
    masked_none[locs[:, 0], locs[:, 1]] = True

    idx = np.where(mask_rgb[locs[:, 0], locs[:, 1]] == True)[0]
    idx = np.intersect1d(idx, unique_row_indices) 
    locs = locs[idx, :]

    batch_size = 10000
    max_value = np.max(idx)
    bins = np.arange(0, max_value + batch_size, batch_size)  
    bin_indices = np.digitize(idx, bins) - 1  
    split_bins = {i: idx[bin_indices == i] for i in range(len(bins) - 1)}

    if os.path.isfile(f"{pref}cnts.parquet"):
        print(f"The cnts file ends with .parquet")
        cnts_prqt = pq.ParquetFile(f"{pref}cnts.parquet")
        umi = []
        batch_counter = 0
        cnts = pd.DataFrame().astype(pd.SparseDtype("int", fill_value=0))  
        for batch in cnts_prqt.iter_batches(batch_size=batch_size):
            df = batch.to_pandas() 
            df = df.iloc[split_bins[batch_counter]-batch_counter*batch_size, :]
            df = df.astype(pd.SparseDtype("int", fill_value=0))  
            print("when batch_counter is", batch_counter, "the shape of df is", df.shape)
            cnts = pd.concat([cnts, df], ignore_index=True)
            umi.extend(df.sum(axis=1).tolist())  # Compute row sums and store in list
            batch_counter = batch_counter + 1
    elif os.path.isfile(f"{pref}cnts.tsv"):
        print(f"The cnts file ends with .tsv")
        cnts = pd.read_csv(f"{pref}cnts.tsv", sep='\t', index_col=0)
        cnts = cnts.iloc[idx, :]
        umi = cnts.sum(axis=1).tolist()
    else:
        raise ValueError("The cnts file is not found")

    umi = pd.Series(umi)
    df = pd.DataFrame()
    df['umi'] = umi
    df['x'] = locs[:, 1]
    df['y'] = locs[:, 0]
    df['clusters_hipt'] = cluster_labels[locs[:, 0], locs[:, 1]] if cluster_labels is not None else None
    assert np.sum(df['clusters_hipt'] == -1) == 0
    print(f"the count table for cluster labels is {df['clusters_hipt'].value_counts()}")
    df.to_parquet(f"{pref}umi_data_filtered.parquet", index=False)

    select_df = pd.DataFrame()
    label_list = np.sort(np.unique(df['clusters_hipt'])).tolist()
    for label in label_list:
        sub_df = df[df['clusters_hipt'] == label]
        idx = select_by_umi_percentile(sub_df['umi'].values, lower_bound=lower, upper_bound=upper)
        select_df = pd.concat([select_df, sub_df.iloc[idx, :]], ignore_index=True)

    select_df.to_parquet(f"{pref}umi_data_filtered_{lower}_{upper}.parquet", index=False)

    mask = np.full(mask_rgb.shape, False, dtype=bool)
    mask[select_df['y'].values, select_df['x'].values] = True
    save_image(mask, f"{pref}mask-small-filter_clustered_umi_{lower}_{upper}.png")
    
if __name__ == '__main__':
    main()