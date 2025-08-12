import pandas as pd
import numpy as np
import os
import seaborn as sns
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from utils import load_image, save_image, load_mask, save_pickle, load_pickle
import argparse
from sklearn.model_selection import GroupKFold


# def plotting(umi):
#     # Find the peak x coordinate of the KDE curve
#     bw_adjust = 0.5  # Adjust this value as needed
#     kde = sns.kdeplot(umi, shade=False, color="blue", linewidth=2, bw_adjust=bw_adjust)
#     x_data, y_data = kde.get_lines()[0].get_data()
#     # Find all peaks (local maxima)
#     peaks, _ = find_peaks(y_data)
#     # Get x-values of all peaks
#     peak_x = x_data[peaks][0]
#     ## peak_x = kde.get_lines()[0].get_data()[0][np.argmax(kde.get_lines()[0].get_data()[1])]
#     print("Peak x coordinate of the KDE curve:", peak_x)

#     # Calculate the percentile of peak_x
#     percentile_peak_x = np.sum(umi <= peak_x) / len(umi) * 100
#     print("Percentile of the peak x coordinate:", percentile_peak_x)
    
#     plt.figure()
#     sns.histplot(umi, kde=True, bins=500, edgecolor='black', kde_kws={'bw_adjust': bw_adjust})
#     plt.title('Histogram with KDE of UMI Distribution')
#     plt.xlabel('UMI Counts')
#     plt.ylabel('Frequency')
#     # Ensure the x-axis is from 0 to 30% quantile
#     plt.xlim(0, np.percentile(umi, 95))
#     quantiles = np.percentile(umi, [1, 5, 10])
#     # Add vertical lines for quantiles
#     for quantile in quantiles:
#         plt.axvline(quantile, color='r', linestyle='--')

#     # Add vertical line for peak value
#     plt.axvline(peak_x, color='blue', linestyle=':', label=f'Peak: {peak_x}')
#     # Save the plot
#     plt.savefig('/home/sicongy/ssh_remote/umi_hist_kde_distribution.png')

#     from scipy.stats import nbinom, fit
#     bounds = [
#         (0.1, 100),  # Bounds for n (must be positive)
#         (0.01, 0.99)  # Bounds for p (must be between 0 and 1)
#     ]

#     # Fit the Negative Binomial distribution to the data with bounds
#     result = fit(nbinom, umi, bounds=bounds)
#     n_fit, p_fit, _ = result.params
#     print(f"Fitted parameters: n = {n_fit}, p = {p_fit}")

#     # Create a range of x values for the fitted distribution
#     x_values = np.arange(0, np.max(umi) + 1)

#     # Calculate the PDF of the fitted Negative Binomial distribution
#     fitted_pdf = nbinom.pmf(x_values, n_fit, p_fit)

#     # Plot the histogram of the data
#     plt.figure(figsize=(10, 6))
#     sns.histplot(umi, bins=50, edgecolor='black', stat='density', label='Histogram')
#     # Plot the fitted Negative Binomial distribution
#     plt.plot(x_values, fitted_pdf, 'r-', lw=2, label=f'Fitted Negative Binomial (n={n_fit:.2f}, p={p_fit:.2f})')

#     # Add labels and legend
#     plt.title('Histogram with Fitted Negative Binomial Distribution')
#     plt.xlabel('UMI Counts')
#     plt.ylabel('Density')
#     plt.legend()

#     # Save the plot
#     plt.savefig('/home/sicongy/ssh_remote/umi_hist_nbinom_fit.png')
#     plt.show()



#     # Calculate the IQR (Interquartile Range)
#     Q1 = umi.quantile(0.25)
#     Q3 = umi.quantile(0.75)
#     IQR = Q3 - Q1

#     # Define outliers as points outside 1.5 * IQR from Q1 and Q3
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     # Find outliers
#     outliers = umi[(umi < lower_bound) | (umi > upper_bound)]
#     print(f"Number of outliers: {len(outliers)}")
#     print("Outliers:", outliers.tolist())

#     # Find points greater than the 99th percentile
#     percentile_99 = np.percentile(umi, 99.5)
#     points_above_99 = umi[umi > percentile_99]
#     print(f"Number of points above 99th percentile: {len(points_above_99)}")
#     print("Points above 99th percentile:", points_above_99.tolist())


def get_block_kfolds_masks(mask, block_size=8, n_folds=10):
    print("Generating block K-Folds masks...")
    grid_data = np.where(mask, 1, np.nan)
    valid_mask = ~np.isnan(mask) & (mask != 0)
    valid_indices = np.where(valid_mask)  # Coordinates of valid spots
    block_ids = np.full(grid_data.shape, -1, dtype=int)
    # Assign block IDs only to valid spots
    for i, j in zip(*valid_indices):
        block_ids[i, j] = (i // block_size) * (grid_data.shape[0] // block_size) + (j // block_size)
    # Flatten block IDs for GroupKFold (only valid spots)
    valid_block_ids = block_ids[valid_mask]
    valid_coords = np.column_stack(valid_indices)  # [x,y] pairs of valid spots
    
    train_masks = []
    test_masks = []
    
    gkf = GroupKFold(n_splits=n_folds)
    for train_idx, test_idx in gkf.split(valid_coords, groups=valid_block_ids):
        # Create empty masks (False by default)
        train_mask = np.zeros_like(valid_mask, dtype=bool)
        test_mask = np.zeros_like(valid_mask, dtype=bool)
        # Get coordinates of train/test spots in this fold
        train_coords = valid_coords[train_idx]  # Shape (n_train, 2)
        test_coords = valid_coords[test_idx]    # Shape (n_test, 2)
        # Convert coordinates to indices in the original grid
        for i, j in train_coords:
            train_mask[i, j] = True  # Mark training spots
        for i, j in test_coords:
            test_mask[i, j] = True   # Mark testing spots
        train_masks.append(train_mask)
        test_masks.append(test_mask)
        
    return train_masks, test_masks

# def generate_bad_mask(pref, umi, locs, shape, lower_bound=None, upper_bound=None):
#     mask = np.full(shape, True, dtype=bool)
#     mask2 = np.full(shape, False, dtype=bool)
#     upper = np.percentile(umi, upper_bound)
#     lower = np.percentile(umi, lower_bound)
#     selected_umi = (umi >= lower) & (umi <= upper)
#     print("the length of selected_umi is", sum(selected_umi), "out of ", len(umi))
#     print("the mean of selected_umi is", np.mean(selected_umi))
#     print("the std of selected_umi is", np.std(selected_umi))
#     print("the median of selected_umi is", np.median(selected_umi))
#     mask2[locs[:, 0], locs[:, 1]] = selected_umi
#     mask = ~mask
#     save_image(mask2, f"{pref}mask-small-filter_bad_{lower_bound}_{upper_bound}_umi.png")
#     return mask, mask2



def generate_mask(pref, umi, locs, shape, percentile=None, lower_bound=None):
    mask = np.full(shape, False, dtype=bool) ## mask to keep 
    mask2 = np.full(shape, False, dtype=bool) ## mask to drop
    
    non_zero_idx = np.where(umi > 0)[0]
    umi_nonzero = umi[non_zero_idx]
    locs_nonzero = locs[non_zero_idx]
    mask[locs_nonzero[:, 0], locs_nonzero[:, 1]] = True
    
    zero_idx = np.where(umi == 0)[0]
    zero_locs = locs[zero_idx]
    mask[zero_locs[:, 0], zero_locs[:, 1]] = False
    mask2[zero_locs[:, 0], zero_locs[:, 1]] = True
    
    if percentile is None:
        bw_adjust = 0.5
        kde = sns.kdeplot(umi_nonzero, shade=False, color="blue", linewidth=2, bw_adjust=bw_adjust)
        x_data, y_data = kde.get_lines()[0].get_data()
        plt.clf()
        
        peaks, _ = find_peaks(y_data)
        peak_x = x_data[peaks][0]
        print("x coordinate of the first local maxima of the KDE curve:", peak_x)
        percentile_peak_x = np.sum(umi_nonzero <= peak_x) / len(umi_nonzero) * 100
        print("Percentile of the peak x coordinate:", percentile_peak_x)
        
        if lower_bound is not None and percentile_peak_x < lower_bound:
            print(f"Percentile of peak x ({percentile_peak_x}) is lower than lower bound ({lower_bound}). Using lower bound.")
            percentile_peak_x = lower_bound
            peak_x = np.percentile(umi_nonzero, lower_bound)
            
        plt.figure()
        sns.histplot(umi_nonzero, kde=True, bins=500, edgecolor='black', kde_kws={'bw_adjust': bw_adjust}, stat="density")
        plt.title('Histogram with KDE of UMI Distribution')
        plt.xlabel('UMI')
        plt.ylabel('Density')
        plt.axvline(peak_x, color='blue', linestyle=':', label=f'Peak: {percentile_peak_x:.1f}th percentile')
        plt.legend()
        plt.savefig(pref + 'umi_hist_kde_distribution.png')
        
        # If the peak_x is still too low, apply hard minimum
        if percentile_peak_x < 30:
            percentile_peak_x = 30
            peak_x = np.percentile(umi_nonzero, 30)
            
        # Find which of the non-zero spots should be dropped
        drop_idx = np.where(umi_nonzero <= peak_x)[0]
        drop_locs = locs_nonzero[drop_idx]
        
        # Update masks
        mask[drop_locs[:, 0], drop_locs[:, 1]] = False
        mask2[drop_locs[:, 0], drop_locs[:, 1]] = True
        
        print("number of spots retained after QC:", mask.sum())
        print("number of spots dropped during QC:", mask2.sum()) 
    else:
        # Apply threshold only on non-zero umi values
        threshold = np.percentile(umi_nonzero, percentile)
        print(f"Using fixed percentile threshold: {percentile} -> value = {threshold}")
        
        # Drop: umi <= threshold or umi == 0
        drop_idx = np.where((umi <= threshold))[0]
        drop_locs = locs[drop_idx]
        mask[drop_locs[:, 0], drop_locs[:, 1]] = False
        mask2[drop_locs[:, 0], drop_locs[:, 1]] = True
        
        print("number of spots retained after QC:", mask.sum())
        print("number of spots dropped during QC:", mask2.sum()) 
    return mask, mask2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pref', type=str)
    parser.add_argument('--n_clusters', type=int, default=None)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    pref = args.pref
    n_clusters = args.n_clusters

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


    mask_hs = load_mask(f"{pref}mask-small-hs.png")
    locs_hs = np.stack(np.where(mask_hs == True), -1)

    idx = np.where(mask_hs[locs[:, 0], locs[:, 1]] == True)[0]
    idx = np.intersect1d(idx, unique_row_indices) 
    locs_intersect_hs = locs[idx, :]


    if os.path.exists(f"{pref}umi.pickle"):
        umi = load_pickle(f"{pref}umi.pickle")
    else:
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
                if batch_counter not in split_bins:
                    continue
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
        save_pickle(umi, f"{pref}umi.pickle")
        
    umi = pd.Series(umi)
    print("the length of umi is", len(umi))
    assert locs_intersect_hs.shape[0] == len(umi)

    info_df = pd.DataFrame({'x': locs_intersect_hs[:, 1], 
                            'y': locs_intersect_hs[:, 0], 
                            'umi': umi})
    dtype = [('y', int), ('x', int)]
    locs_hs_struct = np.array([tuple(pt) for pt in locs_hs], dtype=dtype)
    locs_intersect_struct = np.array([tuple(pt) for pt in locs_intersect_hs], dtype=dtype)
    missing_coords_struct = np.setdiff1d(locs_hs_struct, locs_intersect_struct)
    missing_coords = np.array([[pt[0], pt[1]] for pt in missing_coords_struct])
    missing_df = pd.DataFrame({
        'x': missing_coords[:, 1],
        'y': missing_coords[:, 0],
        'umi': 0
    })
    full_info_df = pd.concat([info_df, missing_df], ignore_index=True)

    labels_mat = load_pickle(f"{os.path.dirname(pref)}/clusters_hipt/nclusters{n_clusters:03d}/labels.pickle")
    # Build a lookup dictionary from locs_hs to labels
    label_dict = {tuple((y, x)): labels_mat[y, x] for y, x in locs_hs}
    # Assign labels by looking up each coordinate in full_info_df
    full_info_df['labels'] = full_info_df[['y', 'x']].apply(lambda row: label_dict.get((row['y'], row['x']), -1), axis=1)

    hipt_cl_dir = os.path.dirname(pref) + '/hipt_cl/'
    os.makedirs(hipt_cl_dir, exist_ok=True)
    mask_final_all = np.full(mask_hs.shape[:2], False, dtype=bool)
    mask_qc_drop_all = np.full(mask_hs.shape[:2], False, dtype=bool)
    for cl in np.unique(full_info_df['labels']):
        if cl == -1:
            continue
        else:
            umi_i = np.array(full_info_df.loc[full_info_df['labels'] == cl, 'umi'])
            locs_i = np.array(full_info_df.loc[full_info_df['labels'] == cl, ['y', 'x']])
            print(f"the shape of umi_i is {umi_i.shape}")
            mask_final_i, mask_qc_drop_i = generate_mask(
                pref, umi_i, locs_i, mask_hs.shape[:2], 
                percentile=30, lower_bound=30
            )
            mask_final_all |= mask_final_i
            mask_qc_drop_all |= mask_qc_drop_i
            save_image(mask_final_i, f"{hipt_cl_dir}/mask-small-filter_he_qc_cl_{cl}.png")
            save_image(mask_qc_drop_i, f"{hipt_cl_dir}/mask-small-pred_he_qc_cl_{cl}.png")

    save_image(mask_final_all, f"{pref}mask-small-filter_he_qc.png")
    save_image(mask_qc_drop_all, f"{pref}mask-small-pred_he_qc.png")
    assert np.array_equal(mask_final_all | mask_qc_drop_all, mask_hs)
        
    train_masks, test_masks = get_block_kfolds_masks(mask_final_all, block_size=8, n_folds=10)
    
    for i, (train_mask, test_mask) in enumerate(zip(train_masks, test_masks)):
        save_image(train_mask, f"{pref}mask-small-filter_he_qc_train_fold_{i}.png")
        save_image(test_mask, f"{pref}mask-small-filter_he_qc_test_fold_{i}.png")

    
if __name__ == '__main__':
    main()