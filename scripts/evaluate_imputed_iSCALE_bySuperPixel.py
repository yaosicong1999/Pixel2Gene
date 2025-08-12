import sys
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import MinMaxScaler
from utils import load_pickle, save_tsv, load_tsv, read_lines, load_mask

scaler = MinMaxScaler()

# Pearson correlation stabilized
def corr_pearson_stablized(x, y, epsilon=1e-6):
    x = standardize(x)
    y = standardize(y)
    x = x - x.mean()
    y = y - y.mean()
    x_std = (x**2).mean()**0.5
    y_std = (y**2).mean()**0.5
    corr = ((x * y).mean() + epsilon) / (x_std * y_std + epsilon)
    return corr

# Pearson correlation
def corr_pearson(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan  # Avoid division by zero
    return pearsonr(x, y)[0]

# Spearman correlation
def corr_spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan  # Return NaN if constant input is detected
    return spearmanr(x, y)[0]

# Uncentered correlation
def corr_uncentered(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan  # Return NaN if constant input is detected
    return np.mean(x * y) / (np.mean(x**2)**0.5 * np.mean(y**2)**0.5)

# RMSE
def rmse(x, y):
    return np.sqrt(np.mean((x - y)**2))

# Peak signal-to-noise ratio
def psnr(x, y):
    mse = np.mean((x - y)**2)
    if mse == 0:
        return np.inf  # Perfect match, infinite PSNR
    return 10 * np.log10(1 / mse)

# Standardization function
def standardize(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)

# Metric calculation function that handles NaNs
def metric_fin(x, y, method='pearson'):
    mask = np.isfinite(x) & np.isfinite(y)  # Handle NaN values
    x, y = x[mask], y[mask]

    if len(x) < 2 or len(y) < 2:
        return np.nan

    method_dict = {
        'pearson': corr_pearson,
        'pearson_stablized': corr_pearson_stablized,
        'spearman': corr_spearman,
        'uncentered': corr_uncentered,
        'psnr': psnr,
        'rmse': rmse
    }

    return method_dict[method](x, y)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pref_train', type=str)
    parser.add_argument('--pref_test', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    return args


def main():
    factor = 16
    args = get_args()
    
    # Load genes to be compared 
    gene_names = read_lines(f'{args.pref_train}gene-names.txt')

    # Load locs
    locs = load_tsv(f'{args.pref_test}locs.tsv')
    locs = locs.astype(float)
    locs = np.stack([locs['y'], locs['x']], -1)
    locs //= factor
    locs = locs.round().astype(int)
    unique_rows, indices, counts = np.unique(locs, axis=0, return_index=True, return_counts=True)
    unique_row_indices = indices[counts == 1]
    locs = locs[unique_row_indices, ]

    # Load mask 
    mask = load_mask(f'{args.pref_test}mask-small-RGB.png')
    mask_flat = mask[locs[:, 0], locs[:, 1]]
    
    # Load ground truth data
    truth_flat = load_tsv(f"{args.pref_test}cnts.tsv") # (num_spots, num_genes)
    truth_flat = np.array(truth_flat).astype(np.float32)
    truth_flat = truth_flat[unique_row_indices, ]
    truth_mat = np.full((mask.shape[0], mask.shape[1], truth_flat.shape[1]), np.nan)
    truth_mat[locs[:, 0], locs[:, 1], :] = truth_flat

    # Load predicted data
    pred_list = []
    for gn in gene_names:
        pred_gn = load_pickle(f'{args.output}cnts-super/{gn}.pickle')            
        if pred_gn.shape == mask.shape:
                pred_list.append(pred_gn)
        else:
            print(f"Skipping {gn} due to incorrect shape: {pred_gn.shape}")
    pred_mat = np.stack(pred_list, axis=-1)
    pred_flat = pred_mat[locs[:, 0], locs[:, 1], :]

    eval_list = []

    # Loop through each row (spatial spot)
    for i in range(pred_flat.shape[0]):
        if i % 10000 == 0:  # Add a progress indicator for large datasets
            print(f"Processing row {i}/{pred_flat.shape[0]}")
        truth_row = truth_flat[i, :]
        pred_row = pred_flat[i, :]

        # Skip if the entire row is NaN for either cnts or predictions
        if np.isnan(truth_row).all() or np.isnan(pred_row).all():
            eval = {key: np.nan for key in ['pearson', 'rmse', 'pearson_stablized', 'spearman', 'uncentered', 'psnr']}
            eval_list.append(eval)
            continue

        # Fill NaNs with 0 and apply MinMax scaling
        truth_row_filled = np.nan_to_num(truth_row).reshape(-1, 1)
        pred_row_filled = np.nan_to_num(pred_row).reshape(-1, 1)
        truth_row_scaled = scaler.fit_transform(truth_row_filled).flatten()
        pred_row_scaled = scaler.fit_transform(pred_row_filled).flatten()

        # Compute metrics using metric_fin
        eval = {
            'pearson': metric_fin(truth_row_scaled, pred_row_scaled, 'pearson'),
            'rmse': metric_fin(truth_row_scaled, pred_row_scaled, 'rmse'),
            'pearson_stablized': metric_fin(truth_row_scaled, pred_row_scaled, 'pearson_stablized'),
            'spearman': metric_fin(truth_row_scaled, pred_row_scaled, 'spearman'),
            'uncentered': metric_fin(truth_row_scaled, pred_row_scaled, 'uncentered'),
            'psnr': metric_fin(truth_row_scaled, pred_row_scaled, 'psnr')
        }

        eval_list.append(eval)

    # Check if eval_list is empty
    if len(eval_list) == 0:
        print("No valid rows to evaluate.")
        return

    # Create DataFrame from eval_list
    df = pd.DataFrame(eval_list)

    # Assign metric names as column headers
    df.columns = ['Pearson', 'RMSE', 'Pearson_Stabilized', 'Spearman', 'Uncentered', 'PSNR']

    # Save the DataFrame to a .tsv file
    os.makedirs(f'{args.output}cnts-super-eval', exist_ok=True)
    df.to_csv(f'{args.output}cnts-super-eval/superPixel{factor:04d}.tsv', sep='\t', index=False)
    print(f"DataFrame saved successfully with shape: {df.shape}")
    
    for col in df.columns:
        fig = plt.subplots()
        ax = sns.violinplot(data=df[col], orient="h", color='cornflowerblue')
        ax = sns.stripplot(data=df[col], orient="h", color='lightskyblue', jitter=True)
        quantiles = [0.25, 0.5, 0.75]
        linestyles = [':', '-', '--']
        for i in range(3):
            q = quantiles[i]
            quantile_line = df.quantile(q)
            ax.axvline(quantile_line[col], linestyle=linestyles[i], color='red',label=f'{int(q * 100)}% Quantile of {col}')
        plt.xlim(0, 1)
        plt.legend()
        plt.title(f'{col}: Predicted vs True', fontsize=16)
        plt.savefig(f'{args.output}cnts-super-eval/{col}.png', dpi=200)
    
    

if __name__ == '__main__':
    main()

