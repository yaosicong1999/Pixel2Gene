import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils import load_pickle, save_image, read_lines, load_image, load_tsv, load_mask, load_parquet, save_pickle
from my_utils import cmapFader, img_reduce, locs_reduce, plot_super
from structural_similarity import structural_similarity
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import jaccard
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations, count
from scipy.ndimage import convolve


def save_comparison_plot(he, oTrue, metrics_df, oPred=None, dTrue=None, dPred=None, outname=None, gn=None, overlay=False):
    """Save comparison plots for downsampled and original images."""
    selected_metric_names = ['Moran\'s I', 'Pearson Correlation', 'Spearman Correlation', 'Cosine Similarity']
    method_names = metrics_df.index.tolist()
        
    if overlay: 
        oTrue_img = plot_super(oTrue / np.nanmax(oTrue), save=False, he=he)
        oPred_img = [plot_super(oPred[i] / np.nanmax(oPred[i]), save=False, he=he) for i in range(len(oPred))] if oPred is not None else None
        dTrue_img = plot_super(dTrue / np.nanmax(dTrue), save=False, he=he) if dTrue is not None else None
        dPred_img = [plot_super(dPred[i] / np.nanmax(dPred[i]), save=False, he=he) for i in range(len(dPred))] if dPred is not None else None
    else:
        oTrue_img = plot_super(oTrue / np.nanmax(oTrue), save=False)
        oPred_img = [plot_super(oPred[i] / np.nanmax(oPred[i]), save=False) for i in range(len(oPred))] if oPred is not None else None
        dTrue_img = plot_super(dTrue / np.nanmax(dTrue), save=False) if dTrue is not None else None
        dPred_img = [plot_super(dPred[i] / np.nanmax(dPred[i]), save=False) for i in range(len(dPred))] if dPred is not None else None
        
    if oPred is not None:
        oPred_method = method_names[1:(len(oPred)+1)]
        dTrue_method = method_names[len(oPred) + 1] if dTrue is not None else None
        dPred_method = method_names[1 + len(oPred) + int(dTrue is not None):] if dPred is not None else None
    else:
        oPred_method = None
        dTrue_method = method_names[1] if dTrue is not None else None
        dPred_method = method_names[1 + int(dTrue is not None):] if dPred is not None else None
 
    if dPred is not None and dTrue is not None:
        nrows = 2; ncols = 2 + len(dPred)
    elif dTrue is not None:
        nrows = 2; ncols = 2 + len(oPred) if oPred is not None else 2
    else:
        nrows = 1; ncols = 2 + len(oPred) if oPred is not None else 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 5*nrows), constrained_layout=True)
    fig.suptitle(f"{gn}", fontsize=16, style='italic')
    print(f"nrow is {nrows}, ncol is {ncols}")
    if nrows == 1:
        axes = np.atleast_1d(axes)  # Ensure axes is always iterable
    else:
        axes = axes.flatten()  # Flatten for easy indexing
    
    axes[0].imshow(he)
    axes[0].axis("off")
    axes[0].set_title("HE Image", fontsize=10)
    
    counter = 1
    axes[counter].imshow(oTrue_img)
    axes[counter].axis("off")
    morans_i_oTrue = metrics_df.loc['Original-Observed', 'Moran\'s I']
    axes[counter].set_title(f"Original-Observed \n Moran\'s I: {morans_i_oTrue:.2f}", fontsize=10)
    counter += 1
    
    if oPred is not None:
        for i in range(len(oPred_img)):
            axes[counter].imshow(oPred_img[i])
            axes[counter].axis("off")
            title = oPred_method[i]
            for metric_name, metric_value in zip(
                selected_metric_names,
                [metrics_df.loc[oPred_method[i], metric_name] for metric_name in selected_metric_names]
            ):
                title += f"\n {metric_name}: {metric_value:.2f} "
            axes[counter].set_title(title.strip(), fontsize=10)
            counter += 1
        if len(oPred_img) < ncols - 2:
            for i in range(len(oPred_img), ncols - 2):
                axes[counter].axis("off")
                counter += 1
        
    if nrows == 2:
        axes[counter].axis("off")
        counter += 1
        
    if dTrue is not None:
        axes[counter].imshow(dTrue_img)
        axes[counter].axis("off")
        title = dTrue_method
        for metric_name, metric_value in zip(
            selected_metric_names,
            [metrics_df.loc[dTrue_method, metric_name] for metric_name in selected_metric_names]
        ):
            title += f"\n {metric_name}: {metric_value:.2f} "
        axes[counter].set_title(title.strip(), fontsize=10)
        counter += 1
           
    if dPred is not None:
        for i in range(len(dPred_img)):
            axes[counter].imshow(dPred_img[i])
            axes[counter].axis("off")
            title = dPred_method[i]
            for metric_name, metric_value in zip(
                selected_metric_names,
                [metrics_df.loc[dPred_method[i], metric_name] for metric_name in selected_metric_names]
            ):
                title += f"\n {metric_name}: {metric_value:.2f} "
            axes[counter].set_title(title.strip(), fontsize=10)
            counter += 1        
     
    plt.tight_layout()
    plt.savefig(outname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outname}")


def standardize_vector(vector):
    return (vector - np.nanmin(vector)) / (np.nanmax(vector) - np.nanmin(vector))


def binarize_data_with_mask(vector, threshold, nan_mask):
    # Apply NaN mask from the true data and then binarize using the threshold
    binarized = np.where(nan_mask, np.nan, np.where(vector > threshold, 1, 0))
    return binarized


def compute_dice_coefficient(true_binarized, imputed_binarized):
    intersection = np.nansum((true_binarized == 1) & (imputed_binarized == 1))
    total_sum = np.nansum(true_binarized == 1) + np.nansum(imputed_binarized == 1)
    dice_coeff = (2 * intersection) / total_sum if total_sum != 0 else np.nan
    return dice_coeff


def compute_jaccard_index(true_binarized, imputed_binarized):
    intersection = np.nansum((true_binarized == 1) & (imputed_binarized == 1))
    union = np.nansum((true_binarized == 1) | (imputed_binarized == 1))
    jaccard_index = intersection / union if union != 0 else np.nan
    return jaccard_index


def compute_chi_square_test(true_binarized, imputed_binarized):
    contingency_table = np.zeros((2, 2))
    
    # Fill contingency table: [ [True 0/0, True 1/0], [Pred 0/1, Pred 1/1] ]
    contingency_table[0, 0] = np.nansum((true_binarized == 0) & (imputed_binarized == 0))
    contingency_table[0, 1] = np.nansum((true_binarized == 0) & (imputed_binarized == 1))
    contingency_table[1, 0] = np.nansum((true_binarized == 1) & (imputed_binarized == 0))
    contingency_table[1, 1] = np.nansum((true_binarized == 1) & (imputed_binarized == 1))
    
    chi2, p, _, _ = chi2_contingency(contingency_table)
    return chi2, p


def create_decaying_kernel(layer=1, decay='gaussian', alpha=1.0, sigma=1.0, exclude_center=True):
    """
    Create a decaying kernel with distance-based weights.

    Parameters:
        layer (int): Number of layers around the center. Kernel size = 2*layer + 1.
        decay (str): 'gaussian' or 'inverse'
        alpha (float): Power for inverse decay (if used)
        sigma (float): Sigma for Gaussian decay
        exclude_center (bool): Whether to set center weight to 0

    Returns:
        np.ndarray: Decaying kernel
    """
    kernel_size = 2 * layer + 1
    center = layer
    ax = np.arange(-layer, layer + 1)
    xx, yy = np.meshgrid(ax, ax)
    distance = np.sqrt(xx**2 + yy**2)
    if decay == 'gaussian':
        kernel = np.exp(- (distance**2) / (2 * sigma**2))
    elif decay == 'inverse':
        with np.errstate(divide='ignore'):
            kernel = 1 / (distance**alpha)
        kernel[distance == 0] = 0  # avoid division by zero
    else:
        raise ValueError("decay must be 'gaussian' or 'inverse'")
    if exclude_center:
        kernel[center, center] = 0  # Set center to 0
    kernel /= np.sum(kernel)  # Normalize to sum to 1
    return kernel

def fast_morans_i_with_nans(grid, kernel_layer=1):
    """
    Computes Moran's I for 2D grid data with NaN values using convolution.
    
    Parameters:
        grid (np.ndarray): 2D array of values (may contain NaN)
    
    Returns:
        Moran's I (float), or NaN if insufficient non-NaN data
    """
    mask = ~np.isnan(grid)
    valid_count = np.sum(mask)
    if valid_count < 2:
        return np.nan    
    grid_filled = np.where(mask, grid, 0)
    if kernel_layer == 1:
        kernel = np.ones((3, 3)) 
        kernel[1, 1] = 0 
    else:
        kernel = create_decaying_kernel(layer=kernel_layer, decay='gaussian', sigma=1.0, exclude_center=True)
    
    neighbor_count = convolve(mask.astype(float), kernel, 
                            mode='constant', cval=0)
    spatial_lag_sum = convolve(grid_filled, kernel, 
                             mode='constant', cval=0)
    spatial_lag = np.divide(spatial_lag_sum, neighbor_count,
                           out=np.zeros_like(grid),
                           where=(neighbor_count > 0))
    x_mean = grid[mask].mean()
    x_dev = np.where(mask, grid - x_mean, 0)
    numerator = np.sum(x_dev * (spatial_lag - x_mean))
    denominator = np.sum(x_dev[mask] ** 2)
    if denominator == 0 or np.sum(neighbor_count[mask]) == 0:
        return np.nan    
    I = (valid_count / np.sum(neighbor_count[mask])) * (numerator / denominator)
    return I


def calculate_binary_metrics(truth, pred, mask=None, thresh=90):
    scaler = MinMaxScaler()
    mask_flat = mask.flatten()
    nrows, ncols = truth.shape[0], truth.shape[1]
    
    truth_flat = truth.reshape((nrows*ncols, 1))
    imputed_flat = pred.reshape((nrows*ncols, 1))
    truth_flat[mask_flat == 0] = np.nan
    imputed_flat[mask_flat == 0] = np.nan
    
    truth_exp_std_flat = scaler.fit_transform(truth_flat)
    imputed_exp_std_flat = scaler.fit_transform(imputed_flat)
    truth_exp_std = truth_exp_std_flat[:,0].reshape(nrows, ncols)
    imputed_exp_std = imputed_exp_std_flat[:,0].reshape(nrows, ncols)
    
    nan_mask = np.isnan(truth_exp_std)
    threshold_truth = np.nanpercentile(truth_exp_std, thresh)
    threshold_pred = np.nanpercentile(imputed_exp_std, thresh)
    print(f'Threshold True: {threshold_truth}, Threshold Pred: {threshold_pred}')
    
    # Binarize the true and predicted expression using their respective thresholds and apply the true NaN mask
    truth_binarized = binarize_data_with_mask(truth_exp_std, threshold_truth, nan_mask)  # Do not flatten for imshow
    imputed_binarized = binarize_data_with_mask(imputed_exp_std, threshold_pred, nan_mask)  # Same NaN mask used for predicted data
    
    # Compute Dice, Jaccard, and Chi-square test
    dice_coeff = compute_dice_coefficient(truth_binarized, imputed_binarized)
    jaccard_index = compute_jaccard_index(truth_binarized, imputed_binarized)
    chi2, p_value = compute_chi_square_test(truth_binarized, imputed_binarized)
    
    return {'Dice Coefficient': dice_coeff,
            'Jaccard Index': jaccard_index,
            'Chi Square': chi2,
            'p_value': p_value}


def calculate_metrics(truth, imputed, mask=None, all_spots=False):
    """Calculate Pearson correlation, Spearman correlation, cosine similarity and SSIM metrics."""
    assert truth.shape == imputed.shape, "Truth and imputed images must have the same shape"
    if mask is not None:
        truth[~mask] = np.nan
        imputed[~mask] = np.nan
    truth = truth / np.nansum(truth) * 1e6
    imputed = imputed / np.nansum(imputed) * 1e6
    
    masked_indices = ~np.isnan(truth) & ~np.isnan(imputed)
    if all_spots:
        valid_indices = masked_indices
    else:
        valid_indices = ~np.isnan(truth) & (truth != 0) & ~np.isnan(imputed)
    # print(f"Number of valid indices: {np.sum(valid_indices)} out of {np.sum(masked_indices)} masked spots out of {truth.shape[0] * truth.shape[1]} total spots")
    
    truth_non_nan = truth[valid_indices] / np.max(truth[valid_indices])
    imputed_non_nan = imputed[valid_indices] / np.max(imputed[valid_indices])
    pearson_corr, _ = stats.pearsonr(truth_non_nan, imputed_non_nan)
    spearman_corr, _ = stats.spearmanr(truth_non_nan, imputed_non_nan)
    cosine_similarity = np.dot(truth_non_nan, imputed_non_nan) / (np.linalg.norm(truth_non_nan) * np.linalg.norm(imputed_non_nan))
    
    truth_mat = (np.nan_to_num(truth, nan=0) / np.nanmax(truth) * 255).astype(int)
    imputed_mat = (np.nan_to_num(imputed, nan=0) / np.nanmax(imputed) * 255).astype(int)
    ssim = structural_similarity(truth_mat, imputed_mat)
    
    return {'Pearson Correlation': pearson_corr, 
            'Spearman Correlation': spearman_corr, 
            'Cosine Similarity': cosine_similarity, 
            'SSIM': ssim}

    
def load_image_data(gn, prefix, locs, unique_row_indices, mask_shape, counts=None):
    """Load image data for a given gene."""
    img = np.full(mask_shape, np.nan)
    if os.path.isfile(f'{prefix}cnts.parquet'):
        counts = pd.read_parquet(f'{prefix}cnts.parquet', columns=[gn])
        counts = counts.iloc[unique_row_indices, :]
    img[locs[:, 0], locs[:, 1]] = counts[gn]
    return img


def handle_locations(prefix, factor=16):
    """Handle location data and reduce resolution."""
    if os.path.isfile(f'{prefix}locs.parquet'):
        locs = load_parquet(f'{prefix}locs.parquet')
    elif os.path.isfile(f'{prefix}locs.tsv'):
        locs = load_tsv(f'{prefix}locs.tsv')
    else:
        raise ValueError(f"Unsupported file format for locations: {prefix}")
    locs = locs.astype(float)
    locs = np.stack([locs['y'], locs['x']], -1)
    locs //= factor
    locs = locs.round().astype(int)
    unique_rows, indices, counts = np.unique(locs, axis=0, return_index=True, return_counts=True)
    unique_row_indices = indices[counts == 1]
    return locs[unique_row_indices], unique_row_indices

        
def plot_comparison_violin(metrics_dict_df, output):
    """Plot violin plot for metrics."""
    metric_names = list(metrics_dict_df.keys())
    method_names = metrics_dict_df[metric_names[0]].columns.tolist()
    colors = ['#FFC4A3', '#B5EAD7', '#E2CFEA', '#FFF2CC', '#AED9E0', '#F4A6A6', '#F9CB9C', '#D0E0E3']
    colors = {method_names[i]: colors[i] for i in range(len(method_names))}
    for metric in metric_names:
        df = metrics_dict_df[metric]
        plt.figure(figsize=(2.5*len(method_names), 6))
        sns.violinplot(data=df, palette=[colors[col] for col in df.columns])
        plt.xlabel("Method", fontweight="bold")  # Make x-axis label bold
        plt.ylabel("Value", fontweight="bold")  # Make y-axis label bold
        plt.title('Violin Plot for ' + metric, fontweight="bold")
        plt.savefig(f"{output}{metric}_violin_plot.png")
        plt.close()
        print(f"Violin plot saved to {output}{metric}_violin_plot.png")


def plot_comparison_boxplot(metrics_dict_df, metric_names, output):
    """Plot boxplot with jittered dots and quantile lines for comparison metrics."""
    method_names = metrics_dict_df[metric_names[0]].columns.tolist()
    colors = ['#FFC4A3', '#B5EAD7', '#E2CFEA', '#FFF2CC', '#AED9E0', '#F4A6A6', '#F9CB9C', '#D0E0E3']
    colors = {method_names[i]: colors[i] for i in range(len(method_names))}
    for metric in metric_names:
        df = metrics_dict_df[metric]    
        df_melted = df.melt(var_name="Method", value_name="Correlation")    
        unique_methods = df_melted["Method"].unique()
        palette = {m: colors[m] for m in unique_methods if m in colors}
        
        all_values = df_melted['Correlation'].dropna()
        min_val = all_values.min()
        max_val = all_values.max()
        
        if metric in ['Pearson Correlation', 'Spearman Correlation', 'Moran\'s I']:
            if min_val < 0:
                ymin = min_val
                ymax = 1.0
            else:
                ymin = max(0, min_val - 0.2)
                ymax = 1.0
        else:
            ymin = min_val * 0.9 if min_val > 0 else min_val * 1.1
            ymax = max_val * 1.1 if max_val > 0 else max_val * 0.9
        
        plt.figure(figsize=(2.5*len(method_names), 6))
        sns.boxplot(data=df_melted, x="Method", y="Correlation", palette=palette, showfliers=False)
        sns.stripplot(
            data=df_melted,
            x="Method",
            y="Correlation",
            color='black',
            dodge=True,
            jitter=True,
            alpha=1,             # fully opaque
            size=3,              # small dot size
            linewidth=0          # no border = solid
        )       
        plt.title('Boxplot for ' + metric, fontweight="bold")
        plt.xlabel("Method", fontweight="bold")  # Make x-axis label bold
        plt.ylabel("Value", fontweight="bold")  # Make y-axis label bold
        plt.ylim(ymin, ymax)  # Set the calculated y-axis limits
        plt.tight_layout()
        outfile = f"{output}{metric.replace(' ', '_')}_boxplot.png"
        plt.savefig(outfile)
        plt.close()
        print(f"Boxplot saved to {outfile}")
            
    
def plot_comparison_scatter(x, y, output, metric, x_label, y_label, gn_list=None, umi=None):
    """Plot scatter plot for two input vectors x and y, with optional point labels and UMI-based coloring."""
    plt.figure(figsize=(6, 6))
    # Map UMI values to colors
    if umi is not None and gn_list is not None:
        umi_values = umi.loc[gn_list].values.flatten()  # Extract UMI values corresponding to name_list
        umi_values = pd.to_numeric(umi_values, errors='coerce')  # Convert to numeric, setting invalid values to NaN
        norm = plt.Normalize(vmin=np.nanmin(np.log1p(umi_values)), vmax=np.nanmax(np.log1p(umi_values)))  # Use log1p scale and nan-safe functions
        colors = plt.cm.turbo(norm(np.log1p(umi_values)))
    else:
        colors = plt.cm.turbo(np.linspace(0, 1, len(x)))  # Default to index-based coloring
    
    scatter = plt.scatter(x, y, color=colors, alpha=0.8)
    plt.xlabel(f'{x_label}')
    plt.ylabel(f'{y_label}')
    plt.title('Scatter Plot for ' + metric)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1)  # y=x line
    plt.gca().set_aspect('equal', adjustable='box')  # Make the plot square
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='turbo', norm=norm if umi is not None else plt.Normalize(vmin=0, vmax=len(x)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical')
    cbar.set_label('Log1p of UMI Value' if umi is not None else 'Index')
    
    if gn_list is not None:
        # Highlight points where y < x and the difference is greater than 0.3
        for i, name in enumerate(gn_list):
            if y[i] < x[i] and (x[i] - y[i]) > 0.3:
                plt.text(x[i], y[i], name, fontsize=8, ha='right', va='bottom', color='red')  # Make the text red
        # Highlight points in the top 5% of y - x
        differences = np.array(y) - np.array(x)
        threshold = np.percentile(differences, 95)  # Top 5% threshold
        for i, name in enumerate(gn_list):
            if differences[i] >= threshold:
                plt.text(x[i], y[i], name, fontsize=8, ha='right', va='bottom', color='green')  # Make the text green
        # Highlight points in the top 5% of y
        y_threshold = np.percentile(y, 95)  # Top 5% threshold for y
        for i, name in enumerate(gn_list):
            if y[i] >= y_threshold:
                plt.text(x[i], y[i], name, fontsize=8, ha='right', va='bottom', color='orange')  # Make the text orange
    
    plt.savefig(f"{output}{metric}_{x_label}_{y_label}_scatter_plot.png")
    plt.close()
    print(f"Scatter plot saved to {output}{metric}_{x_label}_{y_label}_scatter_plot.png")


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--oTrue_pref', type=str)
    parser.add_argument('--oPred_pref', nargs='*', help='Zero or more pref for oPred data', default=None)
    parser.add_argument('--dTrue_pref', type=str, default=None)
    parser.add_argument('--dPred_pref', nargs='*', help='Zero or more pref for dPred data', default=None)
    parser.add_argument('--oPred_method', nargs='*', help='Zero or more methods for oPred', default=None)
    parser.add_argument('--dPred_method', nargs='*', help='Zero or more methods for dPred', default=None)
    parser.add_argument('--output', type=str)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=20000)    
    parser.add_argument('--gene_names', nargs='*', help='Zero or more selected gene names for plotting', default=None)
    parser.add_argument('--overlay', action='store_true')
    parser.add_argument('--all_spots', action='store_true')
    return parser.parse_args()

def main():
    args = get_args()
    oTrue_pref = args.oTrue_pref
    oPred_pref = args.oPred_pref ##  list of oPred prefs
    dTrue_pref = args.dTrue_pref
    dPred_pref = args.dPred_pref ##  list of dPred prefs
    overlay = args.overlay
    all_spots = args.all_spots
    start_idx = args.start_idx
    end_idx = args.end_idx
    oPred_method = args.oPred_method ## list of oPred methods
    dPred_method = args.dPred_method ## list of dPred methods
    mask = load_mask(args.mask) if args.mask else None
    output = args.output + 'gene_metrics_comparison_all_spots/' if all_spots else args.output + 'gene_metrics_comparison/'
    output = output + args.mask.split('/')[-1].split('.')[0] +'/' if args.mask is not None else output + 'no_mask/'
    factor = 16
    assert (dPred_pref is None and dPred_method is None) or (dPred_pref is not None and dPred_method is not None and len(dPred_pref) == len(dPred_method))
    assert (oPred_pref is None and oPred_method is None) or (oPred_pref is not None and oPred_method is not None), "oPred_pref and oPred_method should be both None or both not None"      
    
 
    
    metrics_file = f"{output}No.{start_idx}-No.{end_idx}_metrics_data.pickle"
    if os.path.exists(metrics_file) and args.gene_names is None:
        print(f"Metrics data file already exists: {metrics_file}. Skipping computation.")
        metrics_dict_df = load_pickle(metrics_file)
        
        metrics_names = list(metrics_dict_df.keys())
        oumi_counts_df = load_pickle(f"{output}No.{start_idx}-No.{end_idx}_UMI_counts.pickle")
        
        all_genes = read_lines(f'{oTrue_pref}gene-names.txt')  # Read all genes
        d_genes = read_lines(f'{dTrue_pref}gene-names.txt') if dTrue_pref is not None else None
        all_genes = [g for g in all_genes if g in d_genes] if d_genes is not None else all_genes
        end_idx = len(all_genes) if end_idx > len(all_genes) else end_idx
        gene_names = all_genes[start_idx:end_idx]
        print(f"Number of genes evaluating: {len(gene_names)}")
    else:
        he = img_reduce(load_image(f'{oTrue_pref}he.jpg'), factor=factor)

        ## Load location data for original, imputed, downsampled and downsampled_imputed data
        oTrue_locs, oTrue_row_indices = handle_locations(oTrue_pref, factor=factor)
        if oPred_pref is not None and oPred_method is not None:
            oPred_locs = []; oPred_row_indices = []
            for i in range(len(oPred_pref)):
                if oPred_method[i] != "Original-Imputation":
                    oPred_locs_i, oPred_row_indices_i = handle_locations(oPred_pref[i], factor=factor)
                else:
                    oPred_locs_i, oPred_row_indices_i = None, None
                oPred_locs.append(oPred_locs_i); oPred_row_indices.append(oPred_row_indices_i)
        else:
            oPred_locs = None; oPred_row_indices = None
        dTrue_locs, dTrue_row_indices = handle_locations(dTrue_pref, factor=factor) if dTrue_pref is not None else (None, None)
        if dPred_pref is not None and dPred_method is not None:
            dPred_locs = []; dPred_row_indices = []
            for i in range(len(dPred_pref)):
                if dPred_method[i] != "Downsampled-Imputation":
                    dPred_locs_i, dPred_row_indices_i = handle_locations(dPred_pref[i], factor=factor)
                else:
                    dPred_locs_i, dPred_row_indices_i = None, None
                dPred_locs.append(dPred_locs_i); dPred_row_indices.append(dPred_row_indices_i)
        else:
            dPred_locs = None; dPred_row_indices = None

        ## Load count data if they are .tsv file
        oTrue_cnts = (load_tsv(f'{oTrue_pref}cnts.tsv')).iloc[oTrue_row_indices, ] if os.path.isfile(f'{oTrue_pref}cnts.tsv') else None
        if oPred_pref is not None and oPred_method is not None:
            oPred_cnts = []
            for i in range(len(oPred_pref)):
                oPred_cnts_i = (load_tsv(f'{oPred_pref[i]}cnts.tsv')).iloc[oPred_row_indices[i], ] if os.path.isfile(f'{oPred_pref[i]}cnts.tsv') else None
                oPred_cnts.append(oPred_cnts_i)
        else:
            oPred_cnts = None
        dTrue_cnts = (load_tsv(f'{dTrue_pref}cnts.tsv')).iloc[dTrue_row_indices, ] if os.path.isfile(f'{dTrue_pref}cnts.tsv') else None
        if dPred_pref is not None and dPred_method is not None:
            dPred_cnts = []
            for i in range(len(dPred_pref)):
                dPred_cnts_i = (load_tsv(f'{dPred_pref[i]}cnts.tsv')).iloc[dPred_row_indices[i], ] if os.path.isfile(f'{dPred_pref[i]}cnts.tsv') else None
                dPred_cnts.append(dPred_cnts_i)
        else:
            dPred_cnts = None
        
        print(f"args.gene_names is {args.gene_names}")
        if args.gene_names is not None:
            gene_names = args.gene_names
            o_genes = read_lines(f'{oTrue_pref}gene-names.txt')  # Read all genes
            gene_names = [g for g in gene_names if g in o_genes]
            d_genes = read_lines(f'{dTrue_pref}gene-names.txt') if dTrue_pref is not None else None
            gene_names = [g for g in gene_names if g in d_genes] if d_genes is not None else gene_names
            print(f"Number of genes evaluating: {len(gene_names)}")
            for g in gene_names:
                print(f"Index for gene {g} in o_genes is {np.where(np.array(o_genes) == g)[0][0]}")
            gene_sets = {
            "custom": gene_names
            }
            start_idx=None; end_idx=None
        else:
            o_genes = read_lines(f'{oTrue_pref}gene-names.txt')  # Read all genes
            d_genes = read_lines(f'{dTrue_pref}gene-names.txt') if dTrue_pref is not None else None
            o_genes = [g for g in o_genes if g in d_genes] if d_genes is not None else o_genes
            end_idx = len(o_genes) if end_idx > len(o_genes) else end_idx
            gene_names = o_genes[start_idx:end_idx]
            print(f"Number of genes evaluating: {len(gene_names)}")
            gene_sets = {
                "top_25": o_genes[:len(o_genes) // 4],
                "middle_50": o_genes[len(o_genes) // 4:3 * len(o_genes) // 4],
                "bottom_25": o_genes[3 * len(o_genes) // 4:]
            }
      
        metrics_names = ['Pearson Correlation', 'Spearman Correlation', 'Cosine Similarity', 'SSIM', 'Dice Coefficient', 'Jaccard Index', 'Chi Square', 'p_value']
        method_names = ['Original-Observed']
        if oPred_method is not None:
            for i in range(len(oPred_pref)):
                method_names.append(oPred_method[i])
        if dTrue_pref is not None:
            method_names.append('Downsampled-Observed')
        if dPred_pref is not None:
            for i in range(len(dPred_pref)):
                method_names.append(dPred_method[i])
        metrics_dict_df = {metric: pd.DataFrame(index=gene_names, columns=method_names[1:]) for metric in metrics_names}
        metrics_dict_df['Moran\'s I'] = pd.DataFrame(index=gene_names, columns=method_names)
        oumi_counts_df = pd.DataFrame(index=gene_names, columns=["oTrue_UMI_Count"])


        for key, gene_set in gene_sets.items():
            base_dir = f"{output}/raw/{key}/"
            overlay_dir = f"{output}/overlay/{key}/"
            os.makedirs(base_dir, exist_ok=True)
            os.makedirs(overlay_dir, exist_ok=True)
            
            for gn in gene_names:
                if gn in gene_set:
                    oTrue = load_image_data(gn, oTrue_pref, oTrue_locs, oTrue_row_indices, mask.shape, counts=oTrue_cnts)
                    morans_i_oTrue = fast_morans_i_with_nans(oTrue)
                    metrics_dict_df['Moran\'s I'].loc[gn, 'Original-Observed'] = morans_i_oTrue
                    oumi_counts_df.loc[gn, "oTrue_UMI_Count"] = np.nansum(oTrue[mask])
                    
                    if oPred_pref is not None:
                        oPred = []; oPredDict = []
                        for i in range(len(oPred_pref)):
                            if oPred_method[i] == 'Original-Imputation':
                                oPred_i = load_pickle(f'{oPred_pref[i]}cnts-super/{gn}.pickle')
                                oPred.append(oPred_i)
                            else:
                                oPred_i = load_image_data(gn, oPred_pref[i], oPred_locs[i], oPred_row_indices[i], mask.shape, counts=oPred_cnts[i])
                                oPred.append(oPred_i)
                            oPredDict_i = {**calculate_metrics(oTrue, oPred_i, mask, all_spots),
                                            **calculate_binary_metrics(oTrue, oPred_i, mask=mask, thresh=90)}
                            oPredDict_i['Moran\'s I'] = fast_morans_i_with_nans(oPred_i)
                            oPredDict.append(oPredDict_i)
                            for metric in metrics_names:
                                metrics_dict_df[metric].loc[gn, oPred_method[i]] = oPredDict_i[metric]
                            metrics_dict_df['Moran\'s I'].loc[gn, oPred_method[i]] = oPredDict_i['Moran\'s I']
                    else:
                        oPred = None ; oPredDict = None;                           
                    
                    if dTrue_pref is not None:
                        dTrue = load_image_data(gn, dTrue_pref, dTrue_locs, dTrue_row_indices, mask.shape, counts=dTrue_cnts)
                        dTrueDict = {**calculate_metrics(oTrue, dTrue, mask, all_spots),
                                        **calculate_binary_metrics(oTrue, dTrue, mask=mask, thresh=90)}
                        dTrueDict['Moran\'s I'] = fast_morans_i_with_nans(dTrue)
                        for metric in metrics_names:
                            metrics_dict_df[metric].loc[gn, 'Downsampled-Observed'] = dTrueDict[metric]
                        metrics_dict_df['Moran\'s I'].loc[gn, 'Downsampled-Observed'] = dTrueDict['Moran\'s I'] 
                    else:
                        dTrue = None; dTrueDict = None;
                        
                    if dPred_pref is not None:
                        dPred = []; dPredDict = []
                        for i in range(len(dPred_pref)):
                            if dPred_method[i] == 'Downsampled-Imputation':
                                dPred_i = load_pickle(f'{dPred_pref[i]}cnts-super/{gn}.pickle')
                                dPred.append(dPred_i)
                            else:
                                dPred_i = load_image_data(gn, dPred_pref[i], dPred_locs[i], dPred_row_indices[i], mask.shape, counts=dPred_cnts[i])
                                dPred.append(dPred_i)
                            dPredDict_i = {**calculate_metrics(oTrue, dPred_i, mask, all_spots),
                                            **calculate_binary_metrics(oTrue, dPred_i, mask=mask, thresh=90)}
                            dPredDict_i['Moran\'s I'] = fast_morans_i_with_nans(dPred_i)
                            dPredDict.append(dPredDict_i)
                            for metric in metrics_names:
                                metrics_dict_df[metric].loc[gn, dPred_method[i]] = dPredDict_i[metric] 
                            metrics_dict_df['Moran\'s I'].loc[gn, dPred_method[i]] = dPredDict_i['Moran\'s I']    
                    else:
                        dPred = None ; dPredDict = None;                     
                    
                    metrics_df = pd.DataFrame(columns=['Moran\'s I'] + metrics_names, index=method_names)
                    metrics_df['Moran\'s I'].loc['Original-Observed'] = morans_i_oTrue
                    if oPred_method is not None:
                        for i, method in enumerate(oPred_method):
                            metrics_df.loc[method] = [oPredDict[i][key] for key in ['Moran\'s I'] + metrics_names]                      
                    if dTrue_pref is not None:
                        metrics_df.loc['Downsampled-Observed'] = [dTrueDict[key] for key in ['Moran\'s I'] + metrics_names]
                    if dPred_pref is not None:
                        for i, method in enumerate(dPred_method):
                            metrics_df.loc[method] = [dPredDict[i][key] for key in ['Moran\'s I'] + metrics_names]                    
                             
                    if not os.path.exists(f"{base_dir}{gn}.png"):
                        save_comparison_plot(
                            he=he,
                            oTrue=oTrue,
                            metrics_df=metrics_df,
                            oPred=oPred,
                            dTrue=dTrue,
                            dPred=dPred,
                            outname=base_dir + f"{gn}.png",
                            gn=gn,
                            overlay=False
                        )
                    
                    if overlay:
                        if not os.path.exists(f"{overlay_dir}{gn}.png"):
                            save_comparison_plot(
                                he=he,
                                oTrue=oTrue,
                                metrics_df=metrics_df,
                                oPred=oPred,
                                dTrue=dTrue,
                                dPred=dPred,
                                outname=overlay_dir + f"{gn}.png",
                                gn=gn,
                                overlay=True
                            )
                    
        if start_idx is not None and end_idx is not None:
            save_pickle(metrics_dict_df, metrics_file)
            save_pickle(oumi_counts_df, f"{output}No.{start_idx}-No.{end_idx}_UMI_counts.pickle")

    if start_idx is not None and end_idx is not None:
        # plot_comparison_violin(metrics_dict_df=metrics_dict_df, output=f"{output}No.{start_idx}-No.{end_idx}_")    
        plot_comparison_boxplot(metrics_dict_df=metrics_dict_df, metric_names=['Moran\'s I'], output=f"{output}No.{start_idx}-No.{end_idx}_")
        plot_comparison_boxplot(metrics_dict_df=metrics_dict_df, metric_names=metrics_names, output=f"{output}No.{start_idx}-No.{end_idx}_")

    if oPred_pref is not None:
        for i in range(len(oPred_method)):                
            plot_comparison_scatter(
                x=metrics_dict_df['Moran\'s I']['Original-Observed'],
                y=metrics_dict_df['Moran\'s I'][oPred_method[i]],
                output=f"{output}No.{start_idx}-No.{end_idx}_",
                metric='Moran\'s I',
                x_label='Original-Observed',
                y_label=oPred_method[i],
                gn_list=gene_names,
                umi=oumi_counts_df
            )
        for metrics in metrics_names + ['Moran\'s I']:  
            for pair in combinations(oPred_method, 2):
                plot_comparison_scatter(
                    x=metrics_dict_df[metrics][pair[0]],
                    y=metrics_dict_df[metrics][pair[1]],
                    output=f"{output}No.{start_idx}-No.{end_idx}_",
                    metric=metrics,
                    x_label=pair[0],
                    y_label=pair[1],
                    gn_list=gene_names,
                    umi=oumi_counts_df
                ) 
        
    if dPred_pref is not None and dTrue_pref is not None:        
        for metric in metrics_names + ['Moran\'s I']:   
            for i in range(len(dPred_method)):
                plot_comparison_scatter(
                    x=metrics_dict_df[metric]['Downsampled-Observed'],
                    y=metrics_dict_df[metric][dPred_method[i]],
                    output=f"{output}No.{start_idx}-No.{end_idx}_",
                    metric=metric,
                    x_label='Downsampled-Observed',
                    y_label=dPred_method[i],
                    gn_list=gene_names,
                    umi=oumi_counts_df
                )
            for pair in combinations(dPred_method, 2):
                plot_comparison_scatter(
                    x=metrics_dict_df[metric][pair[0]],
                    y=metrics_dict_df[metric][pair[1]],
                    output=f"{output}No.{start_idx}-No.{end_idx}_",
                    metric=metric,
                    x_label=pair[0],
                    y_label=pair[1],
                    gn_list=gene_names,
                    umi=oumi_counts_df
                )
                                
        
if __name__ == '__main__':
    main()
