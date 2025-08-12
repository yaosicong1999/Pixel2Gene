import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_pickle, save_image, read_lines, load_image, load_tsv, load_mask, load_parquet, save_pickle
from my_utils import cmapFader, img_reduce, locs_reduce, plot_super
from structural_similarity import structural_similarity

def plot_comparison_scatter(x, y, output, name_list=None, metrics='Pearson Correlation'):
    """Plot scatter plot for two input vectors x and y, with optional point labels."""
    plt.figure(figsize=(6, 6))
    colors = plt.cm.turbo(np.linspace(0, 1, len(x)))  # Use a continuous colormap
    scatter = plt.scatter(x, y, color=colors, alpha=0.8)
    plt.xlabel('original observed v.s. downsampled-SAVER')
    plt.ylabel('original observed v.s. downsampled-imputed')
    plt.title('Scatter Plot for ' + metrics)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1)  # y=x line
    plt.gca().set_aspect('equal', adjustable='box')  # Make the plot square
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(vmin=0, vmax=len(x)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical')
    cbar.set_label('Index')
    
    if name_list is not None:
        # Highlight points where y < x and the difference is greater than 0.3
        for i, name in enumerate(name_list):
            if y[i] < x[i] and (x[i] - y[i]) > 0.3:
                plt.text(x[i], y[i], name, fontsize=8, ha='right', va='bottom', color='red')  # Make the text red
        # Highlight points in the top 5% of y - x
        differences = np.array(y) - np.array(x)
        threshold = np.percentile(differences, 95)  # Top 5% threshold
        for i, name in enumerate(name_list):
            if differences[i] >= threshold:
                plt.text(x[i], y[i], name, fontsize=8, ha='right', va='bottom', color='green')  # Make the text green
        # Highlight points in the top 5% of y
        y_threshold = np.percentile(y, 95)  # Top 5% threshold for y
        for i, name in enumerate(name_list):
            if y[i] >= y_threshold:
                plt.text(x[i], y[i], name, fontsize=8, ha='right', va='bottom', color='green')  # Make the text orange
    plt.savefig(f"{output}{metrics}_scatter_plot.png")
    plt.close()
    print(f"Scatter plot saved to {output}{metrics}_scatter_plot.png")



def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--impute_pref', type=str)
    parser.add_argument('--saver_pref', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--n_top', type=int, default=None)
    parser.add_argument('--all_spots', action='store_true')
    return parser.parse_args()

args = get_args()
impute_pref = args.impute_pref
saver_pref = args.saver_pref
output = args.output
mask = args.mask
n_top = args.n_top
all_spots = args.all_spots

impute_pref= "../visiumhd_heg/P1_downsampled_full_0.1_0.03/filter_locs_he_qc/"
saver_pref= "../visiumhd_heg/P1_downsampled_full_0.1_0.03/saver_imputed/"
mask = "../visiumhd_heg/CRC-P1-downsampled6-mask-small-filter_locs_he_qc.png"
all_spots=False
n_top=100

impute_dir = impute_pref  + 'gene_metrics_comparison_all_spots/' if all_spots else impute_pref + 'gene_metrics_comparison/'
impute_dir = impute_dir + mask.split('/')[-1].split('.')[0] + '/top' + str(n_top) + '_metrics_data.pickle' if mask is not None else impute_dir + 'no_mask/top_' + str(n_top) + '_metrics_data.pickle'
saver_dir = saver_pref + 'gene_metrics_comparison_all_spots/' if all_spots else saver_pref + 'gene_metrics_comparison/'
saver_dir = saver_dir + mask.split('/')[-1].split('.')[0] + '/top' + str(n_top) + '_metrics_data.pickle' if mask is not None else saver_dir + 'no_mask/top_' + str(n_top) + '_metrics_data.pickle'

impute_metrics = load_pickle(impute_dir)
saver_metrics = load_pickle(saver_dir)


plot_comparison_scatter(x=saver_metrics['dimputedCosine'], y=impute_metrics['dimputedCosine'], output='', name_list=None, metrics='Cosine Similarity')

plot_comparison_scatter(x=saver_metrics['dimputedRsq'], y=impute_metrics['dimputedRsq'], output='', name_list=None, metrics='Pearson Correlation')
