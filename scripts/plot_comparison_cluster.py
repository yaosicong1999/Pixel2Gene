import argparse
from math import e
import os
from tkinter import font
import numpy as np
import matplotlib.pyplot as plt
from utils import load_mask, load_pickle, read_lines, save_pickle
from sklearn.metrics import adjusted_rand_score
from plot_labels_overlay import plot_overlay
from sklearn.metrics import normalized_mutual_info_score
import argparse
from evaluate_ICC import calculate_icc_per_pc_and_cluster, read_parquet_cnts, read_tsv_cnts, read_imputed_pickles
from reduce_dim import reduce_dim 

def get_n_level_folders_by_structure(dirs, n):
    """
    Retrieve n-level subfolders, keeping track of the parent structure up to (n-1) levels.
    Returns a dictionary where keys are tuples representing the parent structure (up to n-1 levels),
    and values are sets of n-th level folder names.

    Parameters:
        dirs (list): List of root directories to scan.
        n (int): The depth of folder levels to retrieve.

    Returns:
        dict: A dictionary with keys as tuples of parent folder names (up to n-1 levels)
              and values as sets of n-th level folder names.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    def traverse_and_collect(current_path, current_level):
        """
        Recursively traverse directories to collect n-th level folders.
        """
        if current_level == n - 1:
            # Collect n-th level folders
            from pathlib import Path
            current_path = Path(current_path)
            return {tuple(current_path.parts[-(n-1):]): {entry.name for entry in os.scandir(current_path) if entry.is_dir()}}
        else:
            # Traverse deeper
            collected = {}
            for entry in os.scandir(current_path):
                if entry.is_dir():
                    deeper_collected = traverse_and_collect(entry.path, current_level + 1)
                    for key, value in deeper_collected.items():
                        if key in collected:
                            collected[key].update(value)
                        else:
                            collected[key] = value
            return collected
    n_level_dict = {}
    for root_dir in dirs:
        # Ensure root_dir is a valid string path
        if isinstance(root_dir, str) and os.path.isdir(root_dir):
            collected = traverse_and_collect(os.path.abspath(root_dir), 0)
            for key, value in collected.items():
                if key in n_level_dict:
                    n_level_dict[key].update(value)
                else:
                    n_level_dict[key] = value
        else:
            print(f"Warning: {root_dir} is not a valid directory.")
    return n_level_dict


def compare_multiple_folders_and_plot(dirs, data_list, output_dir, subtitles=None, overlay=False, ground_truth=None):
    """
    Compare cluster labels from multiple directories and generate visualizations.
    
    Parameters:
        dirs (list): List of directories containing clustering results.
        output_dir (str): Directory where output images will be saved.
        subtitles (list, optional): Titles for each comparison plot.
        overlay (bool, optional): If True, use "labels_overlayed.png", otherwise use "labels.png".
    """
    num_dirs = len(dirs)
    if num_dirs < 2:
        print("At least two directories are required for comparison.")
        return
    if subtitles is None or len(subtitles) != num_dirs:
        subtitles = [f"Clustering {i + 1}" for i in range(num_dirs)]  
    common_folders = get_n_level_folders_by_structure(dirs, n=2) 
    # Iterate over common parent structures and their n-th level folders
    for parent_structure, nth_layer_folders in common_folders.items():
        for nth_layer in nth_layer_folders:
            labels_list = []
            folder_path = os.path.join(*parent_structure, nth_layer)
            print(f"Processing folder: {folder_path}")
            # Load clustering labels
            for d in dirs:
                labels_path = os.path.join(d, folder_path, "labels.pickle")
                if os.path.exists(labels_path):
                    labels_list.append(load_pickle(labels_path))
                else:
                    labels_list.append(None)             
            if all(labels is not None for labels in labels_list):
                if ground_truth is None:
                    ground_truth = labels_list[0]        
                # Remove labels where the first labels list has -1
                valid_indices = labels_list[0].flatten() != -1
                valid_data_list = [data[valid_indices] for data in data_list]
                flatten_labels_list = [labels.reshape(-1)[valid_indices] for labels in labels_list]
                flatten_ground_truth = ground_truth.reshape(-1)[valid_indices]       
                # Compute scores
                print(f"for icc input the data shape is {valid_data_list[0].shape}, the label shape is {flatten_labels_list[0].shape}")
                iccs = [calculate_icc_per_pc_and_cluster(valid_data_list[i], flatten_labels_list[i]) for i in range(0, num_dirs)]
                icc_array = [icc[1] for icc in iccs]
                icc_mean = [np.round(icc[2],3) for icc in iccs]
                icc_median = [np.round(icc[3],3) for icc in iccs]
                ari_scores = [round(adjusted_rand_score(flatten_ground_truth, flatten_labels_list[i]), 3) for i in range(1, num_dirs)]
                nmi_scores = [round(normalized_mutual_info_score(flatten_ground_truth, flatten_labels_list[i]), 3) for i in range(1, num_dirs)]
                metrics = {
                    "ICC-array": icc_array,
                    "ICC-Mean": icc_mean,
                    'ICC-Median': icc_median,
                    "ARI": ari_scores,
                    "NMI": nmi_scores
                }
                # Create output directory
                output_subdir = os.path.join(output_dir, *parent_structure, nth_layer)
                os.makedirs(output_subdir, exist_ok=True)  
                # Plot cluster images
                fig, axes = plt.subplots(1, num_dirs, figsize=(4 * num_dirs, 5.2))
                if num_dirs == 2:
                    axes = [axes]  # Ensure axes is iterable for two directories
                for i, d in enumerate(dirs):
                    img_filename = "labels_overlayed.png" if overlay else "labels.png"
                    labels_img_path = os.path.join(d, folder_path, img_filename)
                    if os.path.exists(labels_img_path):
                        img = plt.imread(labels_img_path)
                        axes[i].imshow(img)
                        axes[i].axis('off')
                        if i == 0:
                            axes[i].set_title(f"{subtitles[i]}\nICC-Mean: {icc_mean[i]:.3f}\nICC-Median: {icc_median[i]:.3f}")
                        else:
                            axes[i].set_title(f"{subtitles[i]}\nICC-Mean: {icc_mean[i]:.3f}\nICC-Median: {icc_median[i]:.3f}\nARI: {ari_scores[i - 1]:.3f}\nNMI: {nmi_scores[i - 1]:.3f}")                
                # Extract number of clusters from folder name (if applicable)
                num_clusters = nth_layer[-3:].lstrip('0')
                plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space at the top for suptitle
                # plt.suptitle(f"Number of Clusters: {num_clusters}", fontsize=16)
                output_file = os.path.join(output_subdir, f"clustering_comparison_{num_clusters}.png")
                plt.savefig(output_file)
                plt.close()
                save_pickle(metrics, os.path.join(output_subdir, "metrics.pickle"))

def run_plot_overlay_on_all_folders(dirs, data_pref, n_layer=3):
    """
    Runs plot_overlay for each n-th layer folder in the common directory structure.
    
    Parameters:
        dirs (list): List of base directories.
        data_pref (str): The prefix used for data input in plot_overlay.
        n_layer (int): The depth of folder levels to process.
    """
    # Get common n-th level folder structure
    common_folders = get_n_level_folders_by_structure(dirs, n=n_layer)
    for parent_structure, nth_layer_folders in common_folders.items():
        for nth_layer in nth_layer_folders:
            for base_dir in dirs:  # Iterate over base directories
                folder_path = os.path.join(base_dir, *parent_structure, nth_layer)
                # Ensure the folder exists before calling plot_overlay
                if os.path.exists(folder_path):
                    print(f"Running plot_overlay on: {folder_path}")
                    plot_overlay(data_pref=data_pref, label_pref=folder_path, save=True)
                else:
                    print(f"Skipping non-existent folder: {folder_path}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_dir', type=str)
    parser.add_argument('--downsampled_dir', type=str, default=None)
    parser.add_argument('--data_pref', type=str, default=None)
    parser.add_argument('--downsampled_pref', type=str, default=None)
    parser.add_argument('--ground_truth_dir', type=str, default=None)
    parser.add_argument('--imputation_method', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()  
    original_dir = args.original_dir
    downsampled_dir = args.downsampled_dir
    data_pref = args.data_pref
    downsampled_pref = args.downsampled_pref
    ground_truth_dir = args.ground_truth_dir
    
    if downsampled_dir is not None:
        dirs = [original_dir + 'clusters_truth/', 
                original_dir + 'clusters_pred_gene_pickle/',
                downsampled_dir + 'clusters_truth/',
                downsampled_dir + 'clusters_pred_gene_pickle/']
        subtitles = ['Original', 'Original-Imputed', 'Downsampled', 'Downsampled-Imputed']
    else:
        dirs = [original_dir + 'clusters_truth/', 
                original_dir + 'clusters_pred_gene_pickle/']
        subtitles = ['Original', 'Original-Imputed']    
    
    ## whether we should use any cluster labels as ground truth 
    if ground_truth_dir is not None:
        ground_truth = load_pickle(ground_truth_dir + 'labels.pickle')

    if ground_truth_dir is None:        
        base_dir = downsampled_dir + 'cluster_comparison_top12000topc/'
    else:
        base_dir = downsampled_dir + 'cluster_comparison_top12000topc_with_ground_truth/'
        
    ## loading data for ICC comparison 
    n_top = 12000
    gene_names_original = read_lines(data_pref + 'gene-names.txt')
    gene_names_downsampled = read_lines(downsampled_pref + 'gene-names.txt')
    gene_names = [gene for gene in gene_names_original if gene in gene_names_downsampled]
    gene_names = gene_names[:n_top]

    mask = load_mask(f"{data_pref}mask-small-RGB.png")

    if os.path.exists(f'{data_pref}cnts-top{n_top}genes_50pc.npy'):
        print(f"Loading truth data from {data_pref}cnts-top{n_top}genes_50pc.npy")
        truth_flat = np.load(f'{data_pref}cnts-top{n_top}genes_50pc.npy')
    else:
        cnts_truth, locs_truth = read_parquet_cnts(data_pref, gene_names=gene_names) if os.path.exists(data_pref+'cnts.parquet') else read_tsv_cnts(data_pref, gene_names=gene_names)
        mat_truth = np.full((mask.shape[0], mask.shape[1], cnts_truth.shape[1]), np.nan, dtype=np.float32)
        mat_truth[locs_truth[:,0], locs_truth[:,1], :] = cnts_truth.to_numpy()
        truth_flat = mat_truth.reshape(-1, mat_truth.shape[2])
        del cnts_truth, mat_truth
        truth_flat = reduce_dim(truth_flat, n_components=50)[0]
        np.save(f'{data_pref}cnts-top{n_top}genes_50pc.npy', truth_flat)


    if os.path.exists(f'{original_dir}cnts-top{n_top}genes_50pc.npy'):
        print(f"Loading imputed data from {original_dir}cnts-top{n_top}genes_50pc.npy")
        imputed_flat = np.load(f'{original_dir}cnts-top{n_top}genes_50pc.npy')
    else:
        data_imputed = read_imputed_pickles(original_dir, mask.shape, gene_names)
        imputed_flat = data_imputed.reshape(-1, data_imputed.shape[2])    
        del data_imputed
        imputed_flat = reduce_dim(imputed_flat, n_components=50)[0]
        imputed = (imputed_flat.reshape(mask.shape[0], mask.shape[1], -1))
        imputed[~mask, :] = np.nan
        imputed_flat = imputed.reshape(-1, imputed.shape[2])
        np.save(f'{original_dir}cnts-top{n_top}genes_50pc.npy', imputed_flat)

        
    if downsampled_dir is not None:
        if os.path.exists(f'{downsampled_pref}cnts-top{n_top}genes_50pc.npy'):
            print(f'Loading downsampled data from {downsampled_pref}cnts-top{n_top}genes_50pc.npy')
            downsampled_flat = np.load(f'{downsampled_pref}cnts-top{n_top}genes_50pc.npy')
        else:
            cnts_downsampled, locs_downsampled = read_parquet_cnts(downsampled_pref, gene_names=gene_names) if os.path.exists(downsampled_pref+'cnts.parquet') else read_tsv_cnts(downsampled_pref, gene_names=gene_names)
            mat_downsampled = np.full((mask.shape[0], mask.shape[1], cnts_downsampled.shape[1]), np.nan, dtype=np.float32)
            mat_downsampled[locs_downsampled[:,0], locs_downsampled[:,1], :] = cnts_downsampled.to_numpy()
            downsampled_flat = mat_downsampled.reshape(-1, mat_downsampled.shape[2])
            del cnts_downsampled, mat_downsampled
            downsampled_flat = reduce_dim(downsampled_flat, n_components=50)[0]
            np.save(f'{downsampled_pref}cnts-top{n_top}genes_50pc.npy', downsampled_flat)
            
        if os.path.exists(f'{downsampled_dir}cnts-top{n_top}genes_50pc.npy'):
            print(f'Loading imputed downsampled data from {downsampled_dir}cnts-top{n_top}genes_50pc.npy')
            imputed_downsampled_flat = np.load(f'{downsampled_dir}cnts-top{n_top}genes_50pc.npy')
        else:
            data_downsampled_imputed = read_imputed_pickles(downsampled_dir, mask.shape, gene_names)
            imputed_downsampled_flat = data_downsampled_imputed.reshape(-1, data_downsampled_imputed.shape[2])
            del data_downsampled_imputed
            imputed_downsampled_flat = reduce_dim(imputed_downsampled_flat, n_components=50)[0]
            imputed_downsampled = (imputed_downsampled_flat.reshape(mask.shape[0], mask.shape[1], -1))
            imputed_downsampled[~mask, :] = np.nan
            imputed_downsampled_flat = imputed_downsampled.reshape(-1, imputed_downsampled.shape[2])
            np.save(f'{downsampled_dir}cnts-top{n_top}genes_50pc.npy', imputed_downsampled_flat)

    data_list = [truth_flat[:, :50], 
                 imputed_flat[:, :50], 
                 downsampled_flat[:, :50], 
                 imputed_downsampled_flat[:, :50]]
    output_dir = base_dir + '/raw/'    
    os.makedirs(output_dir, exist_ok=True)

    compare_multiple_folders_and_plot(dirs=dirs, data_list=data_list, output_dir=output_dir, subtitles=subtitles, overlay=False, ground_truth=ground_truth)

    if data_pref is not None:
        # run_plot_overlay_on_all_folders(dirs=dirs, data_pref=data_pref, n_layer=3)
        output_dir = base_dir + '/overlay/'
        os.makedirs(output_dir, exist_ok=True)
        
        compare_multiple_folders_and_plot(dirs=dirs, data_list=data_list, output_dir=output_dir, subtitles=subtitles, overlay=True, ground_truth=ground_truth)
                                                                                                                                             
if __name__ == "__main__":
    main()