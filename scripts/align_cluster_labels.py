import os
import pandas as pd
from scipy.optimize import linear_sum_assignment
from utils import load_pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from visual import plot_labels
from PIL import Image, ImageOps
from sklearn.metrics import adjusted_rand_score


def get_aligned_labels(cl0, cl1):
    # Compute confusion matrix between cl0 and cl1
    matrix = pd.crosstab(cl0, cl1)
    cost_matrix = -matrix.values  # Hungarian algorithm minimizes cost
    # Find optimal one-to-one mapping
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Create mapping from cl1 to cl0
    cl0_labels = matrix.index.values
    cl1_labels = matrix.columns.values
    mapping = {cl1_labels[col_ind[i]]: cl0_labels[row_ind[i]] for i in range(len(row_ind))}
    # Apply mapping
    new_cl1 = np.array([mapping[label] for label in cl1])
    return new_cl1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cl0_dir', type=str)
    parser.add_argument('--cl1_dir', type=str)
    parser.add_argument('--save_aligned_plot', action='store_true')
    return parser.parse_args()

def main():
    args = parse_args()
    cl0_dir = args.cl0_dir
    cl1_dir = args.cl1_dir
    save_aligned_plot = args.save_aligned_plot
            
    
    cl0_mat = load_pickle(f'{cl0_dir}/labels.pickle')
    cl1_mat = load_pickle(f'{cl1_dir}/labels.pickle')

    palette = plt.get_cmap('tab20').colors
    
    cl0_flat = cl0_mat.flatten()
    cl1_flat = cl1_mat.flatten()

    select_idx = [i for i in range(len(cl0_flat)) if cl0_flat[i] != -1 and cl1_flat[i] != -1]
    cl0_flat_select = cl0_flat[select_idx]
    cl1_flat_select = cl1_flat[select_idx]
    
    cl1_flat_aligned = get_aligned_labels(cl0_flat_select, cl1_flat_select)
    cl1_flat[select_idx] = cl1_flat_aligned
    
    cl1_mat_aligned = cl1_flat.reshape(cl1_mat.shape)
    
    if save_aligned_plot:
        plot_labels(cl1_mat_aligned, f'{cl1_dir}labels_aligned.png', white_background=True)
        
        # Flatten and filter out -1 labels for ARI calculation
        cl0_valid = cl0_mat.flatten()
        cl1_valid = cl1_mat_aligned.flatten()
        valid_idx = (cl0_valid != -1) & (cl1_valid != -1)
        ari = adjusted_rand_score(cl0_valid[valid_idx], cl1_valid[valid_idx])

        # Save ARI score to a txt file
        ari_output_path = os.path.join(cl1_dir, "ari_score.txt")
        with open(ari_output_path, "w") as f:
            f.write(f"ARI: {ari}\n")

    
if __name__ == '__main__':
    main()
