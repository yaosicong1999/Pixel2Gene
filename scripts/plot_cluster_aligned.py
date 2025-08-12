import os
import pandas as pd
from scipy.optimize import linear_sum_assignment
from utils import load_pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from visual import plot_labels
from PIL import Image, ImageOps


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

def cmap_tab20(x):
    cmap = plt.get_cmap('tab20')
    x = np.asarray(x)
    x_mod = x % 20
    idx = (x_mod // 10) + (x_mod % 10) * 2  # reorder second 10 colors
    return cmap(idx / 20.0)

def cmap_tab30(x):
    x = np.asarray(x)
    n_base = 20
    n_max = 30
    brightness = np.array([0.7, 0.7, 0.7, 1.0])
    isin_base = (x < n_base)[..., np.newaxis]
    isin_extended = ((x >= n_base) & (x < n_max))[..., np.newaxis]
    isin_beyond = (x >= n_max)[..., np.newaxis]
    color = (
        isin_base * cmap_tab20(x)
        + isin_extended * cmap_tab20(x - n_base) * brightness
        + isin_beyond * np.array([0.0, 0.0, 0.0, 1.0])
    )
    return color

def cmap_tab70(x):
    x = np.asarray(x)
    brightness = np.array([0.5, 0.5, 0.5, 1.0])
    brightness_sq = brightness ** 2
    cmap_base = cmap_tab30
    colors = [
        cmap_base(x),                                 # 0–29
        1 - (1 - cmap_base(x - 30)) * brightness,     # 30–39
        cmap_base(x - 30) * brightness,               # 40–49
        1 - (1 - cmap_base(x - 50)) * brightness_sq,  # 50–59
        cmap_base(x - 50) * brightness_sq,            # 60–69
        np.array([0.0, 0.0, 0.0, 1.0])                # 70+
    ]
    isin = [
        (x < 30)[..., np.newaxis],
        ((x >= 30) & (x < 40))[..., np.newaxis],
        ((x >= 40) & (x < 50))[..., np.newaxis],
        ((x >= 50) & (x < 60))[..., np.newaxis],
        ((x >= 60) & (x < 70))[..., np.newaxis],
        (x >= 70)[..., np.newaxis]
    ]
    color_out = sum(isi * col for isi, col in zip(isin, colors))
    return color_out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str)
    parser.add_argument('--hipt_dir', type=str)
    parser.add_argument('--o_dir', type=str)
    parser.add_argument('--p_dir', type=str, default=None)
    parser.add_argument('--d_dir', type=str, default=None)
    parser.add_argument('--dp_dir', type=str, default=None)
    parser.add_argument('--n_cl', type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    hipt_dir = args.hipt_dir
    o_dir = args.o_dir
    p_dir = args.p_dir
    d_dir = args.d_dir
    dp_dir = args.dp_dir
    output = args.output
    n_cl = args.n_cl
    
    print(f'n_cl: {n_cl}')
    os.makedirs(output, exist_ok=True)
    hipt_mat = load_pickle(f'{hipt_dir}/labels.pickle')
    o_mat = load_pickle(f'{o_dir}/labels.pickle')
    hipt_flat = hipt_mat.flatten()
    o_flat = o_mat.flatten()

    select_idx = [i for i in range(len(o_flat)) if hipt_flat[i] != -1 and o_flat[i] != -1]

    hipt_flat_select = hipt_flat[select_idx]    
    o_flat_select = o_flat[select_idx]
    o_aligned = get_aligned_labels(hipt_flat_select, o_flat_select)
    o_flat[select_idx] = o_aligned
    o_mat_aligned = o_flat.reshape(hipt_mat.shape)
    plot_labels(hipt_mat, f'{output}{n_cl}_hipt.png', white_background=True)
    plot_labels(o_mat_aligned, f'{output}{n_cl}_o_aligned.png', white_background=True) 
    
    border_size = 3  # change as needed
    img = Image.open(f'{output}{n_cl}_hipt.png')
    bordered_img = ImageOps.expand(img, border=border_size, fill='black')
    bordered_img.save(f'{output}{n_cl}_hipt.png')
    img = Image.open(f'{output}{n_cl}_o_aligned.png')
    bordered_img = ImageOps.expand(img, border=border_size, fill='black')
    bordered_img.save(f'{output}{n_cl}_o_aligned.png')
    
    
    if p_dir is not None:
        p_mat = load_pickle(f'{p_dir}/labels.pickle') 
        p_flat = p_mat.flatten()
        select_idx = [i for i in range(len(o_flat)) if hipt_flat[i] != -1 and p_flat[i] != -1]
        hipt_flat_select = hipt_flat[select_idx]    
        p_flat_select = p_flat[select_idx]
        p_aligned = get_aligned_labels(hipt_flat_select, p_flat_select)
        p_flat[select_idx] = p_aligned
        p_mat_aligned = p_flat.reshape(hipt_mat.shape)
        plot_labels(p_mat_aligned, f'{output}{n_cl}_p_aligned.png', white_background=True)
        
        img = Image.open(f'{output}{n_cl}_p_aligned.png')
        bordered_img = ImageOps.expand(img, border=border_size, fill='black')
        bordered_img.save(f'{output}{n_cl}_p_aligned.png')
    
    if d_dir is not None:
        d_mat = load_pickle(f'{d_dir}/labels.pickle') 
        d_flat = d_mat.flatten()
        select_idx = [i for i in range(len(o_flat)) if hipt_flat[i] != -1 and d_flat[i] != -1]
        hipt_flat_select = hipt_flat[select_idx]    
        d_flat_select = d_flat[select_idx]
        d_aligned = get_aligned_labels(hipt_flat_select, d_flat_select)
        d_flat[select_idx] = d_aligned
        d_mat_aligned = d_flat.reshape(hipt_mat.shape)
        plot_labels(d_mat_aligned, f'{output}{n_cl}_d_aligned.png', white_background=True)
        
        img = Image.open(f'{output}{n_cl}_d_aligned.png')
        bordered_img = ImageOps.expand(img, border=border_size, fill='black')
        bordered_img.save(f'{output}{n_cl}_d_aligned.png')
    
    if dp_dir is not None:
        dp_mat = load_pickle(f'{dp_dir}/labels.pickle') 
        dp_flat = dp_mat.flatten()
        select_idx = [i for i in range(len(o_flat)) if hipt_flat[i] != -1 and dp_flat[i] != -1]
        hipt_flat_select = hipt_flat[select_idx]    
        dp_flat_select = dp_flat[select_idx]
        dp_aligned = get_aligned_labels(hipt_flat_select, dp_flat_select)
        dp_flat[select_idx] = dp_aligned
        dp_mat_aligned = dp_flat.reshape(hipt_mat.shape)
        plot_labels(dp_mat_aligned, f'{output}{n_cl}_dp_aligned.png', white_background=True)
        
        img = Image.open(f'{output}{n_cl}_dp_aligned.png')
        bordered_img = ImageOps.expand(img, border=border_size, fill='black')
        bordered_img.save(f'{output}{n_cl}_dp_aligned.png')
    
    
    
    
if __name__ == '__main__':
    main()