import sys
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from impute import get_data
from utils import load_pickle, read_lines, load_image, load_mask, read_string, load_tsv
from einops import reduce
from image import get_disk_mask
from structural_similarity import structural_similarity
import argparse

palette = [plt.cm.tab20(i) for i in range(20)]

def plot_super(x, underground=None, truncate=None):
    x = x.copy()
    mask = np.isfinite(x)
    if truncate is not None:
        x -= np.nanmean(x)
        x /= np.nanstd(x) + 1e-12
        x = np.clip(x, truncate[0], truncate[1])
    x -= np.nanmin(x)
    x /= np.nanmax(x) + 1e-12

    cmap = plt.get_cmap('turbo')
    # cmap = cmap_turbo_truncated
    if underground is not None:
        under = underground.mean(-1, keepdims=True)
        under -= under.min()
        under /= under.max() + 1e-12

    img = cmap(x)[..., :3]
    if underground is not None:
        img = img * 0.5 + under * 0.5
    img[~mask] = 1.0
    img = (img * 255).astype(np.uint8)
    return img

def plot_sc_spots(
        img, cnts, locs, radius, cmap='turbo',
        weight=0.8, disk_mask=True, standardize_img=False):
    cnts = cnts.astype(np.float32)

    img = img.astype(np.float32)
    img /= 255.0

    if standardize_img:
        if np.isclose(0.0, np.nanstd(img, (0, 1))).all():
            img[:] = 1.0
        else:
            img -= np.nanmin(img)
            img /= np.nanmax(img) + 1e-12

    cnts -= np.nanmin(cnts)
    cnts /= np.nanmax(cnts) + 1e-12

    cmap = plt.get_cmap(cmap)
    if disk_mask:
        mask_patch = get_disk_mask(radius)
    else:
        mask_patch = np.ones((radius*2, radius*2)).astype(bool)
    indices_patch = np.stack(np.where(mask_patch), -1)
    indices_patch -= radius
    for ij, ct in zip(locs, cnts):
        color = np.array(cmap(ct)[:3])
        indices = indices_patch + ij
        img[indices[:, 0], indices[:, 1]] *= 1 - weight
        img[indices[:, 0], indices[:, 1]] += color * weight
    img = (img * 255).astype(np.uint8)
    return img

    # recale image
    locs = locs.astype(float)
    locs = np.stack([locs['y'], locs['x']], -1)
    locs /= factor
    locs = locs.round().astype(int)
    img = reduce(
            img.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
            h=factor, w=factor).astype(np.uint8)

    # rescale spot
    spot_radius = np.round(spot_radius / factor).astype(int)

    # plot spot-level gene expression measurements
    plot_spots_multi(
            cnts=cnts,
            locs=locs, gene_names=gene_names,
            radius=spot_radius, disk_mask=True,
            img=img, prefix=prefix_sc+'spots/')

def standardize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    standardized_image = (image - min_val) / (max_val - min_val)
    return standardized_image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--overlay', type=str, default='False')
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--optprefix1', type=str, default=None)
    parser.add_argument('--optprefix2', type=str, default=None)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    prefix = args.prefix
    gene_names = read_lines(f'{prefix}gene-names.txt')
    factor = 16

    infile_radius = f'{prefix}test-radius.txt'
    spot_radius = int(read_string(infile_radius))
    spot_radius = np.round(spot_radius / factor).astype(int)

    if args.mask is not None:
        mask = load_mask(args.mask)
        print("mask shape is ", mask.shape)
    else:
        mask = None

    print("the overlay parameter is", args.overlay)
    if args.overlay == 'True':
        infile_he = f'{prefix}test-he-scaled.jpg'
        print("using h&e overlay...")
        he = load_image(infile_he)
        print("he shape before reducing is ", he.shape)
        if he.dtype == bool:
            he = he.astype(np.uint8) * 255
        if he.ndim == 2:
            he = np.tile(he[..., np.newaxis], 3)
        he = reduce(
            he.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
            h=factor, w=factor).astype(np.uint8)
        print("he shape after reducing is ", he.shape)
    else:
        print("not using h&e overlay...")
        he = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) * 255

    cnts_test = load_tsv(f'{prefix}test-cnts.tsv')

    locs_test = load_tsv(f'{prefix}test-locs-raw.tsv')
    locs_test = locs_test.astype(float)
    locs_test = np.stack([locs_test['y'], locs_test['x']], -1)
    locs_test //= factor
    locs_test = locs_test.round().astype(int)

    if not os.path.exists(prefix + 'correlation_comparison'):
        os.mkdir(prefix + 'correlation_comparison')

    n_comparison = 1
    prefix_list = [prefix]
    if args.optprefix1 is not None:
        n_comparison = n_comparison + 1
        prefix_list.append(args.optprefix1)
    if args.optprefix2 is not None:
        n_comparison = n_comparison + 1
        prefix_list.append(args.optprefix2)

    ssim_mat = np.full((gene_names.__len__(), n_comparison), np.nan)
    corr_mat = np.full((gene_names.__len__(), n_comparison), np.nan)

    for gn_idx in range(gene_names.__len__()):
        gn = gene_names[gn_idx]
        img_list = []

        f2 = np.full((he.shape[0], he.shape[1]), np.nan)
        f2[locs_test[:, 0], locs_test[:, 1]] = np.array(cnts_test[gn])
        f2 = standardize_image(f2)

        for i in range(n_comparison):
            # the super-resolution pickle will be in the shape of (width_pixel/16, length_pixel_16)
            predicted = load_pickle(f'{prefix_list[i]}cnts-super/{gn}.pickle')
            new_predicted = np.full((he.shape[0], he.shape[1]), np.nan)
            new_predicted[locs_test[:, 0], locs_test[:, 1]] = predicted[locs_test[:, 0], locs_test[:, 1]]
            predicted = new_predicted
            if mask is not None:
                predicted[~mask] = np.nan
            img = plot_super(predicted)
            img_list.append(img)

            f1 = standardize_image(predicted)
            ssim = structural_similarity(f1, f2, channel_axis=None)
            ssim_mat[gn_idx, i] = ssim
            corr = np.corrcoef(f1[locs_test[:, 0], locs_test[:, 1]], f2[locs_test[:, 0], locs_test[:, 1]])[0, 1]
            corr_mat[gn_idx, i] = corr

        img2 = plot_sc_spots(img=he, cnts=np.array(cnts_test[gn]), locs=locs_test,
                             radius=spot_radius)
        img_list.append(img2)

        nrow = 1; ncol = n_comparison + 1
        fig, axs = plt.subplots(nrow, ncol, figsize=(5*ncol, 5*nrow))
        for idx in range(ncol-1):
            axs[idx].imshow(img_list[idx])
            axs[idx].set_title('Predicted Gene Expression from ' + prefix_list[idx])
        axs[ncol].imshow(img_list[ncol])
        axs[ncol].set_title('True Gene Expression')
        plt.tight_layout()

        fig.suptitle('Gene: ' + gn, fontsize=16)
        fig.text(0.5, 0.9,
                 'SSIMs: ' + str(np.round(ssim_mat[gn_idx,], 3)) +
                 ' Correlations: ' + str(np.round(corr_mat[gn_idx,], 3)),
                 ha='center', va='center', fontsize=14, color='blue')
        filename = prefix +'correlation_comparison/' + gn + ".png"
        plt.savefig(filename, dpi=200)

    df1 = pd.DataFrame(ssim_mat, columns=['SSIM_' + s for s in prefix_list])
    df2 = pd.DataFrame(corr_mat, columns=['Corr_' + s for s in prefix_list])
    df = pd.concat([df1, df2], axis=1)

    ax = sns.violinplot(data=df)
    quantiles = [0.25, 0.5, 0.75]
    for q in quantiles:
        quantile_line = df.quantile(q)
        for ldx in range(quantile_line.__len__()):
            ax.axhline(y=quantile_line[ldx], linestyle='--', color=palette[ldx], label=f'{int(q * 100)}% Quantile of ' + quantile_line.index[ldx])
    plt.legend()
    plt.savefig(prefix + 'violin_comparison_plot.png')

if __name__ == '__main__':
    main()