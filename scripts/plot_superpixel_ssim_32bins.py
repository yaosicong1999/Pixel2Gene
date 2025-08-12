import sys
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_pickle, read_lines, load_mask, load_tsv, load_image, save_pickle
from structural_similarity import structural_similarity
import argparse
import itertools

def plot_super(x, truncate=None):
    x = x.copy()
    mask = np.isfinite(x)
    if truncate is not None:
        x = np.clip(x, truncate[0], truncate[1])
    cmap = plt.get_cmap('turbo')
    img = cmap(x)[..., :3]
    img[~mask] = 1.0
    img = (img * 255).astype(np.uint8)
    return img

def get_clipped(x, min_level=5, percentile=95):
    y = x.copy().flatten()
    y = y[~np.isnan(y)]
    n_levels = np.unique(y).__len__()
    if n_levels <= min_level:
        return x/np.nanmax(x)
    else:
        cutoff_l = np.sort(np.unique(y))[min_level-1]
        cutoff_p = np.percentile(y, percentile)
        y = x.copy()
        y = np.clip(y, a_min=0, a_max=np.max([cutoff_l, cutoff_p]))
        y = y / np.nanmax(y)
        return y

def standardize_image(image):
    min_val = np.nanmin(image)
    max_val = np.nanmax(image)
    standardized_image = (image - min_val) / (max_val - min_val)
    return standardized_image

def plot_img_comparison(img1, img0, gn, title, width, height, save_fig=True, out_dir=None, fig_name='comparison.jpg', dpi=200):
    fig, axs = plt.subplots(1, 2, figsize=(width, height))
    axs[0].imshow(img1)
    axs[0].set_title('Predicted Gene Expression')
    axs[1].imshow(img0)
    axs[1].set_title('True Gene Expression')
    fig.suptitle('Gene: ' + gn, fontsize=16)
    fig.text(0.5, 0.93, title, ha='center', va='center', fontsize=12, color='blue')
    plt.tight_layout()
    if save_fig:
        for d in out_dir:
            plt.savefig(d + fig_name, dpi=dpi)
    else:
        plt.show()
    plt.close(fig)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--clip0', type=float, default=95)
    parser.add_argument('--clip1', type=float, default=99)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    prefix = args.prefix
    gene_names = read_lines(f'{prefix}gene-names.txt')
    mask = args.mask
    clip0 = args.clip0
    clip1 = args.clip1
    factor = 16

    if mask is not None:
        mask = load_mask(mask)
        print("mask shape is ", mask.shape)
    else:
        he_test = load_image(f'{prefix}test-he.jpg')
        mask = np.ones([int(he_test.shape[0] / 32), int(he_test.shape[1] / 32)], dtype=bool)

    locs_test = load_tsv(f'{prefix}test-locs-32px.tsv')
    locs_test = locs_test.astype(float)
    locs_test = np.stack([locs_test['y'], locs_test['x']], -1)
    locs_test //= factor
    locs_test = locs_test.round().astype(int)
    unique_rows, indices, counts = np.unique(locs_test, axis=0, return_index=True, return_counts=True)
    unique_row_indices = indices[counts == 1]
    locs_test = locs_test[unique_row_indices, ]

    cnts_test = load_tsv(f'{prefix}test-cnts-32px.tsv')
    cnts_test = cnts_test.iloc[unique_row_indices, :]

    os.makedirs(prefix + 'ssim_full_32bins', exist_ok=True)
    os.makedirs(prefix + 'ssim_top100_32bins', exist_ok=True)
    ssim_rel_list = []
    ssim_abs_list = []
    gene_name_list = []

    height = 8
    width = mask.shape[1] / mask.shape[0] * (height-1) * 2

    intersected_genes = list(set(gene_names).intersection(cnts_test.columns.to_list()))
    print("number of genes for plotting is", intersected_genes.__len__())

    if os.path.exists(prefix + 'superpixel_ssim_comparison_32bins.tsv'):
        df = pd.read_csv(prefix + 'superpixel_ssim_comparison_32bins.tsv', sep='\t', index_col=0)
        ssim_abs_list = df['SSIM-Abs'].tolist()
        ssim_rel_list = df['SSIM-Rel'].tolist()
    else:
        for gn in gene_names:
            if gn in intersected_genes:
                new_truth = np.full((mask.shape[0], mask.shape[1]), np.nan)
                new_truth[locs_test[:, 0], locs_test[:, 1]] = cnts_test[gn]
                if mask is not None:
                    new_truth[~mask] = np.nan

                predicted = load_pickle(f'{prefix}cnts-super/{gn}.pickle')
                size = 2
                new_predicted0 = np.zeros(
                    (np.floor(predicted.shape[0] / size).astype(int), np.floor(predicted.shape[1] / size).astype(int)))
                df = pd.DataFrame(list(itertools.product(range(predicted.shape[0]), range(predicted.shape[1]))),
                                  columns=['x', 'y'])
                df['ge'] = predicted.flatten()
                df['bin_x'] = np.floor(df.x / size).astype('int')
                df['bin_y'] = np.floor(df.y / size).astype('int')
                df2 = df.groupby(['bin_x', 'bin_y']).agg('sum').reset_index()
                new_predicted0 = df2['ge'].to_numpy().reshape((new_predicted0.shape[0], new_predicted0.shape[1]))

                new_predicted = np.full((mask.shape[0], mask.shape[1]), np.nan)
                new_predicted[locs_test[:, 0], locs_test[:, 1]] = new_predicted0[locs_test[:, 0], locs_test[:, 1]]
                if mask is not None:
                    new_predicted[~mask] = np.nan
                # new_predicted = np.round(new_predicted)

                # new_predicted2 = np.clip(new_predicted, a_min=0, a_max=np.percentile(new_predicted, 95))
                # new_truth2 = np.clip(new_truth, a_min=0, a_max=np.percentile(new_truth, 95))
                new_truth2 = get_clipped(new_truth, percentile=clip0)
                img0 = plot_super(new_truth2 / np.nanmax(new_truth2))
                new_predicted2 = get_clipped(new_predicted, percentile=clip1)
                img1 = plot_super(new_predicted2 / np.nanmax(new_predicted2))

                new_predicted[np.isnan(new_predicted)] = 0
                new_truth[np.isnan(new_truth)] = 0
                ssim_abs = structural_similarity(new_predicted, new_truth, channel_axis=None)
                ssim_abs_list.append(ssim_abs)
                f1 = new_predicted / np.nanmax(new_predicted)
                f0 = new_truth / np.nanmax(new_truth)
                ssim_rel = structural_similarity(f1, f0, channel_axis=None)
                ssim_rel_list.append(ssim_rel)

                gene_name_list.append(gn)

                title = 'SSIM by Relative Value: ' + str(np.round(ssim_rel, 3)) + ', SSIM by Absolute Value: ' + str(
                    np.round(ssim_abs, 3))
                if ssim_rel_list.__len__() <= 100:
                    plot_img_comparison(img1, img0, gn=gn, width=width, height=height,
                                        save_fig=True, out_dir=[prefix + 'ssim_full_32bins/', prefix + 'ssim_top100_32bins/'],
                                        title=title, fig_name=gn + '.jpg')
                else:
                    plot_img_comparison(img1, img0, gn=gn, width=width, height=height,
                                        save_fig=True, out_dir=[prefix + 'ssim_full_32bins/'],
                                        title=title, fig_name=gn + '.jpg')

        df = pd.DataFrame(
            {'Gene': gene_name_list, 'SSIM-Abs': ssim_abs_list, 'SSIM-Rel': ssim_rel_list})
        df.to_csv(prefix + 'superpixel_ssim_comparison.tsv', sep='\t')
        # save_pickle(pd.DataFrame({'Gene': gene_name_list, 'SSIM-Abs': ssim_abs_list}), filename=prefix + "ssim_abs_list.pickle")
        # save_pickle(pd.DataFrame({'Gene': gene_name_list, 'SSIM-Rel': ssim_rel_list}), filename=prefix + "ssim_rel_list.pickle")

    ## All genes
    df = pd.DataFrame({'SSIM-Rel': ssim_rel_list})
    fig = plt.subplots()
    ax = sns.violinplot(data=df, orient="h", color='cornflowerblue')
    ax = sns.stripplot(data=df, orient="h", color='lightskyblue', jitter=True)
    quantiles = [0.25, 0.5, 0.75]
    linestyles = [':', '-', '--']
    for i in range(3):
        q = quantiles[i]
        quantile_line = df.quantile(q)
        ax.axvline(quantile_line['SSIM-Rel'], linestyle=linestyles[i], color='red', label=f'{int(q * 100)}% Quantile of SSIM')
    plt.xlim(0, 1)
    plt.legend()
    plt.title('SSIM by Relative Value: Predicted vs True\n All ' + str(len(ssim_rel_list)) + ' Genes', fontsize=16)
    plt.savefig(prefix + 'violin_plot_rel_full.png', dpi=200)

    df = pd.DataFrame({'SSIM-Abs': ssim_abs_list})
    fig = plt.subplots()
    ax = sns.violinplot(data=df, orient="h", color='cornflowerblue')
    ax = sns.stripplot(data=df, orient="h", color='lightskyblue', jitter=True)
    quantiles = [0.25, 0.5, 0.75]
    linestyles = [':', '-', '--']
    for i in range(3):
        q = quantiles[i]
        quantile_line = df.quantile(q)
        ax.axvline(quantile_line['SSIM-Abs'], linestyle=linestyles[i], color='red', label=f'{int(q * 100)}% Quantile of SSIM')
    plt.xlim(0, 1)
    plt.legend()
    plt.title('SSIM by Absolute Value: Predicted vs True\n All ' + str(len(ssim_abs_list)) + ' Genes', fontsize=16)
    plt.savefig(prefix + 'violin_plot_abs_full.png', dpi=200)

    ## Top 100 HVG
    df = pd.DataFrame({'SSIM-Rel': ssim_rel_list[0:100]})
    fig = plt.subplots()
    ax = sns.violinplot(data=df, orient="h", color='cornflowerblue')
    ax = sns.stripplot(data=df, orient="h", color='lightskyblue', jitter=True)
    quantiles = [0.25, 0.5, 0.75]
    linestyles = [':', '-', '--']
    for i in range(3):
        q = quantiles[i]
        quantile_line = df.quantile(q)
        ax.axvline(quantile_line['SSIM-Rel'], linestyle=linestyles[i], color='red',
                   label=f'{int(q * 100)}% Quantile of SSIM')
    plt.xlim(0, 1)
    plt.legend()
    plt.title('SSIM by Relative Value: Predicted vs True\n Top ' + str(100) + ' HVG', fontsize=16)
    plt.savefig(prefix + 'violin_plot_rel_top100.png', dpi=200)

    df = pd.DataFrame({'SSIM-Abs': ssim_abs_list[0:100]})
    fig = plt.subplots()
    ax = sns.violinplot(data=df, orient="h", color='cornflowerblue')
    ax = sns.stripplot(data=df, orient="h", color='lightskyblue', jitter=True)
    quantiles = [0.25, 0.5, 0.75]
    linestyles = [':', '-', '--']
    for i in range(3):
        q = quantiles[i]
        quantile_line = df.quantile(q)
        ax.axvline(quantile_line['SSIM-Abs'], linestyle=linestyles[i], color='red',
                   label=f'{int(q * 100)}% Quantile of SSIM')
    plt.xlim(0, 1)
    plt.legend()
    plt.title('SSIM by Absolute Value: Predicted vs True\n Top ' + str(100) + ' HVG', fontsize=16)
    plt.savefig(prefix + 'violin_plot_abs_top100.png', dpi=200)


if __name__ == '__main__':
    main()