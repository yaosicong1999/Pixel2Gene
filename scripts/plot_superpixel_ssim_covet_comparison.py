import sys
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_pickle, read_lines, load_mask, load_tsv, load_image, save_pickle
from structural_similarity import structural_similarity
import argparse


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
        return x / np.nanmax(x)
    else:
        cutoff_l = np.sort(np.unique(y))[min_level - 1]
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

def plot_img_comparison(img1, img2, img0, gn, title, width, height, save_fig=True, out_dir=None, fig_name='comparison.jpg',
                        dpi=200):
    fig, axs = plt.subplots(1, 3, figsize=(width, height))
    axs[0].imshow(img1)
    axs[0].set_title('Predicted Gene Expression w. COVET')
    axs[1].imshow(img2)
    axs[1].set_title('Predicted Gene Expression w.o. COVET')
    axs[2].imshow(img0)
    axs[2].set_title('True Gene Expression')
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
    parser.add_argument('--clip2', type=float, default=99)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    prefix = args.prefix
    gene_names = read_lines(f'{prefix}gene-names.txt')
    mask = args.mask
    clip0 = args.clip0
    clip1 = args.clip1
    clip2 = args.clip2
    factor = 16

    if mask is not None:
        mask = load_mask(mask)
        print("mask shape is ", mask.shape)
    else:
        he_test = load_image(f'{prefix}test-he.jpg')
        mask = np.ones([int(he_test.shape[0] / 16), int(he_test.shape[1] / 16)], dtype=bool)

    locs_test = load_tsv(f'{prefix}test-locs.tsv')
    locs_test = locs_test.astype(float)
    locs_test = np.stack([locs_test['y'], locs_test['x']], -1)
    locs_test //= factor
    locs_test = locs_test.round().astype(int)
    unique_rows, indices, counts = np.unique(locs_test, axis=0, return_index=True, return_counts=True)
    unique_row_indices = indices[counts == 1]
    locs_test = locs_test[unique_row_indices,]

    cnts_test = load_tsv(f'{prefix}test-cnts.tsv')
    cnts_test = cnts_test.iloc[unique_row_indices, :]

    os.makedirs(prefix + 'comparison_covet/ssim_full', exist_ok=True)
    os.makedirs(prefix + 'comparison_covet/ssim_top100', exist_ok=True)
    ssim_rel_covet_list = []
    ssim_rel_nocovet_list = []
    ssim_abs_covet_list = []
    ssim_abs_nocovet_list = []
    gene_list = []

    height = 8
    width = mask.shape[1] / mask.shape[0] * (height - 1) * 3

    intersected_genes = list(set(gene_names).intersection(cnts_test.columns.to_list()))
    print("number of genes for plotting is", intersected_genes.__len__())
    for gn in gene_names:
        if gn in intersected_genes:
            new_truth = np.full((mask.shape[0], mask.shape[1]), np.nan)
            new_truth[locs_test[:, 0], locs_test[:, 1]] = cnts_test[gn]
            if mask is not None:
                new_truth[~mask] = np.nan

            covet = load_pickle(f'{prefix}cnts-super-covet/{gn}.pickle')
            new_covet = np.full((mask.shape[0], mask.shape[1]), np.nan)
            new_covet[locs_test[:, 0], locs_test[:, 1]] = covet[locs_test[:, 0], locs_test[:, 1]]
            if mask is not None:
                new_covet[~mask] = np.nan
            # new_covet = np.round(new_covet)

            nocovet = load_pickle(f'{prefix}cnts-super/{gn}.pickle')
            new_nocovet = np.full((mask.shape[0], mask.shape[1]), np.nan)
            new_nocovet[locs_test[:, 0], locs_test[:, 1]] = nocovet[locs_test[:, 0], locs_test[:, 1]]
            if mask is not None:
                new_nocovet[~mask] = np.nan
            # new_covet = np.round(new_covet)

            # new_truth2 = np.clip(new_truth, a_min=0, a_max=np.percentile(new_truth, 95))
            # new_covet2 = np.clip(new_covet, a_min=0, a_max=np.percentile(new_covet, 95))
            # new_nocovet2 = np.clip(new_nocovet, a_min=0, a_max=np.percentile(new_nocovet, 95))
            new_truth2 = get_clipped(new_truth, percentile=clip0)
            img0 = plot_super(new_truth2 / np.nanmax(new_truth2))
            new_covet2 = get_clipped(new_covet, percentile=clip1)
            img1 = plot_super(new_covet2 / np.nanmax(new_covet2))
            new_nocovet2 = get_clipped(new_nocovet, percentile=clip2)
            img2 = plot_super(new_nocovet2 / np.nanmax(new_nocovet2))

            new_truth[np.isnan(new_truth)] = 0
            new_covet[np.isnan(new_covet)] = 0
            new_nocovet[np.isnan(new_nocovet)] = 0
            ssim_abs_covet = structural_similarity(new_covet, new_truth, channel_axis=None)
            ssim_abs_covet_list.append(ssim_abs_covet)
            ssim_abs_nocovet = structural_similarity(new_nocovet, new_truth, channel_axis=None)
            ssim_abs_nocovet_list.append(ssim_abs_nocovet)
            f0 = new_truth / np.nanmax(new_truth)
            f1 = new_covet / np.nanmax(new_covet)
            f2 = new_nocovet / np.nanmax(new_nocovet)
            ssim_rel_covet = structural_similarity(f1, f0, channel_axis=None)
            ssim_rel_covet_list.append(ssim_rel_covet)
            ssim_rel_nocovet = structural_similarity(f2, f0, channel_axis=None)
            ssim_rel_nocovet_list.append(ssim_rel_nocovet)

            gene_list.append(gn)

            df = pd.DataFrame(pd.DataFrame(
                {'Truth': new_truth.flatten(), 'Covet': new_covet.flatten(), 'No-Covet': new_nocovet.flatten()}))
            print(df['Truth'])
            print(df['Covet'])
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            sc = ax.scatter(x=df['Truth'], y=df['Covet'], s=1)
            ax.plot([0, 1], [0, 1], color='red', linestyle='--')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('True expression')
            ax.set_ylabel('Prediction with COVET')
            ax.set_aspect('equal', adjustable='box')
            plt.title('True vs Covet for '+ gn)
            plt.savefig(prefix + "comparison_covet/covet_" + gn + ".png", dpi=200)

            df = pd.DataFrame(pd.DataFrame(
                {'Truth': new_truth.flatten(), 'Covet': new_covet.flatten(), 'No-Covet': new_nocovet.flatten()}))
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            sc = ax.scatter(x=df['Truth'], y=df['No-Covet'], s=1)
            ax.plot([0, 1], [0, 1], color='red', linestyle='--')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('True expression')
            ax.set_ylabel('Prediction without COVET')
            ax.set_aspect('equal', adjustable='box')
            plt.title('True vs No-Covet for '+ gn)
            plt.savefig(prefix + "comparison_covet/nocovet_" + gn + ".png", dpi=200)

            '''
            title = 'SSIM by Relative Value w. COVET: ' + str(np.round(ssim_rel_covet, 3)) + ', SSIM by Relative Value w.o. COVET: ' + str(
                np.round(ssim_rel_nocovet, 3))
            if ssim_rel_covet_list.__len__() <= 100:
                plot_img_comparison(img1, img2, img0, gn=gn, width=width, height=height,
                                    save_fig=True, out_dir=[prefix + 'comparison_covet/ssim_full/', prefix + 'comparison_covet/ssim_top100/'],
                                    title=title, fig_name=gn + '.jpg')
            else:
                plot_img_comparison(img1, img2, img0,  gn=gn, width=width, height=height,
                                    save_fig=True, out_dir=[prefix + 'comparison_covet/ssim_full/'],
                                    title=title, fig_name=gn + '.jpg')


    df = pd.DataFrame(pd.DataFrame({'Gene': gene_list, 'SSIM-Abs-covet': ssim_abs_covet_list, 'SSIM-Abs-nocovet': ssim_abs_nocovet_list, 'SSIM-Rel-covet': ssim_rel_covet_list, 'SSIM-Rel-nocovet': ssim_rel_nocovet_list}))
    df.to_csv(prefix + 'comparison_covet/ssim_comparison.tsv', sep='\t')

    fig = plt.subplots()
    ax = sns.violinplot(data=df[['SSIM-Rel-covet', 'SSIM-Rel-nocovet']], orient="h")
    ax = sns.stripplot(data=df[['SSIM-Rel-covet', 'SSIM-Rel-nocovet']], orient="h", color='black', jitter=True)
    quantiles = [0.25, 0.5, 0.75]
    linestyles = [':', '-', '--']
    for i in range(3):
        q = quantiles[i]
        quantile_line = df.quantile(q)
        ax.axvline(quantile_line['SSIM-Rel-covet'], linestyle=linestyles[i], color='blue',
                   label=f'{int(q * 100)}% Quantile of SSIM-Rel-covet')
        ax.axvline(quantile_line['SSIM-Rel-nocovet'], linestyle=linestyles[i], color='orange',
                   label=f'{int(q * 100)}% Quantile of SSIM-Rel-nocovet')
    plt.xlim(0, 1.2)
    plt.legend()
    plt.title('SSIM for w. and w.o. COVET Predictions for All Genes', fontsize=16)
    plt.savefig(prefix + 'comparison_covet/violin_plot_full.png', dpi=200)

    fig = plt.subplots()
    ax = sns.violinplot(data=df[['SSIM-Rel-covet', 'SSIM-Rel-nocovet']][0:100], orient="h")
    ax = sns.stripplot(data=df[['SSIM-Rel-covet', 'SSIM-Rel-nocovet']][0:100], orient="h", color='black', jitter=True)
    quantiles = [0.25, 0.5, 0.75]
    linestyles = [':', '-', '--']
    for i in range(3):
        q = quantiles[i]
        quantile_line = df[0:100].quantile(q)
        ax.axvline(quantile_line['SSIM-Rel-covet'], linestyle=linestyles[i], color='blue',
                   label=f'{int(q * 100)}% Quantile of SSIM-Rel-covet')
        ax.axvline(quantile_line['SSIM-Rel-nocovet'], linestyle=linestyles[i], color='orange',
                   label=f'{int(q * 100)}% Quantile of SSIM-Rel-nocovet')
    plt.xlim(0, 1.2)
    plt.legend()
    plt.title('SSIM for w. and w.o. COVET Predictions for Top 100 HVG', fontsize=16)
    plt.savefig(prefix + 'comparison_covet/violin_plot_top100.png', dpi=200)

    mean_exp = np.mean(cnts_test[df['Gene']])
    var_exp = np.var(cnts_test[df['Gene']])
    df['mean'] = mean_exp.tolist()
    df['var'] = var_exp.tolist()

    df1 = df.copy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sc = ax.scatter(x=df1['SSIM-Rel-nocovet'], y=df1['SSIM-Rel-covet'], s=1, c=df1['mean'], cmap='turbo')
    for i in range(100):
        if df1['SSIM-Rel-covet'][i] - df1['SSIM-Rel-nocovet'][i] > 0.05:
            ax.annotate(gene_names[i], (df1['SSIM-Rel-nocovet'][i], df1['SSIM-Rel-covet'][i]), textcoords="offset points", xytext=(-2, -3),
                        ha='center', fontsize=6, color='red')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Prediction without COVET')
    ax.set_ylabel('Prediction with COVET')
    ax.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(sc)
    cbar.set_label('Gene Expression Mean')
    plt.title('SSIM for iStar Prediction w./w.o. COVET for All Genes')
    plt.savefig(prefix + "comparison_covet/ssim_scatter_mean_full.png", dpi=200)

    df1 = df[0:100]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sc = ax.scatter(x=df1['SSIM-Rel-nocovet'], y=df1['SSIM-Rel-covet'], s=1, c=df1['mean'], cmap='turbo')
    for i in range(100):
        if df1['SSIM-Rel-covet'][i] - df1['SSIM-Rel-nocovet'][i] > 0.05:
            ax.annotate(gene_names[i], (df1['SSIM-Rel-nocovet'][i], df1['SSIM-Rel-covet'][i]), textcoords="offset points", xytext=(-2, -3),
                        ha='center', fontsize=6, color='red')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Prediction without COVET')
    ax.set_ylabel('Prediction with COVET')
    ax.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(sc)
    cbar.set_label('Gene Expression Mean')
    plt.title('SSIM for iStar Prediction w./w.o. COVET for Top 100 HVG')
    plt.savefig(prefix + "comparison_covet/ssim_scatter_mean_top100.png", dpi=200)
            '''
if __name__ == '__main__':
    main()
