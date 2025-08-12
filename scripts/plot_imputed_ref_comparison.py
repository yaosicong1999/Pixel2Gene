import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import load_pickle, save_image, read_lines, load_image, load_tsv, load_mask, save_pickle
from my_utils import cmapFader, img_reduce
from structural_similarity import structural_similarity


def standardize_image(image):
    min_val = np.nanmin(image)
    max_val = np.nanmax(image)
    standardized_image = (image - min_val) / (max_val - min_val)
    return standardized_image

def plot_super(x, truncate=None, he=None, locs_ref=None):
    x = x.copy()
    mask = np.isfinite(x)
    if truncate is not None:
        x = np.clip(x, truncate[0], truncate[1])
    # col = cmapFader(cmap_name='turbo', start_val=0, stop_val=1)
    # img = col.get_rgb(x)[:, :, :3]
    cmap = plt.get_cmap('turbo')
    img = cmap(x)[..., :3]
    img[~mask] = 1.0
    img = (img * 255).astype(np.uint8)
    if locs_ref is not None:
        if he is not None:
            out = he.copy()
        else:
            out = np.full((x.shape[0], x.shape[1], 3), 255)
        out[locs_ref[:, 0], locs_ref[:, 1], :] = img[locs_ref[:, 0], locs_ref[:, 1], :]
        img = out
    filter = np.isnan(x)
    if he is not None:
        img[filter] = he[filter]
    return img

def bhattacharyya_distance(hist1, hist2):
    p = hist1 / np.sum(hist1)
    q = hist2 / np.sum(hist2)
    bc = np.sum(np.sqrt(p * q))
    return -np.log(bc)


def get_clipped(x, min_level=5, percentile=95, truth=None):
    y = x.copy().flatten()
    y = y[~np.isnan(y)]
    n_levels = np.unique(y).__len__()
    if n_levels <= min_level:
        return x/np.nanmax(x), None
    else:
        cutoff_l = np.sort(np.unique(y))[min_level-1]
        if not isinstance(percentile, list):
            cutoff_p = np.percentile(y, percentile)
            y = x.copy()
            y = np.clip(y, a_min=0, a_max=np.max([cutoff_l, cutoff_p]))
            y = y / np.nanmax(y)
            return y, None
        else:
            if truth is None:
                print("must have a truth expression!")
                cutoff_p = np.percentile(y[y > 0], percentile[0])
                y = x.copy()
                y = np.clip(y, a_min=0, a_max=np.max([cutoff_l, cutoff_p]))
                y = y / np.nanmax(y)
                return y
            else:
                distance = []
                y_list = []
                hist1, bin_edges1 = np.histogram(truth[~np.isnan(truth)], bins=50, density=True)
                for p in percentile:
                    y = x.copy().flatten()
                    y = y[~np.isnan(y)]
                    cutoff_p = np.percentile(y[y > 0], p)
                    y = x.copy()
                    y = np.clip(y, a_min=0, a_max=np.max([cutoff_l, cutoff_p]))
                    y = y / np.nanmax(y)
                    hist2, bin_edges2 = np.histogram(y[~np.isnan(y)], bins=50, density=True)
                    distance.append(bhattacharyya_distance(hist1, hist2))
                    y_list.append(y)
                print(distance)
                return y_list[distance.index(min(distance))], percentile[distance.index(min(distance))]

def get_clipped2(x, min_level=5, percentile=95, truth=None):
    y = x.copy().flatten()
    y = y[~np.isnan(y)]
    n_levels = np.unique(y).__len__()
    if n_levels <= min_level:
        return x/np.nanmax(x), None
    else:
        cutoff_l = np.sort(np.unique(y))[min_level-1]
        if not isinstance(percentile, list):
            cutoff_p = np.percentile(y, percentile)
            y = x.copy()
            y = np.clip(y, a_min=0, a_max=np.max([cutoff_l, cutoff_p]))
            y = y / np.nanmax(y)
            return y, None
        else:
            if truth is None:
                print("must have a truth expression!")
                cutoff_p = np.percentile(y[y >= 0], percentile[0])
                y = x.copy()
                y = np.clip(y, a_min=0, a_max=np.max([cutoff_l, cutoff_p]))
                y = y / np.nanmax(y)
                return y
            else:
                distance = []
                y_list = []
                hist1, bin_edges1 = np.histogram(truth[~np.isnan(truth)], bins=50, density=True)
                for p in percentile:
                    y = x.copy().flatten()
                    y = y[~np.isnan(y)]
                    cutoff_p = np.percentile(y[y > 0], p)
                    y = x.copy()
                    y = np.clip(y, a_min=0, a_max=np.max([cutoff_l, cutoff_p]))
                    y = y / np.nanmax(y)
                    hist2, bin_edges2 = np.histogram(y[~np.isnan(y)], bins=50, density=True)
                    distance.append(bhattacharyya_distance(hist1, hist2))
                    y_list.append(y)
                print(distance)
                return y_list[distance.index(min(distance))], percentile[distance.index(min(distance))]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--pred1', type=str)
    parser.add_argument('--pred2', type=str, default=None)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--locsfilter', type=str, default="False") ## only for predicted
    parser.add_argument('--overlay', type=str, default="False") ## only for predicted
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    prefix = args.prefix
    pred1 = args.pred1
    pred2 = args.pred2
    locsfilter = args.locsfilter
    overlay = args.overlay

    factor = 16
    gene_names = read_lines(f'{prefix}gene-names.txt')
    cnts_ref = load_tsv(f'{prefix}ref-cnts.tsv')
    cnts_ref = cnts_ref[gene_names]

    if args.mask is not None:
        mask = load_mask(args.mask)
        print("mask shape is", mask.shape)
    else:
        mask = None

    locs_ref = load_tsv(f'{prefix}ref-locs.tsv')
    locs_ref = locs_ref.astype(float)
    locs_ref = np.stack([locs_ref['y'], locs_ref['x']], -1)
    locs_ref //= factor
    locs_ref = locs_ref.round().astype(int)

    infile_he = f'{prefix}he.jpg'
    print("using h&e overlay...")
    he = load_image(infile_he)
    print("h&e shape before reducing is ", he.shape)
    he = img_reduce(he, factor=factor)

    ssim_abs1_list = []
    ssim_abs2_list = []
    ssim_rel1_list = []
    ssim_rel2_list = []

    for gn in gene_names:
        if gn in cnts_ref.columns:
            ## for ground truth (always apply locsfilters and he overlay)
            truth = np.full((he.shape[0], he.shape[1]), np.nan)
            truth[locs_ref[:, 0], locs_ref[:, 1]] = cnts_ref[gn]
            if mask is not None:
                truth[~mask] = np.nan
            new_truth = truth.copy()

            ## for pred1
            predicted1 = load_pickle(f'{pred1}cnts-super/{gn}.pickle')
            print("the predicted1 gene expression shape is", predicted1.shape)
            new_predicted1 = predicted1.copy()
            if mask is not None:
                assert mask.shape == predicted1.shape
                new_predicted1[~mask] = np.nan
            # predicted1 = np.round(new_predicted1).copy()

            ## optional: for pred2
            if pred2 is not None:
                predicted2 = load_pickle(f'{pred2}cnts-super/{gn}.pickle')
                print("the predicted2 gene expression shape is", predicted2.shape)
                new_predicted2 = predicted2.copy()
                if mask is not None:
                    assert mask.shape == predicted2.shape
                    new_predicted2[~mask] = np.nan
                # predicted2 = np.round(new_predicted2).copy()

            ## for plotting:
            new_truth, _ = get_clipped(new_truth, percentile=95)
            # ref_expr = np.nanmean(new_truth)
            img0 = plot_super(new_truth / np.nanmax(new_truth), he=he)
            # new_predicted1, p1 = get_clipped(new_predicted1, percentile=[90, 92.5, 95, 97.5, 99, 99.5,99.9,99.95,99.99, 99.995, 99.999], reference_expr=ref_expr)
            # if np.nanmax(new_predicted1)>=6:
            #    new_predicted1 = np.round(new_predicted1)
            new_predicted1, p1 = get_clipped(new_predicted1, percentile=99.9)
            print("For gene ", gn, " the percentile of predicted 1 clipped is ", p1)
            if overlay == "True":
                if locsfilter == "True":
                    img1 = plot_super(new_predicted1 / np.nanmax(new_predicted1), he=he, locs_ref=locs_ref)
                else:
                    img1 = plot_super(new_predicted1 / np.nanmax(new_predicted1), he=he)
            else:
                if locsfilter == "True":
                    img1 = plot_super(new_predicted1 / np.nanmax(new_predicted1), locs_ref=locs_ref)
                else:
                    img1 = plot_super(new_predicted1 / np.nanmax(new_predicted1))

            new_truth = truth.copy()
            new_truth[np.isnan(new_truth)] = 0
            new_predicted1 = predicted1.copy()
            new_predicted1[np.isnan(new_predicted1)] = 0
            ssim_abs1 = structural_similarity(np.round(new_predicted1), new_truth, channel_axis=None)
            ssim_abs1_list.append(ssim_abs1)
            f1 = new_predicted1/np.nanmax(new_predicted1)
            f0 = new_truth/np.nanmax(new_truth)
            ssim_rel1 = structural_similarity(f1, f0, channel_axis=None)
            del f1, f0
            ssim_rel1_list.append(ssim_rel1)

            if pred2 is not None:
              #  if np.nanmax(new_predicted2) >= 6:
                #    new_predicted2 = np.round(new_predicted2)
                new_predicted2, p2 = get_clipped(new_predicted2, percentile=99.9)
                print("For gene ", gn, " the percentile of predicted 2 clipped is ", p2)
                if overlay == "True":
                    if locsfilter == "True":
                        img2 = plot_super(new_predicted2 / np.nanmax(new_predicted2), he=he, locs_ref=locs_ref)
                    else:
                        img2 = plot_super(new_predicted2 / np.nanmax(new_predicted2), he=he)
                else:
                    if locsfilter == "True":
                        img2 = plot_super(new_predicted2 / np.nanmax(new_predicted2), locs_ref=locs_ref)
                    else:
                        img2 = plot_super(new_predicted2 / np.nanmax(new_predicted2))

                new_predicted2 = predicted2.copy()
                new_predicted2[np.isnan(new_predicted2)] = 0
                ssim_abs2 = structural_similarity(np.round(new_predicted2), new_truth, channel_axis=None)
                ssim_abs2_list.append(ssim_abs2)
                f2 = new_predicted2/np.nanmax(new_predicted2)
                f0 = new_truth/np.nanmax(new_truth)
                ssim_rel2 = structural_similarity(f0, f2, channel_axis=None)
                ssim_rel2_list.append(ssim_rel2)

                # for S1R1_vs_S2: (10, 4)
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img0)
                axs[0].set_title('Ground Truth Xenium Expression')
                axs[1].imshow(img1)
                axs[1].set_title('iStar Prediction without Xenium Reference \n SSIM: ' + str(np.round(ssim_rel1,3)))
                axs[2].imshow(img2)
                axs[2].set_title('iStar Prediction with Xenium Reference \n SSIM: ' + str(np.round(ssim_rel2,3)))
                plt.tight_layout()
            else:
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(img0)
                axs[0].set_title('Ground Truth Xenium Expression')
                axs[1].imshow(img1)
                axs[1].set_title('iStar Prediction \n SSIM: ' + str(np.round(ssim_rel1,3)))
                plt.tight_layout()

            if locsfilter == "True":
                foldername = prefix + 'comparison_filter2/'
            else:
                foldername = prefix + 'comparison2/'

            if not os.path.exists(foldername):
                os.mkdir(foldername)
            filename = foldername + gn + ".png"
            plt.suptitle('Gene ' + gn)
            plt.savefig(filename, dpi=400)

    os.makedirs(prefix + 'comparison_ref', exist_ok=True)
    os.makedirs(prefix + 'comparison_ref/ssim_full', exist_ok=True)
    os.makedirs(prefix + 'comparison_ref/ssim_top100', exist_ok=True)
    gene_name_list = []
    ssim_abs1_list = []
    ssim_abs2_list = []
    ssim_rel1_list = []
    ssim_rel2_list = []

    for gn in gene_names:
        if gn in cnts_ref.columns:
            ## for ground truth (always apply locsfilters and he overlay)
            truth = np.full((he.shape[0], he.shape[1]), np.nan)
            truth[locs_ref[:, 0], locs_ref[:, 1]] = cnts_ref[gn]
            if mask is not None:
                truth[~mask] = np.nan
            new_truth = truth.copy()

            ## for pred1
            predicted1 = load_pickle(f'{pred1}cnts-super/{gn}.pickle')
            print("the predicted1 gene expression shape is", predicted1.shape)
            new_predicted1 = predicted1.copy()
            if mask is not None:
                assert mask.shape == predicted1.shape
                new_predicted1[~mask] = np.nan
            # predicted1 = np.round(new_predicted1).copy()

            ## optional: for pred2
            if pred2 is not None:
                predicted2 = load_pickle(f'{pred2}cnts-super/{gn}.pickle')
                print("the predicted2 gene expression shape is", predicted2.shape)
                new_predicted2 = predicted2.copy()
                if mask is not None:
                    assert mask.shape == predicted2.shape
                    new_predicted2[~mask] = np.nan
                # predicted2 = np.round(new_predicted2).copy()

            ## for plotting:
            new_truth, _ = get_clipped2(new_truth, percentile=97.5)
            # ref_expr = np.nanmean(new_truth)
            img0 = plot_super(new_truth / np.nanmax(new_truth), he=he)
            # new_predicted1, p1 = get_clipped(new_predicted1, percentile=[90, 92.5, 95, 97.5, 99, 99.5,99.9,99.95,99.99, 99.995, 99.999], reference_expr=ref_expr)
            # if np.nanmax(new_predicted1)>=6:
            #    new_predicted1 = np.round(new_predicted1)
            new_predicted1, p1 = get_clipped2(new_predicted1, percentile=99.9)
            print("For gene ", gn, " the percentile of predicted 1 clipped is ", p1)
            if overlay == "True":
                if locsfilter == "True":
                    img1 = plot_super(new_predicted1 / np.nanmax(new_predicted1), he=he, locs_ref=locs_ref)
                else:
                    img1 = plot_super(new_predicted1 / np.nanmax(new_predicted1), he=he)
            else:
                if locsfilter == "True":
                    img1 = plot_super(new_predicted1 / np.nanmax(new_predicted1), locs_ref=locs_ref)
                else:
                    img1 = plot_super(new_predicted1 / np.nanmax(new_predicted1))

            new_truth = truth.copy()
            new_truth[np.isnan(new_truth)] = 0
            new_predicted1 = predicted1.copy()
            new_predicted1[np.isnan(new_predicted1)] = 0
            ssim_abs1 = structural_similarity(np.round(new_predicted1), new_truth, channel_axis=None)
            ssim_abs1_list.append(ssim_abs1)
            f1 = new_predicted1/np.nanmax(new_predicted1)
            f0 = new_truth/np.nanmax(new_truth)
            ssim_rel1 = structural_similarity(f1, f0, channel_axis=None)
            del f1, f0
            ssim_rel1_list.append(ssim_rel1)

            gene_name_list.append(gn)

            if pred2 is not None:
              #  if np.nanmax(new_predicted2) >= 6:
                #    new_predicted2 = np.round(new_predicted2)
                new_predicted2, p2 = get_clipped2(new_predicted2, percentile=99.9)
                print("For gene ", gn, " the percentile of predicted 2 clipped is ", p2)
                if overlay == "True":
                    if locsfilter == "True":
                        img2 = plot_super(new_predicted2 / np.nanmax(new_predicted2), he=he, locs_ref=locs_ref)
                    else:
                        img2 = plot_super(new_predicted2 / np.nanmax(new_predicted2), he=he)
                else:
                    if locsfilter == "True":
                        img2 = plot_super(new_predicted2 / np.nanmax(new_predicted2), locs_ref=locs_ref)
                    else:
                        img2 = plot_super(new_predicted2 / np.nanmax(new_predicted2))

                new_predicted2 = predicted2.copy()
                new_predicted2[np.isnan(new_predicted2)] = 0
                ssim_abs2 = structural_similarity(np.round(new_predicted2), new_truth, channel_axis=None)
                ssim_abs2_list.append(ssim_abs2)
                f2 = new_predicted2/np.nanmax(new_predicted2)
                f0 = new_truth/np.nanmax(new_truth)
                ssim_rel2 = structural_similarity(f0, f2, channel_axis=None)
                ssim_rel2_list.append(ssim_rel2)

                # for S1R1_vs_S2: (10, 4)
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(img0)
                axs[0].set_title('Ground Truth Xenium Expression')
                axs[1].imshow(img1)
                axs[1].set_title('iStar Prediction without Xenium Reference \n SSIM: ' + str(np.round(ssim_rel1,3)))
                axs[2].imshow(img2)
                axs[2].set_title('iStar Prediction with Xenium Reference \n SSIM: ' + str(np.round(ssim_rel2,3)))
                plt.tight_layout()
            else:
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(img0)
                axs[0].set_title('Ground Truth Xenium Expression')
                axs[1].imshow(img1)
                axs[1].set_title('iStar Prediction \n SSIM: ' + str(np.round(ssim_rel1,3)))
                plt.tight_layout()

            if ssim_abs1_list.__len__() <= 100:
                filename = ['comparison_ref/ssim_top100' + gn + ".png",
                            'comparison_ref/ssim_full' + gn + ".png"]
            else:
                filename = 'comparison_ref/ssim_full' + gn + ".png"
            plt.suptitle('Gene ' + gn)
            plt.savefig(filename, dpi=400)

    df = pd.DataFrame(pd.DataFrame({'Gene': gene_name_list, 'SSIM-Rel-ref': ssim_rel2_list, 'SSIM-Rel-noref': ssim_rel1_list, 'SSIM-Abs-ref': ssim_abs2_list, 'SSIM-Abs-noref': ssim_abs1_list}))
    df.to_csv(prefix + 'comparison_ref/ssim_comparison.tsv', sep='\t')


    df = pd.DataFrame({'SSIM-Rel': ssim_rel1_list})
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
    plt.title('SSIM by Relative Value: Predicted vs True\n All ' + str(len(ssim_rel1_list)) + ' Genes', fontsize=16)
    plt.savefig(prefix + 'violin_plot_rel1_full.png')

    df = pd.DataFrame({'SSIM-Abs': ssim_abs1_list})
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
    plt.title('SSIM by Absolute Value:: Predicted vs True\n All ' + str(len(ssim_abs1_list)) + ' Genes', fontsize=16)
    plt.savefig(prefix + 'violin_plot_abs1_full.png')

    df = pd.DataFrame({'SSIM-Rel': ssim_rel2_list})
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
    plt.title('SSIM by Relative Value: Predicted vs True\n All ' + str(len(ssim_rel2_list)) + ' Genes', fontsize=16)
    plt.savefig(prefix + 'violin_plot_rel2_full.png')

    df = pd.DataFrame({'SSIM-Abs': ssim_abs2_list})
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
    plt.title('SSIM by Absolute Value:: Predicted vs True\n All ' + str(len(ssim_abs2_list)) + ' Genes', fontsize=16)
    plt.savefig(prefix + 'violin_plot_abs2_full.png')

    ssim_rel1_list = load_pickle(prefix + "ssim_rel1_list.pickle")
    ssim_rel2_list = load_pickle(prefix + "ssim_rel2_list.pickle")
    mean_exp = np.mean(cnts_ref)
    var_exp = np.var(cnts_ref)
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    df = pd.DataFrame({'x': ssim_rel1_list, 'y': ssim_rel2_list, 'm': mean_exp, 'v': var_exp})
    sc = ax.scatter(x=df.x, y=df['y'], s=1, c=df.m, cmap='turbo')
    for i in range(len(mean_exp)):
        if df.m[i] > 0.4:
            ax.annotate(gene_names[i], (df.x[i], df.y[i]), textcoords="offset points", xytext=(-2,-3), ha='center', fontsize=6, color='red')
    for i in range(len(mean_exp)):
        if df.y[i] - df.x[i] > 0.4:
            ax.annotate(gene_names[i], (df.x[i], df.y[i]), textcoords="offset points", xytext=(-2,-3), ha='center', fontsize=6, color='blue')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Prediction without Xenium Reference')
    ax.set_ylabel('Prediction with Xenium Reference')
    ax.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(sc)
    cbar.set_label('Gene Expression Mean')
    plt.title('SSIM for iStar Prediction with/without Xenium Reference')
    plt.savefig(prefix + "ssim_scatter_mean.png", dpi=200)

    fig, ax = plt.subplots(1,1, figsize=(8,8))
    df = pd.DataFrame({'x': ssim_rel1_list, 'y': ssim_rel2_list, 'm': mean_exp, 'v': var_exp})
    sc = ax.scatter(x=df.x, y=df['y'], s=1, c=df.v, cmap='turbo')
    for i in range(len(mean_exp)):
        if df.v[i] > 1:
            ax.annotate(gene_names[i], (df.x[i], df.y[i]), textcoords="offset points", xytext=(-2,-3), ha='center', fontsize=6, color='red')
    for i in range(len(mean_exp)):
        if df.y[i] - df.x[i] > 0.4:
            ax.annotate(gene_names[i], (df.x[i], df.y[i]), textcoords="offset points", xytext=(-2,-3), ha='center', fontsize=6, color='blue')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Prediction without Xenium Reference')
    ax.set_ylabel('Prediction with Xenium Reference')
    ax.set_aspect('equal', adjustable='box')
    cbar = plt.colorbar(sc)
    cbar.set_label('Gene Expression Variance')
    plt.title('SSIM for iStar Prediction with/without Xenium Reference')
    plt.savefig(prefix + "ssim_scatter_var.png", dpi=200)

    fig, ax = plt.subplots(1,1, figsize=(8,5))
    ax.scatter(x=df.m, y=df.y, s=1)
    for i in range(len(mean_exp)):
        if df.y[i] - df.x[i] > 0.4:
            ax.annotate(gene_names[i], (df.m[i], df.y[i]), textcoords="offset points", xytext=(-2,-3), ha='center', fontsize=6, color='red')
    for i in range(len(mean_exp)):
        if df.m[i] > 0.4:
            ax.annotate(gene_names[i], (df.m[i], df.y[i]), textcoords="offset points", xytext=(-2,-3), ha='center', fontsize=6, color='blue')
    ax.set_xlim(0, 2.3)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Gene Expression Mean')
    ax.set_ylabel('SSIM for Prediction w. Xenium Reference')
    # ax.set_aspect('equal', adjustable='box')
    plt.title('SSIM for iStar Prediction with Xenium Reference vs Gene Expression Mean')
    plt.savefig(prefix + "ssim_mean.png", dpi=200)

    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.scatter(x=df.v, y=df.y, s=1)
    for i in range(len(mean_exp)):
        if df.y[i] - df.x[i] > 0.4:
            ax.annotate(gene_names[i], (df.m[i], df.y[i]), textcoords="offset points", xytext=(-2,-3), ha='center', fontsize=6, color='red')
    for i in range(len(var_exp)):
        if df.v[i] > 1:
            ax.annotate(gene_names[i], (df.v[i], df.y[i]), textcoords="offset points", xytext=(-2,-3), ha='center', fontsize=6, color='blue')
    ax.set_xlim(0, 7.3)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Gene Expression Variance')
    ax.set_ylabel('SSIM for Prediction w. Xenium Reference')
    # ax.set_aspect('equal', adjustable='box')
    plt.title('SSIM for iStar Prediction with Xenium Reference vs Gene Expression Variance')
    plt.savefig(prefix + "ssim_var.png", dpi=200)

    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.scatter(x=df.m, y=df.v, s=1)
    for i in range(len(mean_exp)):
        if df.y[i] - df.x[i] > 0.4:
            ax.annotate(gene_names[i], (df.m[i], df.v[i]), textcoords="offset points", xytext=(-2,-3), ha='center', fontsize=6, color='red')
    ax.set_xlim(0, 2.3)
    ax.set_ylim(0, 7.3)
    ax.set_xlabel('Gene Expression Mean')
    ax.set_ylabel('Gene Expression Variance')
    # ax.set_aspect('equal', adjustable='box')
    plt.title('Gene Expression Mean vs Gene Expression Variance')
    plt.savefig(prefix + "mean_var.png", dpi=200)


if __name__ == '__main__':
    main()
