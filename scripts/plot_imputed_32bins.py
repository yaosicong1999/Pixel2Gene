import pickle
import pandas as pd
import itertools
import numpy as np
def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from einops import reduce
from utils import load_pickle, save_image, read_lines, load_image, load_tsv, load_mask
from my_utils import cmapFader, img_reduce

def plot_super(x, outfile, truncate=None, he=None, locs=None):
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
    if locs is not None:
        if he is not None:
            out = he.copy()
        else:
            out = np.full((x.shape[0], x.shape[1], 3), 255)
        out[locs[:, 0], locs[:, 1], :] = img[locs[:, 0], locs[:, 1], :]
        img = out
    filter = np.isnan(x)
    if he is not None:
        img[filter] = he[filter]
    img = img.astype(np.uint8)
    save_image(img, outfile)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--locsfilter', action='store_true')
    parser.add_argument('--overlay', action='store_true')
    parser.add_argument('--out-of-sample', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    prefix = args.prefix
    locsfilter = args.locsfilter
    overlay = args.overlay

    gene_names = read_lines(f'{prefix}gene-names.txt')
    if args.mask is not None:
        mask = load_mask(args.mask)
        print("mask shape is", mask.shape)
    else:
        mask = None
    factor = 16

    if locsfilter:
        if args.out_of_sample:
            locs = load_tsv(f'{prefix}test-locs.tsv')
        else:
            locs = load_tsv(f'{prefix}locs.tsv')
        locs = locs.astype(float)
        locs = np.stack([locs['y'], locs['x']], -1)
        locs //= factor
        locs = locs.round().astype(int)

    if overlay:
        factor = 16
        if args.out_of_sample:
            infile_he = f'{prefix}test-he.jpg'
        else:
            infile_he = f'{prefix}he.jpg'
        print("using h&e overlay...")
        he = load_image(infile_he)
        print("h&e shape before reducing is ", he.shape)
        he = img_reduce(he, factor=factor)


    for gn in gene_names:
        predicted = load_pickle(f'{prefix}cnts-super/{gn}.pickle')
        print("the predicted1 gene expression shape is", predicted.shape)
        size = 2
        new_predicted = np.zeros((np.floor(predicted.shape[0]/size).astype(int), np.floor(predicted.shape[1]/size).astype(int)))

        df = pd.DataFrame(list(itertools.product(range(predicted.shape[0]), range(predicted.shape[1]))), columns=['x', 'y'])
        df['ge'] = predicted.flatten()
        df['bin_x'] = np.floor(df.x / size).astype('int')
        df['bin_y'] = np.floor(df.y / size).astype('int')
        df2 = df.groupby(['bin_x', 'bin_y']).agg('sum').reset_index()
        new_predicted = df2['ge'].to_numpy().reshape((new_predicted.shape[0], new_predicted.shape[1]))

        if mask is not None:
            assert mask.shape == predicted.shape
            new_predicted[~mask] = np.nan

        # new_predicted = np.clip(new_predicted, a_min=0, a_max=np.nanpercentile(new_predicted, 95))
        outname = f'{prefix}cnts-super-plots2/{gn}.png'
        if overlay:
            if locsfilter:
                plot_super(new_predicted / np.nanmax(new_predicted), outfile=outname, he=he, locs=locs)
            else:
                plot_super(new_predicted / np.nanmax(new_predicted), outfile=outname, he=he)
        else:
            if locsfilter:
                plot_super(new_predicted / np.nanmax(new_predicted), outfile=outname, locs=locs)
            else:
                plot_super(new_predicted / np.nanmax(new_predicted), outfile=outname)

if __name__ == '__main__':
    main()
