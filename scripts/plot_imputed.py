import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from einops import reduce
from utils import load_pickle, save_image, read_lines, load_image, load_tsv, load_mask
from my_utils import cmapFader, img_reduce, plot_super

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pref', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--n_top', type=int, default=None)
    parser.add_argument('--gene_names', nargs='*', help='Zero or more selected gene names for plotting', default=None)
    parser.add_argument('--overlay', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    pref = args.pref
    overlay = args.overlay
    output = args.output
    n_top = args.n_top
    gene_names = args.gene_names

    mask = load_mask(args.mask) if args.mask else None
    output = args.output + 'gene_expression_plot_imputed/'
    output = output + args.mask.split('/')[-1].split('.')[0] +'/' if args.mask is not None else output + 'no_mask/'
    factor = 16

    all_genes = read_lines(f'{pref}gene-names.txt')  # Read all genes
    if gene_names is None:
        if n_top is None:
            n_top = len(all_genes)
        gene_names = all_genes[:n_top] 
        print(f"Number of genes evaluating: {len(gene_names)}")
    else:
        gene_names = [gn for gn in gene_names if gn in all_genes]
        if len(gene_names) == 0:
            raise ValueError("No valid gene names provided. Please check the gene names.")
        else:
            print(f"Number of genes evaluating: {len(gene_names)}")

    gene_sets = {
        "top_25": all_genes[:len(all_genes) // 4],
        "middle_50": all_genes[len(all_genes) // 4:3 * len(all_genes) // 4],
        "bottom_25": all_genes[3 * len(all_genes) // 4:]
    }
    
    if overlay:
        he = img_reduce(load_image(f'{pref}he.jpg'), factor=factor)

    for key, gene_set in gene_sets.items():
        base_dir = f"{output}/raw/{key}/"
        overlay_dir = f"{output}/overlay/{key}/"
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
    
        for gn in gene_names:
            if gn in gene_set:
                predicted = load_pickle(f'{args.output}cnts-super/{gn}.pickle')
                new_predicted = predicted.copy()
                if mask is not None:
                    assert mask.shape == predicted.shape
                    new_predicted[~mask] = np.nan
                
                plot_super(new_predicted / np.nanmax(new_predicted), outfile=f'{output}raw/{key}/{gn}.png')
                
                if overlay:
                    plot_super(new_predicted / np.nanmax(new_predicted), outfile=f'{output}overlay/{key}/{gn}.png', he=he)

if __name__ == '__main__':
    main()
