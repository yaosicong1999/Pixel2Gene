import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from einops import reduce
from utils import load_pickle, save_image, read_lines, load_image, load_tsv, load_mask, load_parquet
from my_utils import cmapFader, img_reduce, plot_super

def handle_locations(prefix, factor=16):
    """Handle location data and reduce resolution."""
    if os.path.isfile(f'{prefix}locs.parquet'):
        locs = load_parquet(f'{prefix}locs.parquet')
    elif os.path.isfile(f'{prefix}locs.tsv'):
        locs = load_tsv(f'{prefix}locs.tsv')
    print("Locations shape before reducing is", locs.shape)
    locs = locs.astype(float)
    locs = np.stack([locs['y'], locs['x']], -1)
    locs //= factor
    locs = locs.round().astype(int)
    unique_rows, indices, counts = np.unique(locs, axis=0, return_index=True, return_counts=True)
    unique_row_indices = indices[counts == 1]
    return locs[unique_row_indices], unique_row_indices

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

    mask = load_mask(args.mask) if args.mask is not None else load_mask(f'{pref}mask-small-hs.png')
    output = output + 'gene_expression_plot_truth/'
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

    locs, unique_row_indices = handle_locations(pref, factor=16) 
    print("Locations shape after reducing is", locs.shape)

    if os.path.isfile(f'{pref}cnts.tsv'):
        cnts = load_tsv(f'{pref}cnts.tsv')
        cnts = cnts.iloc[unique_row_indices, :]
        cnts = cnts[gene_names]
    elif os.path.isfile(f'{pref}cnts.parquet'):
        cnts = pd.read_parquet(f'{pref}cnts.parquet', columns=gene_names)
        print("Counts shape is", cnts.shape)
        cnts = cnts.iloc[unique_row_indices, :]
    else:
        raise FileNotFoundError("Counts file not found. Please provide a valid counts file in TSV or Parquet format.")

    ## plot UMI counts
    if args.gene_names is None and args.n_top is None:
        print("print additional umi counts")
        umi = cnts.sum(axis=1)
        new_truth = np.full((mask.shape[0], mask.shape[1]), np.nan)    
        new_truth[locs[:, 0], locs[:, 1]] = umi
        if args.mask is not None:
            new_truth[~mask] = np.nan
        os.makedirs(f'{output}/raw/', exist_ok=True)
        plot_super(new_truth / np.nanmax(new_truth), outfile=f'{output}/raw/umi.png')
        if overlay:
            os.makedirs(f'{output}/overlay/', exist_ok=True)
            plot_super(new_truth / np.nanmax(new_truth), outfile=f'{output}/overlay/umi.png', he=he)

    for key, gene_set in gene_sets.items():
        base_dir = f"{output}/raw/{key}/"
        overlay_dir = f"{output}/overlay/{key}/"
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
    
        for gn in gene_names:
            if gn in gene_set:
                new_truth = np.full((mask.shape[0], mask.shape[1]), np.nan)
            
                if os.path.isfile(f'{pref}cnts.parquet'):
                    cnts = pd.read_parquet(f'{pref}cnts.parquet', columns=[gn])
                    cnts = cnts.iloc[unique_row_indices, :]
                
                new_truth[locs[:, 0], locs[:, 1]] = cnts[gn]
                if args.mask is not None:
                    new_truth[~mask] = np.nan
                
                print(f"Plotting gene {gn} in set {key}")
                plot_super(new_truth / np.nanmax(new_truth), outfile=f'{output}raw/{key}/{gn}.png')
                
                if overlay:
                    plot_super(new_truth / np.nanmax(new_truth), outfile=f'{output}overlay/{key}/{gn}.png', he=he)

if __name__ == '__main__':
    main()
    
    
    
    
    
    

# def main():
#     args = get_args()
#     overlay = args.overlay
#     output = args.output
#     mask = args.mask
#     pref_train = args.pref_train
#     pref_test = args.pref_test

#     gene_names = read_lines(f'{pref_train}gene-names.txt')
#     top_25 = gene_names[:len(gene_names) // 4]
#     middle_50 = gene_names[len(gene_names) // 4:3 * len(gene_names) // 4]
#     bottom_25 = gene_names[3 * len(gene_names) // 4:]

#     if mask is not None:
#         mask_name = mask.split('/')[-1].split('.')[0]
#         mask = load_mask(mask)
#         print("mask shape is", mask.shape)
#         output = output + mask_name + '-' 
#     else:
#         mask = None
#     factor = 16
    
    


#     if overlay:
#         factor = 16
#         infile_he = f'{pref_test}he.jpg'
#         print("using h&e overlay...")
#         he = load_image(infile_he)
#         print("h&e shape before reducing is ", he.shape)
#         he = img_reduce(he, factor=factor)
    
#     dict ={"top_25": top_25, "middle_50": middle_50, "bottom_25": bottom_25}
#     for key, gene_set in dict.items():
#         for gn in gene_set: 
#             new_truth = np.full((mask.shape[0], mask.shape[1]), np.nan)
            
#             if os.path.isfile(f'{pref_test}cnts.parquet'):
#                 cnts_test = pd.read_parquet(f'{pref_test}cnts.parquet', columns=[gn])
#                 cnts_test = cnts_test.iloc[unique_row_indices, :]
            
#             new_truth[locs_test[:, 0], locs_test[:, 1]] = cnts_test[gn]
#             if mask is not None:
#                 new_truth[~mask] = np.nan
#             if overlay:
#                 outname = f'{output}cnts-truth-plots2/{key}/{gn}.png'
#                 plot_truth(new_truth / np.nanmax(new_truth), outfile=outname)
#                 outname = f'{output}cnts-truth-plots2-overlay/{key}/{gn}.png'
#                 plot_truth(new_truth / np.nanmax(new_truth), outfile=outname, he=he)
#             else:
#                 outname = f'{output}cnts-truth-plots2/{key}/{gn}.png'
#                 plot_truth(new_truth / np.nanmax(new_truth), outfile=outname)

# if __name__ == '__main__':
#     main()