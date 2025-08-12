import os
import argparse
import gc
import numpy as np
from utils import load_pickle, load_mask, save_pickle, read_lines, write_lines
from joblib import Parallel, delayed, cpu_count, parallel_backend
from tqdm import tqdm
import shutil
from tqdm_joblib import tqdm_joblib  # requires install

def parse_args():
    parser = argparse.ArgumentParser(description='K-fold aggregation of gene expression data')
    parser.add_argument('--data_pref', type=str, required=True, help='Data prefix path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--mask_pref', type=str, default='', help='Prefix for masks')
    parser.add_argument('--k', type=int, default=10, help='Number of folds')
    return parser.parse_args()

def process_gene(gn, k, output, mask_pref):
    p_list = []
    for fold in range(k):
        fold_path = f'{output}{mask_pref}_fold_{fold}/cnts-super/{gn}.pickle'
        if os.path.exists(fold_path):
            p = load_pickle(fold_path, verbose=False).astype(np.float16)
            p_list.append(p)
            os.remove(fold_path)
        else:
            print(f'[WARN] Missing: {fold_path}')

    # Sum across folds with NaN-safe addition
    p_sum = np.nanmean(p_list, axis=0)
    p_sum = p_sum.astype(np.float16)
    save_pickle(p_sum, f'{output}{mask_pref}/cnts-super/{gn}.pickle')
    del p_list, p_sum
    gc.collect()

def main():
    args = parse_args()
    pref = args.data_pref
    output = args.output
    mask_pref = args.mask_pref
    k = args.k
    
    gene_names = read_lines(f'{output}{mask_pref}_fold_0/predict-gene-names.txt')
    
    os.makedirs(f'{output}{mask_pref}/cnts-super/', exist_ok=True)
    print(f"[INFO] Processing {len(gene_names)} genes with {k} folds...")
    
    write_lines(gene_names, f'{output}{mask_pref}/predict-gene-names.txt')

    num_cpus = cpu_count()
    print(f"[INFO] Using {num_cpus} CPU cores for parallel processing")

    with tqdm_joblib(tqdm(desc="Processing genes", total=len(gene_names))) as progress_bar:
        Parallel(n_jobs=-1)(
            delayed(process_gene)(gn=gn, k=k, output=output, mask_pref=mask_pref) for gn in gene_names
        )

    print("[DONE] All genes processed.")

    for fold in range(k):
        fold_dir = f'{output}{mask_pref}_fold_{fold}'
        if os.path.exists(fold_dir):
            try:
                shutil.rmtree(fold_dir)
                print(f"[INFO] Removed directory: {fold_dir}")
            except Exception as e:
                print(f"[WARN] Failed to remove {fold_dir}: {e}")

if __name__ == '__main__':
    main()