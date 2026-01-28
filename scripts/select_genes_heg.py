import argparse
from numpy.ma import nonzero
import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
from utils import load_tsv, write_lines, read_lines, load_parquet

def get_genenames_in_batch(filepath, batch_size=1000, filtering_cnts=20):
    print(f"Loading and processing in batches: {filepath}")
    table = pq.read_table(filepath)
    columns_to_drop = ['__index_level_0__', 'spot']
    table = table.drop([col for col in columns_to_drop if col in table.column_names])     

    n_cols = table.num_columns
    print(f"Total number of genes columns: {n_cols}")
    column_sums = np.zeros(n_cols, dtype=np.float32)
    column_freqs = np.zeros(n_cols, dtype=np.float32)
    
    for col_start in range(0, n_cols, batch_size):
        print("current column start:", col_start)
        col_end = min(col_start + batch_size, n_cols)
        cols_batch = table.columns[col_start:col_end]
        batch_data = {}
        for idx, col in enumerate(cols_batch):
            col_name = table.schema.field(col_start + idx).name
            batch_data[col_name] = col.to_pandas()
        df_batch = pd.DataFrame(batch_data) 
        column_sums[col_start:col_end] += df_batch.sum(axis=0).values
        column_freqs[col_start:col_end] += (df_batch > 0).sum(axis=0).values

    nonzero_mask = column_sums > filtering_cnts
    nonzero_sums = column_sums[nonzero_mask]
    nonzero_freqs = column_freqs[nonzero_mask]
    
    nonzero_columns = np.array(table.column_names)[nonzero_mask]
    count_df = pd.DataFrame({'gene_name': nonzero_columns, 'umi': nonzero_sums, 'freq': nonzero_freqs})
    print(f"Total genes: {n_cols}, Nonzero genes: {len(nonzero_columns)}")
    sorted_cols = nonzero_columns[np.argsort(nonzero_sums)[::-1]].tolist()
    return sorted_cols, count_df

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pref', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    pref = args.pref

    filtering_cnts = 20
    if os.path.exists(f'{pref}cnts.parquet'):
        sorted_cols, count_df = get_genenames_in_batch(f'{pref}cnts.parquet', filtering_cnts=filtering_cnts)  
        print(count_df.head())
    elif os.path.exists(f'{pref}cnts.tsv'):
        count_df = load_tsv(f'{pref}cnts.tsv')
        sum_expr = count_df.astype(pd.SparseDtype()).sum(axis=0)
        nonzero_mask = sum_expr > filtering_cnts
        
        nonzero_sums = sum_expr[nonzero_mask]
        nonzero_columns = nonzero_sums.index
        print(f"Total genes: {count_df.shape[1]}, Nonzero genes: {len(nonzero_columns)}")
        nonzero_freqs = (count_df[nonzero_columns] > 0).sum(axis=0)
        count_df = pd.DataFrame({'gene_name': nonzero_columns, 'umi': nonzero_sums, 'freq': nonzero_freqs})
        print(count_df.head())
        count_df = count_df.sort_values(by='umi', ascending=False)
        sorted_cols = count_df['gene_name'].tolist()
        
        
    else:
        raise ValueError(f'Unknown file type for the cnts file: {pref}cnts')     

    print("filtering out unwanted gene...")
    sorted_cols = [name for name in sorted_cols if not name.startswith(('BLANK', 'NegControlCodeword', 'NegControlProbe', 
            'UnassignedCodeword', 'DeprecatedCodeword'))]
    write_lines(sorted_cols, f'{pref}gene-names.txt')
    count_df = count_df[count_df['gene_name'].isin(sorted_cols)]
    count_df = count_df.sort_values(by='umi', ascending=False)
    count_df.to_csv(f'{pref}count_df.tsv', sep='\t', index=False)
    
if __name__ == '__main__':
    main()

