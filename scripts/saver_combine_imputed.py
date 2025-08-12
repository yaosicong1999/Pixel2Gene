import pandas as pd
import argparse
import numpy as np
import scipy.sparse as sp
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow as pa
from utils import write_lines

parser = argparse.ArgumentParser(description="Example argparse in Python")
parser.add_argument("--pref", type=str, help="prefix of the input files")
parser.add_argument("--num", type=int, default=4, help="number of splits")
args = parser.parse_args()
n_split = args.num
pref = args.pref

output_file = pref + "saver-imputed-cnts.parquet"
gene_names = np.loadtxt(f"{pref}gene-names.txt", dtype=str)

# Read and process each chunk separately
for i in range(1, n_split + 1):
    # Read the current split into pandas DataFrame
    cnts_split = pq.read_table(f"{pref}saver-split{i}-imputed.parquet").to_pandas()
    assert cnts_split.columns.tolist() == gene_names.tolist(), "The columns of the cnts split do not match the gene names"
    # Convert pandas DataFrame to Arrow Table
    cnts_split_arrow = pa.Table.from_pandas(cnts_split)
    if i == 1:
        # For the first chunk, create the Parquet file
        pq.write_table(cnts_split_arrow, output_file)
    else:
        # For subsequent chunks, append by reading the existing Parquet file and concatenating
        existing_table = pq.read_table(output_file)
        combined_table = pa.concat_tables([existing_table, cnts_split_arrow])
        pq.write_table(combined_table, output_file)

write_lines(gene_names, f"{pref}saver-imputed-gene-names.txt")


output_file = pref + "saver-imputed-locs.tsv"
# Initialize an empty list to store DataFrames
locs_split_list = []
# Read and process each chunk separately
for i in range(1, n_split + 1):
  locs_split = pd.read_csv(f"{pref}saver-split{i}-locs.tsv", sep="\t", index_col=0)
  locs_split_list.append(locs_split)

# Concatenate all DataFrames
combined_locs_df = pd.concat(locs_split_list, ignore_index=True)

# Write the combined DataFrame to a TSV file
combined_locs_df.to_csv(output_file, sep="\t")
    
