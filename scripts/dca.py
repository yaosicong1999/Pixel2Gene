from prometheus_client import g
from dca.api import dca
import scanpy as sc
import numpy as np
import pandas as pd


count = pd.read_parquet('../visiumhd_heg/CRC-P1-downsampled6-cnts.parquet')
locs = pd.read_csv('../visiumhd_heg/CRC-P1-downsampled6-locs.tsv', sep='\t', index_col=0)
gene_names = np.loadtxt('../visiumhd_heg/CRC-P1-downsampled6-gene-names.txt', dtype=str)
count = count[gene_names]
count = count.iloc[:, :100]

count.to_csv("../visiumhd_heg/CRC-P1-downsampled6-top100_cnts.csv", index=False)




adata = sc.AnnData(count, obs=locs, var=pd.DataFrame(index=gene_names))
adata.obs['x'] = adata.obs['x'].astype(float)
adata.obs['y'] = adata.obs['y'].astype(float)

sc.pp.filter_genes(adata, min_counts=1)
dca(adata, threads=1)



