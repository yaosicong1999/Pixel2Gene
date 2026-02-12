from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from my_utils import plot_visium_ge, plot_visium_cluster
import tifffile
import locale
import os
import pyarrow.parquet as pq
import pyarrow as pa
import subprocess

os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

def load_image(filename, verbose=True):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img

def load_mask(filename, verbose=True):
    mask = load_image(filename, verbose=verbose)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    return mask

data_dir = '/Users/sicongy/data/visiumHD/human_lung_cancer_post_xenium_v1/' ## change your folder here:
data_dir0 = data_dir
out_dir = data_dir + 'istar_input_260113/'
os.makedirs(out_dir, exist_ok=True)

foldername = 'square_008um'
bin_size = 8
data_dir = data_dir + "binned_outputs/" + foldername + "/"
out_dir = out_dir + foldername + "/"
os.makedirs(out_dir, exist_ok=True)

## when you want to directly read the full image, depending on the format
## Case 1: .btf
he = tifffile.imread(data_dir0 + "extras/he-raw.btf")
## Case 2: .tif
# he = tifffile.imread(data_dir0 + "extras/he-raw.tif")
## Case 3: .ome.tif
# he = tifffile.imread(data_dir0 + "extras/he-raw.ome.tif", level=0)
## Case 4: .jpg
# he = np.array(Image.open(data_dir0 + "extras/he-raw.jpg"))
print("he's shape is", he.shape)

## preprocess the tissue_positions to be in Visium-like format to be read directly
tissue_positions = pd.read_parquet(data_dir+"/spatial/tissue_positions.parquet")
tissue_positions.set_index(tissue_positions.columns[0], inplace=True)
tissue_positions = tissue_positions.astype(np.float32)
tissue_positions.to_csv(data_dir+"/spatial/tissue_positions_list.csv")

# Case 1: if the data is in complete VisiumHD format
adata = sc.read_visium(path=data_dir)
adata.obsm['spatial'] = adata.obsm['spatial'].astype(np.float32)
adata.var_names_make_unique()
adata.obs_names_make_unique()

'''
# Case 2: if the data is in incomplete VisiumHD format, need to build adata by yourself
matrix = csr_matrix(mmread(data_dir + "filtered_feature_bc_matrix/matrix.mtx.gz").T)
features = pd.read_csv(data_dir + "filtered_feature_bc_matrix/features.tsv.gz", header=None, sep="\t")
gene_names = features[1].values
gene_ids = features[0].values
barcodes = pd.read_csv(data_dir + "filtered_feature_bc_matrix/barcodes.tsv.gz", header=None, sep="\t")
cell_barcodes = barcodes[0].values
adata = sc.AnnData(X=matrix)
adata.var_names = gene_names  # Set gene names
adata.var['gene_ids'] = gene_ids  # Store gene IDs
adata.obs_names = cell_barcodes  # Set cell barcodes
positions = pd.read_csv(data_dir + "spatial/tissue_positions_list.csv", header=None)
positions.columns = [
    "barcode", "in_tissue", "array_row", "array_col",
    "pxl_row_in_fullres", "pxl_col_in_fullres"
]
positions.set_index("barcode", inplace=True)
scalefactors_file = data_dir + "spatial/scalefactors_json.json"
with open(scalefactors_file, 'r') as f:
    scalefactors = json.load(f)
spot_diameter_fullres = scalefactors["spot_diameter_fullres"]
print(f"Spot diameter in full resolution: {spot_diameter_fullres}")
adata.obs = adata.obs.join(positions, how="left")
adata.uns['spatial'] = {}
adata.uns['spatial']['spatial'] = {}
adata.uns['spatial']['spatial']['scalefactors'] = {}
adata.uns['spatial']['spatial']['scalefactors']['spot_diameter_fullres'] = spot_diameter_fullres
adata.obs["pxl_col_in_fullres"] = pd.to_numeric(adata.obs["pxl_col_in_fullres"], errors='coerce')
adata.obs["pxl_row_in_fullres"] = pd.to_numeric(adata.obs["pxl_row_in_fullres"], errors='coerce')
# adata.obs["tissue_x"] = adata.obs["pxl_col_in_fullres"] * scalefactors["tissue_hires_scalef"]
# adata.obs["tissue_y"] = adata.obs["pxl_row_in_fullres"] * scalefactors["tissue_hires_scalef"]
adata.obsm['spatial'] = np.array(adata.obs[['pxl_col_in_fullres','pxl_row_in_fullres']])
adata.obs_names_make_unique()
adata.var_names_make_unique()
'''

# Optional Step 1: HVG selection (Skip if you want to keep all genes)
## If you feel that whole-transcriptome is not needed for the analysis,
## you can just do HVG selection and output a subset of genes,
## recommend: top 3000 genes

## Method 1: default scanpy HVG selection
adata2 = adata.copy()
sc.pp.normalize_total(adata2, inplace=True)
sc.pp.log1p(adata2)
sc.pp.highly_variable_genes(adata2, flavor="seurat", n_top_genes=3000)
hvg = adata2.var_names[adata2.var['highly_variable'] == True]
boo = np.array(adata2.var['highly_variable']) == True
del adata2

## Method 2: manual HVG/HEG selection
mean_expr = np.mean(adata.X.todense(), axis=0).flatten().A1
var_expr = np.var(adata.X.todense(), axis=0).flatten().A1
top_n = 3000
top_mean_indices = np.argsort(mean_expr)[-top_n:]
top_variance_indices = np.argsort(var_expr)[-top_n:]
combined_indices = np.unique(np.concatenate([top_mean_indices, top_variance_indices]))
ranked_by_variance = combined_indices[np.argsort(var_expr[combined_indices])][::-1]
final_selected_indices = ranked_by_variance[:top_n]
print("Final selected indices:", final_selected_indices)
print("Total selected features:", len(final_selected_indices))  # Should be 3000
heghvg = adata.var_names[final_selected_indices].to_list()
adata.var['heghvg'] = [g in heghvg for g in adata.var_names]
boo = np.array(adata.var['heghvg']) == True
colors = np.where(boo, 'red', 'blue')
fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
ax.scatter(x=np.log1p(mean_expr), y=np.log1p(var_expr), c=colors, s=1)
plt.xlabel('Mean')
plt.ylabel('Variance')
plt.show()
adata = adata[:, heghvg]


# Optional Step 2: Cut the image/locs
## As the raw H&E image is very big, you can try to cut the full H&E into the part with expressions only

## Step 1: filter out out-of-image sequences
full_size = [he.shape[0], he.shape[1]]
locs = adata.obsm['spatial']
idx = np.where((locs[:,0] >= 0) & (locs[:, 0] <= he.shape[1]) & (locs[:, 1] >= 0) & (locs[:, 1] <= he.shape[0]))[0]
adata2 = adata[idx]
adata2.obsm['spatial'][:,0] = adata2.obsm['spatial'][:,0]
spot_diameter_fullres = adata2.uns['spatial'][next(iter(adata2.uns['spatial']))]['scalefactors']['spot_diameter_fullres']
radius = spot_diameter_fullres*0.5 ## number of pixels per spot radius
locs = pd.DataFrame(adata2.obsm['spatial'].astype(int), columns=['x', 'y'], index=adata2.obs_names).astype(np.float32)

## Step 1.5: we like to save all full data coordinates before doing the coordinates cutting
## however, the images are typically very large so we need to do rescaling first (on cluster maybe, very memory-consuming on local)
os.makedirs(out_dir+"full_image/", exist_ok=True)
# rescale full-he-raw.jpg on cluster
full_he = np.array(Image.open(out_dir+"full_image/full-he.jpg")) ## rescaled, full he
with open(out_dir+"full_image/full-pixel-size-raw.txt", 'w') as file:
    file.write(str(bin_size/2/radius))
with open(out_dir+"full_image/full-pixel-size.txt", 'w') as file:
    file.write(str(0.5))
locs.to_csv(out_dir +"full_image/full-locs-raw.tsv", sep='\t')
dir_temp = out_dir+"full_image/full-"
rescale_script = "../scripts/rescale.py"
cmd = f"""
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixel2gene
export dir="{dir_temp}"
python {rescale_script} {dir_temp} --locs
"""
subprocess.run(cmd, shell=True, executable="/bin/bash")
full_locs = pd.read_csv(out_dir +"full_image/full-locs.tsv", sep="\t", index_col=0)
full_adata = adata2.copy()
full_adata.obsm['spatial'] = np.array(full_locs)
x = full_adata.X.tocsr()
cnts_sparse = pd.DataFrame.sparse.from_spmatrix(x, columns=full_adata.var_names, index=full_adata.obs_names)
cnts_dense = cnts_sparse.apply(lambda col: col.sparse.to_dense(), axis=0)
table = pa.Table.from_pandas(cnts_dense)
pq.write_table(table, out_dir + 'full_image/full-cnts.parquet', compression='brotli')
plot_visium_ge(out_dir=out_dir+"full_image/", adata=full_adata, gene='REG1A', he=full_he, dpi=0.1, clip_q=[0, 100], log1p_transform=False)


plot_visium_ge(out_dir=out_dir, adata=adata2, gene='CD4', he=he, dpi=0.1, clip_q=[0, 100], log1p_transform=False)
##  (Optional): Step 2: automatically cut the full H&E into the part with expressions
xl = int(np.ceil(max([0, locs.x.min()-2*radius])))
xr = int(np.floor(min([locs.x.max()+2*radius, he.shape[1]])))
yl = int(np.ceil(max([0, locs.y.min()-2*radius])))
yr = int(np.floor(min([locs.y.max()+2*radius, he.shape[0]])))
adata2 = adata2[(adata2.obsm['spatial'][:,0] < xr-2*radius) &
                (adata2.obsm['spatial'][:,1] < yr-2*radius) &
                (adata2.obsm['spatial'][:,0] > xl+2*radius) &
                (adata2.obsm['spatial'][:,1] > yl+2*radius)]
adata2.obsm['spatial'][:, 0] = adata2.obsm['spatial'][:, 0] - xl
adata2.obsm['spatial'][:, 1] = adata2.obsm['spatial'][:, 1] - yl
he_cut = he[yl:yr, xl:xr, :] ## this is the cut image
# Image.fromarray(he_cut, 'RGB').save(data_dir+'he-cut.jpg')
# Image.fromarray(he_cut, 'RGB').save(out_dir+'he-raw.jpg')
locs = pd.DataFrame(adata2.obsm['spatial'].astype(int), columns=['x', 'y'], index=adata2.obs_names).astype(np.float32)
plot_visium_ge(out_dir=out_dir, adata=adata2, gene='CD4', he=he_cut, dpi=0.1, clip_q=[0, 100], log1p_transform=False)


## If any mask is presented
# mask = load_mask(data_dir + "mask-full.png")
# locs = pd.DataFrame(adata2.obsm['spatial'].astype(int), columns=['x', 'y'], index=adata2.obs_names)
# ind = mask[locs.iloc[:,1], locs.iloc[:,0]]
# adata3 = adata2[ind,:]
# plot_visium_ge(out_dir=out_dir, adata=adata3, gene='PIGR', he=he_cut, dpi=0.1, clip_q=[0, 100], log1p_transform=False, savename_prefix="masked")


## output
Image.fromarray(he, 'RGB').save(out_dir+'he-raw.jpg')    ## if not cut!!
Image.fromarray(he_cut, 'RGB').save(out_dir+'he-raw.jpg')   ## if cut!!
x = adata2.X.tocsr()
cnts_sparse = pd.DataFrame.sparse.from_spmatrix(x, columns=adata2.var_names, index=adata2.obs_names)
cnts_dense = cnts_sparse.apply(lambda col: col.sparse.to_dense(), axis=0)
table = pa.Table.from_pandas(cnts_dense)
pq.write_table(table, out_dir + 'cnts.parquet', compression='brotli')
locs.to_csv(out_dir + 'locs-raw.tsv', sep='\t')
with open(out_dir+"radius-raw.txt", 'w') as file:
    file.write(str(radius))
with open(out_dir+"pixel-size-raw.txt", 'w') as file:
    file.write(str(bin_size/2/radius))
with open(out_dir+"pixel-size.txt", 'w') as file:
    file.write(str(0.5))