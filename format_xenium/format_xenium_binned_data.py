from my_utils import (read_gene_expression, read_he, plot_bins_umi, plot_bins_ge)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import os
import shutil
import tifffile
import xml.etree.ElementTree as ET
Image.MAX_IMAGE_PIXELS = None

data_dir = "/Users/sicongy/data/xenium/human_lung_cancer_xenium_v1/"
with tifffile.TiffFile(data_dir + "/extras/he-raw.ome.tif") as tif:
    ome_metadata = tif.ome_metadata
del tif
root = ET.fromstring(ome_metadata)
ns = {"ome": root.tag.split("}")[0].strip("{")}
pixels = root.find(".//ome:Pixels", ns)
pixel_size_x = float(pixels.get("PhysicalSizeX"))
pixel_size_y = float(pixels.get("PhysicalSizeY"))
pixel_size_raw = np.sqrt(pixel_size_x * pixel_size_y)

size = 16 * 0.5 / pixel_size_raw
size = np.round(size, 2)
print(f"the size for binning will be {size}....")
out_dir = data_dir + "istar_input_" + str(size) + "/"
os.makedirs(out_dir, exist_ok=True)
method = "transcript"


'''
## old version
data_dir = "/Users/sicongy/data/xenium/human_artery/P4_D/"
stage_to_morph = np.loadtxt(data_dir + "extras/stage_to_morph.txt", delimiter=',')
morphology_to_he = np.loadtxt(data_dir + "extras/morphology_to_he.txt", delimiter=',')
T = morphology_to_he @ stage_to_morph  # or just use T1 if only one matrix

# Extract scaling from transformation matrix
scale_x = np.linalg.norm(T[0, :2])
scale_y = np.linalg.norm(T[1, :2])
microns_per_pixel_x = 1 / scale_x
microns_per_pixel_y = 1 / scale_y
microns_per_pixel = (microns_per_pixel_x + microns_per_pixel_y) / 2

print(f"Microns per pixel (x): {microns_per_pixel_x:.4f}")
print(f"Microns per pixel (y): {microns_per_pixel_y:.4f}")
print(f"Average microns per pixel: {microns_per_pixel:.4f}")

data_dir = "/Users/sicongy/data/xenium/human_artery/P4_H/"
stage_to_morph = np.loadtxt(data_dir + "extras/stage_to_morph.txt", delimiter=',')
morphology_to_he = np.loadtxt(data_dir + "extras/morphology_to_he.txt", delimiter=',')
T = morphology_to_he @ stage_to_morph  # or just use T1 if only one matrix

# Extract scaling from transformation matrix
scale_x = np.linalg.norm(T[0, :2])
scale_y = np.linalg.norm(T[1, :2])
microns_per_pixel_x = 1 / scale_x
microns_per_pixel_y = 1 / scale_y
microns_per_pixel = (microns_per_pixel_x + microns_per_pixel_y) / 2

print(f"Microns per pixel (x): {microns_per_pixel_x:.4f}")
print(f"Microns per pixel (y): {microns_per_pixel_y:.4f}")
print(f"Average microns per pixel: {microns_per_pixel:.4f}")
'''

## read cnts data, locs data, and h&e image
cnts, locs = read_gene_expression(data_dir, method, size, inimage=True)  ## FOR HE-RAW.JPG
cnts, locs = read_gene_expression(data_dir, method, size, inimage=False) ## ONLY FOR SUPER-LARGE HE-RAW > .JPG
locs['grid_x'] = [int(locs.index[i].split('_')[0]) for i in range(locs.shape[0])]
locs['grid_y'] = [int(locs.index[i].split('_')[1]) for i in range(locs.shape[0])]
locs['grid_index'] = locs['grid_x'].astype(str) + 'x' + locs['grid_y'].astype(str)
locs['spot'] = locs['grid_index']
locs.index = locs['spot']
cnts.index = locs['spot']

boundary = pd.read_parquet(data_dir + "cell_boundary_he_pixel.parquet")
if 'vertex_x' in boundary.columns:
    boundary = boundary.rename(columns={'vertex_x': 'x'})
if 'vertex_y' in boundary.columns:
    boundary = boundary.rename(columns={'vertex_y': 'y'})
if 'cell_id' in boundary.columns:
    boundary = boundary.rename(columns={'cell_id': 'id'})

nucleus = pd.read_csv(data_dir + 'nucleus_boundary_he_pixel.csv')


he = read_he(data_dir, filename='extras/he-raw.jpg') ## FOR HE-RAW.JPG, don't use for cut image!
he = tifffile.imread(data_dir + 'extras/he-raw.ome.tif', level=0) ## ONLY FOR SUPER-LARGE HE-RAW > .JPG
plt.imshow(he)
plt.show()
#Image.fromarray(he, 'RGB').save(out_dir + 'he-raw.jpg')


## As the raw H&E image is very big, you can try to cut the full H&E
# into the part with expressions only

## Step 1: filter out out-of-image sequences
ind = locs.loc[(locs.x >= 0 + size / 2) & (locs.y >= 0 + size / 2) & (locs.x < he.shape[1] - size / 2) & (
            locs.y < he.shape[0] - size / 2)].index
if ind.shape[0] != cnts.shape[0]:
    cnts = cnts.loc[ind]
    locs = locs.loc[ind]
locs = locs[['x', 'y']]
### ENE OF STEP 1


## (Optional) Step 2: automatically cut the full H&E into the part with expressions
xl = int(np.ceil(max([0, locs.x.min()-2*size/2])))
xr = int(np.floor(min([locs.x.max()+2*size/2, he.shape[1]])))
yl = int(np.ceil(max([0, locs.y.min()-2*size/2])))
yr = int(np.floor(min([locs.y.max()+2*size/2, he.shape[0]])))
ind = locs.loc[(locs.x < xr-2*size/2) & (locs.x > xl+2*size/2) &
               (locs.y < yr-2*size/2) & (locs.y > yl+2*size/2)].index
cnts = cnts.loc[ind]
locs = locs.loc[ind]
locs.x = locs.x - xl
locs.y = locs.y - yl

he = he[yl:yr, xl:xr, :] ## this is the cut image
# Image.fromarray(he_cut, 'RGB').save(data_dir+'he-cut.jpg')
# Image.fromarray(he_cut, 'RGB').save(out_dir+'he-raw.jpg')

boundary.x = boundary.x - xl
boundary.y = boundary.y - yl
boundary = boundary.merge(boundary.groupby('id').agg(
    min_x=('x', 'min'),
    max_x=('x', 'max'),
    min_y=('y', 'min'),
    max_y=('y', 'max')), on='id')
boundary = boundary.loc[(boundary.min_x >= 0) & (boundary.max_x < he.shape[1]) & (boundary.min_y >= 0) & (boundary.max_y < he.shape[0])]

nucleus.vertex_x = nucleus.vertex_x - xl
nucleus.vertex_y = nucleus.vertex_y - yl
### ENE OF STEP 2


plot_bins_ge(out_dir=out_dir, size=size, cnts=cnts, locs=locs, method=method, he=he, gene="EPCAM", save_fig=True,
             save_he_copy=False)
plot_bins_umi(out_dir=out_dir, size=size, cnts=cnts, locs=locs, method=method, he=he, save_fig=True, save_he_copy=False)


## output
Image.fromarray(he, 'RGB').save(out_dir+'he-raw.jpg')
cnts.to_parquet(out_dir + "cnts.parquet")
locs.to_csv(out_dir+'locs-raw.tsv', sep='\t')
# locs.to_parquet(out_dir+'locs-raw.parquet', compression="brotli")
with open(out_dir+"pixel-size.txt", 'w') as file:
    file.write("0.5")
with open(out_dir+"pixel-size-raw.txt", 'w') as file:
    file.write(str(16*0.5/size))
with open(out_dir+"radius-raw.txt", 'w') as file:
    file.write(str(0))
shutil.copy(data_dir + "cell_feature_matrix.h5", out_dir + "cell_feature_matrix.h5")
boundary.to_csv(out_dir+'boundary-raw.tsv', sep="\t")
nucleus.to_csv(out_dir+'nucleus-raw.csv.gz', compression="gzip")
