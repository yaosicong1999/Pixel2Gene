import sys
import os
from PIL import Image
import numpy as np
import pandas as pd
import anndata
from scipy import sparse
from itertools import product
from my_utils import (read_he, plot_visium_ge)
from tqdm import tqdm
import scanpy as sc
import tifffile
import xml.etree.ElementTree as ET
Image.MAX_IMAGE_PIXELS = None

def main():
    ## load Xenium transcript data
    data_dir = "/Users/sicongy/data/xenium/human_breast_cancer_FFPE/sample1_rep1/"
    print("reading raw transcripts from " + data_dir + "...")
    transcripts_microns = pd.read_csv(data_dir + 'transcripts.csv.gz')
    print("reading transformed transcripts...")
    transcripts_pixel = pd.read_csv(data_dir + 'transcript_he_pixel.tsv', sep='\t', index_col=0)
    he = read_he(data_dir, filename='extras/he-raw.jpg')
    print("reading he image...")

    ##  new version of getting pixels_per_microns
    with tifffile.TiffFile(data_dir +"/extras/he-raw.ome.tif") as tif:
        ome_metadata = tif.ome_metadata
    del tif
    root = ET.fromstring(ome_metadata)
    ns = {"ome": root.tag.split("}")[0].strip("{")}
    pixels = root.find(".//ome:Pixels", ns)
    pixel_size_x = float(pixels.get("PhysicalSizeX"))
    pixel_size_y = float(pixels.get("PhysicalSizeY"))
    print(f"Pixel Width (PhysicalSizeX) in µm: {pixel_size_x} µm")
    print(f"Pixel Height (PhysicalSizeY) in µm: {pixel_size_y} µm")
    microns_per_pixel = np.sqrt(pixel_size_x * pixel_size_y)
    pixels_per_micron = 1/microns_per_pixel

    ## old version
    pixels_per_micron_x = (transcripts_pixel.x_location.iloc[transcripts_pixel.shape[0]-1] - transcripts_pixel.x_location.iloc[0])/(transcripts_microns.x_location.iloc[transcripts_pixel.shape[0]-1]- transcripts_microns.x_location.iloc[0])
    # pixels_per_micron_y = (transcripts_pixel.y_location.iloc[transcripts_pixel.shape[0]-1] - transcripts_pixel.y_location.iloc[0])/(transcripts_microns.y_location.iloc[transcripts_pixel.shape[0]-1]- transcripts_microns.y_location.iloc[0])
    # pixels_per_micron = (np.abs(pixels_per_micron_x) + np.abs(pixels_per_micron_y))/2
    # microns_per_pixel = 1/pixels_per_micron

    radius = 55*0.5/microns_per_pixel ## radius in pixel level
    distance = 100/microns_per_pixel ## pseudo-visium center-to-center in pixel level

    x_min = transcripts_pixel.x_location.min()
    x_max = np.min([transcripts_pixel.x_location.max(), he.shape[1]])
    n_x = int((x_max - x_min)//(distance*2)*2)
    y_min = transcripts_pixel.y_location.min()
    y_max = np.min([transcripts_pixel.y_location.max(), he.shape[0]])
    n_y = int((y_max - y_min)//(distance*np.sqrt(3)*2)*2)

    center_x_min = x_min + int((x_max-x_min-n_x*distance)/2)
    center_x_max = center_x_min + n_x*distance
    center_x = np.linspace(start=center_x_min, stop=center_x_max, num=n_x+1, endpoint=True)
    center_y_min = y_min + int((y_max-y_min-n_y*(distance*np.sqrt(3)))/2)
    center_y_max = center_y_min + n_y*(distance*np.sqrt(3))
    center_y = np.linspace(start=center_y_min, stop=center_y_max, num=n_y+1, endpoint=True)
    center = pd.DataFrame(list(product(center_x, center_y)), columns=['x', 'y'])

    center_x_min = center_x_min + distance/2
    center_x_max = center_x_max - distance/2
    center_x = np.linspace(start=center_x_min, stop=center_x_max, num=n_x, endpoint=True)
    center_y_min = center_y_min + (distance*np.sqrt(3))/2
    center_y_max = center_y_max - (distance*np.sqrt(3))/2
    center_y = np.linspace(start=center_y_min, stop=center_y_max, num=n_y, endpoint=True)
    center_df = pd.concat([center, pd.DataFrame(list(product(center_x, center_y)), columns=['x', 'y'])])
    center_df.index = range(center_df.shape[0])

    transcripts_df = transcripts_pixel[['x_location', 'y_location', 'feature_name']]
    transcripts_df.columns = ['x', 'y', 'feature_name']
    def find_closest_center(center_points, other_points):
        center_coords = center_points[['x', 'y']].values
        other_coords = other_points[['x', 'y']].values
        closest_centers = np.zeros(len(other_coords), dtype=int)
        for i, (ox, oy) in enumerate(tqdm(other_coords)):
            distances = np.sqrt((center_coords[:, 0] - ox) ** 2 + (center_coords[:, 1] - oy) ** 2)
            if np.min(distances) <= radius:
                closest_centers[i] = np.argmin(distances)
            else:
                closest_centers[i] = -1
        return closest_centers
    print("finding the closest center...")
    closest_center_indices = find_closest_center(center_df, transcripts_df)
    transcripts_df['center_id'] = closest_center_indices

    transcripts_df = transcripts_df[transcripts_df.center_id >= 0]
    transcripts1 = transcripts_df[['center_id', 'feature_name']]
    transcripts1['count'] = 1
    spots = transcripts1.groupby(['center_id', 'feature_name']).agg('sum').reset_index()
    ## filtering out genes: 'BLANK', 'NegControlCodeword', 'NegControlProbe'
    print("generating spots information...")
    spots = spots[[i.split('_')[0] not in ['BLANK', 'NegControlCodeword', 'NegControlProbe'] for i in spots.feature_name]]
    spots = spots.sort_values(by='center_id',ascending=True)
    del transcripts1
    ge_table = spots.pivot_table(index='center_id', columns='feature_name', values="count", aggfunc="sum")
    ge_table = ge_table.sort_index(ascending=True)
    ge_table[np.isnan(ge_table)] = 0
    min_filter = 0
    ge_table = ge_table[ge_table.sum(axis=1) > min_filter]

    cnts = np.array(ge_table)
    adata = anndata.AnnData(X=sparse.csr_matrix(cnts), obs=center_df.loc[ge_table.index])
    adata.var_names = ge_table.columns.to_list()
    adata.uns['spatial'] = {'my_pseudo_visium': {'scalefactors': {'spot_diameter_fullres': 65*pixels_per_micron}}}
    adata.obsm['spatial'] = np.array(center_df.loc[ge_table.index])
    radius = pixels_per_micron*55/2

    ## save psuedo-visium files
    if not os.path.exists(data_dir + 'pseudo_visium_20250512'):
        os.mkdir(data_dir + 'pseudo_visium_20250512')
    adata.write_h5ad(filename = data_dir + 'pseudo_visium_20250512/adata.h5ad')
    ge_table.to_csv(data_dir + 'pseudo_visium_20250512/cnts.tsv', sep='\t')
    center_df.loc[ge_table.index].to_csv(data_dir + 'pseudo_visium_20250512/locs-raw.tsv', sep='\t')
    Image.fromarray(he).save(data_dir + 'pseudo_visium_20250512/he-raw.jpg')
    with open(data_dir + 'pseudo_visium_20250512/pixel-size-raw.txt', 'w') as file:
        file.write(str(microns_per_pixel))
    with open(data_dir + 'pseudo_visium_20250512/pixel-size.txt', 'w') as file:
        file.write("0.5")
    with open(data_dir + 'pseudo_visium_20250512/radius-raw.txt', 'w') as file:
        file.write(str(radius))

    adata = sc.read_h5ad(data_dir + 'pseudo_visium/adata.h5ad')
    plot_visium_ge(out_dir=data_dir + 'pseudo_visium/', adata=adata, gene='ERBB2', he=he, dpi=0.1,
                   log1p_transform=False, save_he_copy=True)
    plot_visium_ge(out_dir=data_dir + 'pseudo_visium/', adata=adata, gene='CD4', he=he, dpi=0.1, log1p_transform=False,
                   save_he_copy=True)


if __name__ == '__main__':
    main()
