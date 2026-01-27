'''
This .py file transforms the coordinates of gene expression/cell from MICRO into
the pixel level of the H&E image file (he.tiff).
Case 1: If the H&E image file (he.tiff) correctly matches one layer of morphology.ome.tif,
a scaling factor will be multiplied directly.
Case 2: If the H&E image file (he.tiff) does NOT match any one layer of morphology.ome.tif,
a stage_to_morph.txt and a morphology_to_he.txt will be needed.
'''

import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from cv2 import perspectiveTransform, findHomography, RANSAC
Image.MAX_IMAGE_PIXELS = None
from tifffile import tifffile
from my_utils import plot_cell_centroid

def transform_coordinates(coords, homography_matrix):
    return(perspectiveTransform(coords.reshape(-1,1,2).astype(np.float32), homography_matrix )[:,0,:] )

def transcripts_transform(in_dir, out_dir=None, scale=None, save=False):
    if out_dir is None:
        out_dir = in_dir
    print("reading transcripts...")
    if os.path.isfile(in_dir + 'transcripts.csv.gz'):
        transcripts = pd.read_csv(in_dir + 'transcripts.csv.gz', index_col=0)
    elif os.path.isfile(in_dir + 'transcripts.parquet'):
        transcripts = pd.read_parquet(in_dir + 'transcripts.parquet')
    ## transcripts are in the units of 'microns', aka 'micrometers' or 'µm'
    '''
    transcripts.shape:
    (n_transcript, 11)
    transcripts.columns: 
    Index(['cell_id', 'overlaps_nucleus', 'feature_name', 'x_location', 'y_location', 'z_location', 'qv', 'fov_name', 'nucleus_distance'], dtype='object')
    '''
    print("transforming transcript coords. into h&e pixel space...")
    if scale is not None:
        '''
        Case 1: H&E matches one layer of morphology.ome.tif.
        '''
        transcripts.x_location = transcripts.x_location * scale
        transcripts.y_location = transcripts.y_location * scale
    else:
        '''
        Case 2: stage_to_morph.txt and morphology_to_he.txt are both available.
        '''
        stage_to_morph = np.genfromtxt(in_dir + 'extras/stage_to_morph.txt',
                                       delimiter=',')  ## this matrix transform microns size to morphology pixel size, by default on level 0
        transcript_coords_in_morph = transform_coordinates(np.array(transcripts[['x_location','y_location']]), stage_to_morph)
        morphology_to_he = np.genfromtxt(in_dir + 'extras/morphology_to_he.txt',
                                         delimiter=',')  ## this matrix transform morphology pixel size, by default on level 0, to h&e pixel size
        transcript_coords_in_he = transform_coordinates(np.array(transcript_coords_in_morph), morphology_to_he)
        transcripts.x_location = transcript_coords_in_he[:,0]
        transcripts.y_location = transcript_coords_in_he[:,1]
        del transcript_coords_in_he
        '''
        Case 3: Only he_imagealignment.csv is available, suppose only 1 file is named as xxx_he_imagealignment.csv
        TODO: How to transform this file?
        matching_files = [filename for filename in os.listdir(in_dir+'extra/') if 'he_imagealignment.csv' in filename]
        morphology_to_he = np.array(pd.read_csv(in_dir+'extra/'+matching_files[0], header=None))
        '''
    if save:
        transcripts.to_parquet(out_dir + 'transcript_he_pixel.parquet', compression="brotli")
        ## transcripts.to_csv(out_dir + 'transcript_he_pixel.tsv', sep='\t')
        return transcripts
    else:
        return transcripts

def cells_transform(in_dir, he_ome_matched=False, scale=None, save=False):
    print("reading cells...")
    if os.path.isfile(in_dir + 'cells.csv.gz'):
        cell_df = pd.read_csv(in_dir + 'cells.csv.gz', index_col=0)
    elif os.path.isfile(in_dir + 'cells.parquet'):
        cell_df = pd.read_parquet(in_dir + 'cells.parquet')
    cell_centroids_micron = cell_df[['x_centroid', 'y_centroid']]
    print("transforming cell coords. into h&e pixel space...")
    if he_ome_matched:
        cell_centroids_he = np.array(pd.concat([cell_centroids_micron['x_centroid']*scale, cell_centroids_micron['y_centroid']*scale], axis=1))
    else:
        stage_to_morph = np.genfromtxt(in_dir + 'extras/stage_to_morph.txt', delimiter=',') ## this matrix transform microns size to morphology pixel size, by default on level 0
        cell_centroids_morph = transform_coordinates(np.array(cell_centroids_micron), stage_to_morph)
        morphology_to_he = np.genfromtxt(in_dir + 'extras/morphology_to_he.txt', delimiter=',') ## this matrix transform morphology pixel size, by default on level 0, to h&e pixel size
        cell_centroids_he = transform_coordinates(np.array(cell_centroids_morph), morphology_to_he)
    cell_df[['x_centroid', 'y_centroid']] = cell_centroids_he
    if save:
        cell_df.to_parquet(in_dir + 'cells_he_pixel.parquet', compression="brotli")
        # cell_df.to_csv(in_dir + 'cells_he_pixel.tsv', sep='\t')
        return cell_df
    else:
        return cell_df

def boundary_transform(in_dir, he_ome_matched=False, scale=None, save=False):
    print("reading cell boundary...")
    boundary_df = pd.read_csv(in_dir + 'cell_boundaries.csv.gz')
    boundary_micron = boundary_df[['vertex_x', 'vertex_y']]
    print("transforming cell boundary coords. into h&e pixel space...")
    if he_ome_matched:
        boundary_he = np.array(pd.concat([boundary_micron['vertex_x']*scale, boundary_micron['vertex_y']*scale, boundary_micron['label_id']], axis=1))
    else:
        stage_to_morph = np.genfromtxt(in_dir + 'extras/stage_to_morph.txt', delimiter=',') ## this matrix transform microns size to morphology pixel size, by default on level 0
        boundary_morph = transform_coordinates(np.array(boundary_micron), stage_to_morph)
        morphology_to_he = np.genfromtxt(in_dir + 'extras/morphology_to_he.txt', delimiter=',') ## this matrix transform morphology pixel size, by default on level 0, to h&e pixel size
        boundary_he = transform_coordinates(np.array(boundary_morph), morphology_to_he)
    boundary_df[['vertex_x', 'vertex_y']] = boundary_he
    if save:
        boundary_df.to_parquet(in_dir + 'cell_boundary_he_pixel.parquet', compression="brotli")
        # boundary_df.to_csv(in_dir + 'cell_boundary_he_pixel.tsv', sep='\t')
        return boundary_df
    else:
        return boundary_df

def nucleus_transform(in_dir, he_ome_matched=False, scale=None, save=False):
    print("reading nucleus boundary...")
    nucleus_df = pd.read_csv(in_dir + 'nucleus_boundaries.csv.gz')
    nucleus_micron = nucleus_df[['vertex_x', 'vertex_y']]
    print("transforming nucleus boundary coords. into h&e pixel space...")
    if he_ome_matched:
        boundary_he = np.array(pd.concat([nucleus_micron['vertex_x']*scale, nucleus_micron['vertex_y']*scale, nucleus_micron['label_id']], axis=1))
    else:
        stage_to_morph = np.genfromtxt(in_dir + 'extras/stage_to_morph.txt', delimiter=',') ## this matrix transform microns size to morphology pixel size, by default on level 0
        boundary_morph = transform_coordinates(np.array(nucleus_micron), stage_to_morph)
        morphology_to_he = np.genfromtxt(in_dir + 'extras/morphology_to_he.txt', delimiter=',') ## this matrix transform morphology pixel size, by default on level 0, to h&e pixel size
        boundary_he = transform_coordinates(np.array(boundary_morph), morphology_to_he)
    nucleus_df[['vertex_x', 'vertex_y']] = boundary_he
    if save:
        nucleus_df.to_csv(in_dir + 'nucleus_boundary_he_pixel.csv', index=False)
        return nucleus_df
    else:
        return nucleus_df


def main():
    data_dir = "/Users/sicongy/data/xenium/human_lung_cancer_xenium_v1/"
    '''
    This .py file is used to bin Xenium transcripts/cells based on certain H&E square grids (in H&E image pixels).
    Hence, a proper alignment is a must.
    All raw files from Xenium outputs are in physical micron units, however, our target units are in H&E pixel level.
    There are possibilities that:
    1. H&E image is aligned with one layer of morphology.ome.tif file. Then there is no need for H&E transformation.
    2. H&E image is NOT aligned with one layer of morphology.ome.tif file, but registration is done, we have both stage_to_morph.txt and morphology_to_he.txt files for H&E transformation.
    ## TODO : 3. H&E image is NOT aligned with one layer of morphology.ome.tif file, but we have a xxx_he_imagealignment.csv for H&E transformation.
    '''
    scale = None  ## search for desired transformation scale or None for manual alignment
    '''
    -   if the h&e size matches one of the ome.tiff series, 
        just search for the desired scale on https://kb.10xgenomics.com/hc/en-us/articles/11636252598925-What-are-the-Xenium-image-scale-factors#:~:text=For%20the%20full%2Dresolution%20(series,pixel%20size%20is%200.2125%20microns
        the scale will be 1/ the third column in that chart.
    -   if the h&e size does not match any ome.tiff series, 
        you must include a file named 'transcript_coords_in_he.tsv'/'stage_to_morph.txt' + 'morphology_to_he.txt' for registration,
        which can be obtained from keypoints_to_homograph.py via keypoints registration.
        See: https://www.10xgenomics.com/support/software/xenium-explorer/latest/tutorials/xe-image-alignment
    '''
    transcripts_he = transcripts_transform(in_dir=data_dir, out_dir=data_dir, scale=scale, save=True)
    cells_he = cells_transform(in_dir=data_dir, scale=scale, save=True)

    he = tifffile.imread(data_dir + 'extras/he-raw.ome.tif', level=1) ## ONLY FOR SUPER-LARGE HE-RAW > .JPG
    cells_he.x_centroid = cells_he.x_centroid / 2
    cells_he.y_centroid = cells_he.y_centroid / 2
    plot_cell_centroid(cells_he, data_dir, transform_dir=None, he=he, color='red', save_name=None, save_fig=True)

    boundary_he = boundary_transform(in_dir=data_dir, scale=scale, save=True)
    nucleus_he = nucleus_transform(in_dir=data_dir, scale=scale, save=True)

if __name__ == '__main__':
    main()