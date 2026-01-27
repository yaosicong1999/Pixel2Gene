'''
This .py file strictly follows the registration.html for transforming
the keypoint coordinates in both (one selected layer of during the anchoring) the morphology.ome.tiff
and the keypoint coordinates in the he.tiff (NOT THE he.ome.tiff).
The inputs are the two .txt/.csv files for coordinates as well as the transcripts.parquet file, and
the outputs will be the two matrices,
1. stage_to_morph.txt which can transform the coordinates of gene expression/cell coordinates in MICRON into the
PIXEL level of the morphology image,
2. morphology_to_he.txt which transform the coordinates of gene expression/cell coordinates in the
PIXEL level of the morphology image into the PIXEL level of the H&E image in the he.tiff
'''
import os
import numpy as np
import pandas as pd
from ome_types import from_tiff
from cv2 import findHomography, RANSAC
def generate_homograph(in_dir,
                       morph_keypoints_level, he_keypoints_level,
                       morph_keypoints_filename, he_keypoints_filename):
    '''
    :param in_dir: input directory
    :param morph_keypoints_level: on which level of morphology the keypoints were anchored. Starting from 0
    :param morph_keypoints_filename: file name for the keypoints coordinates on the morphology image
    :param he_keypoints_filename: file name for the keypoints coordinates on the H&E image
    :return: None
    '''
    if ".tsv" in he_keypoints_filename:
        he_nuclei_centroids = pd.read_csv(in_dir + he_keypoints_filename, sep='\t', header=None).values
    elif ".txt" in he_keypoints_filename:
        he_nuclei_centroids = np.loadtxt(in_dir + he_keypoints_filename)
    # the keypoints is obtained on a subsampled he, for example if we used layer 2, which has 0.25x resolution of layer 0.
    he_nuclei_centroids = he_nuclei_centroids * pow(2, he_keypoints_level)

    if ".tsv" in morph_keypoints_filename:
        morphology_nuclei_centroids = pd.read_csv(in_dir + morph_keypoints_filename, sep='\t', header=None).values
    elif ".txt" in morph_keypoints_filename:
        morphology_nuclei_centroids = np.loadtxt(in_dir + morph_keypoints_filename)
    # the keypoints is obtained on a subsampled morphology, for example if we used layer 2, which has 0.25x resolution of layer 0.
    morphology_nuclei_centroids = morphology_nuclei_centroids * pow(2, morph_keypoints_level)

    if os.path.isfile(in_dir + 'morphology_mip.ome.tif'):
        img_path = in_dir + 'morphology_mip.ome.tif'
    elif os.path.isfile(in_dir + 'morphology.ome.tif'):
        img_path = in_dir + 'morphology.ome.tif'
    elif os.path.isdir(in_dir + "morphology_focus"):
        img_path = in_dir + 'morphology_focus/morphology_focus_0000.ome.tif'
    print(img_path)
    morphology_metadata = from_tiff(path=img_path)
    origin_x = morphology_metadata.plates[0].well_origin_x
    origin_y = morphology_metadata.plates[0].well_origin_y
    physical_size_x = morphology_metadata.images[0].pixels.physical_size_x
    physical_size_y = morphology_metadata.images[0].pixels.physical_size_y
    ## Homography matrix describing scaling by pixelsize and translating by offset.
    stage_to_morph = np.array([[1 / physical_size_x, 0, origin_x],
                               [0, 1 / physical_size_y, origin_y],
                               [0, 0, 1]])
    np.savetxt(
        in_dir + 'extras/stage_to_morph.txt',
        stage_to_morph,
        delimiter=',', )
    morphology_to_he, keypoint_selection = findHomography(morphology_nuclei_centroids, he_nuclei_centroids, RANSAC)
    np.savetxt(
        in_dir + 'extras/morphology_to_he.txt',
        morphology_to_he,
        delimiter=',', )
def main():
    data_dir = "/Users/sicongy/data/xenium/human_lung_cancer_xenium_v1/"
    generate_homograph(in_dir=data_dir,
                       morph_keypoints_level=2,
                       he_keypoints_level=2,
                       morph_keypoints_filename="extras/morphology#2_keypoints.txt",
                       he_keypoints_filename="extras/he#2_tiff_keypoints.txt")

if __name__ == '__main__':
    main()