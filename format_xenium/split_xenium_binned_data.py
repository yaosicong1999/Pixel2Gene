import os
from my_utils import (read_gene_expression, read_istar_gene_expression, read_he, load_pickle, plot_transcript_ge, plot_bins_umi, plot_bins_ge)
from img_enhancement import img_enhance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = 2500000000

palette = [plt.cm.tab20(i) for i in range(20)]
palette2 = [plt.cm.Set3(i) for i in range(12)]
palette3 = [plt.cm.Pastel1(i) for i in range(9)]
palette.extend(palette2)
palette.extend(palette3)

def split_train_test_he(he, out_dir,
             xl_train, xr_train, yl_train, yr_train,
             xl_test, xr_test, yl_test, yr_test,
             enhance=False, save=True):
    he_train, he_test = (he[yl_train:yr_train, xl_train:xr_train, :],
                         he[yl_test:yr_test, xl_test:xr_test, :])
    if enhance:
        he_train = img_enhance(he_train, save=False)
        he_test = img_enhance(he_test,save=False)
    plt.imshow(he_train)
    plt.show()
    plt.imshow(he_test)
    plt.show()
    if save:
        Image.fromarray(he_train, 'RGB').save(out_dir + 'train-he-raw.jpg')
        Image.fromarray(he_test, 'RGB').save(out_dir + 'test-he-raw.jpg')
    return he_train, he_test

def split_train_test_cnts_locs(cnts, locs, out_dir,
                         xl_train, xr_train, yl_train, yr_train,
                         xl_test, xr_test, yl_test, yr_test,
                         save=True):
    # train
    prefix = ['train', 'test']
    return_list = []
    for pref in prefix:
        print("handling the", pref + "ing part...")
        if pref == 'train':
            xl_split = xl_train; xr_split = xr_train; yl_split = yl_train; yr_split = yr_train
        else:
            xl_split = xl_test; xr_split = xr_test; yl_split = yl_test; yr_split = yr_test

        cnts_split = cnts.copy()
        locs_split = locs.copy()
        cnts_split = cnts_split[
            (locs_split['x'] >= xl_split) & (locs_split['x'] < xr_split) & (
                    locs_split['y'] >= yl_split) & (locs_split['y'] < yr_split)]
        locs_split = locs_split[
            (locs_split['x'] >= xl_split) & (locs_split['x'] < xr_split) & (
                    locs_split['y'] >= yl_split) & (locs_split['y'] < yr_split)]
        locs_split['x'] = locs_split['x'] - xl_split
        locs_split['y'] = locs_split['y'] - yl_split
        locs_split['grid_x'] = locs_split['grid_x'] - xl_split / 16
        locs_split['grid_y'] = locs_split['grid_y'] - yl_split / 16

        locs_split['grid_index'] = locs_split['grid_x'].astype(int).astype(str) + 'x' + locs_split['grid_y'].astype(
            int).astype(str)
        locs_split['spot'] = locs_split['grid_index']
        locs_split.index = locs_split['spot']
        cnts_split.index = locs_split['spot']
        if save:
            cnts_split.to_csv(out_dir + pref + '-cnts.tsv', sep="\t")
            locs_split[['x', 'y']].to_csv(out_dir + pref + '-locs-raw.tsv', sep="\t")
            return_list.append([cnts_split, locs_split])
    return return_list

def split_train_test_boundary(boundary, out_dir,
                         xl_train, xr_train, yl_train, yr_train,
                         xl_test, xr_test, yl_test, yr_test,
                         save=True):
    id_count = boundary.cell_id.value_counts()
    id_count = id_count.sort_index(ascending=True)
    prefix = ['train', 'test']
    return_list = []
    for pref in prefix:
        print("handling the", pref + "ing part...")
        if pref == 'train':
            xl_split = xl_train; xr_split = xr_train; yl_split = yl_train; yr_split = yr_train
        else:
            xl_split = xl_test; xr_split = xr_test; yl_split = yl_test; yr_split = yr_test

        boundary_split = boundary.copy()
        boundary_split = boundary_split[
            (boundary_split['vertex_x'] >= xl_split) & (boundary_split['vertex_x'] < xr_split) & (
                    boundary_split['vertex_y'] >= yl_split) & (boundary_split['vertex_y'] < yr_split)]
        split_id_count = boundary_split.cell_id.value_counts()
        split_id_count = split_id_count.sort_index(ascending=True)
        id_count_slice = id_count.loc[split_id_count.index]
        id_select = split_id_count[split_id_count == id_count_slice].dropna().index
        boundary_split = boundary_split.loc[boundary_split.cell_id.isin(id_select)]
        boundary_split['vertex_x'] = boundary_split['vertex_x'] - xl_split
        boundary_split['vertex_y'] = boundary_split['vertex_y'] - yl_split
        if save:
            boundary_split[['cell_id', 'vertex_x', 'vertex_y']].to_csv(out_dir + pref + '-boundary-raw.tsv', sep="\t")
            return_list.append(boundary_split)
    return return_list


def split_train_test_pixel_radius(out_dir):
    with open(out_dir + "train-pixel-size.txt", 'w') as file:
        file.write("1")
    with open(out_dir + "test-pixel-size.txt", 'w') as file:
        file.write("1")
    with open(out_dir + "train-pixel-size-raw.txt", 'w') as file:
        file.write("1")
    with open(out_dir + "test-pixel-size-raw.txt", 'w') as file:
        file.write("1")
    with open(out_dir + "radius.txt", 'w') as file:
        file.write("0")


''' 
# for mouse brain cortex
xl_train = 8960; xr_train = 14080
yl_train = 2816; yr_train = 6144
xl_test = 3840; xr_test = 8960
yl_test = 2816; yr_test = 6144
out_dir = "/Users/sicongy/results/istar-sc/mouse_brain/cortex/"
'''

'''
# for mouse brain half
xl_train = 0; xr_train = 9216
yl_train = 0; yr_train = 12032
xl_test = 9216; xr_test = 18176
yl_test = 0; yr_test = 12032
out_dir = "/Users/sicongy/results/istar-sc/mouse_brain/half/"
'''

# for mouse colon
xl_train = 0; xr_train = 18000
yl_train = 0; yr_train = 28329
xl_test = 18000; xr_test = 32201
yl_test = 0; yr_test = 28329
out_dir = "/Users/sicongy/results/istar-sc/mouse_colon/half/"

'''
## for human breast cancer, two tumors 
xl_train = 8320; xr_train = 10368
yl_train = 8704; yr_train = 12544
xl_test = 22272; xr_test = 24320
yl_test = 8704; yr_test = 12544
out_dir = "/Users/sicongy/results/istar-sc/breast_cancer/S1R1/tumor/"
'''

'''
## for human breast cancer, quarter of the tissue
xl_train = 4206; xr_train = 13934
yl_train = 4356; yr_train = 11780
xl_test = 14446; xr_test = 24174
yl_test = 12292; yr_test = 19716
out_dir = "/Users/sicongy/results/istar-sc/breast_cancer/S1R1/quarter/"
'''

'''
## for human breast cancer S1R1, half of the tissue
xl_train = 4206; xr_train = 13934
yl_train = 4356; yr_train = 19716
xl_test = 14446; xr_test = 24174
yl_test = 4356; yr_test = 19716
out_dir = "/Users/sicongy/results/istar-sc/breast_cancer/S1R1/half/"
'''

'''
## for human colorectal cancer P1, in area cut
xl_train = 0; xr_train = 25000
yl_train = 19800; yr_train = 44500
xl_test = 0; xr_test = 1
yl_test = 0; yr_test = 1
out_dir = "/Users/sicongy/data/xenium/human_colorectal_cancer_CRC/P1/istar_input_32/"
'''

'''
## for human colorectal cancer P1, half of the tissue
xl_train = 0; xr_train = 25000
yl_train = 0; yr_train = 12000
xl_test = 0; xr_test = 25000
yl_test = 12000; yr_test = 24700
out_dir = "/Users/sicongy/data/xenium/human_colorectal_cancer_CRC/P1_split/"
'''

'''
## for human colorectal cancer P2, in area cut
xl_train = 14000; xr_train = 40800
yl_train = 0; yr_train = 23800
xl_test = 0; xr_test = 1
yl_test = 0; yr_test = 1
out_dir = "/Users/sicongy/data/xenium/human_colorectal_cancer_CRC/P2/istar_input_32/"
'''

'''
## for human colorectal cancer P2, half of the tissue
xl_train = 0; xr_train = 26800
yl_train = 0; yr_train = 13000
xl_test = 0; xr_test = 26800
yl_test = 13000; yr_test = 23800
out_dir = "/Users/sicongy/data/xenium/human_colorectal_cancer_CRC/P2_split/"
'''

'''
## for human colorectal cancer P5, in area cut
xl_train = 20600; xr_train = 43600
yl_train = 23200; yr_train = 47800
xl_test = 0; xr_test = 1
yl_test = 0; yr_test = 1
out_dir = "/Users/sicongy/data/xenium/human_colorectal_cancer_CRC/P5/istar_input_32/"
'''

'''
## for human colorectal cancer P5, half of the tissue
xl_train = 0; xr_train = 23000
yl_train = 0; yr_train = 12000
xl_test = 0; xr_test = 23000
yl_test = 12000; yr_test = 24600
out_dir = "/Users/sicongy/data/xenium/human_colorectal_cancer_CRC/P5_split/"
'''

os.makedirs(out_dir, exist_ok=True)
data_dir = "/Users/sicongy/data/xenium/human_colorectal_cancer_CRC/P1/"
method = "transcript"
size = 32
## read cnts data, locs data, and h&e image
cnts, locs = read_gene_expression(data_dir, method, size)
# cnts, locs = read_istar_gene_expression(data_dir)
import re
if re.match(r'^\d+_\d+$', locs.index[0]):
    locs['grid_x'] = [int(locs.index[i].split('_')[0]) for i in range(locs.shape[0])]
    locs['grid_y'] = [int(locs.index[i].split('_')[1]) for i in range(locs.shape[0])]
elif re.match(r'^\d+x\d+$', locs.index[0]):
    locs['grid_x'] = [int(locs.index[i].split('x')[0]) for i in range(locs.shape[0])]
    locs['grid_y'] = [int(locs.index[i].split('x')[1]) for i in range(locs.shape[0])]
locs['grid_index'] = locs['grid_x'].astype(str) + 'x' + locs['grid_y'].astype(str)
locs['spot'] = locs['grid_index']
locs.index = locs['spot']
cnts.index = locs['spot']
he = read_he(data_dir, filename='extras/he-raw.jpg')
# he = read_he(data_dir, filename='he-raw.jpg')
plt.imshow(he)
plt.show()

## double check whether H&E and cnts are correctly aligned!
plot_bins_umi(out_dir=out_dir, size=size, cnts=cnts, locs=locs[['x', 'y']], method=method, he=he, save_fig=True)
plot_bins_ge(out_dir=out_dir, size=size, cnts=cnts, locs=locs[['x', 'y']], method=method, he=he, gene="Nrn1", save_fig=True)
transcripts = pd.read_csv(data_dir + 'transcripts.csv.gz', index_col=0)
plot_transcript_ge(transcripts, gene='Nrn1', out_dir=out_dir, transform_dir=data_dir + 'extras/', he=he, save_fig=True)

he_train, he_test = split_train_test_he(he, out_dir,
                                        xl_train, xr_train, yl_train, yr_train,
                                        xl_test, xr_test, yl_test, yr_test)

split_list = split_train_test_cnts_locs(cnts, locs, out_dir,
                                        xl_train, xr_train, yl_train, yr_train,
                                        xl_test, xr_test, yl_test, yr_test)

split_train_test_pixel_radius(out_dir)

plot_bins_umi(out_dir=out_dir, size=size, cnts=split_list[0][0], locs=split_list[0][1][['x', 'y']], method=method, he=he_train, save_fig=True, savename_prefix="train")
plot_bins_umi(out_dir=out_dir, size=size, cnts=split_list[1][0], locs=split_list[1][1][['x', 'y']], method=method, he=he_test, save_fig=True, savename_prefix="test")

# boundary = pd.read_csv(data_dir + "cell_boundary_he_pixel.tsv", sep='\t', index_col=0)
boundary = pd.read_csv(data_dir + "boundary-raw.tsv", sep='\t', index_col=0)
train_boundary, test_boundary = split_train_test_boundary(boundary, out_dir,
                                                          xl_train, xr_train, yl_train, yr_train,
                                                          xl_test, xr_test, yl_test, yr_test)
