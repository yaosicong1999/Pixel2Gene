import os
import sys
import pandas as pd
import anndata
import numpy as np
import scipy as sp
from scipy.io import mmread
import scanpy as sc
from PIL import Image
from my_utils import transform_coordinates, plot_bins_umi, plot_bins_ge, plot_transcript_ge
from scipy.sparse import csr_matrix
import tifffile
import xml.etree.ElementTree as ET
Image.MAX_IMAGE_PIXELS = None

def transcripts_binning(in_dir, out_dir, size, he_shape, apply_mask=False, min_filtering=10, save_parquet=True, save_tsv=False, save_adata=False):
    '''
    :param in_dir: input directory
    :param out_dir: out directory
    :param size: size in pixel for binned squares
    :param he_ome_matched: whether the h&e size matches any one of the morphology sires. if not, registration needs to be done.
    :param scale: scaling factor if he_ome_matched is true, which can be obtained from https://kb.10xgenomics.com/hc/en-us/articles/11636252598925-What-are-the-Xenium-image-scale-factors-#:~:text=For%20the%20full%2Dresolution%20(series,pixel%20size%20is%200.2125%20microns.
    :param apply_mask: whether the binning is done within the area of the tissue mask. mask is read from the input directory as well in the name of 'mask.png'
    :param min_filtering: the minimum UMI counts in the binned squares for filtering
    :param save_tsv: whether cnts, locs and radius will be saved into .tsv/.txt or not
    :param save_adata: whether adata will be created and saved or not
    :return: cnts and locs for binned squares
    '''
    print("reading transcripts...")
    if os.path.isfile(in_dir + 'transcript_he_pixel.parquet'):
        transcripts = pd.read_parquet(in_dir + 'transcript_he_pixel.parquet')
    else:
        transcripts = pd.read_csv(in_dir + 'transcript_he_pixel.tsv', sep='\t', index_col=0)
    print("putting all transcripts in bins ...")
    transcripts['bin_x'] = np.floor(transcripts.x_location / size).astype('int')
    transcripts['bin_y'] = np.floor(transcripts.y_location / size).astype('int')
    transcripts1 = transcripts[['feature_name', 'bin_x', 'bin_y']]
    transcripts1['count'] = 1
    transcripts1 = transcripts1[transcripts1['feature_name'].apply(lambda x: x.split('_')[0] not in
                                                     ['BLANK', 'NegControlCodeword', 'NegControlProbe',
                                                      'UnassignedCodeword', 'DeprecatedCodeword'])]
    del transcripts
    '''
    transcripts1.columns: Index(['feature_name', 'bin_x', 'bin_y', 'count'], dtype='object')
    '''
    gene_set = np.unique(transcripts1.feature_name)
    spots = transcripts1.groupby(['bin_x', 'bin_y', 'feature_name']).agg('sum').reset_index()
    spots['id'] = [str(spots.bin_x[i]) + '_' + str(spots.bin_y[i]) for i in range(spots.shape[0])]
    ## filtering out genes: 'BLANK', 'NegControlCodeword', 'NegControlProbe'
    print("generating spots information...")
    spots = spots.sort_values(by='id',ascending=True)
    del transcripts1

    id_unique = spots.id.unique()
    x_min = [int(i.split('_')[0]) * size for i in id_unique]
    x_max = [int(i.split('_')[0]) * size + size for i in id_unique]
    y_min = [int(i.split('_')[1]) * size for i in id_unique]
    y_max = [int(i.split('_')[1]) * size + size for i in id_unique]
    coords1 = np.array((y_min, x_min)).transpose()
    coords1 = coords1.astype('int')  ## represents the up-left corner coord. of all squares
    coords2 = np.array((y_max, x_max)).transpose()
    coords2 = coords2.astype('int')  ## represents the bottom-right corner coord. of all squares

    chunk_size = 100
    total_chunks = (len(gene_set) + chunk_size - 1) // chunk_size  # Total chunks, rounding up
    ge_table = pd.DataFrame(index=id_unique)
    for i in range(0, len(gene_set), chunk_size):
        chunk = gene_set[i:i + chunk_size]
        current_chunk = i // chunk_size + 1
        print(f"Processing chunk {current_chunk}/{total_chunks}: {chunk[:5]} ... {len(chunk)} items")

        sub_spots = spots[spots['feature_name'].isin(chunk)]
        '''
        spots.shape:  (n_bins, 5)
        spots.columns: Index(['bin_x', 'bin_y', 'feature_name', 'count', 'id'], dtype='object')
        '''
        # MASK
        if apply_mask:
            mask = np.array(Image.open(in_dir + 'mask.png'))
            print("applying mask...")
            within_image = ((coords1[:, 0] < mask.shape[0]).tolist() and
                            (coords1[:, 1] < mask.shape[1]).tolist() and
                            (coords2[:, 0] < mask.shape[0]).tolist() and
                            (coords2[:, 1] < mask.shape[1]).tolist())
            index1 = [mask[i[0], i[1]] for i in coords1[within_image]]
            index2 = [mask[i[0], i[1]] for i in coords2[within_image]]
            in_mask_index = index1 and index2  ## whether the up-left corner and the bottom-right corner are within the mask
            sub_ge_table = sub_spots.pivot_table(index='id', columns='feature_name', values="count", aggfunc="sum")
            sub_ge_table = sub_ge_table.sort_index(ascending=True)
            sub_ge_table = sub_ge_table[within_image][in_mask_index]
            sub_ge_table[np.isnan(sub_ge_table)] = 0
            sub_ge_table = sub_ge_table.reindex(id_unique, fill_value=0)

            # sub_ge_table.to_parquet(out_dir + "ge_table_chunk_" + str(current_chunk) + ".parquet", compression="brotli")
            ## sub_ge_table = pd.read_parquet(out_dir + "ge_table_chunk_" + str(current_chunk) + ".parquet")

            sub_ge_table = sub_ge_table.astype(pd.SparseDtype("int", fill_value=0))
            sub_ge_table = sub_ge_table[in_mask_index]
            print(f"gene expression table of trunk {i // chunk_size + 1} generated...")

            ge_table = ge_table.join(sub_ge_table, how='left')
            ge_table = ge_table.astype(pd.SparseDtype("int", fill_value=0))

        else:
            in_mask_index = [True for i in coords1]
            sub_ge_table = sub_spots.pivot_table(index='id', columns='feature_name', values="count", aggfunc="sum")
            sub_ge_table = sub_ge_table.sort_index(ascending=True)
            sub_ge_table[np.isnan(sub_ge_table)] = 0
            sub_ge_table = sub_ge_table.reindex(id_unique, fill_value=0)

            sub_ge_table.to_parquet(out_dir + "ge_table_chunk_" + str(current_chunk) + ".parquet", compression="brotli")
            ## sub_ge_table = pd.read_parquet(out_dir + "ge_table_chunk_" + str(current_chunk) + ".parquet")

            sub_ge_table = sub_ge_table.astype(pd.SparseDtype("int", fill_value=0))
            sub_ge_table = sub_ge_table[in_mask_index]
            print(f"gene expression table of trunk {i // chunk_size + 1} generated...")

            ge_table = ge_table.join(sub_ge_table, how='left')
            ge_table = ge_table.astype(pd.SparseDtype("int", fill_value=0))

    n_expression = ge_table.sum(axis=1)
    filter = n_expression >= min_filtering
    cnts = ge_table.loc[filter, :]
    cnts = cnts.sparse.to_dense()
    locs = pd.DataFrame(columns=['x', 'y'])
    locs['x'] = [int(i.split('_')[0]) * size + size/2 for i in cnts.index]
    locs['y'] = [int(i.split('_')[1]) * size + size/2 for i in cnts.index]
    locs.index = cnts.index
    if save_parquet:
        print("writing out gene expression...")
        cnts.to_parquet(out_dir + 'transcript_' + 'cnts' + str(size) + '.parquet', compression="brotli")
        print("writing out location information...")
        # locs.to_parquet(out_dir + 'transcript_' + 'locs' + str(size) + '.parquet', compression="brotli")
        locs.to_csv(out_dir + 'transcript_' + 'locs' + str(size) + '.tsv', sep='\t')

        print("writing out in_image gene expression...")
        ind = locs.loc[(locs.x >= 0 + size / 2) & (locs.y >= 0 + size / 2) & (locs.x < he_shape[1] - size / 2) & (
                    locs.y < he_shape[0] - size / 2)].index
        cnts = cnts.astype(pd.SparseDtype("int", fill_value=0))
        cnts = cnts.loc[ind]
        cnts = cnts.sparse.to_dense()
        cnts.to_parquet(out_dir + 'transcript_' + 'cnts' + str(size) + '_inimage.parquet', compression="brotli")
        print("writing out in_image location information...")
        locs = locs.loc[ind]
        # locs.to_parquet(out_dir + 'transcript_' + 'locs' + str(size) + '_inimage.parquet', compression="brotli")
        locs.to_csv(out_dir + 'transcript_' + 'locs' + str(size) + '_inimage.tsv', sep='\t')


    if save_tsv:
        print("writing out gene expression...")
        cnts.to_csv(out_dir + 'transcript_' + 'cnts' + str(size) + '.tsv', sep='\t')
        print("writing out location information...")
        locs.to_csv(out_dir + 'transcript_' + 'locs' + str(size) + '.tsv', sep='\t')

        print("writing out in_image gene expression...")
        ind = locs.loc[(locs.x >= 0 + size / 2) & (locs.y >= 0 + size / 2) & (locs.x < he_shape[1] - size / 2) & (
                    locs.y < he_shape[0] - size / 2)].index
        cnts = cnts.astype(pd.SparseDtype("int", fill_value=0))
        cnts = cnts.loc[ind]
        cnts.to_csv(out_dir + 'transcript_' + 'cnts' + str(size) + '_inimage.tsv', sep='\t')
        print("writing out in_image location information...")
        locs = locs.loc[ind]
        locs.to_csv(out_dir + 'transcript_' + 'locs' + str(size) + '_inimage.tsv', sep='\t')

    if save_adata:
        print("processing and saving adata...")
        assert (locs.index == cnts.index).all()
        adata = anndata.AnnData(X=csr_matrix(cnts), obs=locs, var=pd.DataFrame(cnts.columns.to_list(), columns=['gene_name']))
        adata.var.index.name = None
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        adata.obsm['spatial'] = np.array(adata.obs)
        adata.layers['raw'] = adata.X.copy()
        sc.pp.log1p(adata)
        adata.layers['log1p'] = adata.X.copy()
        adata.X = adata.layers['raw'].copy()
        adata.write(out_dir + 'transcript_' + 'adata' + str(size) + '.h5ad')

    return cnts, locs

def cell_binning(in_dir, out_dir, size, he_ome_matched=True, scale=None, apply_mask=True, min_filtering=10, save_tsv=True, save_adata=True):
    '''
    :param in_dir: input directory
    :param out_dir: out directory
    :param size: size in pixel for binned squares
    :param he_ome_matched: whether the h&e size matches any one of the morphology sires. if not, registration needs to be done.
    :param scale: scaling factor if he_ome_matched is true, which can be obtained from https://kb.10xgenomics.com/hc/en-us/articles/11636252598925-What-are-the-Xenium-image-scale-factors-#:~:text=For%20the%20full%2Dresolution%20(series,pixel%20size%20is%200.2125%20microns.
    :param apply_mask: whether the binning is done within the area of the tissue mask. mask is read from the input directory as well in the name of 'mask.png'
    :param min_filtering: the minimum UMI counts in the binned squares for filtering
    :param save_tsv: whether cnts, locs and radius will be saved into .tsv/.txt or not
    :param save_adata: whether adata will be created and saved or not
    :return: cnts and locs for binned squares
    '''
    print("reading cells...")
    if os.path.isfile(in_dir + 'transcript_he_pixel.parquet'):
        cell_df = pd.read_parquet(in_dir + 'cells_he_pixel.parquet')
    else:
        cell_df = pd.read_csv(in_dir + 'cells_he_pixel.tsv', sep='\t', index_col=0)
    cell_centroids_he = cell_df[['x_centroid', 'y_centroid']]
    print("transforming cell coords. into h&e pixel space...")

    cell_centroids = pd.DataFrame(columns=['bin_x', 'bin_y', 'count'], index=cell_centroids_he.index)
    cell_centroids['bin_x'] = np.floor(cell_centroids_he['x_centroid'] / size).astype('int')
    cell_centroids['bin_y'] = np.floor(cell_centroids_he['y_centroid'] / size).astype('int')
    cell_centroids['count'] = cell_df['transcript_counts']

    print("reading cell-gene information...")
    mtx = mmread(in_dir + '/cell_feature_matrix/matrix.mtx.gz').todense()
    gene_name = pd.read_csv(in_dir + '/cell_feature_matrix/features.tsv.gz', index_col=0, sep="\t", header=None).iloc[:,0]
    barcodes = pd.read_csv(in_dir + '/cell_feature_matrix/barcodes.tsv.gz', index_col=0, sep="\t", header=None).index
    cells = pd.DataFrame(mtx.T, index=barcodes, columns=gene_name)
    cells['bin_x'] = cell_centroids['bin_x']
    cells['bin_y'] = cell_centroids['bin_y']
    ge_table = cells.groupby(['bin_x', 'bin_y']).agg('sum').reset_index()
    ge_table['id'] = [str(ge_table.bin_x[i]) + '_' + str(ge_table.bin_y[i]) for i in range(ge_table.shape[0])]
    ge_table.index = ge_table['id']
    ge_table = ge_table.iloc[:,[i.split('_')[0] not in ['BLANK', 'NegControlCodeword', 'NegControlProbe', 'UnassignedCodeword', 'DeprecatedCodeword'] for i in ge_table.columns]]
    ge_table = ge_table.iloc[:,[i not in ['bin_x', 'bin_y', 'id'] for i in ge_table.columns]]
    ## filtering out genes: 'BLANK', 'NegControlCodeword', 'NegControlProbe'
    print("generating gene expression...")
    x_min = [int(i.split('_')[0]) * size for i in ge_table.index]
    x_max = [int(i.split('_')[0]) * size + size for i in ge_table.index]
    y_min = [int(i.split('_')[1]) * size for i in ge_table.index]
    y_max = [int(i.split('_')[1]) * size + size for i in ge_table.index]
    coords1 = np.array((y_min, x_min)).transpose()
    coords1 = coords1.astype('int') ## represents the up-left corner coord. of all squares
    coords2 = np.array((y_max, x_max)).transpose()
    coords2 = coords2.astype('int') ## represents the bottom-right corner coord. of all squares
    # MASK
    if apply_mask:
        mask = np.array(Image.open(in_dir + 'mask.png'))
        print("applying mask...")
        within_image = ((coords1[:,0]<mask.shape[0]).tolist() and
                        (coords1[:,1]<mask.shape[1]).tolist() and
                        (coords2[:,0]<mask.shape[0]).tolist() and
                        (coords2[:,1]<mask.shape[1])).tolist()
        index1 = [mask[i[0], i[1]] for i in coords1[within_image]]
        index2 = [mask[i[0], i[1]] for i in coords2[within_image]]
        in_mask_index = index1 and index2  ## whether the up-left corner and the bottom-right corner are within the mask
        ge_table = ge_table[within_image][in_mask_index]
    else:
        in_mask_index = [True for i in coords1]
        ge_table = ge_table[in_mask_index]
    ge_table[np.isnan(ge_table)] = 0
    print("gene expression table generated...")

    n_expression = ge_table.sum(axis=1)
    filter = n_expression >= min_filtering
    cnts = ge_table.loc[filter, :]
    locs = pd.DataFrame(columns=['x', 'y'])
    locs['x'] = [int(i.split('_')[0]) * size + size/2 for i in cnts.index]
    locs['y'] = [int(i.split('_')[1]) * size + size/2 for i in cnts.index]
    locs.index = cnts.index
    if save_tsv:
        print("writing out gene expression...")
        cnts.to_csv(out_dir + 'cell_' + 'cnts' + str(size) + '.tsv', sep='\t')
        print("writing out location information...")
        locs.to_csv(out_dir + 'cell_' + 'locs' + str(size) + '.tsv', sep='\t')
        with open(out_dir + 'cell_' + 'radius' + str(size) + '.txt', 'w') as f:
            print("writing out radius...")
            f.write(str(size))

    if save_adata:
        print("processing and saving adata...")
        assert (locs.index == cnts.index).all()
        adata = anndata.AnnData(X=csr_matrix(cnts), obs=locs, var=pd.DataFrame(cnts.columns, columns=['gene_name']))
        adata.var.index.name = None
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        adata.obsm['spatial'] = np.array(adata.obs)
        adata.layers['raw'] = adata.X.copy()
        sc.pp.log1p(adata)
        adata.layers['log1p'] = adata.X.copy()
        adata.X = adata.layers['raw'].copy()
        adata.write(out_dir + 'cell_' + 'adata' + str(size) + '.h5ad')
    return cnts, locs

def main():
    '''
    This .py file is used to bin Xenium transcripts/cells based on certain H&E square grids (in H&E image pixels).
    Hence, a proper alignment is a must.
    All raw files from Xenium outputs are in physical micron units, however, our target units are in H&E pixel level.
    There are possibilities that:
    1. H&E image is aligned with one layer of morphology.ome.tif file. Then there is no need for H&E transformation.
    2. H&E image is NOT aligned with one layer of morphology.ome.tif file, but registration is done, we have both stage_to_morph.txt and morphology_to_he.txt files for H&E transformation.
    3. H&E image is NOT aligned with one layer of morphology.ome.tif file, but we have a xxx_he_imagealignment.csv for H&E transformation.
    '''
    data_dir = "/Users/sicongy/data/xenium/human_lung_cancer_xenium_v1/"
    method = "transcript"

    with tifffile.TiffFile(data_dir +"/extras/he-raw.ome.tif") as tif:
        ome_metadata = tif.ome_metadata
    del tif
    root = ET.fromstring(ome_metadata)
    ns = {"ome": root.tag.split("}")[0].strip("{")}
    pixels = root.find(".//ome:Pixels", ns)
    pixel_size_x = float(pixels.get("PhysicalSizeX"))
    pixel_size_y = float(pixels.get("PhysicalSizeY"))
    print(f"Pixel Width (PhysicalSizeX): {pixel_size_x} µm")
    print(f"Pixel Height (PhysicalSizeY): {pixel_size_y} µm")
    size = 16*0.5/np.sqrt(pixel_size_x*pixel_size_y)
    size = np.round(size, 2)
    print(f"the size for binning will be {size}....")

    min_filtering = 0
    he = tifffile.imread(data_dir +"/extras/he-raw.ome.tif")
    # he = np.array(Image.open(data_dir + 'extras/he-raw.jpg'))

    if os.path.isfile(data_dir + method + '_cnts' + str(size) + '_inimage.parquet'):
        cnts = pd.read_parquet(data_dir + method + '_cnts' + str(size) + '_inimage.parquet')
        locs = pd.read_csv(data_dir + method + '_locs' + str(size) + '_inimage.tsv', sep='\t', index_col=0)
        # locs = pd.read_parquet(data_dir + method + '_locs' + str(size) + '_inimage.parquet')
    elif os.path.isfile(data_dir + method + '_cnts' + str(size) + '_inimage.tsv'):
        cnts = pd.read_csv(data_dir + method + '_cnts' + str(size) + '_inimage.tsv', sep='\t', index_col=0)
        locs = pd.read_csv(data_dir + method + '_locs' + str(size) + '_inimage.tsv', sep='\t', index_col=0)
    else:
        if method == "transcript":
            cnts, locs = transcripts_binning(in_dir=data_dir, out_dir=data_dir, size=size, he_shape=he.shape,
                                             apply_mask=False, min_filtering=min_filtering)
        elif method == "cell":
            cnts, locs = cell_binning(in_dir=data_dir, out_dir=data_dir, size=size, apply_mask=False,
                                      min_filtering=min_filtering)

    plot_bins_umi(out_dir=data_dir, size=size, cnts=cnts, locs=locs, method=method, he=he, save_fig=True, save_he_copy=True)
    plot_bins_ge(out_dir=data_dir, size=size, cnts=cnts, locs=locs, method=method, he=he, gene="CD4", save_fig=True, save_he_copy=True)
    ## transcripts = pd.read_csv(data_dir + 'transcripts.csv.gz', index_col=0)
    ## plot_transcript_ge(transcripts, gene='Pvalb', out_dir=data_dir, transform_dir=data_dir+'extras/', he=he)

if __name__ == '__main__':
    main()