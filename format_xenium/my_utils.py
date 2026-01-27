import anndata
import cv2
from cv2 import perspectiveTransform, findHomography, RANSAC
import numpy as np
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.collections import PatchCollection
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from PIL import Image
import scanpy as sc
import sklearn
import sklearn.neighbors
import tifffile
from tqdm import tqdm
from einops import reduce
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)

def load_pickle(filename):
    """
    :param filename: pickle filename
    :return: data loaded from .pickle
    """
    with open(filename, 'rb') as file:
        x = pickle.load(file)
        print(f'Pickle loaded from {filename}...')
    return x

def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines

def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)

def read_gene_expression(in_dir, method, size, inimage=True):
    """
    Read gene expression from cnts.tsv and locs.tsv files
    :param in_dir: directory for reading cnts.tsv and locs.tsv
    :param method: either 'transcript' or 'cell' for different types of binning
    :param size: size for each square
    :return: cnts dataframe and locs dataframe
    """
    print("reading cnts.tsv and locs.tsv of size " + str(size) + "px...")
    if inimage:
        if os.path.isfile(in_dir + method + '_cnts' + str(size) + '_inimage.parquet'):
            cnts = pd.read_parquet(in_dir + method + '_cnts' + str(size) + '_inimage.parquet')
            # locs = pd.read_parquet(in_dir + method + '_locs' + str(size) + '_inimage.parquet')
        else:
            cnts = pd.read_csv(in_dir + method + '_cnts' + str(size) + '_inimage.tsv', sep='\t', index_col=0)
        locs = pd.read_csv(in_dir + method + '_locs' + str(size) + '_inimage.tsv', sep='\t', index_col=0)
    else:
        if os.path.isfile(in_dir + method + '_cnts' + str(size) + '.parquet'):
            cnts = pd.read_parquet(in_dir + method + '_cnts' + str(size) + '.parquet')
            # locs = pd.read_parquet(in_dir + method + '_locs' + str(size) + '.parquet')
        else:
            cnts = pd.read_csv(in_dir + method + '_cnts' + str(size) + '.tsv', sep='\t', index_col=0)
        locs = pd.read_csv(in_dir + method + '_locs' + str(size) + '.tsv', sep='\t', index_col=0)
    assert (cnts.index == locs.index).all()
    return cnts, locs

def read_istar_gene_expression(in_dir):
    """
    Read gene expression from cnts.tsv and locs.tsv files
    :param in_dir: directory for reading cnts.tsv and locs.tsv
    :param method: either 'transcript' or 'cell' for different types of binning
    :param size: size for each square
    :return: cnts dataframe and locs dataframe
    """
    print("reading cnts.tsv and locs-raw.tsv from input_istar...")
    cnts = pd.read_csv(in_dir + 'cnts.tsv', sep="\t", index_col=0)
    locs = pd.read_csv(in_dir + 'locs-raw.tsv', sep="\t", index_col=0)
    assert (cnts.index == locs.index).all()
    return cnts, locs

def read_he(in_dir, filename='he-raw.jpg'):
    """
    Read H&E file from input directory
    :param in_dir: directory for reading he-raw.jpg
    :return: numpy array for the he image
    """
    print("reading he image...")
    he = np.array(Image.open(in_dir + filename))
    return he

def read_hipt_features(in_dir, filename="embeddings-hist.pickle"):
    """
    Read saved image HIPT features from .pickle file
    :param in_dir: directory for reading HIPT features
    :param filename: pickle file name
    :return:  a dictionary with cls features in shape of (H/ 256, W/ 256, 192) and sub features in shape of (H/ 16, W/ 16, 384)
    """
    print("reading hipt features from " + in_dir + filename + "...")
    data = load_pickle(in_dir + filename)
    cls = np.dstack(data['cls'])
    sub = np.dstack(data['sub'])
    return {'cls': cls, 'sub': sub}

def build_adata(cnts, locs, nn, n_hvg=None):
    """
    Build adata object from cnts dataframe and locs dataframe
    :param cnts: cnts dataframe
    :param locs: locs dataframe
    :param nn: number of nearest neighbors for building graphs
    :param n_hvg: number of HVG, must be integer if input. if not input or by default HVG detection will NOT be performed
    :return:
    """
    print("building adata from cnts and locs...")
    adata = anndata.AnnData(X=cnts, obs=locs, dtype=np.float32)
    adata = adata[(adata.obs['x'] > 0) & (adata.obs['y'] > 0), :]
    adata.obs.index = adata.obs['grid_index']
    adata.obsm['spatial'] = np.array(adata.obs[['x','y']])
    adata.layers['raw'] = adata.X.copy()
    sc.pp.log1p(adata)
    adata.layers['log1p'] = adata.X.copy()
    adata.X = adata.layers['raw'].copy()
    sc.pp.neighbors(adata, n_neighbors=nn)
    if n_hvg is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor='seurat_v3')
    print("adata built...")
    return adata

def colorFader(color1, color2, proportion):
    '''
    :param color1: first color to be mixed with
    :param color2: second color to be mixed with
    :param proportion: proportion of the second color
    :return: a hex representation of the mixed color
    '''
    c1 = np.array(mpl.colors.to_rgb(color1))
    c2 = np.array(mpl.colors.to_rgb(color2))
    return mpl.colors.to_hex((1 - proportion) * c1 + proportion * c2)

class cmapFader:
  def __init__(self, cmap_name, start_val, stop_val):
    self.cmap_name = cmap_name
    self.cmap = plt.get_cmap(cmap_name)
    self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
    self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
  def get_rgb(self, val):
    return self.scalarMap.to_rgba(val)

def rgb_to_hex(rgb):
    """
    Convert RGB color to hexadecimal.
    Parameters:
        rgb (tuple): Tuple containing three integers representing RGB values.
    Returns:
        str: Hexadecimal representation of the RGB color.
    """
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

def he_enhance(img, save=True, out_dir=None, out_filename=None):
    '''
    :param img: input image that will be enhanced
    :param save: whether to save the enhanced image or not
    :param out_dir: output directory for enhanced image to be saved
    :param out_filename: filename for enhanced image to be saved
    :return: enhanced image numpy array
    '''
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))
    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    print('image enhancement completed...')
    ## Stacking the original image with the enhanced image
    ## result = np.hstack((img, enhanced_img))
    if save:
        ## cv2.imwrite(data_dir + 'enhanced_with_raw.jpg', result)
        print('saving enhanced image...')
        cv2.imwrite(out_dir + out_filename, enhanced_img)
    return enhanced_img

def he_tiff2jpg(in_dir, in_filename, out_dir, out_filename, level=0):
    '''
    :param in_dir: input directory for reading tiff
    :param in_filename: input file name ending in '.tiff'
    :param out_dir: output directory for saving jpg
    :param out_filename: output file name ending in '.jpg'
    :param level: the level of image read from the .ome.tif, by default 0 and no need to input for one-level .tif file
    :return: image in numpy array
    '''
    print('reading tiff file...')
    img = tifffile.imread(in_dir + in_filename, level=level)
    if img.shape[2] == 3:
        print('this he.tif is in the shape of ' + str(img.shape) + '...')
        im = Image.fromarray(img, 'RGB')
    else:
        print('this he.tif is in the shape of ' + str(img.shape) + '...')
        img = np.ascontiguousarray(img.transpose(1, 2, 0))
        print("transposing image...")
        im = Image.fromarray(img, 'RGB')
    print('saving jpg file...')
    im.save(out_dir + out_filename)
    return img



def bisect_cluster(adata, n_clusters, remaining_iter, lower=0.01, upper=1, method="leiden"):
    '''
    :param adata: input adata for clustering
    :param n_clusters: desired number of clusters
    :param remaining_iter: number of remaining iterations that will be attempted
    :param lower: lower boundary resolution for clustering
    :param upper: upper boundary resolution for clustering
    :param method: clustering method, either 'louvain' or 'leiden'. by default use 'leiden'
    :return: the number of clusters accepted in the end and its corresponding resolution
    '''
    if method == 'leiden':
        def adata_cluster(adata, resolution, key_added):
            sc.tl.leiden(adata, resolution=resolution, key_added=key_added)
    elif method == 'louvain':
        def adata_cluster(adata, resolution, key_added):
            sc.tl.louvain(adata, resolution=resolution, key_added=key_added)
    else:
        raise Exception(
            "method is not correctly input, must be louvain or leiden!")

    cl_res_dict = {}
    for key in adata.uns.keys():
        if key.startswith(method) and key[len(method + '_'):].isdigit():
            n_cluster = int(key[len(method+'_'):])
            cl_res_dict[int(n_cluster)] = adata.uns[key]['params']['resolution']
    if n_clusters in cl_res_dict.keys():
        return n_clusters, cl_res_dict[n_clusters]
    def find_closest_res(cl_res_dict, target_n_cluster):
        sorted_clusters = sorted(cl_res_dict.keys())
        if target_n_cluster in cl_res_dict:
            return {
                "exact": (target_n_cluster, cl_res_dict[target_n_cluster]),
                "smaller": None,
                "larger": None
            }
        smaller = None
        larger = None
        for n_cluster in sorted_clusters:
            if n_cluster < target_n_cluster:
                smaller = n_cluster  # Update the smaller cluster
            elif n_cluster > target_n_cluster and larger is None:
                larger = n_cluster  # First larger cluster found
                break
        result = {
            "exact": None,
            "smaller": (smaller, cl_res_dict[smaller]) if smaller is not None else None,
            "larger": (larger, cl_res_dict[larger]) if larger is not None else None
        }
        return result
    res_dict = find_closest_res(cl_res_dict, n_clusters)
    if res_dict['exact'] is not None:
        adata_cluster(adata, resolution=res_dict['exact'], key_added=method + '_' + str(n_clusters))
        return n_clusters, res_dict['exact']
    if res_dict['smaller'] is not None:
        lower = np.max([lower, res_dict['smaller'][1]])
    if res_dict['larger'] is not None:
        upper = np.min([upper, res_dict['larger'][1]])

    if method + '_mid' not in adata.obs.columns:
        ## check whether we need to do cluster on either boundary again:
        print("starting " + method + " clustering...")
        print("remaining iter: " + str(remaining_iter) + ", now clustering on the lower boundary resolution = " + str(lower) + "...")
        adata_cluster(adata, resolution=lower, key_added=method + '_lower')
    lower_clusters = adata.obs[method + '_lower'].cat.categories.__len__()
    print("the n_clusters on the lower boundary is ", lower_clusters)
    adata.obs[method + '_' + str(lower_clusters)] = adata.obs[method + '_lower']
    adata.uns[method + '_' + str(lower_clusters)] = {'params': {'resolution': lower}}

    if lower_clusters > n_clusters:
        del adata.obs[method + '_lower']
        raise Exception(
            "the resolution lower bound is too high, please input smaller lower bound...")

    if method + '_mid' not in adata.obs.columns:
        print("remaining iter: " + str(remaining_iter) + ", now clustering on the upper boundary resolution = " + str(upper) + "...")
        adata_cluster(adata, resolution=upper, key_added=method + '_upper')
    upper_clusters = adata.obs[method + '_upper'].cat.categories.__len__()
    print("the n_clusters on the upper boundary is ", upper_clusters)
    adata.obs[method + '_' + str(upper_clusters)] = adata.obs[method + '_upper']
    adata.uns[method + '_' + str(upper_clusters)] = {'params': {'resolution': upper}}

    if upper_clusters < n_clusters:
        del adata.obs[method + '_lower']
        del adata.obs[method + '_upper']
        raise Exception(
            "the resolution upper bound is too low, please input bigger upper bound...")

    m = (lower + upper) / 2
    print("remaining iter: " + str(remaining_iter) + ", now clustering on the middle point resolution = " + str(m) + "...")
    adata_cluster(adata, resolution=m, key_added=method + '_mid')
    m_clusters = adata.obs[method + '_mid'].cat.categories.__len__()
    adata.obs[method + '_' + str(m_clusters)] = adata.obs[method + '_mid']
    adata.uns[method + '_' + str(m_clusters)] = {'params': {'resolution': m}}
    print("current middle point clusters is " + str(m_clusters) + "...")

    if m_clusters == n_clusters:
        adata.obs[method + '_' + str(m_clusters)] = adata.obs[method + '_mid']
        print("the resolution for " + str(m_clusters) + " clusters is", m, '...')
        del adata.obs[method + '_upper']
        del adata.obs[method + '_lower']
        del adata.obs[method + '_mid']
        return m_clusters, m
    elif remaining_iter == 0:
        adata.obs[method + '_' + str(m_clusters)] = adata.obs[method + '_mid']
        print("max iteration reached but desired resolution hasn't been found yet...")
        print("the latest resolution for " + str(m_clusters) + " clusters is " + str(m) + "...")
        del adata.obs[method + '_upper']
        del adata.obs[method + '_lower']
        del adata.obs[method + '_mid']
        return m_clusters, m
    else:
        remaining_iter = remaining_iter - 1
        if m_clusters > n_clusters:
            print("current middle point resolution is too big, trying smaller resolution...")
            adata.obs[method + '_upper'] = adata.obs[method + '_mid']
            return bisect_cluster(adata, n_clusters=n_clusters, remaining_iter=remaining_iter, lower=lower,
                                  upper=m, method=method)
        elif m_clusters < n_clusters:
            print("current middle point resolution is too small, trying bigger resolution...")
            adata.obs[method + '_lower'] = adata.obs[method + '_mid']
            return bisect_cluster(adata, n_clusters=n_clusters, remaining_iter=remaining_iter, lower=m,
                                  upper=upper, method=method)

def plot_spatial_cluster(adata, cluster_key, palette,
                         marker_size=0.3, marker_type="o",
                         save_fig=True, title=None, out_dir=None, fig_name=None, dpi=1000,
                         he_overlay=None):
    """
    :param adata: input adata which contains clustering information in adata.obs[cluster_key] and spatial information in adata.obs['x'] and adata.obs['y']
    :param cluster_key: key in adata.obs that represents the clustering result
    :param palette: coloring palette for plotting
    :param marker_size: marker size
    :param marker_type: marker type
    :param save_fig: whether to save the figure or not
    :param title: figure title
    :param out_dir: output figure directory
    :param fig_name: output figure file name
    :param dpi: dpi for the figure to be saved
    :param he_overlay: h&e image to overlay
    :return: none
    """
    fig, ax = plt.subplots()
    if he_overlay is not None:
        im = ax.imshow(he_overlay)
    for i in np.sort(adata.obs[cluster_key].astype(int).unique()):
        data = adata[adata.obs[cluster_key] == str(i), :]
        ax.scatter(data.obs['x'], data.obs['y'], s=marker_size, marker=marker_type,
                   c=np.repeat([palette[i][0:3]], [data.shape[0]], axis=0),
                   edgecolors='none', label=str(i))
    ax.legend(scatterpoints=1, fontsize=5, markerscale=5, title='cluster', bbox_to_anchor=(1.02, 1), prop={'size': 5})
    if he_overlay is None:
        ax.invert_yaxis()
    if title is None:
        title = "Spatial Cluster Results using " + cluster_key
    plt.title(title)
    plt.tight_layout()
    if save_fig:
        plt.savefig(out_dir + fig_name, dpi=dpi)
    plt.show()

def plot_spatial_gene(adata, gene_name,
                         marker_size=0.3, marker_type="o",
                         save_fig=True, title=None, out_dir=None, fig_name=None, dpi=1000,
                         he_overlay=None):
    """
    :param adata: input adata which contains clustering information in adata.obs[cluster_key] and spatial information in adata.obs['x'] and adata.obs['y']
    :param gene_name: gene_name in adata.var
    :param marker_size: marker size
    :param marker_type: marker type
    :param save_fig: whether to save the figure or not
    :param title: figure title
    :param out_dir: output figure directory
    :param fig_name: output figure file name
    :param dpi: dpi for the figure to be saved
    :param he_overlay: h&e image to overlay
    :return: none
    """
    fig, ax = plt.subplots()
    if he_overlay is not None:
        im = ax.imshow(he_overlay, alpha=0.2)
    ax.scatter(adata.obs['x'], adata.obs['y'], s=marker_size, marker=marker_type,
               c=adata[:, gene_name].X / adata[:, gene_name].X.sum(), cmap="inferno", edgecolors='none')
    # plt.colorbar()
    if he_overlay is None:
        ax.invert_yaxis()
    if title is None:
        title = "Gene Expression for Gene " + gene_name
    plt.title(title)
    plt.tight_layout()
    if save_fig:
        plt.savefig(out_dir + fig_name, dpi=dpi)
    plt.show()


def knn_neighborhood(adata, k=None, layer_str=None, n_layers=None, max_distance=None, min_neighbors=2):
    """
    Get kNN neighborhood information of adata based on x, y coordinates with maximum distance and minimum number of neighbors requirement
    :param adata: input adata with spatial coordinates in adata.obsm['spatial']
    :param k: if spatial coordinates are not in squares or hexagons, must input the number of nearest neighbors manually
    :param layer_str: Layer structure. If spatial coordinates are in squares or hexagons, input either 'square' or 'hex' as well as the number of layers to calculate k value automatically
    :param n_layer: Number of layers. If spatial coordinates are in squares or hexagons, the number of layers to calculate k value automatically
    :param max_distance: the maximum distance to be considered as a neighbor
    :param max_distance: the minimum number of neighbors that are required for all spots, if not, filter out invalid spots and recursively fit this funtion
    :return: dictionary with four objects 1. 'adata': adata after filtering all invalid spots
                                          2. 'index': matrix (nobs, k), i-th row represents k-nearest neighbors' index for i-th spot. -1 for filling void for out-of-range neighbors
                                          3. 'distance': matrix (nobs, k), (i, j) represents the distance between i-th spot and its j-th neighbor. -1 for filling void for out-of-range neighbors
                                          4. 'k': value for the number of neighbors considered before discarding out-of-range neighbors
    """
    if k is None:
        if layer_str == "square":
            assert n_layers > 1
            k = (n_layers - 1) * 8
            print("now considering square spots, the number of nearest neighbors is " + str(k) + "...")
        elif layer_str == "hex":
            assert n_layers > 1
            k = (n_layers - 1) * 6
            print("now considering hexagon spots, the number of nearest neighbors is " + str(k) + "...")
    X = adata.obsm['spatial']
    assert k > min_neighbors
    stop = True

    distance = sklearn.neighbors.kneighbors_graph(X, n_neighbors=k, mode='distance', n_jobs=-1, include_self=False).tocoo()
    distance = np.reshape(np.asarray(distance.data), [X.shape[0], k])

    adj = sklearn.neighbors.kneighbors_graph(X, n_neighbors=k, mode='connectivity', n_jobs=-1, include_self=False).tocoo()
    ind = np.reshape(np.asarray(adj.col), [X.shape[0], k])

    if max_distance is not None:
        ## re-code out-of-range neighbors as '-1'
        discard_row, discard_col = np.where(distance > max_distance)
        print("marking " + str(discard_row.__len__()) + " neighbors which are out of max distance " + str(max_distance) + "...")
        distance[discard_row, discard_col] = -1
        ind[discard_row, discard_col] = -1
        if np.where(np.sum(ind!=-1, axis=1) < min_neighbors)[0].__len__() > 0:
            stop = False
    if not stop:
        keep_spot_ind = np.where(np.sum(ind != -1, axis=1) >= min_neighbors)[0]
        print("filtering out spots that don't have enough neighbors...re-fitting knn model...")
        adata = adata[keep_spot_ind, :]
        adata.obs.index = range(adata.shape[0])
        knn = knn_neighborhood(adata=adata, k=k, max_distance=max_distance, min_neighbors=min_neighbors)
    else:
        print("all spots now have enough neighbors...")
        knn = {"adata": adata, "index": ind, "distance": distance, "k": k}
    return knn

def plot_visium_ge(out_dir, adata, gene, he=None, dpi=0.1, clip_q=[0, 99], log1p_transform=False, savename_prefix=None, save_colorbar=True, save_fig=True, save_he_copy=False):
    locs = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'])
    if he is not None:
        height_pixels, width_pixels = he.shape[0:2]
    else:
        width_pixels = ((locs.x.max()-np.min([0, locs.x.min()]))//100 + 1)*100
        height_pixels = ((locs.y.max()-np.min([0, locs.y.min()]))//100 + 1)*100
    width_fig, height_fig = width_pixels / 1, height_pixels / 1

    fig, ax = plt.subplots(1, 1, figsize=(width_fig, height_fig), dpi=dpi)
    if he is not None:
        im = ax.imshow(he, extent=[0, width_pixels, 0, height_pixels], alpha=0.8)
        if save_he_copy and save_fig:
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            if savename_prefix is None:
                plt.savefig(out_dir + 'visium_spot_he_rescaled.jpg', dpi=dpi)
            else:
                plt.savefig(out_dir + savename_prefix + '_he_rescaled.jpg', dpi=dpi)
            fig, ax = plt.subplots(1, 1, figsize=(width_fig, height_fig), dpi=dpi)
            im = ax.imshow(he, extent=[0, width_pixels, 0, height_pixels],
                           alpha=0.8)
    spot_diameter_fullres = adata.uns['spatial'][next(iter(adata.uns['spatial']))]['scalefactors'][
        'spot_diameter_fullres']
    radius_in_pixel = spot_diameter_fullres * 0.5 / (width_pixels/width_fig/dpi)
    s = (radius_in_pixel*2*72/fig.dpi)**2
    ge = np.array(adata[:, gene].X.todense()).flatten()
    qmin, qmax = clip_q[0], clip_q[1]
    ge = np.clip(ge, a_min=np.percentile(ge, qmin), a_max=np.percentile(ge, qmax))
    if log1p_transform:
        print("Doing log1p transformation...")
        ge = np.log1p(ge)
    sct = ax.scatter(locs.x, height_pixels - locs.y, s=s, marker='o', c=ge/np.max(ge), cmap='turbo')
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if savename_prefix is None:
        save_name = "visium_spot_" + gene + ".jpg"
    else:
        save_name = savename_prefix + "_" + gene + ".jpg"
    if save_fig:
        print("saving the plot...")
        plt.savefig(out_dir + save_name, dpi=dpi)
    else:
        plt.show()

    if save_colorbar and save_fig:
        if savename_prefix is None:
            cbr_name = "visium_spot_" + gene + "_colorbar.jpg"
        else:
            last_dot_index = save_name.rfind('.')
            if last_dot_index != -1:
                substring = save_name[:last_dot_index]
                cbr_name = substring + '_colorbar.jpg'
            else:
                cbr_name = save_name + '_colorbar.jpg'
        plot_detected_colorbar(out_dir=out_dir, axs=sct, width_fig=10.24, height_fig=7.936, label_name="Relative Expression Level", save_name=cbr_name, dpi=100, label_size=20)
def plot_visium_cluster(out_dir, adata, cluster_key, palette, dpi=0.1,
                        he=None, savename_prefix=None, save_fig=True, save_he_copy=False, save_legend=True):
    locs = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'])
    if he is not None:
        height_pixels, width_pixels = he.shape[0:2]
    else:
        width_pixels = ((locs.x.max() - np.min([0, locs.x.min()])) // 100 + 1) * 100
        height_pixels = ((locs.y.max() - np.min([0, locs.y.min()])) // 100 + 1) * 100
    width_fig, height_fig = width_pixels / 1, height_pixels / 1

    fig, ax = plt.subplots(1, 1, figsize=(width_fig, height_fig), dpi=dpi)
    if he is not None:
        im = ax.imshow(he, extent=[0, width_pixels, 0, height_pixels], alpha=0.8)
        if save_he_copy and save_fig:
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            if savename_prefix is None:
                plt.savefig(out_dir + 'visium_spot_he_rescaled.jpg', dpi=dpi)
            else:
                plt.savefig(out_dir + savename_prefix + '_he_rescaled.jpg', dpi=dpi)
            fig, ax = plt.subplots(1, 1, figsize=(width_fig, height_fig), dpi=dpi)
            im = ax.imshow(he, extent=[0, width_pixels, 0, height_pixels],
                           alpha=1)
    spot_diameter_fullres = adata.uns['spatial'][next(iter(adata.uns['spatial']))]['scalefactors'][
        'spot_diameter_fullres']
    radius_in_pixel = spot_diameter_fullres * 0.5 / (width_pixels / width_fig / dpi)
    s = (radius_in_pixel * 2 * 72 / fig.dpi) ** 2
    for i in np.sort(adata.obs[cluster_key].astype(int).unique()):
        data = adata[adata.obs[cluster_key] == str(i), :]
        ax.scatter(locs.x[(adata.obs[cluster_key] == str(i)).tolist()],
                   height_pixels - locs.y[(adata.obs[cluster_key] == str(i)).tolist()],
                   s=s, marker='o',
                   c=np.repeat([palette[i][0:3]], [data.shape[0]], axis=0),
                   edgecolors='none', label=str(i), alpha=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if savename_prefix is None:
        save_name = "visium_cluster_" + cluster_key + ".jpg"
    else:
        save_name = savename_prefix + "_" + cluster_key + ".jpg"
    if save_fig:
        print("saving the plot...")
        plt.savefig(out_dir + save_name, dpi=dpi)
    else:
        plt.show()
        plt.close(fig)
    if save_legend:
        spot_labels = [str(i) for i in np.sort(adata.obs[cluster_key].astype(int).unique())]
        plot_cluster_legend(out_dir=out_dir, spot_colors=palette, cluster_key=cluster_key, spot_labels=spot_labels, savename_prefix=savename_prefix, dpi=300)

def transform_coordinates(coords, homography_matrix):
    return(perspectiveTransform(coords.reshape(-1,1,2).astype(np.float32), homography_matrix )[:,0,:] )

def plot_transcript_ge(transcripts, gene, out_dir, transform_dir = None, he=None, color='red', save_name=None, save_fig=True):
    '''
    :param transcripts: input transcript df for plotting, including x, y coord.
    :param gene: specific gene name for plotting
    :param out_dir: output directory for saved plot
    :param transform_dir: if not none, read the transformation matrix, otherwise suppose the transformation has already been done
    :param he: input h&e image for overlay
    :param save_name: saved plot file name. by default is "binned_" + method + "_" + str(size) + "_" + gene + ".jpg"
    :param color: color for plotting, by default is 'red'
    :return: none
    '''
    if transform_dir is not None:
        stage_to_morph = np.genfromtxt(transform_dir + 'stage_to_morph.txt', delimiter=',')  ## this matrix transform microns size to morphology pixel size, by default on level 0
        transcript_coords_in_morph = transform_coordinates(np.array(transcripts[['x_location', 'y_location']]), stage_to_morph)
        morphology_to_he = np.genfromtxt(transform_dir + 'morphology_to_he.txt', delimiter=',')  ## this matrix transform morphology pixel size, by default on level 0, to h&e pixel size
        transcript_coords_in_he = transform_coordinates(np.array(transcript_coords_in_morph), morphology_to_he)
    transcripts.x_location = transcript_coords_in_he[:, 0]
    transcripts.y_location = transcript_coords_in_he[:, 1]

    transcripts = transcripts[transcripts['feature_name'] == gene]

    if he is not None:
        height_pixels, width_pixels = he.shape[0:2]
    else:
        width_pixels = ((transcripts.x_location.max()-np.min([0, transcripts.x_location.min()]))//100 + 1)*100
        height_pixels = ((transcripts.y_location.max()-np.min([0, transcripts.y_location.min()]))//100 + 1)*100
    width_fig, height_fig = width_pixels // 1000, height_pixels // 1000

    fig, ax = plt.subplots(1, 1, figsize=(width_fig, height_fig), dpi=100)
    if he is not None:
        im = ax.imshow(he)
    ax.scatter(transcripts.x_location, transcripts.y_location, s=0.1, marker='o', color=color)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    print("saving the plot...")
    if save_name is None:
        save_name = "xenium_transcript_" + gene + ".jpg"
    if save_fig:
        print("saving the plot...")
        plt.savefig(out_dir + save_name, dpi=100)
    else:
        plt.show()

def plot_cell_centroid(cells, out_dir, transform_dir = None, he=None, color='red', save_name=None, save_fig=True):
    '''
    :param cells: input cells df for plotting, including x, y coord.
    :param gene: specific gene name for plotting
    :param out_dir: output directory for saved plot
    :param transform_dir: if not none, read the transformation matrix, otherwise suppose the transformation has already been done
    :param he: input h&e image for overlay
    :param save_name: saved plot file name. by default is "binned_" + method + "_" + str(size) + "_" + gene + ".jpg"
    :param color: color for plotting, by default is 'red'
    :return: none
    '''
    if transform_dir is not None:
        stage_to_morph = np.genfromtxt(transform_dir + 'stage_to_morph.txt', delimiter=',')  ## this matrix transform microns size to morphology pixel size, by default on level 0
        transcript_coords_in_morph = transform_coordinates(np.array(cells[['x_centroid', 'y_centroid']]), stage_to_morph)
        morphology_to_he = np.genfromtxt(transform_dir + 'morphology_to_he.txt', delimiter=',')  ## this matrix transform morphology pixel size, by default on level 0, to h&e pixel size
        transcript_coords_in_he = transform_coordinates(np.array(transcript_coords_in_morph), morphology_to_he)
        cells.x_centroid = transcript_coords_in_he[:, 0]
        cells.y_centroid = transcript_coords_in_he[:, 1]
    else:
        print("Directly working on transformed coordinates...")
    if he is not None:
        height_pixels, width_pixels = he.shape[0:2]
    else:
        width_pixels = ((cells.x_centroid.max()-np.min([0, cells.x_centroid.min()]))//100 + 1)*100
        height_pixels = ((cells.y_centroid.max()-np.min([0, cells.y_centroid.min()]))//100 + 1)*100
    width_fig, height_fig = width_pixels // 1000, height_pixels // 1000

    fig, ax = plt.subplots(1, 1, figsize=(width_fig, height_fig), dpi=100)
    if he is not None:
        im = ax.imshow(he)
    ax.scatter(cells.x_centroid, cells.y_centroid, s=0.1, marker='o', color=color)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    print("saving the plot...")
    if save_name is None:
        save_name = "xenium_cells_centroid.jpg"
    if save_fig:
        print("saving the plot...")
        plt.savefig(out_dir + save_name, dpi=100)
    else:
        plt.show()



def plot_bins_umi_patch(cnts, locs, size, method, out_dir, he=None, cmap='turbo', clip_q=[0, 99], log1p_transform=False, clean=False, savename_prefix=None, save_colorbar=True, save_fig=True, save_he_copy=False):
    '''
    :param cnts: input cnts df for plotting
    :param locs: input locs df for plotting
    :param method: either 'cell' or 'transcript'
    :param size: size in pixel for binned squares
    :param out_dir: output directory for saved plot
    :param he: input h&e image for overlay
    :param save_name: saved plot file name. by default is "binned_" + method + "_" + str(size) + "_overlay.jpg"
    :param cmap: cmap palette for plotting, by default is 'turbo'
    :return: none
    '''
    ## locs represents the center coords. for each square
    assert he.shape[2] == 3
    umi = cnts.sum(axis=1)
    qmin, qmax = clip_q[0], clip_q[1]
    umi = np.clip(umi, a_min=np.percentile(umi, qmin), a_max=np.percentile(umi, qmax))
    umi_max = umi.max()
    if log1p_transform:
        print("Doing log1p transformation...")
        umi = np.log1p(umi)
    col = cmapFader(cmap_name=cmap, start_val=0, stop_val=1)

    if he is not None:
        height_pixels, width_pixels = he.shape[0:2]
    width_fig, height_fig = width_pixels/1000, height_pixels/1000

    fig, ax = plt.subplots(1, 1, figsize=(width_fig, height_fig), dpi=100)
    patches_list = []
    color_list = []
    print("adding patches...")
    for i in tqdm(range(locs.shape[0])):
        patches_list.append(mpl.patches.Rectangle((locs.iloc[i,0]-size/2, locs.iloc[i,1]-size/2), size, size))
        color_list.append(rgb_to_hex(tuple(int(i*255.0) for i in col.get_rgb(umi[i]/umi_max)[0:3])))
    my_cmap = ListedColormap(color_list)
    patches_collection = PatchCollection(patches_list, cmap=my_cmap)
    patches_collection.set_array(np.arange(len(patches_list)))
    if he is not None:
        im = ax.imshow(he)
        if save_he_copy and save_fig:
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            if savename_prefix is None:
                plt.savefig(out_dir + 'xenium_binned_he_rescaled.jpg', dpi=100)
            else:
                plt.savefig(out_dir + savename_prefix + '_he_rescaled.jpg', dpi=100)
            fig, ax = plt.subplots(1, 1, figsize=(width_fig, height_fig), dpi=100)
            im = ax.imshow(he, extent=[0, width_pixels, 0, height_pixels], alpha=0.8)
    ax.add_collection(patches_collection)
    print("plotting...")
    if clean:
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
    else:
        plt.title("Overlay plot for UMI counts with bin-size " + str(size) + " using " + method + " binning")
    if savename_prefix is None:
        save_name = "xenium_binned_" + method + "_" + str(size) + "_UMI.jpg"
    if save_fig:
        print("saving the plot...")
        plt.savefig(out_dir + save_name, dpi=100)
    else:
        plt.show()

    if save_colorbar and save_fig:
        if savename_prefix is None:
            cbr_name = "xenium_binned_" + method + "_" + str(size) + "_UMI_colorbar.jpg"
        else:
            last_dot_index = save_name.rfind('.')
            if last_dot_index != -1:
                substring = save_name[:last_dot_index]
                cbr_name = substring +'_colorbar.jpg'
            else:
                cbr_name = save_name + '_colorbar.jpg'
        fig, ax = plt.subplots(1, 1, figsize=(10.24, 7.936), dpi=100)
        cbr = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='turbo'),
                           ax=ax, orientation='vertical')
        cbr.set_label(label='Relative Expression Level', size=20, weight='bold')
        cbr.ax.tick_params(labelsize=20)
        ax.remove()
        plt.savefig(out_dir + cbr_name, bbox_inches='tight')

def plot_bins_ge_patch(cnts, locs, size, method, gene, out_dir, he=None, cmap='turbo', clip_q=[0, 99], log1p_transform=False, clean=False, savename_prefix=None, save_colorbar=True, save_fig=True, save_he_copy=False):
    '''
    :param cnts: input cnts df for plotting
    :param locs: input locs df for plotting
    :param size: size in pixel for binned squares
    :param method: either 'cell' or 'transcript'
    :param gene: specific gene name for plotting
    :param out_dir: output directory for saved plot
    :param he: input h&e image for overlay
    :param save_name: saved plot file name. by default is "binned_" + method + "_" + str(size) + "_" + gene + ".jpg"
    :param cmap: cmap palette for plotting, by default is 'turbo'
    :return: none
    '''
    ## locs represents the center coords. for each square
    assert he.shape[2] == 3
    ge = np.array(cnts[gene])
    qmin, qmax = clip_q[0], clip_q[1]
    ge = np.clip(ge, a_min=np.percentile(ge, qmin), a_max=np.percentile(ge, qmax))
    ge_max = ge.max()
    if log1p_transform:
        print("Doing log1p transformation...")
        ge = np.log1p(ge)
    col = cmapFader(cmap_name=cmap, start_val=0, stop_val=1)

    if he is not None:
        height_pixels, width_pixels = he.shape[0:2]
    width_fig, height_fig = width_pixels / 1000, height_pixels / 1000

    fig, ax = plt.subplots(1, 1, figsize=(width_fig, height_fig), dpi=100)
    if he is not None:
        im = ax.imshow(he)
        if save_he_copy and save_fig:
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            if savename_prefix is None:
                plt.savefig(out_dir + 'xenium_binned_he_rescaled.jpg', dpi=100)
            else:
                plt.savefig(out_dir + savename_prefix + '_he_rescaled.jpg', dpi=100)
            fig, ax = plt.subplots(1, 1, figsize=(width_fig, height_fig), dpi=100)
            im = ax.imshow(he, extent=[0, width_pixels, 0, height_pixels], alpha=0.8)
    patches_list = []
    color_list = []
    print("adding patches...")
    for i in tqdm(range(locs.shape[0])):
        patches_list.append(mpl.patches.Rectangle((locs.iloc[i,0]-size/2, locs.iloc[i,1]-size/2), size, size))
        color_list.append(rgb_to_hex(tuple(int(i*255.0) for i in col.get_rgb(ge[i]/ge_max)[0:3])))
    my_cmap = ListedColormap(color_list)
    patches_collection = PatchCollection(patches_list, cmap=my_cmap)
    patches_collection.set_array(np.arange(len(patches_list)))
    ax.add_collection(patches_collection)
    print("plotting...")
    if clean:
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
    else:
        plt.title("Overlay plot for gene " + gene + " with bin-size " + str(size) + " using " + method + " binning")
    if savename_prefix is None:
        save_name = "xenium_binned_" + method + "_" + str(size) + "_" + gene + ".jpg"
    else:
        save_name = savename_prefix + "_" + gene + ".jpg"
    if save_fig:
        print("saving the plot...")
        plt.savefig(out_dir + save_name, dpi=100)
    else:
        plt.show()

    if save_colorbar and save_fig:
        if savename_prefix is None:
            cbr_name = "xenium_binned_" + method + "_" + str(size) + "_" + gene + "_colorbar.jpg"
        else:
            last_dot_index = save_name.rfind('.')
            if last_dot_index != -1:
                substring = save_name[:last_dot_index]
                cbr_name = substring +'_colorbar.jpg'
            else:
                cbr_name = save_name + '_colorbar.jpg'
        fig, ax = plt.subplots(1, 1, figsize=(10.24, 7.936), dpi=100)
        cbr = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='turbo'),
                           ax=ax, orientation='vertical')
        cbr.set_label(label='Relative Expression Level', size=20, weight='bold')
        cbr.ax.tick_params(labelsize=20)
        ax.remove()
        plt.savefig(out_dir + cbr_name, bbox_inches='tight')

def plot_detected_colorbar(out_dir, axs, width_fig, height_fig, label_name, save_name, dpi=100, label_size=20):
    fig, ax = plt.subplots(1, 1, figsize=(width_fig, height_fig), dpi=dpi)
    cbr = plt.colorbar(axs, ax=ax)
    cbr.set_label(label=label_name, size=label_size, weight='bold')
    cbr.ax.tick_params(labelsize=label_size)
    ax.remove()
    plt.savefig(out_dir + save_name, bbox_inches='tight')

def plot_label_cluster(out_dir, color_map, save_name, width_fig=5, height_fig=50,dpi=500, prop_size=20, marker_size=5):
    fig, ax = plt.subplots(1, 1, figsize=(100, 100), dpi=1)
    for cluster_label, color in color_map.items():
        ax.scatter(0, 0, color=color, label=f'Cluster {cluster_label}', s=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [int(s[s.rfind(' ') + 1:]) for s in labels]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    fig, ax = plt.subplots(1, 1, figsize=(width_fig, height_fig), dpi=dpi)
    fig.legend(prop={'size': prop_size}, handles=handles, labels=labels, loc='center', frameon=False, markerscale=marker_size)
    ax.remove()
    plt.savefig(out_dir + save_name, bbox_inches='tight')

def plot_cluster_legend(out_dir, spot_colors, spot_labels, cluster_key, savename_prefix=None, dpi=300):
    fig, ax = plt.subplots(figsize=(2, len(spot_labels) * 0.5))
    ax.axis('off')
    legend_handles = [mlines.Line2D([], [], color=spot_colors[i], marker='o', linestyle='None', markersize=10, label=spot_labels[i]) for i in range(len(spot_labels))]
    ax.legend(handles=legend_handles, loc='center', frameon=False)
    if savename_prefix is None:
        save_name = "visium_cluster_" + cluster_key + "_legend.jpg"
    else:
        save_name = savename_prefix + "_" + cluster_key + "_legend.jpg"
    plt.tight_layout()
    plt.savefig(out_dir + save_name, dpi=dpi)
    plt.close(fig)

def plot_bins_ge(cnts, locs, he, size, method, gene, out_dir, cmap='turbo', clip_q=[0, 99],
                 log1p_transform=False, savename_prefix=None, save_colorbar=True, save_fig=True,
                 save_he_copy=False):
        if save_he_copy and save_fig:
            out_image = (Image.fromarray(he).copy())
            if savename_prefix is None:
                out_image.save(out_dir + 'xenium_binned_he_rescaled.jpg', quality=10)
            else:
                out_image.save(out_dir + savename_prefix + '_he_rescaled.jpg', quality=10)
        assert he.shape[2] == 3
        ge = np.array(cnts[gene])
        qmin, qmax = clip_q[0], clip_q[1]
        ge = np.clip(ge, a_min=np.percentile(ge, qmin), a_max=np.percentile(ge, qmax))
        ge_max = ge.max()
        if log1p_transform:
            print("Doing log1p transformation...")
            ge = np.log1p(ge)
        col = cmapFader(cmap_name=cmap, start_val=0, stop_val=1)

        he = he.copy()
        locs = np.array(locs.copy())
        print("plotting bins on original H&E...")

        x_start = (np.floor(locs[:, 0] - size / 2)).astype(int)
        x_end = (np.floor(locs[:, 0] + size / 2)).astype(int)
        y_start = (np.floor(locs[:, 1] - size / 2)).astype(int)
        y_end = (np.floor(locs[:, 1] + size / 2)).astype(int)
        color = (255 * col.get_rgb(ge/ge_max)).astype(int)

        for i in tqdm(range(len(locs))):
            he[y_start[i]:y_end[i], x_start[i]:x_end[i], :] = color[i, 0:3]
        out_image = (Image.fromarray(he).copy()).resize((he.shape[1], he.shape[0]))

        if savename_prefix is None:
            save_name = "xenium_binned_" + method + "_" + str(size) + "_" + gene + ".jpg"
        else:
            save_name = savename_prefix + "_" + gene + ".jpg"
        if save_fig:
            print("saving the plot...")
            out_image.save(out_dir + save_name, quality=10)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(he.shape[1]/1000, he.shape[0]/1000), dpi=100)
            plt.title("Overlay plot for gene " + gene + " with bin-size " + str(size) + " using " + method + " binning")
            plt.imshow(he)
            plt.axis('off')
            plt.show()

        if save_colorbar and save_fig:
            if savename_prefix is None:
                cbr_name = "xenium_binned_" + method + "_" + str(size) + "_" + gene + "_colorbar.jpg"
            else:
                last_dot_index = save_name.rfind('.')
                if last_dot_index != -1:
                    substring = save_name[:last_dot_index]
                    cbr_name = substring + '_colorbar.jpg'
                else:
                    cbr_name = save_name + '_colorbar.jpg'
            fig, ax = plt.subplots(1, 1, figsize=(10.24, 7.936), dpi=100)
            cbr = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='turbo'),
                               ax=ax, orientation='vertical')
            cbr.set_label(label='Relative Expression Level', size=20, weight='bold')
            cbr.ax.tick_params(labelsize=20)
            ax.remove()
            plt.savefig(out_dir + cbr_name, bbox_inches='tight')

def plot_bins_umi(cnts, locs, he, size, method, out_dir, cmap='turbo', clip_q=[0, 99],
                 log1p_transform=False, savename_prefix=None, save_colorbar=True, save_fig=True,
                 save_he_copy=False):
    locs = np.array(locs.copy())

    # --- handle missing H&E input ---
    if he is None:
        print("No H&E image provided — creating white placeholder image...")
        # Compute required dimensions
        x_end = int(np.ceil(np.max(locs[:, 0] + size / 2)))
        y_end = int(np.ceil(np.max(locs[:, 1] + size / 2)))
        he = np.ones((y_end, x_end, 3), dtype=np.uint8) * 255  # white RGB background

    if save_he_copy and save_fig:
        out_image = (Image.fromarray(he).copy())
        if savename_prefix is None:
            out_image.save(out_dir + 'xenium_binned_he_rescaled.jpg', quality=10)
        else:
            out_image.save(out_dir + savename_prefix + '_he_rescaled.jpg', quality=10)
    umi = cnts.sum(axis=1)
    qmin, qmax = clip_q[0], clip_q[1]
    umi = np.clip(umi, a_min=np.percentile(umi, qmin), a_max=np.percentile(umi, qmax))
    umi_max = umi.max()
    if log1p_transform:
        print("Doing log1p transformation...")
        umi = np.log1p(umi)
    col = cmapFader(cmap_name=cmap, start_val=0, stop_val=1)

    he = he.copy()
    print("plotting bins on original H&E...")

    x_start = (np.floor(locs[:, 0] - size / 2)).astype(int)
    x_end = (np.floor(locs[:, 0] + size / 2)).astype(int)
    y_start = (np.floor(locs[:, 1] - size / 2)).astype(int)
    y_end = (np.floor(locs[:, 1] + size / 2)).astype(int)
    color = (255 * col.get_rgb(umi / umi_max)).astype(int)

    for i in tqdm(range(len(locs))):
        he[y_start[i]:y_end[i], x_start[i]:x_end[i], :] = color[i, 0:3]
    out_image = (Image.fromarray(he).copy()).resize((he.shape[1], he.shape[0]))

    if savename_prefix is None:
        save_name = "xenium_binned_" + method + "_" + str(size) + "_UMI.jpg"
    else:
        save_name = savename_prefix + "_UMI.jpg"
    if save_fig:
        print("saving the plot...")
        if he.shape[0]>10000:
            out_image.save(out_dir + save_name, quality=10)
        else:
            out_image.save(out_dir + save_name, quality=50)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(he.shape[1] / 1000, he.shape[0] / 1000), dpi=100)
        plt.title("Overlay plot for UMI with bin-size " + str(size) + " using " + method + " binning")
        plt.imshow(he)
        plt.axis('off')
        plt.show()

    if save_colorbar and save_fig:
        if savename_prefix is None:
            cbr_name = "xenium_binned_" + method + "_" + str(size) + "_UMI_colorbar.jpg"
        else:
            last_dot_index = save_name.rfind('.')
            if last_dot_index != -1:
                substring = save_name[:last_dot_index]
                cbr_name = substring + '_colorbar.jpg'
            else:
                cbr_name = save_name + '_colorbar.jpg'
        fig, ax = plt.subplots(1, 1, figsize=(10.24, 7.936), dpi=100)
        cbr = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='turbo'),
                           ax=ax, orientation='vertical')
        cbr.set_label(label='Relative Expression Level', size=20, weight='bold')
        cbr.ax.tick_params(labelsize=20)
        ax.remove()
        plt.savefig(out_dir + cbr_name, bbox_inches='tight')

def img_crop(img, extent, mode='edge', constant_values=None):
    extent = np.array(extent)
    pad = np.zeros((img.ndim, 2), dtype=int)
    for i, (lower, upper) in enumerate(extent):
        if lower < 0:
            pad[i][0] = 0 - lower
        if upper > img.shape[i]:
            pad[i][1] = upper - img.shape[i]
    if (pad != 0).any():
        kwargs = {}
        if mode == 'constant' and constant_values is not None:
            kwargs['constant_values'] = constant_values
        img = np.pad(img, pad, mode=mode, **kwargs)
        extent += pad[:extent.shape[0], [0]]
    for i, (lower, upper) in enumerate(extent):
        img = img.take(range(lower, upper), axis=i)
    return img

def img_adjust_margins(img, pad, pad_value=None):
    extent = np.stack([[0, 0], img.shape[:2]]).T
    # make size divisible by pad without changing coords
    remainder = (extent[:, 1] - extent[:, 0]) % pad
    complement = (pad - remainder) % pad
    extent[:, 1] += complement
    if pad_value is None:
        mode = 'edge'
    else:
        mode = 'constant'
    img = img_crop(
            img, extent, mode=mode, constant_values=pad_value)
    return img

def img_reduce(img, factor):
    img_red = reduce(
        img.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
        h=factor, w=factor).astype(np.uint8)
    print("he shape after reducing is ", img_red.shape)
    return img_red

def locs_reduce(locs, factor):
    locs = locs.copy()
    locs //= factor
    locs = locs.round().astype(int)
    return locs

def load_tiff(filename, level=0):
    import tifffile
    img = tifffile.imread(filename, level=level)
    return img

def resize_image(input_image, new_width):
    width_percent = (new_width / float(input_image.shape[1]))
    new_height = int((float(input_image.shape[0]) * float(width_percent)))
    resized_image = Image.fromarray(input_image, 'RGB').resize((new_width, new_height), Image.LANCZOS)
    return resized_image