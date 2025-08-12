import sys
import argparse
import numpy as np

from utils import load_image, save_image, load_pickle
from cluster_hist import cluster
from connected_components import relabel_small_connected
from image import crop_image
from tissue_mask import compute_tissue_mask

def remove_margins(embs, mar):
    for ke, va in embs.items():
        embs[ke] = [
                v[mar[0][0]:-mar[0][1], mar[1][0]:-mar[1][1]]
                for v in va]
def get_img_mask(img):
    print('Computing tissue mask...')
    mask = compute_tissue_mask(
            img, size_threshold=0.001, initial_mask=None,
            max_iter=100)
    return mask

def get_mask_embeddings(embs, mar=16, min_connected=4000):

    n_clusters = 2

    # remove margins to avoid border effects
    remove_margins(embs, ((mar, mar), (mar, mar)))

    # get features
    x = np.concatenate(list(embs.values()))

    # segment image
    labels, __ = cluster(x, n_clusters=n_clusters, method='km')
    labels = relabel_small_connected(labels, min_size=min_connected)

    # select cluster for foreground
    rgb = np.stack(embs['rgb'], -1)
    i_foreground = np.argmax([
        rgb[labels == i].std() for i in range(n_clusters)])
    mask = labels == i_foreground

    # restore margins
    extent = [(-mar, s+mar) for s in mask.shape]
    mask = crop_image(
            mask, extent,
            mode='constant', constant_values=mask.min())

    return mask

def get_border_mask(he, n_margin = 1):
    img = np.zeros([int(he.shape[0] / 16), int(he.shape[1] / 16), 3], dtype=np.uint8)
    img.fill(255)
    img[0:(n_margin*16), :, :] = 0
    img[(img.shape[0]-n_margin*16):(img.shape[0]), :, :] = 0
    img[:, 0:(n_margin*16), :] = 0
    img[:, (img.shape[1]-n_margin*16):(img.shape[1]), :] = 0
    return img

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('--bordermask', action='store_true')
    parser.add_argument('--margin', type=int, default=1)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    inpfile = args.infile
    outfile = args.outfile
    use_emb = inpfile.endswith('.pickle')

    if use_emb:
        embs = load_pickle(inpfile)
        mask = get_mask_embeddings(embs)
    else:
        if args.bordermask:
            he = load_image(args.infile)
            mask = get_border_mask(he, args.margin)
        else:
            img = load_image(inpfile)
            mask = get_img_mask(img)

    save_image(mask, outfile)

if __name__ == '__main__':
    main()
