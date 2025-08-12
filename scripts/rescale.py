import argparse
import os
from time import time
from skimage.transform import rescale
import numpy as np
import pandas as pd
from utils import (
        load_image, save_image, read_string, write_string,
        load_tsv, save_tsv, load_parquet, save_parquet)


def get_image_filename(prefix):
    file_exists = False
    for suffix in ['.jpg', '.png', '.tiff', '.tif', '.btf', '.ome.tiff', '.ome.tif']:
        filename = prefix + suffix
        if os.path.exists(filename):
            file_exists = True
            break
    if not file_exists:
        raise FileNotFoundError('Image not found')
    return filename


# def rescale_image(img, scale):
#     if img.ndim == 2:
#         img = rescale(img, scale, preserve_range=True)
#     elif img.ndim == 3:
#         channels = img.transpose(2, 0, 1)
#         channels = [rescale_image(c, scale) for c in channels]
#         img = np.stack(channels, -1)
#     else:
#         raise ValueError('Unrecognized image ndim')
#     return img


def rescale_image(img, scale):
    if img.ndim == 2:
        scale = [scale, scale]
    elif img.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError('Unrecognized image ndim')
    img = rescale(img, scale, preserve_range=True)
    return img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--locs', action='store_true')
    parser.add_argument('--radius', action='store_true')
    parser.add_argument('--boundary', action='store_true')
    parser.add_argument('--nucleus', action='store_true')
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    if args.image:
        print('Rescaling image...')
    if args.mask:
        print('Rescaling mask...')
    if args.locs:
        print('Rescaling locs...')
    if args.radius:
        print('Rescaling radius...')

    filename = args.prefix+'pixel-size-raw.txt'
    print(f'Reading pixel size from {filename}...')
    pixel_size_raw = float(read_string(args.prefix+'pixel-size-raw.txt'))
    pixel_size = float(read_string(args.prefix+'pixel-size.txt'))
    scale = pixel_size_raw / pixel_size

    if args.image:
        img = load_image(get_image_filename(args.prefix+'he-raw'))
        img = img.astype(np.float32)
        print(f'Rescaling image (scale: {scale:.3f})...')
        t0 = time()
        img = rescale_image(img, scale)
        print(int(time() - t0), 'sec')
        img = img.astype(np.uint8)
        save_image(img, args.prefix+'he-scaled.jpg')

    if args.mask:
        mask = load_image(args.prefix+'mask-raw.png')
        mask = mask > 0
        if mask.ndim == 3:
            mask = mask.any(2)
        print(f'Rescaling mask (scale: {scale:.3f})...')
        t0 = time()
        mask = rescale_image(mask.astype(np.float32), scale)
        print(int(time() - t0))
        mask = mask > 0.5
        save_image(mask, args.prefix+'mask-scaled.png')

    if args.locs:
        if os.path.isfile(args.prefix+'locs-raw.tsv'):
            locs = load_tsv(args.prefix+'locs-raw.tsv')
            locs = locs * scale
            locs = locs.round().astype(int)
            save_tsv(locs, args.prefix + 'locs.tsv')
        elif os.path.isfile(args.prefix+'locs-raw.parquet'):
            locs = load_parquet(args.prefix+'locs-raw.parquet')
            locs = locs * scale
            locs = locs.round().astype(int)
            save_parquet(locs, args.prefix + 'locs.parquet')
    
    if args.boundary:
        boundary = load_tsv(args.prefix+'boundary-raw.tsv')
        boundary = boundary[['x', 'y', 'id', 'max_x', 'max_y', 'min_x', 'min_y']]
        boundary['x'] = (boundary['x'] * scale).round().astype(int)
        boundary['y'] = (boundary['x'] * scale).round().astype(int)
        boundary['max_x'] = (boundary['max_x'] * scale).round().astype(int)
        boundary['max_y'] = (boundary['max_y'] * scale).round().astype(int)
        boundary['min_x'] = (boundary['min_x'] * scale).round().astype(int)
        boundary['min_y'] = (boundary['min_y'] * scale).round().astype(int)
        save_tsv(boundary, args.prefix + 'boundary.tsv')
    
    if args.nucleus:
        nucleus = pd.read_csv(args.prefix+'nucleus-raw.csv.gz')
        print("nuceleus.head():", nucleus.head())
        nucleus['vertex_x'] = (nucleus['vertex_x'] * scale).round().astype(int)
        nucleus['vertex_y'] = (nucleus['vertex_y'] * scale).round().astype(int)
        nucleus.to_csv(args.prefix + "nucleus.csv.gz", index=False, compression="gzip")
        
    if args.radius:
        radius = float(read_string(args.prefix+'radius-raw.txt'))
        radius = radius * scale
        radius = np.round(radius).astype(int)
        write_string(radius, args.prefix+'radius.txt')

if __name__ == '__main__':
    main()
