import sys
import argparse
import numpy as np
import os
from einops import reduce
from PIL import Image
from my_utils import cmapFader, img_reduce
from utils import load_pickle
Image.MAX_IMAGE_PIXELS = None


def plot_overlay(data_pref, label_pref, save=True):
    if isinstance(label_pref, str):
        label_pref = [label_pref]

    he_rgb_path = f'{data_pref}he.jpg'
    he_rgb_image = Image.open(he_rgb_path)
    he_rgb_array = np.array(he_rgb_image)[:, :, 0:3]
    factor = 16
    he_rgb_array = img_reduce(he_rgb_array, factor=factor)

    for label_path in label_pref:
        print("now processing label folder:", label_path)
        for pickle_name in os.listdir(label_path):
            if pickle_name.endswith(".pickle"):
                pickle_path = os.path.join(label_path, pickle_name)
                label_pickle = load_pickle(f'{pickle_path}')
                pref = os.path.splitext(pickle_path)[0]
                labeled_image = Image.open(f'{pref}.png')
                labeled_array = np.array(labeled_image)[:, :, 0:3]

                mask = (label_pickle == -1)
                output_array = labeled_array.copy()
                output_array[mask] = he_rgb_array[mask]
                output_img = Image.fromarray((output_array).astype(np.uint8))
                if save:
                    output_img.save(f'{pref}_overlayed.png')
                


def plot_overlay_old(data_pref, label_pref, save=True):
    if isinstance(label_pref, str):
        label_pref = [label_pref]  # Convert to list if it's a single string

    he_rgb_path = f'{data_pref}he.jpg'
    he_rgb_image = Image.open(he_rgb_path)
    he_rgb_array = np.array(he_rgb_image)[:, :, 0:3]
    he_gray_image = he_rgb_image.convert('L')
    he_gray_array = np.array(he_gray_image)

    transparency_array = he_gray_array
    factor = 16
    transparency_array = reduce(
        transparency_array.astype(float), '(h1 h) (w1 w) -> h1 w1', 'mean',
        h=factor, w=factor).astype(np.uint8)

    for label in label_pref:
        print("now processing label folder:", label)
        label_path = f'{label}labels.png'
        label_image = Image.open(label_path)
        label_array = np.array(label_image)[:, :, 0:3]

        # Create overlay variations
        normalized_transparency = transparency_array / 255.0
        label_with_alpha = np.dstack((label_array / 255.0, normalized_transparency))
        result_image_1 = Image.fromarray((label_with_alpha * 255).astype(np.uint8))

        normalized_transparency = 1 - transparency_array / 255.0
        label_with_alpha = np.dstack((label_array / 255.0, normalized_transparency))
        result_image_2 = Image.fromarray((label_with_alpha * 255).astype(np.uint8))

        normalized_transparency = 1 - transparency_array / 255.0
        normalized_transparency *= np.exp(-normalized_transparency)
        normalized_transparency /= normalized_transparency.max()
        label_with_alpha = np.dstack((label_array / 255.0, normalized_transparency))
        label_with_alpha_image = (label_with_alpha * 255).astype(np.uint8)
        result_image_3 = Image.fromarray(label_with_alpha_image).copy()

        he_rgb_array = reduce(
            he_rgb_array.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
            h=factor, w=factor).astype(np.uint8)
        transparent_mask = np.where(label_with_alpha_image[:, :, 3] == 0)
        label_with_alpha_image[transparent_mask[0], transparent_mask[1], 0:3] = he_rgb_array[transparent_mask[0], transparent_mask[1], :]
        label_with_alpha_image[transparent_mask[0], transparent_mask[1], 3] = 255
        white_mask = np.all(label_array[:, :, :3] == [255, 255, 255], axis=-1)
        label_with_alpha_image[white_mask, 0:3] = he_rgb_array[white_mask, 0:3]
        label_with_alpha_image[white_mask, 3] = 255
        result_image_4 = Image.fromarray(label_with_alpha_image).copy()

        label_with_alpha_image[transparent_mask[0], transparent_mask[1], 0:3] = (128, 128, 128)
        label_with_alpha_image[transparent_mask[0], transparent_mask[1], 3] = 255
        label_with_alpha_image[white_mask, 0:3] = (128, 128, 128)
        label_with_alpha_image[white_mask, 3] = 255
        result_image_5 = Image.fromarray(label_with_alpha_image).copy()

        if save:
            result_image_1.save(f'{label}overlay_flipped.png')
            result_image_2.save(f'{label}overlay_unweighted.png')
            result_image_3.save(f'{label}overlay_weighted_bwbg.png')
            result_image_4.save(f'{label}overlay_weighted_rgbbg.png')
            result_image_5.save(f'{label}overlay_weighted_bkbg.png')
        
            
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pref', type=str)
    parser.add_argument('--label_pref', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    data_pref = args.data_pref
    label_pref = args.label_pref
    plot_overlay(data_pref, label_pref, save=True)

if __name__ == '__main__':
    main()
