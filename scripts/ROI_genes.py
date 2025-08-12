from ast import arg
from cv2 import add
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
import numpy as np
from utils import load_image, load_tsv, load_mask, save_image
from my_utils import img_reduce

def plot_truth(x, outfile, truncate=None, he=None, locs=None, save=True):
    x = x.copy()
    mask = np.isfinite(x)
    if truncate is not None:
        x = np.clip(x, truncate[0], truncate[1])
    # col = cmapFader(cmap_name='turbo', start_val=0, stop_val=1)
    # img = col.get_rgb(x)[:, :, :3]
    cmap = plt.get_cmap('turbo')
    img = cmap(x)[..., :3]
    img[~mask] = 1.0
    img = (img * 255).astype(np.uint8)
    if locs is not None:
        if he is not None:
            out = he.copy()
        else:
            out = np.full((x.shape[0], x.shape[1], 3), 255)
        out[locs[:, 0], locs[:, 1], :] = img[locs[:, 0], locs[:, 1], :]
        img = out
    filter = np.isnan(x)
    if he is not None:
        img[filter] = he[filter]
    img = img.astype(np.uint8)
    if save:
        save_image(img, outfile)
    return img


def get_all_files(base_dir):
    """ Recursively get all .jpg and .png files under base_dir and return a dictionary with relative paths. """
    file_dict = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                file_dict[rel_path] = os.path.join(root, file)  # Store full path
    return file_dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pref', type=str, default=None)
    parser.add_argument('--pref1', type=str, default=None)
    parser.add_argument('--pref2', type=str, default=None)
    parser.add_argument('--pref_xenium', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--x', type=int, default=None)
    parser.add_argument('--y', type=int, default=None)
    parser.add_argument('--hx', type=int, default=None)
    parser.add_argument('--hy', type=int, default=None)
    parser.add_argument('--drophe', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    data_pref = args.pref
    apple_dir = args.pref1
    banana_dir = args.pref2
    output_dir = args.output
    x = args.x
    y = args.y
    hx = args.hx
    hy = args.hy
    drophe = args.drophe
    
    if args.pref_xenium is not None:
        cnts_xenium = load_tsv(f'{args.pref_xenium}cnts.tsv')
        locs_xenium = load_tsv(f'{args.pref_xenium}locs.tsv')
        locs_xenium = locs_xenium.astype(float)
        locs_xenium = np.stack([locs_xenium['y'], locs_xenium['x']], -1)
        locs_xenium //= 16
        locs_xenium = locs_xenium.round().astype(int)
        he_xenium = load_image(f'{args.pref_xenium}he.jpg')
        he_xenium = img_reduce(he_xenium, factor=16)
        mask_xenium = load_mask(f'{args.pref_xenium}mask-small-RGB.png')
    
    os.makedirs(output_dir, exist_ok=True)        
    if not os.path.exists(apple_dir) or not os.path.exists(banana_dir):
        print("One or both base directories do not exist.")
        exit()

    apple_files = get_all_files(apple_dir)
    banana_files = get_all_files(banana_dir)
    common_files = set(apple_files.keys()) & set(banana_files.keys())

    if not drophe:
        he = load_image(f"{data_pref}he.jpg")
        he = img_reduce(he, factor = 16)
        if isinstance(he, np.ndarray):
            he = Image.fromarray(he) 
        if hx is not None and hy is not None and hx > 0 and hy > 0: 
            he = he.crop((x, y, x + hx, y + hy))
         
    for rel_path in common_files:
        print(f"Processing {rel_path}")
        img1_path, img2_path = apple_files[rel_path], banana_files[rel_path]
        gene_name = os.path.splitext(os.path.basename(rel_path))[0]
        
        add_xenium = False
        if hx is None and hy is None and args.pref_xenium is not None and gene_name in cnts_xenium.columns:
            img_xenium = np.full((mask_xenium.shape[0], mask_xenium.shape[1]), np.nan)
            img_xenium[locs_xenium[:, 0], locs_xenium[:, 1]] = cnts_xenium[gene_name]
            img_xenium[~mask_xenium] = np.nan
            add_xenium = True
            print(f"adding xenium data for gene {gene_name}")
            img_xenium = plot_truth(img_xenium / np.nanmax(img_xenium), outfile="temp.png", save=False, he=he_xenium)
            new_folder = os.path.join(os.path.dirname(rel_path), "matching_xenium/")
            os.makedirs(os.path.join(f"{output_dir}{x}_{y}_{hx}_{hy}/", new_folder), exist_ok=True)
            rel_path = os.path.join(new_folder, os.path.basename(rel_path))
        os.makedirs(os.path.join(f"{output_dir}{x}_{y}_{hx}_{hy}/", os.path.dirname(rel_path)), exist_ok=True)
        output_path = os.path.join(f"{output_dir}{x}_{y}_{hx}_{hy}/", rel_path)
        
        try:
            img1, img2 = Image.open(img1_path), Image.open(img2_path)
            # Perform cropping if hx and hy are valid
            if hx and hy and hx > 0 and hy > 0:
                print("Cropping...")
                img1, img2 = img1.crop((x, y, x + hx, y + hy)), img2.crop((x, y, x + hx, y + hy))
            else:
                print("Invalid cropping parameters, keeping the original size...")

            num_subplots = 3 if not drophe else 2
            if add_xenium:
                num_subplots += 1
            if num_subplots == 2:
                fig, axes = plt.subplots(1, num_subplots, figsize=(10, 5))
            elif num_subplots == 3:
                fig, axes = plt.subplots(1, num_subplots, figsize=(15, 5))
            else:
                fig, axes = plt.subplots(1, num_subplots, figsize=(20, 5))
            fig.suptitle(f"Gene: $\\it{{{os.path.splitext(os.path.basename(rel_path))[0]}}}$", fontsize=16)

            axes[0].imshow(img1)
            axes[0].axis("off")
            axes[0].set_title("Imputed")

            axes[1].imshow(img2)
            axes[1].axis("off")
            axes[1].set_title("Observed")

            if not drophe:
                axes[2].imshow(he)
                axes[2].axis("off")
                axes[2].set_title("H&E")
                if add_xenium:
                    axes[3].imshow(img_xenium)
                    axes[3].axis("off")
                    axes[3].set_title("Observed (Xenium)")
            else:
                if add_xenium:
                    axes[2].imshow(img_xenium)
                    axes[2].axis("off")
                    axes[2].set_title("Observed (Xenium)")

            plt.tight_layout()
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved: {output_path}")

        except Exception as e:
            print(f"Error processing {rel_path}: {e}")

if __name__ == '__main__':
    main()