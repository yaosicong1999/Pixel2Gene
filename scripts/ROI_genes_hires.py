import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
import numpy as np
import pandas as pd
from utils import load_image, load_tsv, load_mask, save_image, read_lines, load_parquet, load_pickle
from my_utils import img_reduce


def plot_super(x, outfile=None, truncate=None, he=None, locs=None, save=True):
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
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--mask', type=str, default=None)
    parser.add_argument('--x', type=int, default=None, help="x coordinate of the top left corner of the ROI in superpixel")
    parser.add_argument('--y', type=int, default=None, help="y coordinate of the top left corner of the ROI in superpixel")
    parser.add_argument('--hx', type=int, default=None, help="width of the ROI in superpixel")
    parser.add_argument('--hy', type=int, default=None, help="height of the ROI in superpixel")
    parser.add_argument('--drophe', action='store_true')
    parser.add_argument('--pref_xenium', type=str, default=None)
    parser.add_argument('--xenium_x', type=int, default=None, help="x coordinate of the top left corner of the ROI in superpixel for the xenium")
    parser.add_argument('--xenium_y', type=int, default=None, help="y coordinate of the top left corner of the ROI in superpixel for the xenium")
    parser.add_argument('--xenium_hx', type=int, default=None, help="width of the ROI in superpixel for the xenium")
    parser.add_argument('--xenium_hy', type=int, default=None, help="height of the ROI in superpixel for the xenium")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    data_pref = args.pref
    output_dir = args.output
    x = args.x
    y = args.y
    hx = args.hx
    hy = args.hy
    drophe = args.drophe
    xenium_x = args.xenium_x
    xenium_y = args.xenium_y
    xenium_hx = args.xenium_hx
    xenium_hy = args.xenium_hy
        
    if args.pref_xenium is not None:
        if os.path.isfile(f'{args.pref_xenium}cnts.parquet'):
            cnts_xenium = pd.read_parquet(f'{args.pref_xenium}cnts.parquet')
            print("the shape of cnts_xenium is: ", cnts_xenium.shape)
        elif os.path.isfile(f'{args.pref_xenium}cnts.tsv'):
            cnts_xenium = load_tsv(f'{args.pref_xenium}cnts.tsv')
        else:
            print("No expression data found for xenium")
        if os.path.isfile(f'{args.pref_xenium}locs.parquet'):
            locs_xenium = load_parquet(f'{args.pref_xenium}locs.parquet')
        else:    
            locs_xenium = load_tsv(f'{args.pref_xenium}locs.tsv')    
        locs_xenium = np.stack([locs_xenium['y'], locs_xenium['x']], -1)
        locs_xenium_small = locs_xenium // 16
        locs_xenium_small = locs_xenium_small.round().astype(int)
        locs_xenium = locs_xenium.round().astype(int)
        print("the shape of locs_xenium_small is: ", locs_xenium_small.shape)
        assert cnts_xenium.shape[0] == locs_xenium_small.shape[0]
        mask_small_xenium = load_mask(f'{args.pref_xenium}mask-small-hs.png')
        shape_small_xenium = mask_small_xenium.shape
        mask_full_xenium = np.kron(mask_small_xenium, np.ones((16, 16), dtype=mask_small_xenium.dtype))
        he_xenium = load_image(f'{args.pref_xenium}he.jpg')
        roi_he_xenium = Image.fromarray(he_xenium)
        if xenium_hx and xenium_hy and xenium_hx > 0 and xenium_hy > 0:
            roi_he_xenium = roi_he_xenium.crop((16*xenium_x, 16*xenium_y, 16*(xenium_x + xenium_hx), 16*(xenium_y + xenium_hy)))
        
    he = load_image(f"{data_pref}he.jpg")
    shape_small = load_mask(f"{data_pref}mask-small-hs.png").shape
    if hx is not None and hy is not None and hx > 0 and hy > 0: 
        roi_he = Image.fromarray(he)
        roi_he = roi_he.crop((16*x, 16*y, 16*(x + hx), 16*(y + hy)))
        
    gene_names = read_lines(f"{data_pref}gene-names.txt")
    gene_names_xenium = read_lines(f"{args.pref_xenium}gene-names.txt") if args.pref_xenium is not None else []
    if gene_names_xenium is not None:
        gene_names = [gn for gn in gene_names if gn in gene_names_xenium]
    top_25 = gene_names[:len(gene_names)//4]
    middle_50 = gene_names[len(gene_names)//4:3*len(gene_names)//4]
    bottom_25 = gene_names[3*len(gene_names)//4:]
    
    if os.path.isfile(f'{data_pref}locs.parquet'):
        locs_test = load_parquet(f'{data_pref}locs.parquet')
    elif os.path.isfile(f'{data_pref}locs.tsv'):
        locs_test = load_tsv(f'{data_pref}locs.tsv')
    locs_test = locs_test.astype(float)
    locs_test = np.stack([locs_test['y'], locs_test['x']], -1)
    locs_test = locs_test.round().astype(int)
    unique_rows, indices, counts = np.unique(locs_test, axis=0, return_index=True, return_counts=True)
    unique_row_indices = indices[counts == 1]
    locs_test = locs_test[unique_row_indices, ]
    locs_test_small = locs_test // 16
    locs_test_small = locs_test_small.round().astype(int)
    if os.path.isfile(f'{data_pref}cnts.tsv'):
        cnts_test = load_tsv(f'{data_pref}cnts.tsv')

    dict = {"top_25": top_25, "middle_50": middle_50, "bottom_25": bottom_25}
    for key, gene_set in dict.items():
        os.makedirs(f"{output_dir}/ROI_hires/{x}_{y}_{hx}_{hy}/{key}", exist_ok=True)
        
        mask_small = load_mask(f"{data_pref}mask-small-hs.png")
        mask_full = np.kron(mask_small, np.ones((16, 16), dtype=mask_small.dtype))
        
        for gn in gene_set: 
            if os.path.isfile(f'{data_pref}cnts.parquet'):
                cnts_test = pd.read_parquet(f'{data_pref}cnts.parquet', columns=[gn])
                cnts_test = cnts_test.iloc[unique_row_indices, :]

            truth_small = np.full(shape_small, np.nan, dtype=float)
            truth_small[locs_test_small[:, 0], locs_test_small[:, 1]] = cnts_test[gn]
            truth_img_small_no_he = plot_super(truth_small/np.nanmax(truth_small), he=None, save=False)
            truth_img_big_he = he.copy()
            for i, j in locs_test_small:
                truth_img_big_he[(16*i-8):(16*i+8), (16*j-8):(16*j+8), :] = truth_img_small_no_he[i, j, :]
            truth_img_big_he[~mask_full, :] = he[~mask_full, :]
    
            imputed_small = load_pickle(f'{output_dir}/cnts-super/{gn}.pickle')
            imputed_img_small_no_he = plot_super(imputed_small/np.nanmax(imputed_small), he=None, save=False)
            imputed_img_big_he = he.copy()
            for i, j in locs_test_small:
                imputed_img_big_he[(16*i-8):(16*i+8), (16*j-8):(16*j+8), :] = imputed_img_small_no_he[i, j, :]
            imputed_img_big_he[~mask_full, :] = he[~mask_full, :]

            try:
                print("Cropping...")
                img1 = Image.fromarray(imputed_img_big_he)
                img2 = Image.fromarray(truth_img_big_he)
                if hx and hy and hx > 0 and hy > 0:
                    img1, img2 = img1.crop((16*x, 16*y, 16*(x + hx), 16*(y + hy))), img2.crop((16*x, 16*y, 16*(x + hx), 16*(y + hy)))
                else:
                    print("Invalid cropping parameters, keeping the original size...")
            except Exception as e:
                print(f"Error processing gene {gn}: {e}")
        
            include_xenium = False
            if args.pref_xenium is not None and gn in cnts_xenium.columns:
                include_xenium = True
                xenium_small = np.full(shape_small_xenium, np.nan, dtype=float)
                xenium_small[locs_xenium_small[:, 0], locs_xenium_small[:, 1]] = cnts_xenium[gn]
                xenium_img_small_no_he = plot_super(xenium_small/np.nanmax(xenium_small), he=None, save=False)
                xenium_img_big_he = he_xenium.copy()
                for i, j in locs_xenium_small:
                    xenium_img_big_he[(16*i-8):(16*i+8), (16*j-8):(16*j+8), :] = xenium_img_small_no_he[i, j, :]
                # xenium_img_big_he[~mask_full_xenium, :] = he_xenium[~mask_full_xenium, :]

                try:
                    print("Cropping xenium expression image...")
                    img4 = Image.fromarray(xenium_img_big_he)
                    if hx and hy and hx > 0 and hy > 0:
                        img4 = img4.crop((16*xenium_x, 16*xenium_y, 16*(xenium_x + xenium_hx), 16*(xenium_y + xenium_hy)))
                    else:
                        print("Invalid cropping parameters, keeping the original size...")
                except Exception as e:
                    print(f"Error processing gene {gn}: {e}")

            if not drophe:
                if include_xenium:
                    fig, axes =  plt.subplots(2, 3, figsize=(12, 8))
                    fig.suptitle(f"Gene: {gn}", fontsize=16)
                    axes[0, 0].imshow(img1)
                    save_image(np.array(img1), f"{output_dir}/ROI_hires/{x}_{y}_{hx}_{hy}/{key}/{gn}_imputed.jpg")
                    axes[0, 0].axis("off")
                    axes[0, 0].set_title("Visium HD Imputed")
                    axes[0, 1].imshow(img2)
                    save_image(np.array(img2), f"{output_dir}/ROI_hires/{x}_{y}_{hx}_{hy}/{key}/{gn}_observed.jpg")
                    axes[0, 1].axis("off")
                    axes[0, 1].set_title("Visium HD Observed")
                    axes[0, 2].imshow(roi_he)
                    save_image(np.array(roi_he), f"{output_dir}/ROI_hires/{x}_{y}_{hx}_{hy}/{key}/{gn}_he.jpg")
                    axes[0, 2].axis("off")
                    axes[0, 2].set_title("Visium HD H&E")
                    axes[1, 0].axis("off") 
                    axes[1, 1].imshow(img4)
                    save_image(np.array(img4), f"{output_dir}/ROI_hires/{x}_{y}_{hx}_{hy}/{key}/{gn}_xenium_observed.jpg")
                    axes[1, 1].axis("off")
                    axes[1, 1].set_title("Xenium Observed")
                    axes[1, 2].imshow(roi_he_xenium)
                    save_image(np.array(roi_he_xenium), f"{output_dir}/ROI_hires/{x}_{y}_{hx}_{hy}/{key}/{gn}_xenium_he.jpg")
                    axes[1, 2].axis("off")
                    axes[1, 2].set_title("Xenium H&E")
                    out_dir = f"{output_dir}/ROI_hires/{x}_{y}_{hx}_{hy}/{key}/matching_xenium/"
                    os.makedirs(out_dir, exist_ok=True)
                else:
                    fig, axes =  plt.subplots(1, 3, figsize=(15, 5))
                    fig.suptitle(f"Gene: {gn}", fontsize=16)
                    axes[0].imshow(img1)
                    axes[0].axis("off")
                    axes[0].set_title("Visium HD Imputed")
                    axes[1].imshow(img2)
                    axes[1].axis("off")
                    axes[1].set_title("Visium HD Observed")
                    axes[2].imshow(roi_he)
                    axes[2].axis("off")
                    axes[2].set_title("Visium HD H&E")
                    out_dir = f"{output_dir}/ROI_hires/{x}_{y}_{hx}_{hy}/{key}/"                    
            else:
                if include_xenium:
                    fig, axes =  plt.subplots(2, 2, figsize=(10, 18))
                    fig.suptitle(f"Gene: {gn}", fontsize=16)
                    axes[0, 0].imshow(img1)
                    axes[0, 0].axis("off")
                    axes[0, 0].set_title("Visium HD Imputed")
                    axes[0, 1].imshow(img2)
                    axes[0, 1].axis("off")
                    axes[0, 1].set_title("Visium HD Observed")
                    axes[1, 0].axis("off") 
                    axes[1, 1].imshow(img4)
                    axes[1, 1].axis("off")
                    axes[1, 1].set_title("Xenium Observed")
                    out_dir = f"{output_dir}/ROI_hires/{x}_{y}_{hx}_{hy}/{key}/matching_xenium/"
                    os.makedirs(out_dir, exist_ok=True)
                else:
                    fig, axes =  plt.subplots(1, 2, figsize=(10, 10))
                    fig.suptitle(f"Gene: {gn}", fontsize=16)
                    axes[0].imshow(img1)
                    axes[0].axis("off")
                    axes[0].set_title("Visium HD Imputed")
                    axes[1].imshow(img2)
                    axes[1].axis("off")
                    axes[1].set_title("Visium HD Observed")
                    out_dir = f"{output_dir}/ROI_hires/{x}_{y}_{hx}_{hy}/{key}/"       
                
            plt.savefig(f"{out_dir}/{gn}.jpg", dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out_dir}/{gn}.jpg")   
        
if __name__ == '__main__':
    main()