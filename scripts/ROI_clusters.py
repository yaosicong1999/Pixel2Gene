import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
import numpy as np
from utils import load_image
from my_utils import img_reduce

def get_all_files(base_dir):
    """ Recursively get all .jpg and .png files under base_dir and return a dictionary with relative paths. """
    file_dict = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(('labels_overlayed.png', 'labels.png')):
                rel_path = os.path.relpath(os.path.join(root, file), base_dir)
                file_dict[rel_path] = os.path.join(root, file)  # Store full path
    return file_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pref', type=str, default=None)
    parser.add_argument('--pref1', type=str, default=None)
    parser.add_argument('--pref2', type=str, default=None)
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
    
    os.makedirs(output_dir, exist_ok=True)        
    if not os.path.exists(apple_dir) or not os.path.exists(banana_dir):
        print("One or both base directories do not exist.")
        exit()

    apple_files = get_all_files(apple_dir)
    banana_files = get_all_files(banana_dir)
    common_files = set(apple_files.keys()) & set(banana_files.keys())
    print(apple_files)
    print(banana_files)

    if not drophe:
        he = load_image(f"{data_pref}he.jpg")
        he = img_reduce(he, factor = 16)
        if isinstance(he, np.ndarray):
            he = Image.fromarray(he) 
        if hx is not None and hy is not None and hx > 0 and hy > 0: 
            he = he.crop((x, y, x + hx, y + hy))
    
    for rel_path in common_files:
        img1_path = apple_files[rel_path]
        img2_path = banana_files[rel_path]
        output_path = os.path.join(output_dir + str(x) +'_' + str(y) + '_' + str(hx) + '_' + str(hy) + '/', rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  
        try:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            if hx !=0 and hy !=0:       
                if hx is not None and hy is not None and hx > 0 and hy > 0:          
                    print("cropping....")
                    img1 = img1.crop((x, y, x + hx, y + hy))
                    img2 = img2.crop((x, y, x + hx, y + hy))
                else:
                    print("none value detected, not cropping.... keeping the original size...")
                if not drophe:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
                    fig.suptitle(f"{os.path.splitext(os.path.basename(rel_path))[0]}$", fontsize=16)   
                    axes[0].imshow(img1)
                    axes[0].axis("off")
                    axes[0].set_title("Imputed")
                    axes[1].imshow(img2)
                    axes[1].axis("off")
                    axes[1].set_title("Observed")
                    axes[2].imshow(he)
                    axes[2].axis("off")
                    axes[2].set_title("H&E")
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    fig.suptitle(f"Clusters: $\\it{{{os.path.splitext(os.path.basename(rel_path))[0]}}}$", fontsize=16)   
                    axes[0].imshow(img1)
                    axes[0].axis("off")
                    axes[0].set_title("Imputed")
                    axes[1].imshow(img2)
                    axes[1].axis("off")
                    axes[1].set_title("Observed")
                plt.tight_layout()
                plt.savefig(output_path, dpi=200, bbox_inches="tight")
                plt.close()
                print(f"Saved: {output_path}")
            else:
                print(f"Error parameters of hx and hy for cropping {rel_path}")
        except Exception as e:
            print(f"Error processing {rel_path}: {e}")

if __name__ == '__main__':
    main()