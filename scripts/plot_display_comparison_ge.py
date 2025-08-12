import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-pref', type=str, default=None)
    parser.add_argument('--output-pref', type=str, default=None)
    parser.add_argument('--overlay', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    data_pref = args.data_pref
    output_pref = args.output_pref
    overlay = args.overlay

    for set in ['top_25', 'middle_50', 'bottom_25']:
        with open(f"{data_pref}gene-names-{set}.txt", "r") as f:
            genes = [line.strip() for line in f] 
        if overlay:
            out_dir = f"{output_pref}comparison-overlay/{set}/"
        else:
            out_dir = f"{output_pref}comparison/{set}/"
        os.makedirs(out_dir, exist_ok=True)
        for gene in genes:
            if overlay:
                img1_path = f"{output_pref}cnts-super-plots2-overlay/{set}/{gene}.png"
                img2_path = f"{output_pref}cnts-truth-plots2-overlay/{set}/{gene}.png"
            else:
                img1_path = f"{output_pref}cnts-super-plots2/{set}/{gene}.png"
                img2_path = f"{output_pref}cnts-truth-plots2/{set}/{gene}.png"
            output_path = f"{out_dir}/{gene}.jpg"
            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                print(f"Skipping {gene}: missing image file. {img1_path} or {img2_path} not found.")
                continue
            try:
                img1 = Image.open(img1_path)
                img2 = Image.open(img2_path)
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(f"Gene: $\\it{{{gene}}}$", fontsize=16)
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
            except Exception as e:
                print(f"Error processing {gene}: {e}")

if __name__ == '__main__':
    main()