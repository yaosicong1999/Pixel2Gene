import os
import sys
sys.path.append(os.path.abspath("HistoSweep"))
from utils import load_image
from saveParameters import saveParams
from computeMetrics import compute_metrics
from densityFiltering import compute_low_density_mask
from textureAnalysis import run_texture_analysis
from ratioFiltering import run_ratio_filtering
from generateMask import generate_final_mask
from additionalPlots import generate_additionalPlots
import shutil
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    pref = args.prefix
    
    img = load_image(pref + 'he.jpg')

    patch_size = 16
    density_thresh = 100 
    clean_background_flag = True # Set to False if you want to preserve fibrous regions that are otherwise being incorrectly filtered out
    min_size = 10 # Decrease if there are many fibrous areas (e.g. adipose) in the tissue that you wish to retain (e.g. 5), increase if lots of/larger debris you wish to remove (e.g.50)

    he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = compute_metrics(img, patch_size=patch_size)

    # identify low density superpixels
    mask1_lowdensity = compute_low_density_mask(z_v_image_, he_std_image_, ratio_norm_, density_thresh=density_thresh)
    print('Total selected for density filtering: ', mask1_lowdensity.sum())
    mask1_lowdensity_update = run_texture_analysis(prefix=pref, image=img, tissue_mask=mask1_lowdensity, patch_size=patch_size, glcm_levels=64)
    # identify low ratio superpixels
    mask2_lowratio, otsu_thresh = run_ratio_filtering(ratio_norm_, mask1_lowdensity_update)
    print('Total selected for ratio filtering: ', mask2_lowratio.sum())

    generate_final_mask(prefix=pref, he=img, 
                        mask1_updated = mask1_lowdensity_update, mask2 = mask2_lowratio, 
                        clean_background = clean_background_flag, 
                        super_pixel_size=patch_size, minSize = min_size)

    shutil.copy2(f"{pref}HistoSweep_Output/mask-small.png", f"{pref}mask-small-hs.png")

    output_folder = f"{pref}HistoSweep_Output"
    if os.path.exists(output_folder) and os.path.isdir(output_folder):
        shutil.rmtree(output_folder)


if __name__ == '__main__':
    main()



























