# Pixel2Gene unites enhancement and inference in spatial transcriptomics from histology
 
Pixel2Gene is  a deep learning framework that enhances existing spatial gene expression measurements at single-cell resolution while inferring expression de novo in uncharacterized regions, leveraging co-registered histology images to enable comprehensive, histology-guided spatial profiling.

## 📂 Required Input Files

In order to apply the Pixel2Gene platform, you will need the following files:

### **Training Folder** (prefix: `${train_pref}`)

1. `${train_pref}cnts.tsv` or `${train_pref}cnts.parquet`  
   Gene expression matrix with genes as columns and bins/cells as rows.
2. `${train_pref}locs-raw.tsv` or `${train_pref}locs-raw.parquet`  
   Spatial coordinates of bins/cells in the original H&E pixel space after registration, with columns `x` and `y`.
3. `${train_pref}he-raw.jpg` or `${`train_pref}he-raw.tif`  
   Original H&E image.
4. `${train_pref}radius-raw.txt`  
   Single value specifying the physical diameter (in microns) of each bin or cell.
5. `${train_pref}pixel-size-raw.txt`  
   Microns per pixel in the original H&E image.
6. `${train_pref}pixel-size.txt`  
   Desired microns per pixel after rescaling (applied consistently to both image and coordinates).  
   Default: `0.5` (i.e., 1 pixel ≈ 0.5 µm)

---

### **Testing Folder** (prefix: `${test_pref}`)

1. `${test_pref}he-raw.jpg` or `${test_pref}he-raw.tif`  
   Original H&E image.
2. `${test_pref}radius-raw.txt`  
   Physical diameter (in microns) of each bin or cell.
3. `${test_pref}pixel-size-raw.txt`  
   Microns per pixel in the original H&E image.
4. `${test_pref}pixel-size.txt`  
   Desired microns per pixel after rescaling (default: `0.5`).

---

## ⚠️ Generating Input Files for Visium HD and Xenium Data

This section describes how to generate Pixel2Gene-compatible inputs from Visium HD and Xenium datasets.

---

### Visium HD

1. Prepare the following files:
   - Binned Visium HD outputs stored in `binned_outputs/`
   - Raw H&E image stored as `extras/he-raw.btf`  
     (also supports `.jpg`, `.tif`, `.ome.tif`)

2. Run the formatting script:
   ```bash
   python format_visiumHD/format_visiumHD_data.py
   ```
   
3. The final formatted dataset should contain:
	- `cnts.parquet`
	- `locs-raw.tsv`
	- `he-raw.jpg`
    - `radius-raw.txt`
	- `pixel-size-raw.txt`
	- `pixel-size.txt`

---

### Xenium
Xenium data require explicit registration between morphology/DAPI images and H&E.
1. Prepare anchor point coordinates in `extras/`, for example:
   - `extras/morphology#2_keypoints.txt`
   - `extras/he#2_tiff_keypoints.txt`  
   - also need the pyramid level of anchor point coordinates for both `.ome.tif` images
2. Provide a morphology image, one of:
   -	`morphology_mip.ome.tif`
   -	`morphology.ome.tif`
   -	`morphology_focus/morphology_focus_0000.ome.tif`
3. Run the formatting script:
   ```bash
   python format_xenium/keypoints_to_homograph.py
   ```
   This generates 
   - `extras/stage_to_morph.txt`
   - `extras/morphology_to_he.txt`
4. Prepare the following files:
    - `extras/he-raw.ome.tif`
    - `transcripts.parquet` or `transcripts.csv.gz`
5. Map transcript coordinates into H&E pixel space:
   ```bash
   python format_xenium/coord_he_alignment.py
   ```
   This generates
   - `transcript_he_pixel.parquet`
6. Generate binned transcript:
   ```bash
   python format_xenium/bin_superpixel.py
   ```
   This generates super-pixel level (16px x 16px, after rescaling to 0.5µm/pixel) binned data 
   - `transcript_cnts_xxx_inimage.parquet`
7. Generate format outputs:
   ```bash
   python format_xenium/format_xenium_binned_data.py
   ```
8. The final formatted dataset should contain:
	- `cnts.parquet`
	- `locs-raw.tsv`
	- `he-raw.jpg`
    - `radius-raw.txt`
	- `pixel-size-raw.txt`
	- `pixel-size.txt`




---

## ⚙️ Setup

We recommend installing Pixel2Gene in a dedicated Conda environment. 
```bash
conda env create -f environment.core.yml
conda activate pixel2gene
bash install_pixel2gene.sh
```
Then download Pixel2Gene scripts:
```bash
git clone https://github.com/yaosicong1999/Pixel2Gene.git
```
or download the .zip from GitHub website. Then just:
```bash
cd Pixel2Gene/scripts
```
---

## 🚀 Usage

1️⃣ Data Preprocessing & Training

Update the variables in `demo_train.sh` (lines 2–17) to match your dataset. Suppose you are on an interactive GPU node, then you can directly use:
```bash 
bash demo_train.sh
```

**Estimated runtime:**
For the provided demo configuration (600 training epochs), training typically takes approximately 3 hours on a single high-end GPU (e.g., NVIDIA A100 or equivalent). Runtime may vary depending on dataset size and hardware specifications.

2️⃣ Prediction

Update the variables in `demo_predict.sh` (lines 2–18) to match your dataset. Suppose you are on an interactive GPU node, then you can directly use:
```bash
bash demo_predict.sh
```
The predicted gene expression will be saved to ` ${output_predict}/cnts-super/` (as specified in `demo_predict.sh`), with one `.pickle` file per gene.

**Estimated runtime:**
For the provided demo configuration, prediction typically takes approximately 10 minutes on a single high-end GPU (e.g., NVIDIA A100 or equivalent). Runtime may vary depending on dataset size and hardware specifications.

3️⃣ Visualization

To visualize predicted or observed gene expression (examples):
```bash
pref="../data/xenium/CRC-P2/CRC-P2-" ## change your pref here 
output="../data/xenium/CRC-P2_self_predict_hipt_raw/filter_he_qc/" ## change your folder contains  ${output_predict}/cnts-super/ here 
mask=${pref}"mask-small-hs.png" 

python plot_truth.py --pref=${pref} --output=${output} --overlay
python plot_truth.py --pref=${pref} --output=${output} --mask=${mask}  --overlay
python plot_imputed.py --pref=${pref} --output=${output} --n_top=100  --mask=${mask} --overlay
python plot_imputed.py --pref=${pref} --output=${output} --mask=${mask} --overlay
```