# Pixel2Gene unites enhancement and inference in spatial transcriptomics from histology
 
Pixel2Gene is  a deep learning framework that enhances existing spatial gene expression measurements at single-cell resolution while inferring expression de novo in uncharacterized regions, leveraging co-registered histology images to enable comprehensive, histology-guided spatial profiling.

## 📂 Required Input Files

In order to apply the Pixel2Gene platform, you will need the following files:

### **Training Folder** (with training prefix `${train_pref}`):
1. `${train_pref}cnts.tsv` or `${train_pref}cnts.parquet`  
   Contains gene expression information, with columns representing genes and rows representing bins/cells.
2. `${train_pref}locs-raw.tsv` or `${train_pref}locs-raw.parquet`  
   Contains bin/cell location information in the original H&E pixel space after registration, with columns `x` and `y`.
3. `${train_pref}he-raw.jpg` or `${train_pref}he-raw.tif`  
   Original H&E image.
4. `${train_pref}radius-raw.txt`  
   Single value: how many microns one bin/cell spans.
5. `${train_pref}pixel-size-raw.txt`  
   Single value: microns per pixel in the original H&E space.
6. `${train_pref}pixel-size.txt`  
   Single value: desired microns per pixel after rescaling. Default is `0.5` so that 1 pixel ≈ 0.5 microns.

---

### **Testing Folder** (with testing prefix `${test_pref}`):
1. `${test_pref}he-raw.jpg` or `${test_pref}he-raw.tif`  
   Original H&E image.
2. `${test_pref}radius-raw.txt`  
   Single value: how many microns one bin/cell spans.
3. `${test_pref}pixel-size-raw.txt`  
   Single value: microns per pixel in the original H&E space.
4. `${test_pref}pixel-size.txt`  
   Single value: desired microns per pixel after rescaling. Default is `0.5`.

---

## ⚙️ Setup

First, activate the environment containing the required packages, if any (suppose the enviroment is named Pixel2Gene):

```bash
conda activate Pixel2Gene
```


## 🚀 Usage

1️⃣ Data Preprocessing (including rescaling) & Training

Edit train_demo.sh — update the variables in lines 10–15 to match your dataset, then run:

```bash 
bash train_demo.sh
```

2️⃣ Prediction

Edit predict_demo.sh — update the variables in lines 10–16, then run:
```bash
bash predict_demo.sh
```
The results will be stored in the `${output_predict}/cnts-super/` specified in `predict_demo.sh` as .pickle file for each gene.

