import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ssim = pd.read_csv("/Users/sicongy/results/new/S1R1_vs_S1R2/superpixel_ssim_comparison.tsv", sep="\t", index_col=0)
S1R2_cnts = pd.read_csv("/Users/sicongy/results/new/S1R1_vs_S1R2/test-cnts.tsv", sep="\t", index_col=0)
ssim.index = ssim['Gene']
ssim['avg_expr'] = np.mean(S1R2_cnts[ssim['Gene']], axis=0).astype(np.float32)

fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
ax.scatter(x=ssim['avg_expr'], y=ssim['SSIM-Rel'], s=1, cmap='turbo')
for i in range(ssim.shape[0]):
    if (ssim['avg_expr'][i]>0.3) and (ssim['SSIM-Rel'][i]>0.75):
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-2, -3), ha='center',
                    fontsize=6, color='red')
for i in range(ssim.shape[0]):
    if ssim['SSIM-Rel'][i]>0.95:
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-5, -3), ha='center',
                    fontsize=6, color='green')
for i in range(ssim.shape[0]):
    if ssim['SSIM-Rel'][i]<0.65:
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-2, -3), ha='center',
                    fontsize=6, color='blue')
ax.plot([0, 1], [0, 1], color='red', linestyle='--')
ax.set_xlim(0, np.max(ssim['avg_expr']))
ax.set_ylim(0, 1)
ax.set_xlabel('Mean Gene Expression')
ax.set_ylabel('SSIM')
plt.show()


ssim = pd.read_csv("/Users/sicongy/results/new/S1R1_vs_S2/superpixel_ssim_comparison.tsv", sep="\t", index_col=0)
S2_cnts = pd.read_csv("/Users/sicongy/results/new/S1R1_vs_S2/test-cnts.tsv", sep="\t", index_col=0)
ssim.index = ssim['Gene']
ssim['avg_expr'] = np.mean(S2_cnts[ssim['Gene']], axis=0).astype(np.float32)

fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
ax.scatter(x=ssim['avg_expr'], y=ssim['SSIM-Rel'], s=1, cmap='turbo')
for i in range(ssim.shape[0]):
    if (ssim['avg_expr'][i]>0.2) and (ssim['SSIM-Rel'][i]>0.75):
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-2, -3), ha='center',
                    fontsize=6, color='red')
for i in range(ssim.shape[0]):
    if ssim['SSIM-Rel'][i]>0.95:
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-5, -3), ha='center',
                    fontsize=6, color='green')
for i in range(ssim.shape[0]):
    if ssim['SSIM-Rel'][i]<0.65:
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-2, -3), ha='center',
                    fontsize=6, color='blue')
ax.plot([0, 1], [0, 1], color='red', linestyle='--')
ax.set_xlim(0, np.max(ssim['avg_expr']))
ax.set_ylim(0, 1)
ax.set_xlabel('Mean Gene Expression')
ax.set_ylabel('SSIM')
plt.show()




ssim = pd.read_csv("/Users/sicongy/results/new/CRC_P1_split/superpixel_ssim_comparison.tsv", sep="\t", index_col=0)
P1_cnts = pd.read_csv("/Users/sicongy/results/new/CRC_P1_split/test-cnts.tsv", sep="\t", index_col=0)
ssim.index = ssim['Gene']
ssim['avg_expr'] = np.mean(P1_cnts[ssim['Gene']], axis=0).astype(np.float32)

fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
ax.scatter(x=ssim['avg_expr'], y=ssim['SSIM-Rel'], s=1, cmap='turbo')
for i in range(ssim.shape[0]):
    if (ssim['avg_expr'][i]>0.2) and (ssim['SSIM-Rel'][i]>0.75):
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-2, -3), ha='center',
                    fontsize=6, color='red')
for i in range(ssim.shape[0]):
    if ssim['SSIM-Rel'][i]>0.95:
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-5, -3), ha='center',
                    fontsize=6, color='green')
for i in range(ssim.shape[0]):
    if ssim['SSIM-Rel'][i]<0.65:
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-2, -3), ha='center',
                    fontsize=6, color='blue')
ax.plot([0, 1], [0, 1], color='red', linestyle='--')
ax.set_xlim(0, np.max(ssim['avg_expr']))
ax.set_ylim(0, 1)
ax.set_xlabel('Mean Gene Expression')
ax.set_ylabel('SSIM')
plt.show()

ssim = pd.read_csv("/Users/sicongy/results/new/CRC_P2_split/superpixel_ssim_comparison.tsv", sep="\t", index_col=0)
P2_cnts = pd.read_csv("/Users/sicongy/results/new/CRC_P2_split/test-cnts.tsv", sep="\t", index_col=0)
ssim.index = ssim['Gene']
ssim['avg_expr'] = np.mean(P2_cnts[ssim['Gene']], axis=0).astype(np.float32)

fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
ax.scatter(x=ssim['avg_expr'], y=ssim['SSIM-Rel'], s=1, cmap='turbo')
for i in range(ssim.shape[0]):
    if (ssim['avg_expr'][i]>0.2) and (ssim['SSIM-Rel'][i]>0.75):
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-2, -3), ha='center',
                    fontsize=6, color='red')
for i in range(ssim.shape[0]):
    if ssim['SSIM-Rel'][i]>0.95:
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-5, -3), ha='center',
                    fontsize=6, color='green')
for i in range(ssim.shape[0]):
    if ssim['SSIM-Rel'][i]<0.65:
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-2, -3), ha='center',
                    fontsize=6, color='blue')
ax.plot([0, 1], [0, 1], color='red', linestyle='--')
ax.set_xlim(0, np.max(ssim['avg_expr']))
ax.set_ylim(0, 1)
ax.set_xlabel('Mean Gene Expression')
ax.set_ylabel('SSIM')
plt.show()



ssim = pd.read_csv("/Users/sicongy/results/new/CRC_P5_split/superpixel_ssim_comparison.tsv", sep="\t", index_col=0)
P5_cnts = pd.read_csv("/Users/sicongy/results/new/CRC_P5_split/test-cnts.tsv", sep="\t", index_col=0)
ssim.index = ssim['Gene']
ssim['avg_expr'] = np.mean(P5_cnts[ssim['Gene']], axis=0).astype(np.float32)

fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
ax.scatter(x=ssim['avg_expr'], y=ssim['SSIM-Rel'], s=1, cmap='turbo')
for i in range(ssim.shape[0]):
    if (ssim['avg_expr'][i]>0.2) and (ssim['SSIM-Rel'][i]>0.75):
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-2, -3), ha='center',
                    fontsize=6, color='red')
for i in range(ssim.shape[0]):
    if ssim['SSIM-Rel'][i]>0.95:
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-5, -3), ha='center',
                    fontsize=6, color='green')
for i in range(ssim.shape[0]):
    if ssim['SSIM-Rel'][i]<0.65:
        ax.annotate(ssim['Gene'][i], (ssim['avg_expr'][i], ssim['SSIM-Rel'][i]), textcoords="offset points", xytext=(-2, -3), ha='center',
                    fontsize=6, color='blue')
ax.plot([0, 1], [0, 1], color='red', linestyle='--')
ax.set_xlim(0, np.max(ssim['avg_expr']))
ax.set_ylim(0, 1)
ax.set_xlabel('Mean Gene Expression')
ax.set_ylabel('SSIM')
plt.show()


from utils import load_pickle, load_mask, load_image

prefix = "/Users/sicongy/results/32px/S1R1_vs_S2/"
hipt = load_pickle(prefix + "train-embeddings-hist.pickle")
for i in range(384):
    hipt_sub = hipt['sub'][i]
    mask = load_mask(prefix + "train-mask-small.png")
    hipt_sub[~mask] = np.nan
    plot_super(hipt_sub/np.nanmax(hipt_sub), outfile=prefix + "train_sub" + str(i) + ".jpg")

for i in range(192):
    hipt_cls = hipt['cls'][i]
    mask = load_mask(prefix + "train-mask-small.png")
    hipt_cls[~mask] = np.nan
    plot_super(hipt_cls/np.nanmax(hipt_cls), outfile=prefix + "train_cls" + str(i) + ".jpg")


he = load_image(prefix + "train-he.jpg")
factor = 16
from einops import reduce
he = reduce(
    he.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
    h=factor, w=factor).astype(np.uint8)
## he[~mask] = 0
save_image(he, prefix+"train-he-small.jpg")

def plot_super(x, outfile, truncate=None, he=None, locs=None):
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
    img[~mask] = 0
    save_image(img, outfile)


prefix = "/Users/sicongy/results/32px/visium/"
hipt = load_pickle(prefix + "embeddings-hist.pickle")
for i in range(384):
    hipt_sub = hipt['sub'][i]
    mask = load_mask(prefix + "mask-small.png")
    hipt_sub[~mask] = np.nan
    plot_super(hipt_sub/np.nanmax(hipt_sub), outfile=prefix + "train_sub" + str(i) + ".jpg")

for i in range(192):
    hipt_cls = hipt['cls'][i]
    mask = load_mask(prefix + "mask-small.png")
    hipt_cls[~mask] = np.nan
    plot_super(hipt_cls/np.nanmax(hipt_cls), outfile=prefix + "train_cls" + str(i) + ".jpg")

he = load_image(prefix + "he.jpg")
factor = 16
from einops import reduce
he = reduce(
    he.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
    h=factor, w=factor).astype(np.uint8)
he[~mask] = 0
save_image(he, prefix+"he-small.jpg")



prefix = "/Users/sicongy/results/32px/xenium_artery/"
hipt = load_pickle(prefix + "embeddings-hist.pickle")
for i in range(384):
    hipt_sub = hipt['sub'][i]
    mask = load_mask(prefix + "mask-small.png")
    hipt_sub[~mask] = np.nan
    plot_super(hipt_sub/np.nanmax(hipt_sub), outfile=prefix + "train_sub" + str(i) + ".jpg")

for i in range(192):
    hipt_cls = hipt['cls'][i]
    mask = load_mask(prefix + "mask-small.png")
    hipt_cls[~mask] = np.nan
    plot_super(hipt_cls/np.nanmax(hipt_cls), outfile=prefix + "train_cls" + str(i) + ".jpg")

he = load_image(prefix + "he.jpg")
factor = 16
from einops import reduce
he = reduce(
    he.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
    h=factor, w=factor).astype(np.uint8)
## he[~mask] = 0
save_image(he, prefix+"he-small.jpg")


# import cv2
# import numpy as np
# gray_image = cv2.imread(prefix + "mask-small.png", cv2.IMREAD_GRAYSCALE)
# mask = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
# factor = 16
# from einops import repeat
# mask = repeat(mask, 'h w c -> (h repeat_h) (w repeat_w) c', repeat_h=factor, repeat_w=factor)
# save_image(mask, prefix+"mask-full.png")
#
# print(enlarged_image.shape)  # The shape will be (6, 6, 3) for an enlarged image
#


prefix = "/Users/sicongy/results/32px/visiumHD_PIGR/"
he = load_image(prefix + "he.jpg")
factor = 16
from einops import reduce
he = reduce(
    he.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
    h=factor, w=factor).astype(np.uint8)
# mask = load_mask(prefix + "mask-small.png")
# he[~mask] = 0
save_image(he, prefix+"he-small.jpg")