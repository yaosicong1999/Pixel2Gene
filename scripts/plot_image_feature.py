import pickle
from my_utils import load_pickle, plot_super, img_reduce
from utils import load_mask, load_image
import numpy as np
import os

prefix = '../visiumhd_heg/CRC-P2/CRC-P2-'
output = '../visiumhd_heg/CRC-P2/'
embs = load_pickle(f'{prefix}embeddings-hipt-raw.pickle')
embs = np.concatenate([embs['cls'], embs['sub'], embs['rgb']])
embs = embs.transpose(1, 2, 0)

os.makedirs(output + 'plot_hipt_features', exist_ok=True)
for i in range(embs.shape[2]):
    plot_super(embs[:, :, i]/np.nanmax(embs[:, :, i]), outfile=f'{output}plot_hipt_features/feature_{i}.png', save=True)

print('Plotting done.')


mask = load_mask(f'{prefix}mask-small-filter_he_qc.png')
embs[~mask,:] = np.nan
os.makedirs(output + 'plot_hipt_features_masked', exist_ok=True)
for i in range(embs.shape[2]):
    plot_super(embs[:, :, i]/np.nanmax(embs[:, :, i]), outfile=f'{output}plot_hipt_features_masked/feature_{i}.png', save=True)

print('Plotting done.')


from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
def resize_image(input_image, new_width):
    width_percent = (new_width / float(input_image.shape[1]))
    new_height = int((float(input_image.shape[0]) * float(width_percent)))
    resized_image = Image.fromarray(input_image, 'RGB').resize((new_width, new_height), Image.LANCZOS)
    return resized_image
he = load_image(f'{prefix}he.jpg')
he_min = resize_image(he, 848)
he_min.save(f'{prefix}he-min.jpg')


he_select = he[(20*256):(21*256), (20*256):(21*256), :]
Image.fromarray(he_select, 'RGB').save(f'{prefix}he_select.jpg')