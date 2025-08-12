import argparse
import multiprocessing
import os.path
import torch
torch.set_float32_matmul_precision('medium')
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import numpy as np
from tqdm import tqdm
from impute_by_basic import get_gene_counts, get_embeddings, get_locs
from utils import read_lines, read_string, save_pickle, write_lines, load_mask
from image import get_disk_mask
import pickle
import shutil
import zarr


class FeedForward(nn.Module):

    def __init__(
            self, n_inp, n_out,
            activation=None, residual=False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        if activation is None:
            activation = nn.LeakyReLU(0.1, inplace=True)
        self.activation = activation
        self.residual = residual

    def forward(self, x, indices=None):
        if indices is None:
            y = self.linear(x)
        else:
            weight = self.linear.weight[indices]
            bias = self.linear.bias[indices]
            y = nn.functional.linear(x, weight, bias)
        y = self.activation(y)
        if self.residual:
            y = y + x
        return y


class ELU(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta


class ForwardSumModel(pl.LightningModule):

    def __init__(self, lr, n_inp, n_out, n_ref, true_covet, nn_idx_mat):
        super().__init__()
        self.true_covet = true_covet
        self.pred_covet = None
        self.nn_idx_mat = nn_idx_mat

        self.lr = lr
        self.n_ref = n_ref
        self.net_lat = nn.Sequential(
                FeedForward(n_inp, 256),
                FeedForward(256, 256),
                FeedForward(256, 256),
                FeedForward(256, 256))
        self.net_out = FeedForward(
                256, n_out,
                activation=ELU(alpha=0.01, beta=0.01))
        self.save_hyperparameters()

    def inp_to_lat(self, x):
        return self.net_lat.forward(x)

    def lat_to_out(self, x, indices=None):
        x = self.net_out.forward(x, indices)
        return x

    def forward(self, x, indices=None):
        x = self.inp_to_lat(x)
        x = self.lat_to_out(x, indices)
        return x

    def training_step(self, batch):
        x, y_mean = batch
        y_pred = self.forward(x)
        y_mean_pred = y_pred.mean(-2)
        mse = ((y_mean_pred - y_mean)**2).mean()
        loss = mse
        self.log('rmse', mse**0.5, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        # optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': scheduler
        # }
        return optimizer

    def load_reference_weights_old(self, state_dict):
        assert self.n_ref == state_dict['net_out.linear.weight'].shape[0]
        rep_weight = self.net_out.linear.weight.clone()
        rep_bias = self.net_out.linear.bias.clone()
        rep_weight[:self.n_ref, ] = state_dict['net_out.linear.weight']
        rep_bias[:self.n_ref, ] = state_dict['net_out.linear.bias']
        rep_dict = state_dict.copy()
        rep_dict['net_out.linear.weight'] = rep_weight
        rep_dict['net_out.linear.bias'] = rep_bias
        self.load_state_dict(rep_dict, strict=False)
        assert (self.net_out.linear.weight[:self.n_ref,] == state_dict['net_out.linear.weight']).all()
        assert (self.net_out.linear.bias[:self.n_ref,] == state_dict['net_out.linear.bias']).all()

    def load_reference_weights(self, state_dict):
        # Replace weights for all layers before the last layer
        for name, param in self.named_parameters():
            if 'net_out.linear' not in name:  # Skip last layer
                assert param.shape == state_dict[name].shape, f"Shape mismatch for {name}"
                param.data.copy_(state_dict[name])
        # Handle the last layer separately (only update the first n_ref rows)
        assert self.n_ref == state_dict['net_out.linear.weight'].shape[0]
        self.net_out.linear.weight.data[:self.n_ref].copy_(state_dict['net_out.linear.weight'])
        self.net_out.linear.bias.data[:self.n_ref].copy_(state_dict['net_out.linear.bias'])
        # Verify last layer update
        assert (self.net_out.linear.weight[:self.n_ref] == state_dict['net_out.linear.weight']).all().item()
        assert (self.net_out.linear.bias[:self.n_ref] == state_dict['net_out.linear.bias']).all().item()

class SpotDataset(Dataset):

    def __init__(self, x_all, y, locs, radius):
        super().__init__()
        print("In SpotDataset: embs_train's shape is", x_all.shape)
        print("In SpotDataset: cnts_train's shape is", y.shape)
        print("In SpotDataset: locs_train's shape is", locs.shape)

        mask = get_disk_mask(radius)
        x = get_patches_flat(x_all, locs, mask)
        isin = np.isfinite(x).all((-1, -2)).flatten()

        x = x[isin]
        y = y[isin]
        locs = locs[isin]
        if x.shape.__len__() > 3:
            x = x.reshape([x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])

        self.x = x
        self.y = y
        self.locs = locs
        self.radius = radius
        self.mask = mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]
        if mask.all():
            x = patch
        else:
            x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list

def get_predict_data(pref_predict, emb_type="uni"):
    embs_predict = get_embeddings(f'{pref_predict}', emb_type=emb_type)
    embs_predict = embs_predict.astype(np.float32)
    return embs_predict


def normalize(embs, cnts):
    embs = embs.copy()
    cnts = cnts.copy()

    # TODO: check if adjsut_weights in extract_features can be skipped
    embs_mean = np.nanmean(embs, (0, 1))
    embs_std = np.nanstd(embs, (0, 1))
    embs -= embs_mean
    embs /= embs_std + 1e-12

    cnts_min = cnts.min(0)
    cnts_max = cnts.max(0)
    cnts -= cnts_min
    cnts /= (cnts_max - cnts_min) + 1e-12

    return embs, cnts, (embs_mean, embs_std), (cnts_min, cnts_max)


def predict_single_out(model, z, indices, names, y_range):
    z = torch.tensor(z, device=model.device)
    y = model.lat_to_out(z, indices=indices)
    y = y.cpu().detach().numpy()
    # y[y < 0.01] = 0.0
    # y[y > 1.0] = 1.0
    y *= y_range[:, 1] - y_range[:, 0]
    y += y_range[:, 0]
    return y


def predict_single_lat(model, x):
    x = torch.tensor(x, dtype=torch.float32, device=model.device)
    z = model.inp_to_lat(x)
    z = z.cpu().detach().numpy()
    return z


def predict(
        model_states, x_batches, name_list, y_range, output,
        device='cuda'):

    # states: different initial values for training
    # batches: subsets of observations
    # groups: subsets outcomes

    batch_size_outcome = 100

    model_states = [mod.to(device) for mod in model_states]

    # get features of second last layer
    z_states_batches = [
            [predict_single_lat(mod, x_bat) for mod in model_states]
            for x_bat in x_batches]
    z_point = np.concatenate([
        np.median(z_states, 0)
        for z_states in z_states_batches])
    z_dict = dict(cls=z_point.transpose(2, 0, 1))
    # save_pickle(
    #         z_dict,
    #         output+'embeddings-gene.pickle')
    del z_point

    # zarr configuration
    output_zarr_path = f'{output}/cnts-super.zarr'
    print(f"Output Zarr path: {output_zarr_path}")
    batch_size_outcome = 100
    dtype = np.float16 
    H = sum(z_states[0].shape[0] for z_states in z_states_batches)
    W = z_states_batches[0][0].shape[1]
    D = np.min([len(name_list), 3000]) # Limit to 3000 outcomes for memory efficiency
    full_shape = (H, W, D)
    print(f"Full shape for Zarr store: {full_shape}")
    chunk_shape = (256, 256, 10)    # Safe chunking to avoid Blosc >2GB
    print(f"Chunk shape for Zarr store: {chunk_shape}")

    # === Create Zarr store (no chunk shape specified) ===
    z_arr = zarr.open(
        output_zarr_path,
        mode='w',
        shape=full_shape,
        dtype=dtype,
        chunks=chunk_shape,  # <-- Let Zarr determine default chunking
        compressor=zarr.Blosc()
    )

    sub_batch_size = 10  # Adjust based on available memory and file size constraints

    # predict and save y by batches in outcome dimension
    if H * W >= 1500*1500:
        idx_list = np.arange(np.min([len(name_list), 3000]))
        write_lines(name_list[:np.min([len(name_list), 3000])], f'{output}predict-gene-names.txt')
    else:
        idx_list = np.arange(len(name_list))
        write_lines(name_list, f'{output}predict-gene-names.txt')
    n_groups_outcome = len(idx_list) // batch_size_outcome + 1
    
    
    idx_groups = np.array_split(idx_list, n_groups_outcome)
    print(f"Number of groups for outcomes: {len(idx_groups)}")
    for idx_grp in tqdm(idx_groups, desc="Predicting & Saving"):
        name_grp = name_list[idx_grp]
        y_ran = y_range[idx_grp]
        y_grp = np.concatenate([
            np.median([
                predict_single_out(mod, z, idx_grp, name_grp, y_ran)
                for mod, z in zip(model_states, z_states)], 0)
            for z_states in z_states_batches])        
        for i, name in enumerate(name_grp):
                save_pickle(y_grp[..., i].astype(np.float16), f'{output}cnts-super/{name}.pickle')
            # print("Saving y_grp to Zarr store...")
            # for sub_start in range(0, y_grp.shape[2], sub_batch_size):
            #     sub_end = min(sub_start + sub_batch_size, y_grp.shape[2])
            #     sub_slice = slice(sub_start, sub_end)
            #     global_slice = slice(idx_grp[0] + sub_start, idx_grp[0] + sub_end)
            #     z_arr[:, :, global_slice] = y_grp[:, :, sub_slice].astype(dtype)
            # print("Zarr store written and flushed.")

def predict_from_saved(embs_predict, train_output, predict_output, device='cuda', asreference=False):
    if not asreference:
        print("asreference is False, loading model states and metadata...") 
        meta = np.load(f"{train_output}/meta.npz", allow_pickle=True)
        with open(f"{train_output}/model_meta.pkl", "rb") as f:
            model_meta = pickle.load(f)

        names = meta['names']
        cnts_train_min = meta['cnts_min']
        cnts_train_max = meta['cnts_max']
        mask_size = meta['mask_size']
        cnts_train_range = np.stack([cnts_train_min, cnts_train_max], -1) / mask_size

        model_class_name = model_meta['model_class']
        model_kwargs = model_meta['model_kwargs']

        # Get class object from name
        if model_class_name == 'ForwardSumModel':
            model_class = ForwardSumModel  # You must import this

        model_list = []
        i = 0
        while True:
            model_path = f"{train_output}states/{i:02d}/model.pt"
            if not os.path.exists(model_path):
                break
            model = model_class.load_from_checkpoint(model_path, **model_kwargs)
            model.eval()
            model_list.append(model)
            i += 1

        batch_size_row = 50
        n_batches_row = embs_predict.shape[0] // batch_size_row + 1
        embs_predict_batches = np.array_split(embs_predict, n_batches_row)

        predict(
            model_states=model_list, x_batches=embs_predict_batches,
            name_list=names, y_range=cnts_train_range,
            output=predict_output, device=device)

        print("Prediction complete.")
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pref_predict', type=str)
    parser.add_argument('--output_train', type=str)
    parser.add_argument('--output_predict', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--emb_type', type=str, default='uni')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    pref_predict = args.pref_predict
    emb_type = args.emb_type

    """loading to-be-predicted embs"""
    embs_predict = get_predict_data(pref_predict=pref_predict, emb_type=emb_type)
    print("embs_predict's shape is", embs_predict.shape)
    
    os.makedirs(args.output_predict, exist_ok=True)
    
    predict_from_saved(
        embs_predict=embs_predict, 
        train_output=args.output_train, predict_output=args.output_predict, 
        device=args.device, asreference=False)


if __name__ == '__main__':
    main()