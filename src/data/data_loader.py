import os

import torch
from torch.utils.data import DataLoader, random_split
from .dataset import TTVid, TTData

def load_data(data_path, batch_size, src_fps, target_fps, labeled_start, window_size, seed, validation, shuffle=True, transforms=[], num_workers=0):
    # TODO: use all data once it is finished
    tst_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if 'test' in d]
    trn_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if 'train' in d and 'train_2' not in d]

    trn_vids = [TTVid(d, src_fps=src_fps, target_fps=target_fps, labeled_start=labeled_start, window_size=window_size) for d in trn_dirs]
    tst_vids = [TTVid(d, src_fps=src_fps, target_fps=target_fps, labeled_start=labeled_start, window_size=window_size) for d in tst_dirs]

    trn_ds = TTData(trn_vids, window_size, transforms=transforms)
    tst_ds = TTData(tst_vids, window_size, transforms=transforms)
    trn_ds, val_ds = random_split(trn_ds, [1-validation, validation], generator=torch.Generator().manual_seed(seed))

    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trn_loader, val_loader, tst_loader
