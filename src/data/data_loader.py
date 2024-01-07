import os
import pdb

import torchvision
import torch
from torch.utils.data import DataLoader, random_split

from .dataset import TTVid, TTData

def load_data(data_path, batch_size, src_fps, target_fps, labeled_start, window_size, seed, validation, shuffle=False, transforms=[], num_workers=0):
    # TODO: use all data once it is finished
    tst_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if 'test' in d]
    trn_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if 'train' in d]

    trn_vids = [TTVid(d, src_fps=src_fps, target_fps=target_fps, labeled_start=labeled_start, window_size=window_size) for d in trn_dirs if 'train_5' in d]
    tst_vids = [TTVid(d, src_fps=src_fps, target_fps=target_fps, labeled_start=labeled_start, window_size=window_size) for d in tst_dirs if 'test_3' in d]

    trn_ds = TTData(trn_vids, window_size, transforms=transforms)
    tst_ds = TTData(tst_vids, window_size, transforms=transforms)
    # trn_ds, val_ds = random_split(trn_ds, [1-validation, validation], generator=torch.Generator().manual_seed(seed))

    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # num_zeros = len(val_loader.dataset)
    # for masks, labels in val_loader:
    #     num_zeros -= labels.sum().item()
    # num_ones = len(val_loader.dataset) - num_zeros
    
    return trn_loader, None, tst_loader

'''
    Custom transforms
'''
class T3P3Flip(object):
    def __init__(self, prob):
        assert type(prob) == float and prob <= 1
        self.prob = prob
        self.flip = torchvision.transforms.RandomHorizontalFlip(prob)
    
    def __call__(self, sample):
        pdb.set_trace()