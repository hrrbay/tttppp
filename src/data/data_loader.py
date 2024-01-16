import os
import pdb

import torchvision
import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset

from .dataset import TTVid, TTData

import random

def custom_random_split(dataset, val_split=0.1, seed=0):  # only split on sequence level (not on window level)
    sequence_wins_indices = [] # tuple of start and end idx of sequence wins
    for vid in dataset.vids:
        for i in range(len(vid.sequences)):
            if len(sequence_wins_indices) == 0:
                sequence_wins_indices.append((0, vid.wins_per_seq[i]))
            else:
                start = sequence_wins_indices[-1][1]
                sequence_wins_indices.append((start, start + vid.wins_per_seq[i]))

    random.seed(seed)
    random.shuffle(sequence_wins_indices)

    total_wins = sum([end - start for start, end in sequence_wins_indices])
    val_size = int(total_wins * val_split)
    assert total_wins == len(dataset)

    split_idx = 0
    current_size = 0
    for i in range(len(sequence_wins_indices)):
        current_size += sequence_wins_indices[i][1] - sequence_wins_indices[i][0]
        if current_size >= val_size:
            split_idx = i
            break

    val_wins = sequence_wins_indices[:split_idx]
    train_wins = sequence_wins_indices[split_idx:]

    val_indices = []
    for start, end in val_wins:
        val_indices.extend(list(range(start, end)))

    train_indices = []
    for start, end in train_wins:
        train_indices.extend(list(range(start, end)))

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    return train_ds, val_ds


def load_data(data_path, batch_size, src_fps, target_fps, labeled_start, window_size, seed, validation, fixed_seq_len, shuffle=False, transforms=[], num_workers=0):
    # TODO: use all data once it is finished
    tst_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if 'test' in d]
    trn_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if 'train' in d]

    # trn_vids = [TTVid(d, src_fps=src_fps, target_fps=target_fps, labeled_start=labeled_start, window_size=window_size) for d in trn_dirs if 'train_5' in d]
    # tst_vids = [TTVid(d, src_fps=src_fps, target_fps=target_fps, labeled_start=labeled_start, window_size=window_size) for d in tst_dirs if 'test_3' in d]
    trn_vids = [TTVid(d, src_fps=src_fps, target_fps=target_fps, labeled_start=labeled_start, window_size=window_size, fixed_seq_len=fixed_seq_len) for d in trn_dirs]
    tst_vids = [TTVid(d, src_fps=src_fps, target_fps=target_fps, labeled_start=labeled_start, window_size=window_size, fixed_seq_len=fixed_seq_len) for d in tst_dirs]

    trn_ds = TTData(trn_vids, window_size, transforms=transforms)
    tst_ds = TTData(tst_vids, window_size, transforms=transforms)
    val_loader = None
    if validation > 0:
        assert validation <= 1
        #trn_ds, val_ds = random_split(trn_ds, [1-validation, validation], generator=torch.Generator().manual_seed(seed))
        trn_ds, val_ds = custom_random_split(trn_ds, val_split=validation, seed=seed)
        print(f'Loaded {len(trn_ds)} training samples and {len(val_ds)} validation samples.')
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return trn_loader, val_loader, tst_loader

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