import os
import pdb
import random

from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .dataset import TTData, TTVid


def custom_random_split(dataset, val_split=0.1, seed=0):
    """
        Custom random split that ensures that sequences are not split across train/val
    """
    # tuple of start and end idx of sequence wins
    sequence_wins_indices = [] 
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

    # determine the number of sequences to put in the validation set
    split_idx = 0
    current_size = 0
    for i in range(len(sequence_wins_indices)):
        current_size += sequence_wins_indices[i][1] - sequence_wins_indices[i][0]
        if current_size >= val_size:
            split_idx = i
            break
    
    # split the sequence wins into train and validation
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


def load_data(data_path, batch_size, src_fps, target_fps, labeled_start, window_size, seed, validation,
              fixed_seq_len, flip_prob, validation_vid=None, shuffle=False, transforms=[], num_workers=0, use_poses=False):
    """
        Load data from data_path and return train, val and test dataloaders.
    """

    # define train and test videos
    tst_vids = ['test_1', 'test_3', 'test_4', 'test_6', 'test_7']
    trn_vids = ['train_1', 'train_2', 'train_3', 'train_4', 'train_5', 'test_2', 'test_5']

    tst_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if 'test' in d]
    tst_dirs = sorted(tst_dirs)
    val_dirs = []
    if validation_vid is not None:
        # if specific validation video is specified, use this one instead of random splitting
        val_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if 'train' in d and str(validation_vid) in d]
    trn_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if 'train' in d and d not in val_dirs]

    # Creates instances of TTVid for each video
    trn_vids = [TTVid(d, src_fps=src_fps, target_fps=target_fps, labeled_start=labeled_start, window_size=window_size, fixed_seq_len=fixed_seq_len, use_poses=use_poses) for d in trn_dirs]
    tst_vids = [TTVid(d, src_fps=src_fps, target_fps=target_fps, labeled_start=labeled_start, window_size=window_size, fixed_seq_len=fixed_seq_len, use_poses=use_poses) for d in tst_dirs]

    # Creates a custom dataset for the train and test videos
    trn_ds = TTData(trn_vids, window_size, transforms=transforms, flip_prob=flip_prob)
    tst_ds = TTData(tst_vids, window_size, transforms=transforms)
    
    # If validation videos are specified, create a custom dataset for them
    val_loader = None 
    if val_dirs:
        val_vids = [TTVid(d, src_fps=src_fps, target_fps=target_fps, labeled_start=labeled_start, window_size=window_size, fixed_seq_len=fixed_seq_len) for d in val_dirs]
        val_ds = TTData(val_vids, window_size, transforms=transforms)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f'{len(val_loader)=}')

    # If no validation videos are specified, create a random split
    if validation > 0 and not validation_vid:
        # no validation vid specified --> random split
        assert validation <= 1
        trn_ds, val_ds = custom_random_split(trn_ds, val_split=validation, seed=seed)
        print(f'Loaded {len(trn_ds)} training samples and {len(val_ds)} validation samples.')
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f'{len(val_loader)=}')
    
    # Creates a DataLoader for the train and test datasets
    tst_loader = DataLoader(tst_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print(f'{len(trn_loader)=}')
    print(f'{len(tst_loader)=}')
    return trn_loader, val_loader, tst_loader
