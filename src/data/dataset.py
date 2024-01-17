import random
import torch
import numpy as np
import os
import time
import pdb
from torch.utils.data import Dataset
from PIL import Image
import cv2
import scipy.sparse as sp
from torch.utils.data import DataLoader
import copy
import argparse
import psutil
import random
import torchvision

from torchvision.transforms.v2 import ToTensor, Normalize, RandomHorizontalFlip, Compose
from PIL import Image

class TTVid():
    def post_process(self, data, downscale):
        data = data[0::downscale]
        return data

    def __init__(self, path, src_fps, target_fps, labeled_start=False, window_size=30, fixed_seq_len=0):
        self.path = path
        self.src_fps = src_fps
        self.target_fps = target_fps

        # seg-masks are stored and compressed using zarr
        self.seg_masks = np.lib.format.open_memmap(os.path.join(path, 't3p3_masks.npy'))

        # load all frames, create annotations
        # TODO: (!!) offset point labels by 4 (5?) frames as we are missing some frames at the start due to sliding window of ttnet
        self.point_labels = np.loadtxt(os.path.join(path, 'point_labels.txt')).astype(int)
        print(f'{path=}')

        self.sequences = []
        self.sequence_idx = []
        self.next_points = torch.empty(len(self.point_labels), 1, dtype=int)
        for i in range(len(self.point_labels)):
            end = self.point_labels[i][1]
            if not labeled_start:
                # include dead frames
                if i == 0:
                    start = 0
                else:
                    start = self.point_labels[i - 1][1] + 1 # maybe use different offset
                if fixed_seq_len > 0:
                    start = end - fixed_seq_len if end - fixed_seq_len >= 0 else start
            else:
                # (not-)dead frames are labeled
                self.serve_labels = np.loadtxt(os.path.join(path, 'serve_labels.txt')).astype(int)
                if self.serve_labels.shape[1] > 1:
                    # remove useless label
                    self.serve_labels = self.serve_labels[:, 1]
                serve_label = self.serve_labels[self.serve_labels < end]
                assert serve_label.shape[0] > 0
                start = serve_label[-1] # take serve preceeding current point
                # start = self.serve_labels[serve_idx[-1]]
            next_point_player = self.point_labels[i][0]
            self.next_points[i] = next_point_player
            self.sequence_idx.append((start,end))
            self.sequences.append(self.seg_masks[start:end])


        # post-process data
        downscale = src_fps // target_fps
        self.num_sequences = len(self.sequences)
        self.num_frames = 0
        for i in range(self.num_sequences):
            self.sequences[i] = self.post_process(self.sequences[i], downscale)
            self.num_frames += self.sequences[i].shape[0]

        self.wins_per_seq = [seq.shape[0] + 1 - window_size for seq in self.sequences]
        self.num_wins = sum(wins for wins in self.wins_per_seq)

        # # TODO: need to clean this up again. had to store indices and using them to load when trying smth else. Functionality is the same as before
        # self.sequence_idx = [np.arange(seq[0], seq[1], self.src_fps // self.target_fps) for seq in self.sequence_idx]
        # self.num_sequences = len(self.sequence_idx)
        # self.seq_lens = [len(seq) for seq in self.sequence_idx]
        # self.num_frames = sum(self.seq_lens)
        # self.wins_per_seq = [seq_len + 1 - window_size for seq_len in self.seq_lens]
        # self.num_wins = sum(wins for wins in self.wins_per_seq)
        

class TTData(Dataset):
    def __init__(self, tt_vids, win_size=30, transforms=[], flip_prob=0):
        self.vids = tt_vids
        self.win_size = win_size
        self.wins_per_vid = [vid.num_wins for vid in self.vids]
        self.transforms = Compose(transforms)
        self.flip_prob = flip_prob
        print(f'{self.transforms=}')
        print(f'Dataset created. Total number of windows: {sum(v.num_wins for v in self.vids)}')

    def __len__(self):
        return sum(wins for wins in self.wins_per_vid) # num of sliding windows over all sequences of all videos
    
    def __getitem__(self, idx): # idx is the index of the sliding window over all sequences of all videos
        vid_idx = 0
        for i, wins in enumerate(self.wins_per_vid):
            if idx < wins:
                break
            idx -= wins
            vid_idx += 1

        vid = self.vids[vid_idx]
        # print(f'vid_idx: {vid_idx}')
        seq_idx = 0
        for i, wins in enumerate(vid.wins_per_seq):
            if idx < wins:
                break
            idx -= wins
            seq_idx += 1

        seq = vid.sequences[seq_idx]
        seg_masks = seq[idx:idx+self.win_size]
        label = vid.next_points[seq_idx]

        '''
            Random flip segmentation masks
            - Also need to "flip" the grayscale-encoding and flip the label
            - Do this before transforms because normalization changes values
        '''
        if self.flip_prob != 0 and self.flip_prob >= random.uniform(0,1): # 1 can be included due to rounding, therefore >= but also add prob != 0 to if
            val_left = 0.2
            val_right = 0.4
            ball_add = 0.6
            flipped_tmp = np.flip(seg_masks, -1).copy()
            flipped = np.zeros_like(seg_masks)
            flipped[flipped_tmp == val_left] = val_right
            flipped[flipped_tmp == val_right] = val_left
            flipped[flipped_tmp == ball_add] = ball_add
            flipped[flipped_tmp == val_left + ball_add] = val_right + ball_add
            flipped[flipped_tmp == val_right + ball_add] = val_left + ball_add
            seg_masks = flipped

            # TODO: I'll leave this here for the report maybe?
            # import matplotlib.pyplot as plt
            # fig, axs = plt.subplots(3, 1)
            # axs[0].imshow(seg_masks[0], cmap='gray', vmin=0, vmax=1)
            # axs[1].imshow(flipped_tmp[0], cmap='gray', vmin=0, vmax=1)
            # axs[2].imshow(flipped[0], cmap='gray', vmin=0, vmax=1)
            # plt.show()

            label = 1 - label 

        # apply transforms
        seg_masks = [self.transforms(mask) for mask in seg_masks]
        # reshape to CDHW for 3d conv -- don't know if explicit channel-dimension of 1 is necessary but better be safe
        seg_masks = torch.stack(seg_masks, dim=1).to(torch.float32)
        assert seg_masks.shape[1] == self.win_size

        # TODO: remove this when good
        # uncomment this if you want to check how one window looks
        # print(f'seq_idx_range: {vid.sequence_idx[seq_idx]}, seq_idx: {seq_idx}')
        # for d in range(seg_masks.shape[1]):
        #     m = seg_masks[:,d]
        #     # pdb.set_trace()
        #     cv2.imshow('asdf', (m*255).cpu().numpy().astype(np.uint8).transpose(1,2,0))
        #     cv2.waitKey(10)
        # cv2.waitKey(0)
        # Or this if you only want to see middle frame
        # middle = (seg_masks[:,seg_masks.shape[1]//2]*255).cpu().numpy().astype(np.uint8).transpose(1,2,0)
        # # print(f'{middle.shape=}')
        # print(f'{seq_idx=}', end='\r')
        # cv2.imshow('asdf', middle)
        # cv2.waitKey(0)


        return seg_masks, label

def test_load():
    '''
        Some tests
    '''
    # path = '/mnt/data/datasets/t3p3/annotations/test_2'
    vids = []
    for n in range(5, 6):
        vids.append(TTVid(f'/home/jakob/datasets/t3p3/train_{n}', 120, 30, labeled_start=True))
    # path = '/home/jakob/uni/ivu/data/annotations/test_2'
    # path = '/mnt/data/datasets/t3p3/annotations/test_1/'

    # vid1 = TTVid(path, 120, 120)
    # vid2 = TTVid(path2, 120, 120)
    print(f'video created')
    dataset = TTData(vids, transforms=[ToTensor()])
    print(f'ds created')
    loader = DataLoader(dataset, batch_size=64, num_workers=0, shuffle=False)
    print(f'dl created')
    print(f'len(loader): {len(loader)}')
    import time
    t = time.time()
    total_time = 0
    n = 0
    for a in loader:
        cur_t = time.time() - t
        # print(f'{cur_t=:02.4f} ({n=})', end='\r')
        total_time += cur_t
        t = time.time()
        n += 1
    print()
    print(f'avg load ({n} batches): {total_time / n}')
# test_load()

def show_masks():
    path = '/home/jakob/datasets/t3p3/test_1'
    masks = np.load(os.path.join(path, 't3p3_masks.npy'))

    for mask in masks:
        show = (mask*255).astype(np.uint8)[..., None]
        cv2.imshow('asdf', show)
        cv2.waitKey(1)
# show_masks()