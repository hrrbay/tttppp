import json
import os
import pdb
import random

import cv2
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import (Compose, Normalize,
                                       RandomHorizontalFlip, ToTensor)


class TTVid():
    """
        Encapsulates the data for a single video
        Handles loading and post-processing
    """
    def post_process(self, data, downscale):
        data = data[0::downscale]
        return data

    def __init__(self, path, src_fps, target_fps, labeled_start=False, window_size=30, fixed_seq_len=0, use_poses=False):
        self.path = path
        self.src_fps = src_fps
        self.target_fps = target_fps

        # depending on the mode, either load segmentation masks or poses
        if not use_poses:
            self.seg_masks = np.lib.format.open_memmap(os.path.join(path, 't3p3_masks.npy'))
        else:
            self.poses = np.load(os.path.join(path, 'poses.npy'))
            poses_shape = self.poses.shape
            self.poses = np.reshape(self.poses, (poses_shape[0], poses_shape[1] * poses_shape[2], poses_shape[3]))
            # remove first 4 frames
            self.poses = self.poses[4:, :, :]  
            # remove last 10 frames
            self.poses = self.poses[:-10, :, :]  

            f = open(os.path.join(path, 'ball_pos.json'))
            ball_pos_dict = json.load(f)
            ball_pos = np.zeros((len(ball_pos_dict.keys()), 2))

            for frame in ball_pos_dict.keys():
                ball_pos[int(frame) - 5] = np.array(ball_pos_dict[frame])

            # normalize ball pos
            ball_pos[:, 0] = ball_pos[:, 0] / 1920
            ball_pos[:, 1] = ball_pos[:, 1] / 1080

            # concat ball pos to poses
            self.poses = np.concatenate((self.poses, np.expand_dims(ball_pos, 1)), axis=1)


        # load all frames, create annotations
        self.point_labels = np.loadtxt(os.path.join(path, 'point_labels.txt')).astype(int)
        print(f'{path=}')

        # split the video into sequences based on the timestamps of the point labels
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
                    start = self.point_labels[i - 1][1] + 1 
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
                # take serve preceeding current point
                start = serve_label[-1]
            next_point_player = self.point_labels[i][0]
            self.next_points[i] = next_point_player
            self.sequence_idx.append((start,end))
            if not use_poses:
                self.sequences.append(self.seg_masks[start:end])
            else:
                self.sequences.append(self.poses[start:end])


        # post-process data (downscale)
        downscale = src_fps // target_fps
        self.num_sequences = len(self.sequences)
        self.num_frames = 0
        for i in range(self.num_sequences):
            self.sequences[i] = self.post_process(self.sequences[i], downscale)
            self.num_frames += self.sequences[i].shape[0]

        self.wins_per_seq = [seq.shape[0] + 1 - window_size for seq in self.sequences]
        self.num_wins = sum(wins for wins in self.wins_per_seq)

        

class TTData(Dataset):
    """
        Custom dataset for a list of TTVid objects
    """
    def __init__(self, tt_vids, win_size=30, transforms=[], flip_prob=0):
        self.vids = tt_vids
        self.win_size = win_size
        self.wins_per_vid = [vid.num_wins for vid in self.vids]
        self.transforms = Compose(transforms)
        self.flip_prob = flip_prob
    
    # num of sliding windows over all sequences of all videos
    def __len__(self):
        return sum(wins for wins in self.wins_per_vid) 
    
    # idx is the index of the sliding window over all sequences of all videos
    def __getitem__(self, idx): 
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

        
        #Random flip segmentation masks
        #   - Also need to "flip" the grayscale-encoding and flip the label
        #   - Do this before transforms because normalization changes values
        #   - 1 can be included due to rounding, therefore >= but also add prob != 0 to if
        if self.flip_prob != 0 and self.flip_prob >= random.uniform(0,1): 
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
            label = 1 - label 

        # apply transforms
        seg_masks = [self.transforms(mask) for mask in seg_masks]
        # reshape to CDHW for 3d conv -- don't know if explicit channel-dimension of 1 is necessary but better be safe
        seg_masks = torch.stack(seg_masks, dim=1).to(torch.float32)
        assert seg_masks.shape[1] == self.win_size

        return seg_masks, label


def test_load():
    '''
        Debugging function to test the loading of the dataset
    '''
    vids = []
    for n in range(5, 6):
        vids.append(TTVid(f'/home/jakob/datasets/t3p3/train_{n}', 120, 30, labeled_start=True))
    
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
        total_time += cur_t
        t = time.time()
        n += 1
    print()
    print(f'avg load ({n} batches): {total_time / n}')

def show_masks():
    """
        Debugging function to show the segmentation masks
    """
    path = '/home/jakob/datasets/t3p3/test_1'
    masks = np.load(os.path.join(path, 't3p3_masks.npy'))

    for mask in masks:
        show = (mask*255).astype(np.uint8)[..., None]
        cv2.imshow('asdf', show)
        cv2.waitKey(1)
