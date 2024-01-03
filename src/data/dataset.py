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

from torchvision.transforms.v2 import ToTensor, Normalize, RandomHorizontalFlip, Compose
from PIL import Image
import zarr

class CSRMemMap():
    '''
        Unused
    '''
    def __init__(self, path):
        self.shape = np.loadtxt(os.path.join(path, 'shape.txt')).astype(int)
        self.data = np.lib.format.open_memmap(os.path.join(path, 'data.npy'), mode='r')
        self.indptr = np.lib.format.open_memmap(os.path.join(path, 'indptr.npy'), mode='r')
        self.indices = np.lib.format.open_memmap(os.path.join(path, 'indices.npy'), mode='r')
        self.reshape = (128, 320)

    def __len__(self):
        return self.shape[0]-1

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            r = range(idx.stop)[idx]
            arr = np.stack([self[i] for i in range(idx.stop)[idx]])
            return arr
        elif isinstance(idx, list) or isinstance(idx, np.ndarray):
            arr = np.stack([self[i] for i in idx])
            return arr
        else:
            arr = np.zeros((self.shape[1]))
            col_idx = self.indices[self.indptr[idx]]
            arr[col_idx] = self.data[self.indptr[idx]]
            return arr.reshape(self.reshape)

    @staticmethod
    def save(path, csr_array):
        assert type(csr_array) == sp.csr_array or type(csr_array) == sp.csr_matrix
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path, 'data.npy'), csr_array.data)
        np.save(os.path.join(path, 'indptr.npy'), csr_array.indptr)
        np.save(os.path.join(path, 'indices.npy'), csr_array.indices)
        np.savetxt(os.path.join(path, 'shape.txt'), csr_array.shape)

class SparseTTVid():
    '''
        Unused
    '''
    def post_process(self, data, downscale):
        data = data[0::downscale]
        return data

    def __init__(self, path, src_fps, target_fps, labeled_start=False, window_size=30):
        self.path = path
        self.src_fps = src_fps
        self.target_fps = target_fps

        # seg-masks are stored and compressed using zarr
        self.seg_masks = CSRMemMap(os.path.join('/tmp/test'))


        # load all frames, create annotations
        # TODO: (!!) offset point labels by 4 (5?) frames as we are missing some frames at the start due to sliding window of ttnet
        self.point_labels = np.loadtxt(os.path.join(path, 'point_labels.txt')).astype(int)
        print(f'point_labels: {self.point_labels}')
        self.sequences = []
        self.sequence_idx = []
        if labeled_start: # start of sequences are labeled
            self.next_points = torch.empty(len(self.point_labels) // 2, 1, dtype=int) # len of annotations is doubled because start is labeled
            for i in range(0, len(self.point_labels), 2):
                start = self.point_labels[i][1]
                end = self.point_labels[i + 1][1]
                next_point_player = self.point_labels[i + 1][0]
                self.next_points[i // 2] = next_point_player
                self.sequence_idx.append((start,end))
                self.sequences.append(self.seg_masks[start:end])
        else: # start of sequences are not labeled -> use point of previous sequence as start
            self.next_points = torch.empty(len(self.point_labels), 1, dtype=int)
            for i in range(len(self.point_labels)):
                if i == 0:
                    start = 0
                else:
                    start = self.point_labels[i - 1][1] + 1 # maybe use different offset
                end = self.point_labels[i][1]
                next_point_player = self.point_labels[i][0]
                self.next_points[i] = next_point_player
                self.sequence_idx.append((start,end))
                self.sequences.append(self.seg_masks[start:end])

        self.sequence_idx = [np.arange(seq[0], seq[1], self.src_fps // self.target_fps) for seq in self.sequence_idx]
        self.num_sequences = len(self.sequence_idx)
        self.seq_lens = [len(seq) for seq in self.sequence_idx]
        self.num_frames = sum(self.seq_lens)
        self.wins_per_seq = [seq_len + 1 - window_size for seq_len in self.seq_lens]
        self.num_wins = sum(wins for wins in self.wins_per_seq)
        
    def __getitem__(self, idx):
        frame_idx = self.sequence_idx[idx]
        pdb.set_trace()
        return self.seg_masks[frame_idx], self.next_points[idx]

class TTVid():
    def post_process(self, data, downscale):
        data = data[0::downscale]
        return data

    def __init__(self, path, src_fps, target_fps, labeled_start=False, window_size=30):
        self.path = path
        self.src_fps = src_fps
        self.target_fps = target_fps

        # seg-masks are stored and compressed using zarr
        # self.zarr_root = zarr.load(os.path.join(path, 't3p3_masks.zarr'))
        # self.seg_masks = self.zarr_root['masks']
        self.seg_masks = np.lib.format.open_memmap(os.path.join(path, 't3p3_masks.npy'))

        # load all frames, create annotations
        # TODO: (!!) offset point labels by 4 (5?) frames as we are missing some frames at the start due to sliding window of ttnet
        self.point_labels = np.loadtxt(os.path.join(path, 'point_labels.txt')).astype(int)

        self.sequences = []
        self.sequence_idx = []
        if labeled_start: # start of sequences are labeled
            self.next_points = torch.empty(len(self.point_labels) // 2, 1, dtype=int) # len of annotations is doubled because start is labeled
            for i in range(0, len(self.point_labels), 2):
                start = self.point_labels[i][1]
                end = self.point_labels[i + 1][1]
                next_point_player = self.point_labels[i + 1][0]
                self.next_points[i // 2] = next_point_player
                self.sequence_idx.append((start,end))
                self.sequences.append(self.seg_masks[start:end])
        else: # start of sequences are not labeled -> use point of previous sequence as start
            self.next_points = torch.empty(len(self.point_labels), 1, dtype=int)
            for i in range(len(self.point_labels)):
                if i == 0:
                    start = 0
                else:
                    start = self.point_labels[i - 1][1] + 1 # maybe use different offset
                end = self.point_labels[i][1]
                next_point_player = self.point_labels[i][0]
                self.next_points[i] = next_point_player
                self.sequence_idx.append((start,end))
                self.sequences.append(self.seg_masks[start:end])

        # TODO: need to clean this up again. had to store indices and using them to load when trying smth else. Functionality is the same as before
        self.sequence_idx = [np.arange(seq[0], seq[1], self.src_fps // self.target_fps) for seq in self.sequence_idx]
        self.num_sequences = len(self.sequence_idx)
        self.seq_lens = [len(seq) for seq in self.sequence_idx]
        self.num_frames = sum(self.seq_lens)
        self.wins_per_seq = [seq_len + 1 - window_size for seq_len in self.seq_lens]
        self.num_wins = sum(wins for wins in self.wins_per_seq)
        
    def __getitem__(self, idx):
        '''
            Unused. Artifact from trying somethong else
        '''
        frame_idx = self.sequence_idx[idx]
        return self.seg_masks[frame_idx], self.next_points[idx]

class TTData(Dataset):
    def __init__(self, tt_vids, win_size=30, transforms=[]):
        self.vids = tt_vids
        self.win_size = win_size
        self.wins_per_vid = [vid.num_wins for vid in self.vids]
        self.transforms = Compose(transforms)
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

        # seq, label = vid[seq_idx]
        # print(f'vid_idx: {vid_idx} idx: {idx}, seq_idx: {seq_idx}')
        seq = vid.sequences[seq_idx]
        seg_masks = seq[idx:idx+self.win_size]  

        # apply transforms
        seg_masks = [self.transforms(mask) for mask in seg_masks]

        # reshape to CDHW for 3d conv -- don't know if explicit channel-dimension of 1 is necessary but better be safe
        seg_masks = torch.stack(seg_masks, dim=1).to(torch.float32)

        # TODO: remove this when good
        # uncomment this if you want to check how it looks
        # middle = (seg_masks[:,seg_masks.shape[1]//2]*255).cpu().numpy().astype(np.uint8).transpose(1,2,0)
        # print(f'{middle.shape=}')
        # cv2.imshow('asdf', middle)
        # cv2.waitKey(1)

        label = vid.next_points[seq_idx]
        assert seg_masks.shape[1] == self.win_size

        return seg_masks, label

def test_load():
    '''
        Some tests
    '''
    # path = '/mnt/data/datasets/t3p3/annotations/test_2'
    vids = []
    for n in range(1, 8):
        vids.append(TTVid(f'/home/jakob/datasets/t3p3/test_{n}', 120, 120))
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
        print(f'{cur_t=:02.4f} ({n=})', end='\r')
        total_time += cur_t
        t = time.time()
        n += 1
    print()
    print(f'avg load ({n} batches): {total_time / n}')
test_load()

def show_masks():
    path = '/home/jakob/datasets/t3p3/test_2'
    masks = np.load(os.path.join(path, 't3p3_masks.npy'))

    for mask in masks:
        show = (mask*255).astype(np.uint8)[..., None]
        cv2.imshow('asdf', show)
        cv2.waitKey(1)
# show_masks()