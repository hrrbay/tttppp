import torch
import numpy as np
import os
import pdb
from torch.utils.data import Dataset
from PIL import Image
import cv2
import scipy.sparse as sp
from torch.utils.data import DataLoader
import copy
import argparse

from torchvision.transforms.v2 import ToTensor, Normalize, RandomHorizontalFlip, Compose
from PIL import Image
import zarr

class SparseCSR():
    def __init__(self, shape, data, indptr, indices):
        self.data = data
        self.shape = shape
        self.indptr = indptr
        self.indices = indices

    def __getitem__(self, idx):
        arr = np.zeros((self.shape[0]))
        col_idx = self.indices[self.indptr[idx]:self.indptr[idx+1]]
        arr[col_idx] = self.data[self.indptr[idx]:self.indptr[idx+1]]
        return arr

class TTVid():
    def post_process(self, data, downscale):
        data = data[0::downscale]
        return data

    def __init__(self, path, src_fps, target_fps, labeled_start=False, window_size=30, mode='mmap'):
        self.path = path
        self.src_fps = src_fps
        self.target_fps = target_fps

        self.in_size = (320, 128) 
        self.resize = (320, 128) # in case we want to downsample masks -- NOTE that data has to be properly extracted again (and will be of higher size)

        # whether to load images from sparse array or from actual jpg-files
        # NOTE: use mmap, all others pretty useless. 
        self.mode = mode
        if self.mode == 'sparse':
            self.seg_masks = sp.load_npz(os.path.join(path,'t3p3_sparse_masks.npz')).tocsr()
        elif self.mode == 'mmap':
            self.seg_masks = np.lib.format.open_memmap(os.path.join(path, 't3p3_masks.npy'), mode='r')
            shape = (self.seg_masks.shape[0] // (self.in_size[0] * self.in_size[1]), self.in_size[1], self.in_size[0])
            self.seg_masks = np.lib.format.open_memmap(os.path.join(path, 't3p3_masks.npy'), shape=shape)
        elif self.mode == 'disk':
            self.seg_masks = [os.path.join(path, 't3p3_masks', f) for f in os.listdir(os.path.join(path, 't3p3_masks'))]
        elif self.mode == 'zarr':
            zarr_root = zarr.open(os.path.join(path, 't3p3_masks.zarr'), mode='r')
            # num_masks = zarr_root.attrs['num_masks']
            # mask_shape = zarr_root.attrs['mask_shape']
            self.seg_masks = zarr_root['masks']
            pdb.set_trace()
            print(f'info:\n{self.seg_masks.info}')

        # load all frames, create annotations
        # TODO: (!!) offset point labels by 4 (5?) frames as we are missing some frames at the start due to sliding window of ttnet
        self.point_labels = np.loadtxt(os.path.join(path, 'point_labels.txt')).astype(int)

        self.sequences = []
        if labeled_start: # start of sequences are labeled
            self.next_points = torch.empty(len(self.point_labels) // 2, 1, dtype=int) # len of annotations is doubled because start is labeled
            for i in range(0, len(self.point_labels), 2):
                start = self.point_labels[i][1]
                end = self.point_labels[i + 1][1]
                next_point_player = self.point_labels[i + 1][0]
                self.next_points[i // 2] = next_point_player
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
                self.sequences.append(self.seg_masks[start:end])

        # post-process data 
        downscale = src_fps // target_fps
        self.num_sequences = len(self.sequences)
        self.num_frames = 0
        for i in range(self.num_sequences):
            self.sequences[i] = self.post_process(self.sequences[i], downscale)
            self.num_frames += self.sequences[i].shape[0] if self.mode != 'disk' else len(self.sequences[i])

        self.wins_per_seq = [(seq.shape[0] if self.mode != 'disk' else len(seq)) + 1 - window_size for seq in self.sequences]
        self.num_wins = sum(wins for wins in self.wins_per_seq)

class TTData(Dataset):
    def __init__(self, tt_vids, win_size=30, transforms=[]):
        self.vids = tt_vids
        self.win_size = win_size
        self.wins_per_vid = [vid.num_wins for vid in self.vids]
        self.transforms = Compose(transforms)

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
        seq_idx = 0
        for i, wins in enumerate(vid.wins_per_seq):
            if idx < wins:
                break
            idx -= wins
            seq_idx += 1

        seq = vid.sequences[seq_idx]
        seg_masks = copy.deepcopy(seq[idx:idx+self.win_size])

        # NOTE: all pretty useless except mmap and zarr
        if vid.mode == 'sparse':
            seg_masks = seg_masks.toarray().reshape(-1, vid.resize[1], vid.resize[0])
        elif vid.mode == 'mmap' or vid.mode == 'zarr':
            pass
        elif vid.mode == 'disk':
            seg_masks = [Image.open(f).resize((vid.resize[0], vid.resize[1])) for f in seg_masks]

        # apply transforms
        seg_masks = [self.transforms(mask) for mask in seg_masks]

        # reshape to CDHW for 3d conv -- don't know if explicit channel-dimension of 1 is necessary but better be safe+
        seg_masks = torch.stack(seg_masks, dim=1).to(torch.float32)

        # uncomment this if you want to check how it looks
        # for f in seg_masks.squeeze():
        #     if idx < 100:
        #         break
        #     import matplotlib.pyplot as plt
        #     plt.imshow(f, cmap='gray', vmin=0,vmax=1)
        #     print(f'show')
        #     plt.show()

        label = vid.next_points[seq_idx]
        assert seg_masks.shape[1] == self.win_size

        # TODO: replace window_frames with seg-masks, ball positions
        return seg_masks, label

def test_load():
    '''
        Some tests
    '''
    path = '/mnt/data/datasets/t3p3/annotations/test_1'
    # path = '/home/jakob/uni/ivu/data/annotations/test_2'
    # path = '/mnt/data/datasets/t3p3/annotations/test_1/'

    vid = TTVid(path, 120, 60, mode='zarr')
    print(f'video created')
    dataset = TTData([vid], transforms=[ToTensor()])
    print(f'ds created')
    loader = DataLoader(dataset, batch_size=64, num_workers=0)
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
# test_load()

# path = '/mnt/data/datasets/t3p3/annotations/train_2/t3p3_masks.npy'
# arr = np.lib.format.open_memmap(path, mode='r')
# sp_arr = sp.lil_array((arr.shape[0], arr.shape[1]*arr.shape[2]))
# for row in range(arr.shape[0]):
#     print(f'{row}/{arr.shape[0]}', end='\r')
#     sp_arr[row] = arr[row].flatten()
# pdb.set_trace()