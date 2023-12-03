import torch
import numpy as np
import os
import pdb
from torch.utils.data import Dataset
from PIL import Image

class TTVid():

    def post_process(self, data, last_point, downscale):
        data = data[:last_point]
        data = data[0::downscale]
        return data

    def __init__(self, path, annotations, src_fps, target_fps):
        self.path = path
        self.src_fps = src_fps;
        self.target_fps = target_fps;

        # load all frames, create annotations
        self.frames = sorted(os.listdir(path))
        self.frames = [os.path.join(path, frame) for frame in self.frames]

        # set next points
        last_point = 0
        self.next_points = torch.empty(len(self.frames), 1, dtype=int)
        for player, frame in annotations['points']:
            self.next_points[last_point:frame] = player
            last_point = frame

        # post-process data
        downscale = src_fps // target_fps
        self.frames = self.post_process(self.frames, last_point, downscale)
        self.next_points = self.post_process(self.next_points, last_point, downscale)
        
        self.num_frames = len(self.frames)




class TTData(Dataset):
    def __init__(self, tt_vids, win_size=30):
        self.vids = tt_vids
        self.win_size = win_size

    def __len__(self):
        len = sum(np.ceil(vid.num_frames / self.win_size).astype(int) for vid in self.vids)
        return sum(np.ceil(vid.num_frames / self.win_size).astype(int) for vid in self.vids)
    
    def __getitem__(self, idx):
        # find correct video -- treat seperately in order to avoid overlapping windows
        vid_lens = np.cumsum([np.ceil(vid.num_frames / self.win_size).astype(int) for vid in self.vids])
        vid_idx = np.nonzero(vid_lens > idx)[0][0]
        vid = self.vids[vid_idx]
        
        # fix idx to current video
        if vid_idx > 0:
            idx -= vid_lens[vid_idx-1]

        # load images as PIL -- TODO: transforms (need ToTensor at least, could also H-flip with point-label flip=)
        window = vid.frames[idx*self.win_size:idx*self.win_size+self.win_size]
        window = [Image.open(frame).convert('RGB') for frame in window] # This is slow

        # pad window with zeros
        assert len(window) <= self.win_size
        for _ in range(len(window), self.win_size):
            window.append(Image.new('RGB', window[0].size))
        return window




train_path = '/home/jakob/uni/ivu/ttnet/dataset/training/images'
vids = []
for game in os.listdir(train_path):
    points = np.loadtxt(f'point_labels/train_{game[-1]}.txt')
    annotations = {
        'points': points.astype(int)
    }
    vids.append(TTVid(os.path.join(train_path, game), annotations, 120, 60))

dataset = TTData(vids, win_size=30)
for win in dataset:
    print(len(win))