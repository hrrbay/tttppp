import torch
import numpy as np
import os
import pdb
from torch.utils.data import Dataset
from PIL import Image
import cv2


class TTVid():

    def post_process(self, data, last_point, downscale):
        data = data[:last_point]
        data = data[0::downscale]
        return data

    def __init__(self, path, annotations, src_fps, target_fps, sequences=False, window_size=30):
        self.path = path
        self.src_fps = src_fps
        self.target_fps = target_fps

        # load all frames, create annotations
        self.frames = sorted(os.listdir(path))
        self.frames = [os.path.join(path, frame) for frame in self.frames]

        if sequences:
            frames_temp = self.frames.copy()
            self.frames = []
            self.next_points = torch.empty(len(annotations['points'])//2, 1, dtype=int)
            self.start_end = torch.empty(len(annotations['points'])//2, 2, dtype=int)
            for i in range(0, len(annotations['points']), 2):
                start = annotations['points'][i][1]
                end = annotations['points'][i + 1][1]
                next_point_player = annotations['points'][i + 1][0]
                self.next_points[i // 2] = next_point_player
                self.start_end[i // 2] = torch.tensor([start, end])
                self.frames.extend(frames_temp[start:end])
        else:
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

        # annoations
        self.annotations = {
            'next_points': self.next_points
        }

def extract_player_mask(img):
    # convert to HSV 
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   
    # get green blobs and sort by area to get players
    green = np.array([60, 255, 255])
    green_mask = cv2.inRange(hsv_image, green, green)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # extract the 2 largest green blobs
    player_masks = []
    for contour in contours[:2]:
         # find the centers of the contours
        M = cv2.moments(contour)
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        # compute the bounding box for the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        mask = np.zeros((h, w), dtype="uint8")
        contour = contour - [x, y]
        cv2.drawContours(mask, [contour], -1, 255, 1)
    
        player_masks.append((center_x, center_y, mask))

    
    player_masks = sorted(player_masks, key=lambda x: x[0])
    return player_masks[0], player_masks[1]

class TTData(Dataset):
    def __init__(self, tt_vids, win_size=30):
        self.vids = tt_vids
        self.win_size = win_size

    def __len__(self):
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
        window_range = (idx*self.win_size, idx*self.win_size+self.win_size)

        window_frames = vid.frames[window_range[0]:window_range[1]]
        window_frames = [Image.open(frame).convert('RGB') for frame in window_frames]                  # This is slow

        labels = vid.annotations['next_points'][window_range[0]:window_range[1]]

        # pad window with empty images -- How to handle annotations? OR just skip windows with size < win_size OR actually overlap (see first comment)
        assert len(window_frames) <= self.win_size
        for _ in range(len(window_frames), self.win_size):
            window_frames.append(Image.new('RGB', window_frames[0].size))
            labels.append(0) # I don't know, see above
        assert len(window_frames) == self.win_size

        # TODO: replace window_frames with seg-masks, ball positions
        return window_frames, labels




'''train_path = '/home/christian/Desktop/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/dataset/training/images'
vids = []
for game in os.listdir(train_path):
    points = np.loadtxt(f'../point_labels/train_{game[-1]}.txt')
    annotations = {
        'points': points.astype(int)# TODO
    }
    vids.append(TTVid(os.path.join(train_path, game), annotations, 120, 60, sequences=True))

train_dataset = TTData(vids, win_size=30)'''

test_path = '/home/christian/Desktop/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/dataset/test/images'
vids = []
for game in os.listdir(test_path):
    points = np.loadtxt(f'../point_labels/test_{game[-1]}.txt')
    annotations = {
        'points': points.astype(int) # TODO
    }
    vids.append(TTVid(os.path.join(test_path, game), annotations, 120, 60, sequences=True))

test_dataset = TTData(vids, win_size=30)

for frames, annotations in test_dataset:
    print(len(frames))
    # print(annotations)