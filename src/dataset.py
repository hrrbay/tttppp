import torch
import numpy as np
import os
import pdb
from torch.utils.data import Dataset
from PIL import Image
import cv2


class TTVid():

    def post_process(self, data, downscale):
        data = data[0::downscale]
        return data

    def __init__(self, path, annotations, src_fps, target_fps, labeled_start=False, window_size=30):
        self.path = path
        self.src_fps = src_fps
        self.target_fps = target_fps

        # load all frames, create annotations
        total_frames = [os.path.join(path, frame) for frame in sorted(os.listdir(path))]

        self.sequences = []
        if labeled_start: # start of sequences are labeled
            self.next_points = torch.empty(len(annotations['points']) // 2, 1, dtype=int) # len of annotations is doubled because start is labeled
            #self.start_end = torch.empty(len(annotations['points']) // 2, 2, dtype=int) # start end pairs of sequences, not needed right now
            for i in range(0, len(annotations['points']), 2):
                start = annotations['points'][i][1]
                end = annotations['points'][i + 1][1]
                next_point_player = annotations['points'][i + 1][0]
                self.next_points[i // 2] = next_point_player
                #self.start_end[i // 2] = torch.tensor([start, end])
                self.sequences.append(total_frames[start:end])
        else: # start of sequences are not labeled -> use point of previous sequence as start
            self.next_points = torch.empty(len(annotations['points']), 1, dtype=int)
            #self.start_end = torch.empty(len(annotations['points']), 2, dtype=int)
            for i in range(len(annotations['points'])):
                if i == 0:
                    start = 0
                else:
                    start = annotations['points'][i - 1][1] + 1 # maybe use different offset
                end = annotations['points'][i][1]
                next_point_player = annotations['points'][i][0]
                self.next_points[i] = next_point_player
                #self.start_end[i] = torch.tensor([start, end])
                self.sequences.append(total_frames[start:end])

        # post-process data
        downscale = src_fps // target_fps
        self.num_sequences = len(self.sequences)
        self.num_frames = 0
        for i in range(self.num_sequences):
            self.sequences[i] = self.post_process(self.sequences[i], downscale)
            self.num_frames += len(self.sequences[i])

        self.wins_per_seq = [len(seq) + 1 - window_size for seq in self.sequences]
        self.num_wins = sum(wins for wins in self.wins_per_seq)
        # annoations
        '''self.annotations = { # redundant right now?
            'next_points': self.next_points,
            'start_end': self.start_end
        }'''

def extract_player_mask(img):
    # convert to HSV 
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # we mask out the middle third to get rid of the referee
    hsv_image[:,img.shape[1]//3:2*img.shape[1]//3, :] = 0
    
    # get green blobs and sort by area to get players
    green = np.array([60, 255, 255])
    green_mask = cv2.inRange(hsv_image, green, green)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # when players are missing fill up with empty masks
    player_masks = []
    for _ in range(2-len(contours)):
        player_masks.append((0, img.shape[1]//2, img.shape[0]//2, None))
    
    # extract the 2 largest green blobs
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
        
        # first member of tuple indicates that the player in in frame
        player_masks.append((1, center_x, center_y, mask))

    player_masks = sorted(player_masks, key=lambda x: x[1])
    
    return player_masks[0], player_masks[1]

class TTData(Dataset):
    def __init__(self, tt_vids, win_size=30):
        self.vids = tt_vids
        self.win_size = win_size
        self.wins_per_vid = [vid.num_wins for vid in self.vids]

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
        window_frames = seq[idx:idx+self.win_size]


        window_frames = [Image.open(frame).convert('RGB') for frame in window_frames]                  # This is slow
        label = vid.next_points[seq_idx]
        assert len(window_frames) == self.win_size

        # TODO: replace window_frames with seg-masks, ball positions
        return window_frames, label




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
    vids.append(TTVid(os.path.join(test_path, game), annotations, 120, 60, labeled_start=False))

test_dataset = TTData(vids, win_size=30)
frames, label = test_dataset[0]

for frames, annotations in test_dataset:
    print(len(frames))
    # print(annotations)