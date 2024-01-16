import os
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
BASE_PATH = '/mnt/d/ivu/vids'



def extract_poses(video_path):
    
    cap = cv2.VideoCapture(video_path)
    poses = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            result = model(frame, show=False,verbose=False)[0]
            keypoints = result.keypoints
            names = result.names
            probs = result.probs
            poses.append((keypoints, names, probs))
        else:
            break

    cap.release()

    with open(video_path + '_poses.pkl', 'wb') as f:
        pickle.dump(poses, f)

    print(f'Extracting poses from {video_path} done.')

# load poses from pkl file
def load_poses(video_path):
    with open(video_path + '_poses.pkl', 'rb') as f:
        poses = pickle.load(f)
    return poses


model = YOLO('yolov8m-pose.pt')

for root, dirs, files in os.walk(BASE_PATH):
    for file in files:
        if file.endswith('.mp4'):
            video_path = os.path.join(root, file)
            extract_poses(video_path)
            print(load_poses(video_path))
