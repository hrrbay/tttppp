import sys

import cv2
import numpy as np
from scipy.spatial import distance_matrix
from ultralytics import YOLO
from src.data.data_config import BASE_PATH



min_conf = 0.4
eps = 0.1

# removes poses that are too close to each other
def merge_poses(poses):
    x_coords = poses[:, :, 0]
    dist_x = np.tril(distance_matrix(x_coords, x_coords))
    merge_idx = np.argwhere((dist_x < eps)&(dist_x!=0))
    return np.delete(poses, np.max(merge_idx, axis=1), axis=0)
    

def extract_poses(video_path):
    cap = cv2.VideoCapture(video_path)
    number_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    poses_np = np.zeros((number_frames, 2, 17, 2), dtype='float32')
    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if frame_count % (120*30) == 0:
            print(f'Second {frame_count//120}')

        if ret:
            # mask out the referee
            frame[:, 2*frame.shape[1]//5:3*frame.shape[1]//5] = 0
            
            # detect poses and remove poses with low confidence
            results = model(frame, show=False, verbose=False)[0]
            avg_conf = results.keypoints.conf.mean(axis=1).cpu().numpy()
            conf_sorted = np.argsort(-avg_conf)
            avg_conf = avg_conf[conf_sorted]
            
            poses = results.keypoints.xyn.cpu().numpy()[conf_sorted]
            poses = poses[avg_conf > min_conf]

            # merge poses that are too close to each other
            if poses.shape[0] > 2:
                poses = merge_poses(poses)
            
            # store the poses correctly
            if poses.shape[0] == 1:
                if poses[0,:,0].mean() < 0.5:
                    poses_np[frame_count, 0] = poses[0]
                else:
                    poses_np[frame_count, 1] = poses[0]
            elif poses.shape[0] >= 2:
                poses_np[frame_count, 0] = poses[1]
                poses_np[frame_count, 1] = poses[0]
                if poses[0,:,0].mean() < poses[1,:,0].mean():
                    poses_np[frame_count, 0] = poses[0]
                    poses_np[frame_count, 1] = poses[1]
        else:
            break
        
        frame_count += 1
        
        #safe guard. apparently the opencv framenumber is not always correct
        if frame_count >= number_frames:
            break
        
    
    cap.release()

    # write the poses_np to a .npy file
    np.save(video_path.removesuffix('.mp4') + '_poses.npy', poses_np)

    print(f'Extracting poses from {video_path} done.')

model = YOLO('yolov8m-pose.pt')
def main():
    if len(sys.argv) < 2:
        print("please specify a video path")
        return
    video_path = sys.argv[1]
    print(f'Extracting poses from {video_path}...')
    extract_poses(BASE_PATH + video_path)

if __name__ == '__main__':
    main()