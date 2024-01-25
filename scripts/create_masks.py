import os 
import sys
import json

import numpy as np
from PIL import Image

data_path = os.path.join(os.path.dirname(__file__), '..', 't3p3_data')
data_path = os.path.abspath(data_path)
if 'T3P3_PATH' in os.environ:
    data_path = os.environ['T3P3_PATH']

print(f'Loading TTNet-masks from and saving T3P3-masks to {data_path}')

# Frame sizes of original and resized frames. 
# Output of ball-pos by TTNet is 1920x1080 --> need to change it 
w_original, h_original = 1920, 1080
w_resize, h_resize = 320, 128
w_ratio = w_original / w_resize
h_ratio = h_original / h_resize

def get_grayscale_mask(seg_img, ball_pos=None, ball_radius=10, val_left=0.2, val_right=0.4, ball_add=0.6):
    """
    Convert RGB-mask created by TTNet to grayscale-mask used for T3P3

    Parameters
    ----------
    seg_img : array-like
        Segmentation-mask provided by TTNet
    ball_pos : tuple, optional
        Ball-position provided by TTNet (at 1920x1080 resolution). If `None`, ball will not be drawn on mask
    ball_radius : int, optional
        Radius of (filled) circle to draw at ball-position, by default 10
    val_left : float, optional
        Value to use for left player, by default 0.2
    val_right : float, optional
        Value to use for right player, by default 0.4
    ball_add : float, optional
        Value to use for ball-circle. This will be added on top of player masks, by default 0.6

    Returns
    -------
    array-like
        T3P3-mask in grayscale
    """

    # ball-position to resized dimensions
    ball_pos = (ball_pos[0]/w_ratio, ball_pos[1]/h_ratio)

    # only use green channel to remove table
    seg_img = seg_img[..., 1]
    seg_img[seg_img != 255] = 0
    seg_img = seg_img.astype(float) / 255

    # remove referee; middle fifth of image
    seg_img[:, 2*seg_img.shape[1]//5:3*seg_img.shape[1]//5] = 0 

    # Just assume left player in left half, right player in right half. Apply corresponding values
    seg_img[:,:seg_img.shape[1]//2] *= val_left
    seg_img[:,seg_img.shape[1]//2:] *= val_right

    # draw the ball
    if ball_pos and ball_radius > 0:
        x = np.arange(0, seg_img.shape[1])
        y = np.arange(0, seg_img.shape[0])
        mask = (x[None,:]-ball_pos[0])**2 + (y[:,None]-ball_pos[1])**2 <= ball_radius**2
        seg_img[mask] += ball_add
    
    return seg_img



def store_npy():
    out_size = (w_resize, h_resize)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # find valid paths
    vid_dirs = []
    for vid_dir in sorted(os.listdir(data_path)):
        vid_dir_abs = os.path.join(data_path, vid_dir)
        
        if not os.path.isdir(vid_dir_abs):
            continue
        if not os.path.exists(os.path.join(vid_dir_abs, 'ball_pos.json')):
            print(f'* WARNING: {os.path.join(vid_dir_abs, "ball_pos.json")} does not exist. Skipping {vid_dir}..')
            continue
        vid_dirs.append(vid_dir_abs)

    total_num_masks = sum([len(os.listdir(os.path.join(d, 'rgb_masks'))) for d in vid_dirs])
    print(f'Total masks to process: {total_num_masks}')
    for vid_dir in vid_dirs:
        

        # load the ball-positions
        with open(os.path.join(vid_dir, 'ball_pos.json'), 'r') as pos_f:
            ball_positions = json.load(pos_f)
        ball_positions = list(ball_positions.values())

        # need number of masks for mmap
        ttnet_mask_dir = os.path.join(vid_dir, 'rgb_masks')
        num_masks = len(os.listdir(ttnet_mask_dir))
        t3p3_masks = np.lib.format.open_memmap(os.path.join(vid_dir, 't3p3_masks.npy'), mode='w+', shape=(num_masks, out_size[1], out_size[0]))

        # create t3p3-grayscale-mask for every rgb-mask
        for i, im in enumerate([f for f in os.listdir(ttnet_mask_dir) if f.endswith('jpg')]):
            pil_im = Image.open(os.path.join(ttnet_mask_dir, im)).resize(out_size)
            t3p3_masks[i] = get_grayscale_mask(np.array(pil_im), ball_positions[i], ball_radius=10)
            print(f'{os.path.basename(vid_dir)}: {i}/{num_masks}', end='\r')
        print()
    
store_npy()