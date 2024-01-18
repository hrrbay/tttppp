import numpy as np
import cv2
import torchvision
import torch
from data import data_config

video = '/home/jakob/uni/ivu/ttnet/dataset/test/videos/test_1.mp4'
masks = '/mnt/shared/datasets/t3p3/test_1/t3p3_masks.npy'
masks = np.load(masks)


model = getattr(torchvision.models.video, 'r3d_18')(pretrained=False, progress=True)
model.eval()
head_name, head_var = list(model.named_modules())[-1]
assert type(head_var) == torch.nn.Linear, 'Fix this.'
setattr(model, head_name, torch.nn.Linear(in_features=head_var.in_features, out_features=1))
model.to('cuda:0')


data_conf = data_config.data_config['r3d_18']
transforms = data_conf['transforms']
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()] + transforms)
# masks = [transforms(mask) for mask in masks]
cap = cv2.VideoCapture(video)
frame_idx = 0

window = []
window_size = 16
pred = -1
while True:
    flag, frame = cap.read()
    print(frame_idx)

    masks_left = True
    if frame_idx >= 4:
        # missing some frames due to sliding window of TTNet
        if len(masks) <= frame_idx:
            masks_left = False
        elif((frame_idx-4)%8 == 0):
            window.append(masks[frame_idx-4])

    
    if len(window) == 16 and masks_left:
        seg_masks = [transforms(seg_mask) for seg_mask in window]
        seg_masks = torch.stack(seg_masks, dim=1).to(torch.float32)
        seg_masks = seg_masks[None, ...].to('cuda:0')
        with torch.no_grad():
            pred = torch.nn.functional.sigmoid(model(seg_masks.to('cuda:0'))).item()
        window = window[1:]
    cv2.putText(frame, f'{1-pred:.2f}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'{pred:.2f}', (frame.shape[1]-100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow('video', frame)
    cv2.waitKey(1)
    frame_idx += 1




