import torch
import torchvision
import cv2
import numpy as np
from torchvision.models.video import R2Plus1D_18_Weights

# Pre-trained (2+1)D ResNet model with 18 layers
# Paper https://arxiv.org/pdf/1711.11248.pdf
# https://pytorch.org/vision/main/models/generated/torchvision.models.video.r2plus1d_18.html#torchvision.models.video.r2plus1d_18
model = torchvision.models.video.r2plus1d_18(weights='DEFAULT')
model = torch.nn.Sequential(*(list(model.children())[:-1])).cuda() # remove last layer
model.eval()
transform = R2Plus1D_18_Weights.KINETICS400_V1.transforms()

def read_video_as_tensor(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or counter >= 10:
            break

        frames.append(frame)
        counter += 1

    cap.release()
    video_tensor = torch.tensor(np.array(frames)).permute(0, 3, 1, 2)  # TxHxWxC -> TxCxHxW
    return transform(video_tensor).cuda()  # output dimensions are permuted to (..., C, T, H, W) tensors

print(torch.cuda.is_available())
video_file_path = '../test_2.mp4'
frames_tensor = read_video_as_tensor(video_file_path)
print(frames_tensor.shape)
frames_tensor = frames_tensor.unsqueeze(0)  # BxCxTxHxW
output = model(frames_tensor)
print(output.shape)

