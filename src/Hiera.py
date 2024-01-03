import torchvision.transforms as transforms
import cv2
import numpy as np
import torch, hiera

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_video_as_tensor(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or counter >= 150: # TODO hack to not read the whole video
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        counter += 1

    cap.release()
    video_tensor = torch.tensor(np.array(frames))  # TxHxWxC -> TxCxHxW
    return video_tensor  # output dimensions are permuted to (..., C, T, H, W) tensors

def preprocess(frames, transform):
    frames = frames.permute(0, 3, 1, 2).contiguous()  # THWC -> TCHW
    frames = transform(frames)
    frames = torch.stack([frames[:64], frames[64:128]], dim=0)  # BTCHW
    frames = frames[:, ::4]  # Sample every 4 frames -> 64/4 = 16 frames per clip
    frames = frames.permute(0, 2, 1, 3, 4).contiguous()  # BTCHW -> BCTWH
    frames = frames - torch.tensor([0.45, 0.45, 0.45]).view(1, -1, 1, 1, 1)  # Subtract mean
    frames = frames / torch.tensor([0.225, 0.225, 0.255]).view(1, -1, 1, 1, 1)  # Divide by std
    return frames

vid_path = "..."

# Load the frames
frames = read_video_as_tensor(vid_path)
frames = frames.float() / 255  # Convert from byte to float
resize = transforms.Resize((224, 224), )
frames = preprocess(frames, transform=resize)


model = hiera.hiera_base_16x224(pretrained=True, checkpoint="mae_k400_ft_k400").to(device)

# Get kinetics classes as output
out, intermediates = model(frames.to(device), return_intermediates=True) # intermediate results for each block
print(out.shape)

# Average results over the clips
out = out.mean(0)
print(out.argmax(dim=-1).item())
