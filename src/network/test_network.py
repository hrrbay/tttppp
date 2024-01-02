import torch
import pdb

class TestNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1, 1, 2, 3)
        self.conv2 = torch.nn.Conv3d(1, 1, 2, 3)
        self.conv3 = torch.nn.Conv3d(1,1,1,3)
        self.fc1 = torch.nn.Linear(5*12, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x
