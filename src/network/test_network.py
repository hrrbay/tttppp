import torch
import pdb

class TestNet(torch.nn.Module):
    def __init__(self, model_config=None):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1, 1, 2, 3)
        self.bn1 = torch.nn.BatchNorm3d(1)
        self.conv2 = torch.nn.Conv3d(1, 1, 2, 3)
        self.bn2 = torch.nn.BatchNorm3d(1)
        self.conv3 = torch.nn.Conv3d(1, 1, 1, 3)
        self.bn3 = torch.nn.BatchNorm3d(1)
        self.fc1 = torch.nn.Linear(5*12, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.nn.functional.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x
