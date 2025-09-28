# core/models.py
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    """
    Lightweight classifier for 13 classes (empty + 6 white + 6 black).
    Input: 3xHxW images (HxW ~ 64).
    """
    def __init__(self, num_classes=13):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32 x H/2 x W/2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64 x H/4 x W/4
        x = F.relu(self.bn3(self.conv3(x)))             # 128 x H/4 x W/4
        x = self.avgpool(x)                             # 128 x 1 x 1
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
