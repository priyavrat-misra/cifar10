import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Bx3x32x32 ──>
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5,
                      stride=1, padding=2, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # └─> Bx16x16x16 ──>
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,
                      stride=1, padding=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # └─> Bx32x8x8 ──>
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.Dropout2d(p=0.25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # └─> Bx64x4x4 ──>
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64*4*4, out_features=512),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        # └─> Bx512 ──>
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU()
        )
        # └─> Bx128 ──>
        self.out = nn.Linear(in_features=128, out_features=10)  # ──> Bx10

    def forward(self, t):
        t = self.layer1(t)
        t = self.layer2(t)
        t = self.layer3(t)
        t = self.fc1(t.reshape(t.size(0), -1))
        t = self.fc2(t)
        t = self.out(t)

        return t
