import torch.nn as nn


class Network(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.features = nn.Sequential(
            *list(pretrained_model.features.children())
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, t):
        t = self.features(t)
        t = t.reshape(t.size(0), -1)
        t = self.classifier(t)

        return t
