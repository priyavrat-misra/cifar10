import torchvision
import torch.nn as nn

vgg16 = torchvision.models.vgg16(pretrained=False)


class ModifiedVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*list(vgg16.features.children())[:19])
        self.need_train = nn.Sequential(*list(vgg16.features.children())[19:])
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, t):
        t = self.features(t)
        t = self.need_train(t)
        t = t.reshape(t.size(0), -1)
        t = self.classifier(t)

        return t
