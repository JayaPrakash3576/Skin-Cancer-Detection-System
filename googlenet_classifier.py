import torch.nn as nn
from torchvision import models

class GoogLeNetClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(GoogLeNetClassifier, self).__init__()
        self.model = models.googlenet(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
