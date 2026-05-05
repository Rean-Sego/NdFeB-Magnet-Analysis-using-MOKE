import torch
import torch.nn as nn
from torchvision import models


class MOKEModel(nn.Module):
    def __init__(self, num_metadata_features):
        super().__init__()

        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # outputs 512 features

        self.meta_mlp = nn.Sequential(
            nn.Linear(num_metadata_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(512 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # regression output
        )

    def forward(self, image, metadata):
        image_features = self.cnn(image)
        meta_features = self.meta_mlp(metadata)

        combined = torch.cat([image_features, meta_features], dim=1)

        return self.final(combined)