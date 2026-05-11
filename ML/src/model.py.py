# model.py

import torch
import torch.nn as nn
from torchvision import models


class MOKEModel(nn.Module):
    def __init__(self, num_metadata_features: int, num_classes: int):
        super().__init__()

        # Image branch: LC + HC + MOKE as 3-channel input
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # ResNet18 outputs 512 features

        # Metadata branch
        self.meta_mlp = nn.Sequential(
            nn.Linear(num_metadata_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Combined classifier
        self.final = nn.Sequential(
            nn.Linear(512 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, metadata):
        image_features = self.cnn(image)
        meta_features = self.meta_mlp(metadata)

        combined = torch.cat([image_features, meta_features], dim=1)

        return self.final(combined)