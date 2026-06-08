# model.py

import torch
import torch.nn as nn
from torchvision import models


class MOKEModel(nn.Module):
    def __init__(self, num_metadata_features: int, num_classes: int, use_images: bool = True):
        super().__init__()

        self.use_images = use_images

        if self.use_images:
            self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.cnn.fc = nn.Identity()
            image_feature_size = 512
        else:
            self.cnn = None
            image_feature_size = 0

        self.meta_mlp = nn.Sequential(
            nn.Linear(num_metadata_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(image_feature_size + 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, image, metadata):
        meta_features = self.meta_mlp(metadata)

        if self.use_images:
            image_features = self.cnn(image)
            combined = torch.cat([image_features, meta_features], dim=1)
        else:
            combined = meta_features

        return self.final(combined)