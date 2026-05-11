# dataset.py

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MOKEDataset(Dataset):
    def __init__(self):
        super().__init__()

        src_dir = Path(__file__).resolve().parent
        ml_dir = src_dir.parent

        self.csv_path = ml_dir / "inputs" / "metadata" / "metadata.csv"
        self.img_dir = ml_dir / "inputs" / "grain_img"

        self.df = pd.read_csv(self.csv_path)

        self.image_cols = [
            "low_contrast_img",
            "high_contrast_img",
            "moke_img",
        ]

        self.label_col = "target_grade"
        self.ignore_cols = ["uid", *self.image_cols, self.label_col]

        self.metadata_cols = [
            col for col in self.df.columns
            if col not in self.ignore_cols
        ]

        self.grade_names = sorted(self.df[self.label_col].unique())
        self.grade_to_idx = {
            grade: idx for idx, grade in enumerate(self.grade_names)
        }

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        channels = []

        for col in self.image_cols:
            img_path = self.img_dir / row[col]

            img = Image.open(img_path).convert("L")
            img = self.transform(img)  # shape: [1, 224, 224]

            channels.append(img)

        image = torch.cat(channels, dim=0)  # shape: [3, 224, 224]

        metadata_values = row[self.metadata_cols].astype(float).values
        metadata = torch.tensor(metadata_values, dtype=torch.float32)

        label = self.grade_to_idx[row[self.label_col]]
        label = torch.tensor(label, dtype=torch.long)

        return image, metadata, label