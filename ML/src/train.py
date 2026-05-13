# train.py

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import MOKEDataset
from model import MOKEModel


def main():
    src_dir = Path(__file__).resolve().parent
    ml_dir = src_dir.parent
    models_dir = ml_dir / "models"
    models_dir.mkdir(exist_ok=True)

    dataset = MOKEDataset()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    num_metadata_features = len(dataset.metadata_cols)
    num_classes = len(dataset.grade_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MOKEModel(
        num_metadata_features=num_metadata_features,
        num_classes=num_classes
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float("inf")
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, metadata, labels in train_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images, metadata)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, metadata, labels in val_loader:
                images = images.to(device)
                metadata = metadata.to(device)
                labels = labels.to(device)

                outputs = model(images, metadata)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total if total > 0 else 0

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_path = models_dir / "best_model.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata_cols": dataset.metadata_cols,
                    "grade_names": dataset.grade_names,
                    "num_metadata_features": num_metadata_features,
                    "num_classes": num_classes,
                },
                save_path
            )

            print(f"Saved best model to: {save_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()