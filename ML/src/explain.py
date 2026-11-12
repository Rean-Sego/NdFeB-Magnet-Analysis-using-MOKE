# explain.py

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MOKEDataset
from model import MOKEModel


CSV_NAME = "metadata_predict.csv"   # change to "metadata.csv" if you want training-set importance
NUM_REPEATS = 5


def evaluate_accuracy(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, metadata, labels in loader:
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)

            outputs = model(images, metadata)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0


def evaluate_with_shuffled_metadata(model, dataset, feature_index, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()

    all_metadata = []
    all_labels = []
    all_images = []

    for images, metadata, labels in loader:
        all_images.append(images)
        all_metadata.append(metadata)
        all_labels.append(labels)

    all_images = torch.cat(all_images, dim=0)
    all_metadata = torch.cat(all_metadata, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    shuffled_metadata = all_metadata.clone()
    perm = torch.randperm(shuffled_metadata.size(0))
    shuffled_metadata[:, feature_index] = shuffled_metadata[perm, feature_index]

    batch_size = 4
    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(all_labels), batch_size):
            end = start + batch_size

            images = all_images[start:end].to(device)
            metadata = shuffled_metadata[start:end].to(device)
            labels = all_labels[start:end].to(device)

            outputs = model(images, metadata)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0


def evaluate_with_shuffled_images(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()

    all_metadata = []
    all_labels = []
    all_images = []

    for images, metadata, labels in loader:
        all_images.append(images)
        all_metadata.append(metadata)
        all_labels.append(labels)

    all_images = torch.cat(all_images, dim=0)
    all_metadata = torch.cat(all_metadata, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    shuffled_images = all_images.clone()
    perm = torch.randperm(shuffled_images.size(0))
    shuffled_images = shuffled_images[perm]

    batch_size = 4
    correct = 0
    total = 0

    with torch.no_grad():
        for start in range(0, len(all_labels), batch_size):
            end = start + batch_size

            images = shuffled_images[start:end].to(device)
            metadata = all_metadata[start:end].to(device)
            labels = all_labels[start:end].to(device)

            outputs = model(images, metadata)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0


def main():
    src_dir = Path(__file__).resolve().parent
    ml_dir = src_dir.parent

    model_path = ml_dir / "models" / "best_model.pth"
    output_path = ml_dir / "outputs" / "feature_importance.csv"
    output_path.parent.mkdir(exist_ok=True)

    checkpoint = torch.load(model_path, map_location="cpu")

    use_images = checkpoint.get("use_images", True)

    dataset = MOKEDataset(
        csv_name=CSV_NAME,
        use_images=use_images
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MOKEModel(
        num_metadata_features=checkpoint["num_metadata_features"],
        num_classes=checkpoint["num_classes"],
        use_images=use_images
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    baseline_acc = evaluate_accuracy(model, loader, device)

    print(f"Baseline accuracy: {baseline_acc:.4f}")

    results = []

    for feature_index, feature_name in enumerate(dataset.metadata_cols):
        drops = []

        for _ in range(NUM_REPEATS):
            shuffled_acc = evaluate_with_shuffled_metadata(
                model=model,
                dataset=dataset,
                feature_index=feature_index,
                device=device
            )

            drops.append(baseline_acc - shuffled_acc)

        avg_drop = sum(drops) / len(drops)

        results.append({
            "feature": feature_name,
            "group": "metadata",
            "baseline_accuracy": baseline_acc,
            "mean_accuracy_drop": avg_drop,
            "importance_score": avg_drop
        })

        print(f"{feature_name}: accuracy drop = {avg_drop:.4f}")

    if use_images:
        drops = []

        for _ in range(NUM_REPEATS):
            shuffled_acc = evaluate_with_shuffled_images(
                model=model,
                dataset=dataset,
                device=device
            )

            drops.append(baseline_acc - shuffled_acc)

        avg_drop = sum(drops) / len(drops)

        results.append({
            "feature": "image_input_combined",
            "group": "images",
            "baseline_accuracy": baseline_acc,
            "mean_accuracy_drop": avg_drop,
            "importance_score": avg_drop
        })

        print(f"image_input_combined: accuracy drop = {avg_drop:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="importance_score",
        ascending=False
    )

    results_df.to_csv(output_path, index=False)

    print(f"\nSaved feature importance to: {output_path}")


if __name__ == "__main__":
    main()