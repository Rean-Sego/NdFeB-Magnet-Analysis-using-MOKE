# predict.py

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MOKEDataset
from model import MOKEModel


def main():
    src_dir = Path(__file__).resolve().parent
    ml_dir = src_dir.parent

    model_path = ml_dir / "models" / "best_model.pth"
    predict_csv = ml_dir / "inputs" / "metadata" / "metadata_predict.csv"

    checkpoint = torch.load(model_path, map_location="cpu")

    dataset = MOKEDataset(csv_name="metadata_predict.csv")
    print(predict_csv)
    print(predict_csv.exists())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_images = checkpoint.get("use_images", True)

    model = MOKEModel(
        num_metadata_features=checkpoint["num_metadata_features"],
        num_classes=checkpoint["num_classes"],
        use_images=use_images
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []

    with torch.no_grad():
        for i, (images, metadata, labels) in enumerate(loader):
            images = images.to(device)
            metadata = metadata.to(device)

            outputs = model(images, metadata)
            probabilities = torch.softmax(outputs, dim=1)

            confidence, pred_idx = torch.max(probabilities, dim=1)
            pred_grade = checkpoint["grade_names"][pred_idx.item()]

            uid = dataset.df.iloc[i]["uid"]

            results.append({
                "uid": uid,
                "predicted_grade": pred_grade,
                "confidence": confidence.item()
            })

            print(f"{uid} → {pred_grade} | confidence = {confidence.item():.4f}")

    results_df = pd.DataFrame(results)
    output_path = ml_dir / "outputs" / "predictions.csv"
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved predictions to: {output_path}")


if __name__ == "__main__":
    main()