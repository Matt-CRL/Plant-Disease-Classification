from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data_pipeline import build_dataloaders
from src.models.mobilenetv3_model import build_mobilenetv3_small
from src.models.simple_cnn import SimpleLeafCNN
from src.utils.io import save_json
from src.utils.plots import save_confusion_matrix
from src.utils.seed import set_seed


def build_model(model_name: str, num_classes: int):
    if model_name == "scratch":
        return SimpleLeafCNN(num_classes=num_classes)
    if model_name == "mobilenetv3":
        return build_mobilenetv3_small(num_classes=num_classes, freeze_features=True)
    raise ValueError(f"Unsupported model: {model_name}")


def genus_from_label(label: str) -> str:
    return label.split("_")[0].lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/PlantVillage")
    parser.add_argument("--model_name", type=str, choices=["scratch", "mobilenetv3"], default="mobilenetv3")
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, classes = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=0,
    )

    model = build_model(args.model_name, len(classes))
    checkpoint = torch.load(args.weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    prediction_records = []

    with torch.no_grad():
        for images, labels in loaders[args.split]:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            confs, preds = torch.max(probs, dim=1)

            for i in range(len(labels)):
                pred_idx = int(preds[i].cpu().item())
                true_idx = int(labels[i].cpu().item())
                conf = float(confs[i].cpu().item())

                pred_label = classes[pred_idx]
                true_label = classes[true_idx]

                prediction_records.append({
                    "predicted_idx": pred_idx,
                    "true_idx": true_idx,
                    "predicted_label": pred_label,
                    "true_label": true_label,
                    "confidence": conf,
                    "correct": int(pred_idx == true_idx),
                    "genus_match": int(genus_from_label(pred_label) == genus_from_label(true_label)),
                })

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    print(f"{args.split.upper()} Accuracy: {acc:.2f}%")
    print(f"{args.split.upper()} Macro-F1: {macro_f1:.4f}")

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(
        cm,
        classes,
        str(results_dir / f"{args.model_name}_{args.split}_confusion_matrix.png"),
        f"{args.model_name} {args.split.upper()} Confusion Matrix (Acc: {acc:.2f}%)",
    )

    save_json(
        {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "classes": classes,
        },
        results_dir / f"{args.model_name}_{args.split}_metrics.json",
    )

    save_json(
        prediction_records,
        results_dir / f"{args.model_name}_{args.split}_predictions.json",
    )


if __name__ == "__main__":
    main()