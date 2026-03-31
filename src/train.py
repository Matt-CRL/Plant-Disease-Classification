from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from src.data_pipeline import build_dataloaders
from src.models.mobilenetv3_model import build_mobilenetv3_small
from src.models.simple_cnn import SimpleLeafCNN
from src.utils.io import save_json
from src.utils.plots import save_training_curves
from src.utils.seed import set_seed


def build_model(model_name: str, num_classes: int):
    if model_name == "scratch":
        return SimpleLeafCNN(num_classes=num_classes)
    if model_name == "mobilenetv3":
        return build_mobilenetv3_small(num_classes=num_classes, freeze_features=False)
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds) * 100
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/PlantVillage")
    parser.add_argument("--model_name", type=str, choices=["scratch", "mobilenetv3"], default="mobilenetv3")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loaders, classes = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=0,
    )

    model = build_model(args.model_name, len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []

    weights_dir = Path("experiments/logs")
    results_dir = Path("experiments/results")
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_weights_path = weights_dir / f"{args.model_name}_best.pt"

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(loaders["train"])
        val_loss, val_acc, val_f1 = evaluate(model, loaders["val"], device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "model_name": args.model_name,
                    "image_size": args.image_size,
                },
                best_weights_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    save_training_curves(
        train_losses,
        val_losses,
        val_accs,
        str(results_dir / f"{args.model_name}_training_curves.png"),
    )

    save_json(
        {
            "model_name": args.model_name,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "val_macro_f1s": val_f1s,
        },
        results_dir / f"{args.model_name}_training_metrics.json",
    )

    print(f"Best weights saved to: {best_weights_path}")


if __name__ == "__main__":
    main()