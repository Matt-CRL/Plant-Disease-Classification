from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import datasets

from src.utils.io import save_json
from src.utils.plots import save_confusion_matrix


def extract_color_histogram(image_path: str, bins: int = 16, image_size: int = 128) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    arr = np.array(img)

    features = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0, 255), density=True)
        features.extend(hist.tolist())

    return np.array(features, dtype=np.float32)


def build_random_split_indices(total_size: int, seed: int = 42):
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_size, generator=generator).tolist()

    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/PlantVillage")
    parser.add_argument("--max_train_samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = datasets.ImageFolder(args.data_dir)
    classes = dataset.classes
    targets = dataset.targets
    image_paths = [p for p, _ in dataset.imgs]

    train_idx, val_idx, _ = build_random_split_indices(len(dataset), seed=args.seed)

    train_idx = train_idx[: args.max_train_samples]

    print("Extracting non-DL baseline features...")
    X_train = np.array([extract_color_histogram(image_paths[i]) for i in train_idx])
    y_train = np.array([targets[i] for i in train_idx])

    X_val = np.array([extract_color_histogram(image_paths[i]) for i in val_idx])
    y_val = np.array([targets[i] for i in val_idx])

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred) * 100
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    cm = confusion_matrix(y_val, y_pred)

    print(f"Non-DL Baseline Accuracy: {acc:.2f}%")
    print(f"Non-DL Baseline Macro-F1: {macro_f1:.4f}")

    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(
        cm,
        classes,
        str(results_dir / "non_dl_baseline_confusion_matrix.png"),
        "Non-DL Baseline Confusion Matrix",
    )

    save_json(
        {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "max_train_samples": args.max_train_samples,
        },
        results_dir / "non_dl_baseline_metrics.json",
    )


if __name__ == "__main__":
    main()