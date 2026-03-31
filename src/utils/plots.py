from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    val_accs: Sequence[float],
    path: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.plot(epochs, val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_confusion_matrix(cm, class_names, path: str, title: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Greens", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_rl_curve(mean_rewards: np.ndarray, std_rewards: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(mean_rewards))
    plt.figure(figsize=(10, 5))
    plt.plot(x, mean_rewards, label="Mean Reward")
    plt.fill_between(x, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, label="Std Dev")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("RL Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()