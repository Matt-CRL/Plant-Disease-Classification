from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    seed: int = 42,
    num_workers: int = 0,
):
    full_dataset = datasets.ImageFolder(
        data_dir,
        transform=get_train_transform(image_size),
    )
    classes = full_dataset.classes

    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    # Make val/test use eval transforms
    val_subset.dataset = datasets.ImageFolder(
        data_dir,
        transform=get_eval_transform(image_size),
    )
    test_subset.dataset = datasets.ImageFolder(
        data_dir,
        transform=get_eval_transform(image_size),
    )

    loaders = {
        "train": DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    return loaders, classes