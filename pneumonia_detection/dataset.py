import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from pneumonia_detection.augmentation.transformations import build_resnet_transform
from pathlib import Path
import numpy as np
from pneumonia_detection.augmentation.transformations import (
    build_densenet_train_augment_transform,
    build_densenet_transform,  # <-- ADD THIS IMPORT
)

from pneumonia_detection.config import DATA_DIR, BATCH_SIZE  # IMAGE_SIZE not used here


from pneumonia_detection.augmentation.transformations import (
    build_eval_transform,
    build_train_augment_transform,
    build_resnet_transform,
    build_resnet_train_augment_transform,
)

IMG_PATTERNS = ("*.jpeg", "*.jpg", "*.png")


class ChestXRayDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.split_dir = DATA_DIR / split
        self.transform = transform
        self.images, self.labels = [], []

        # NORMAL -> label 0
        for pat in IMG_PATTERNS:
            for p in (self.split_dir / "NORMAL").glob(pat):
                self.images.append(p)
                self.labels.append(0)

        # PNEUMONIA -> label 1
        for pat in IMG_PATTERNS:
            for p in (self.split_dir / "PNEUMONIA").glob(pat):
                self.images.append(p)
                self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        # keep label as scalar float; trainer will do .unsqueeze(1)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------- CNN without augmentation ----------
def get_dataloaders_cnn_no_aug():
    eval_tf = build_eval_transform()
    train_ds = ChestXRayDataset("train", transform=eval_tf)
    val_ds = ChestXRayDataset("val", transform=eval_tf)
    test_ds = ChestXRayDataset("test", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


# ---------- CNN with augmentation ----------
def get_dataloaders_cnn_aug(use_weighted_sampler: bool = True):
    train_tf = build_train_augment_transform()
    eval_tf = build_eval_transform()

    train_ds = ChestXRayDataset("train", transform=train_tf)
    val_ds = ChestXRayDataset("val", transform=eval_tf)
    test_ds = ChestXRayDataset("test", transform=eval_tf)

    if use_weighted_sampler:
        labels = np.array(train_ds.labels)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


# ---------- ResNet with augmentation ----------


def get_dataloaders_resnet_aug(use_weighted_sampler: bool = True):
    train_tf = build_resnet_train_augment_transform()  # <-- USE THIS FOR TRAIN
    eval_tf = build_resnet_transform()  # <-- USE THIS FOR VAL/TEST

    train_ds = ChestXRayDataset("train", transform=train_tf)
    val_ds = ChestXRayDataset("val", transform=eval_tf)
    test_ds = ChestXRayDataset("test", transform=eval_tf)

    if use_weighted_sampler:
        labels = np.array(train_ds.labels)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


def get_dataloaders_densenet_aug(use_weighted_sampler: bool = True):
    train_tf = build_densenet_train_augment_transform()
    eval_tf = build_densenet_transform()

    train_ds = ChestXRayDataset("train", transform=train_tf)
    val_ds = ChestXRayDataset("val", transform=eval_tf)
    test_ds = ChestXRayDataset("test", transform=eval_tf)

    if use_weighted_sampler:
        labels = np.array(train_ds.labels)
        class_counts = np.bincount(labels)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader

