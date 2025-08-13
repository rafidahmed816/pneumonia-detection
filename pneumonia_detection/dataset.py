import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pneumonia_detection.config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE
from pneumonia_detection.augmentation.transformations import (
    train_transform, test_val_transform
)

class ChestXRayDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.split = split
        split_dir = DATA_DIR / split
        self.transform = transform
        self.images, self.labels = [], []

        normal_dir = split_dir / "NORMAL"
        pneu_dir   = split_dir / "PNEUMONIA"

        for p in normal_dir.glob("*.jpeg"):
            self.images.append(p)
            self.labels.append(0)

        for p in pneu_dir.glob("*.jpeg"):
            self.images.append(p)
            self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # return scalar label; trainer will unsqueeze to (N,1)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

def get_dataloaders():
    train_ds = ChestXRayDataset(split="train", transform=train_transform)
    val_ds   = ChestXRayDataset(split="val",   transform=test_val_transform)
    test_ds  = ChestXRayDataset(split="test",  transform=test_val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader
