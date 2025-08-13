import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pneumonia_detection.config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE

class ChestXRayDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.split_dir = DATA_DIR / split
        self.transform = transform
        self.images = []
        self.labels = []

        # NORMAL -> 0
        for p in (self.split_dir / "NORMAL").glob("*.jpeg"):
            self.images.append(p)
            self.labels.append(0)

        # PNEUMONIA -> 1
        for p in (self.split_dir / "PNEUMONIA").glob("*.jpeg"):
            self.images.append(p)
            self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # scalar label; cnn_trainer will unsqueeze to (N,1)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_ds = ChestXRayDataset("train", transform)
    val_ds   = ChestXRayDataset("val",   transform)
    test_ds  = ChestXRayDataset("test",  transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader
