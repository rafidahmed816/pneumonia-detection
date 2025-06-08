import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from .config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE

class ChestXRayDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.split_dir = DATA_DIR / split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load NORMAL images (label 0)
        normal_path = self.split_dir / "NORMAL"
        for img_path in normal_path.glob("*.jpeg"):
            self.images.append(img_path)
            self.labels.append(0)
        
        # Load PNEUMONIA images (label 1)
        pneumonia_path = self.split_dir / "PNEUMONIA"
        for img_path in pneumonia_path.glob("*.jpeg"):
            self.images.append(img_path)
            self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloaders():
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Only resize and normalize for validation/test
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = ChestXRayDataset(split="train", transform=train_transform)
    val_dataset = ChestXRayDataset(split="val", transform=eval_transform)
    test_dataset = ChestXRayDataset(split="test", transform=eval_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader, test_loader