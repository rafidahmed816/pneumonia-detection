import argparse
import torch
from pneumonia_detection.CNN.cnn_trainer import run_training
from pneumonia_detection.config import MODEL_DIR
from pneumonia_detection.CNN.model import PneumoniaCNN
from pneumonia_detection.resnet import ResNet18Binary
from pneumonia_detection.dataset import (
    get_dataloaders_cnn_no_aug,
    get_dataloaders_cnn_aug,
    get_dataloaders_resnet_aug,
)

def train_cnn_no_aug(save_path="models/best_cnn_model_noaug.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Mode: CNN without augmentation")
    train_loader, val_loader, _ = get_dataloaders_cnn_no_aug()

    model = PneumoniaCNN().to(device)
    run_training(model, train_loader, val_loader, device, save_path=save_path)
    print(f"Done. Saved: {save_path}")

def train_cnn_with_aug(save_path="models/best_cnn_model_aug.pth", use_weighted_sampler=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Mode: CNN with augmentation")
    train_loader, val_loader, _ = get_dataloaders_cnn_aug(use_weighted_sampler=use_weighted_sampler)

    model = PneumoniaCNN().to(device)
    run_training(model, train_loader, val_loader, device, save_path=save_path)
    print(f"Done. Saved: {save_path}")

def train_resnet_aug(save_path = "models/best_resnet18_aug.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Mode: ResNet18 with augmentation")

    train_loader, val_loader, _ = get_dataloaders_resnet_aug(use_weighted_sampler=True)
    model = ResNet18Binary(pretrained=True).to(device)
    # save_path = MODEL_DIR / "best_resnet18_aug.pth"  
    run_training(model, train_loader, val_loader, device, save_path=str(save_path))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["noaug", "aug", "resnet"], default="noaug")
    args = ap.parse_args()

    if args.mode == "noaug":
        # your existing no-aug CNN
        train_loader, val_loader, _ = get_dataloaders_cnn_no_aug()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device} | Mode: CNN without augmentation")
        model = PneumoniaCNN().to(device)
        save_path = MODEL_DIR / "best_cnn_model_noaug.pth"
        run_training(model, train_loader, val_loader, device, save_path=str(save_path))

    elif args.mode == "aug":
        # your existing aug CNN
        train_loader, val_loader, _ = get_dataloaders_cnn_aug(use_weighted_sampler=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device} | Mode: CNN with augmentation")
        model = PneumoniaCNN().to(device)
        save_path = MODEL_DIR / "best_cnn_model_aug.pth"
        run_training(model, train_loader, val_loader, device, save_path=str(save_path))

    elif args.mode == "resnet":
        train_resnet_aug()

if __name__ == "__main__":
    main()
