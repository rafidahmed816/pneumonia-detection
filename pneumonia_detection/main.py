import argparse
import torch
from pneumonia_detection.CNN.cnn_trainer import run_training
from pneumonia_detection.resnet.resnet import ResNet18Binary
from pneumonia_detection.CNN.model import PneumoniaCNN
from pneumonia_detection.config import MODEL_DIR
from pneumonia_detection.CNN.model import PneumoniaCNN
from pneumonia_detection.densenet.densenet import DenseNet121Binary
from pneumonia_detection.dataset import (
    get_dataloaders_cnn_no_aug,
    get_dataloaders_cnn_aug,
    get_dataloaders_resnet_aug,
    get_dataloaders_densenet_aug,
)


def train_cnn_no_aug(save_path="models/best_cnn_model_noaug.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Mode: CNN without augmentation")
    train_loader, val_loader, _ = get_dataloaders_cnn_no_aug()

    model = PneumoniaCNN().to(device)
    run_training(model, train_loader, val_loader, device, save_path=save_path)
    print(f"Done. Saved: {save_path}")


def train_cnn_with_aug(
    save_path="models/best_cnn_model_aug.pth", use_weighted_sampler=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Mode: CNN with augmentation")
    train_loader, val_loader, _ = get_dataloaders_cnn_aug(
        use_weighted_sampler=use_weighted_sampler
    )

    model = PneumoniaCNN().to(device)
    run_training(model, train_loader, val_loader, device, save_path=save_path)
    print(f"Done. Saved: {save_path}")


def train_resnet_with_aug(
    save_path="models/best_resnet_model_aug.pth", use_weighted_sampler=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Mode: ResNet with augmentation")
    train_loader, val_loader, _ = get_dataloaders_resnet_aug(
        use_weighted_sampler=use_weighted_sampler
    )
    model = ResNet18Binary(pretrained=False).to(device)
    run_training(model, train_loader, val_loader, device, save_path=save_path)
    print(f"Done. Saved: {save_path}")


def train_densenet_with_aug(
    save_path="models/best_densenet_model_aug.pth", use_weighted_sampler=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Mode: DenseNet with augmentation")

    # Get DataLoaders for DenseNet with augmentation
    train_loader, val_loader, _ = get_dataloaders_densenet_aug(
        use_weighted_sampler=use_weighted_sampler
    )
    model = DenseNet121Binary(pretrained=False).to(device)
    run_training(model, train_loader, val_loader, device, save_path=save_path)
    print(f"Done. Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["noaug", "aug", "resnet_aug", "densenet_aug"],
        default="noaug",
    )
    args = parser.parse_args()

    if args.mode == "aug":
        train_cnn_with_aug()
    elif args.mode == "noaug":
        train_cnn_no_aug()
    elif args.mode == "resnet_aug":
        train_resnet_with_aug()  # Train ResNet with augmentation
    elif args.mode == "densenet_aug":
        train_densenet_with_aug()


if __name__ == "__main__":
    main()
