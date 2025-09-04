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
    PneumoniaDataset,
)
from pneumonia_detection.supcon.model import SupConModel
from pneumonia_detection.supcon.trainer import CrossValidationSupConTrainer

""" CNN Training without Augmentation """


def train_cnn_no_aug(save_path="models/best_cnn_model_noaug.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Mode: CNN without augmentation")
    train_loader, val_loader, _ = get_dataloaders_cnn_no_aug()

    model = PneumoniaCNN().to(device)
    run_training(model, train_loader, val_loader, device, save_path=save_path)
    print(f"Done. Saved: {save_path}")


""" CNN Training with Augmentation """


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


""" ResNet Training with Augmentation """


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


""" DenseNet Training with Augmentation """


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


""" SupCon Training with Augmentation """


def train_supcon_with_aug(
    save_path="models/best_supcon_model_cv.pth", use_weighted_sampler=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Mode: SupCon with Cross-Validation")

    # Create the full dataset (not data loaders)
    full_dataset = PneumoniaDataset(split="train", transform=None)

    # Create trainer with improved hyperparameters
    trainer = CrossValidationSupConTrainer(
        device=device,
        learning_rate=0.0005,  # Lower learning rate for stability
        temperature=0.05,  # Lower temperature for harder negatives
        epochs_stage1=40,  # More epochs for contrastive learning
        epochs_stage2=30,  # More epochs for fine-tuning
        save_path=save_path,
        patience=10,  # More patience
        k_folds=5,
        backbone="resnet18",
        feat_dim=256,  # Larger feature dimension
        min_improvement=0.001,
    )

    # Use cross-validation training
    model, avg_metrics = trainer.cross_validate_training(full_dataset)
    print(f"Cross-validation completed. Best model saved: {save_path}")
    print(
        f"Average accuracy: {avg_metrics['avg_accuracy']:.4f} ± {avg_metrics['std_accuracy']:.4f}"
    )

    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["noaug", "aug", "resnet_aug", "densenet_aug", "supcon_aug"],
        default="noaug",
    )
    args = parser.parse_args()

    if args.mode == "aug":
        train_cnn_with_aug()
    elif args.mode == "noaug":
        train_cnn_no_aug()
    elif args.mode == "resnet_aug":
        train_resnet_with_aug()
    elif args.mode == "densenet_aug":
        train_densenet_with_aug()
    elif args.mode == "supcon_aug":
        train_supcon_with_aug()


if __name__ == "__main__":
    main()
