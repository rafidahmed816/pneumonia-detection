import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import os
from torchvision import transforms

from pneumonia_detection.supcon.model import PneumoniaSupConModel, SupConLoss
from pneumonia_detection.config import MODEL_DIR


class SupConTrainer:
    def __init__(
        self,
        model,
        device,
        learning_rate=0.001,
        temperature=0.1,  # Increased temperature for better gradients
        epochs_stage1=100,
        epochs_stage2=50,
        save_path="models/best_supcon_model_aug.pth",
        patience=15,  # Early stopping patience
    ):
        self.model = model.to(device)
        self.device = device
        self.temperature = temperature
        self.epochs_stage1 = epochs_stage1
        self.epochs_stage2 = epochs_stage2
        self.save_path = save_path
        self.patience = patience

        # Loss functions with class weighting for imbalanced data
        pos_weight = torch.tensor([3.0]).to(device)  # Pneumonia is less frequent
        self.contrastive_loss = SupConLoss(temperature=temperature)
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizers with better learning rates
        self.optimizer_stage1 = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.optimizer_stage2 = optim.AdamW(
            self.model.parameters(), lr=learning_rate * 0.5, weight_decay=1e-4
        )

        # Schedulers
        self.scheduler_stage1 = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_stage1, mode="min", factor=0.5, patience=8
        )
        self.scheduler_stage2 = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_stage2, mode="min", factor=0.5, patience=5
        )

        # Training history
        self.history = {
            "stage1_contrastive_loss": [],
            "stage1_val_loss": [],
            "stage2_classification_loss": [],
            "stage2_val_loss": [],
            "stage2_val_accuracy": [],
            "stage2_val_f1": [],
        }

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.early_stop_counter = 0

        # Setup stronger augmentations for contrastive learning
        self.strong_aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomAffine(
                    degrees=20, translate=(0.15, 0.15), scale=(0.85, 1.15)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                ),
                transforms.RandomAdjustSharpness(2.0, p=0.5),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _create_augmented_views(self, images):
        """Create two different augmented views of the same batch"""
        batch_size = images.size(0)
        view1_list = []
        view2_list = []

        for i in range(batch_size):
            img = images[i]
            # Convert back to PIL-friendly format
            img_np = img.cpu().permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array(
                [0.485, 0.456, 0.406]
            )
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            # Create two different augmented views
            view1 = self.strong_aug(img_np).to(self.device)
            view2 = self.strong_aug(img_np).to(self.device)

            view1_list.append(view1)
            view2_list.append(view2)

        return torch.stack(view1_list), torch.stack(view2_list)

    def train_stage1_contrastive(self, train_loader, val_loader):
        """Stage 1: Train with contrastive loss only"""
        print("=== Stage 1: Contrastive Learning ===")

        for epoch in range(self.epochs_stage1):
            self.model.train()
            total_loss = 0.0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                # Create two different augmented views for better contrastive learning
                view1_batch, view2_batch = self._create_augmented_views(images)

                # Get contrastive features for both views
                features1 = self.model(view1_batch, mode="contrastive")
                features2 = self.model(view2_batch, mode="contrastive")

                # Stack features: [bsz, 2, feat_dim]
                features = torch.stack([features1, features2], dim=1)

                # Compute contrastive loss
                loss = self.contrastive_loss(features, labels)

                self.optimizer_stage1.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer_stage1.step()

                total_loss += loss.item()

                if batch_idx % 20 == 0:
                    print(
                        f"Stage 1 Epoch {epoch+1}/{self.epochs_stage1}, "
                        f"Batch {batch_idx}/{len(train_loader)}, "
                        f"Loss: {loss.item():.6f}"
                    )

            avg_train_loss = total_loss / len(train_loader)
            val_loss = self._validate_stage1(val_loader)

            self.history["stage1_contrastive_loss"].append(avg_train_loss)
            self.history["stage1_val_loss"].append(val_loss)

            self.scheduler_stage1.step(val_loss)

            print(
                f"Stage 1 Epoch {epoch+1}/{self.epochs_stage1}: "
                f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Early stopping for stage 1
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print("Stage 1 completed successfully!")

    def _validate_stage1(self, val_loader):
        """Validation for stage 1 (contrastive learning)"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                # Create augmented views for validation too
                view1_batch, view2_batch = self._create_augmented_views(images)

                features1 = self.model(view1_batch, mode="contrastive")
                features2 = self.model(view2_batch, mode="contrastive")
                features = torch.stack([features1, features2], dim=1)

                loss = self.contrastive_loss(features, labels)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train_stage2_classification(self, train_loader, val_loader):
        """Stage 2: Fine-tune with classification loss"""
        print("\n=== Stage 2: Classification Fine-tuning ===")

        # Reset early stopping for stage 2
        self.best_val_acc = 0.0
        self.early_stop_counter = 0

        for epoch in range(self.epochs_stage2):
            self.model.train()
            total_loss = 0.0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)

                # Get classification predictions (logits, not sigmoid)
                logits = self.model.classify(images)  # Use direct classify method

                # Compute classification loss (BCEWithLogitsLoss handles sigmoid internally)
                loss = self.classification_loss(logits, labels)

                self.optimizer_stage2.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer_stage2.step()

                total_loss += loss.item()

                if batch_idx % 20 == 0:
                    print(
                        f"Stage 2 Epoch {epoch+1}/{self.epochs_stage2}, "
                        f"Batch {batch_idx}/{len(train_loader)}, "
                        f"Loss: {loss.item():.6f}"
                    )

            avg_train_loss = total_loss / len(train_loader)
            val_metrics = self._validate_stage2(val_loader)

            self.history["stage2_classification_loss"].append(avg_train_loss)
            self.history["stage2_val_loss"].append(val_metrics["loss"])
            self.history["stage2_val_accuracy"].append(val_metrics["accuracy"])
            self.history["stage2_val_f1"].append(val_metrics["f1"])

            self.scheduler_stage2.step(val_metrics["loss"])

            # Save best model based on validation accuracy (better metric for medical data)
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(
                    f"*** New best model saved with val_acc: {val_metrics['accuracy']:.6f} ***"
                )
            else:
                self.early_stop_counter += 1

            print(
                f"Stage 2 Epoch {epoch+1}/{self.epochs_stage2}: "
                f"Train Loss: {avg_train_loss:.6f}, "
                f'Val Loss: {val_metrics["loss"]:.6f}, '
                f'Val Acc: {val_metrics["accuracy"]:.4f}, '
                f'Val F1: {val_metrics["f1"]:.4f}'
            )

            # Early stopping
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print("Stage 2 completed successfully!")

    def _validate_stage2(self, val_loader):
        """Validation for stage 2 (classification)"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)

                # Get logits (not sigmoid predictions)
                logits = self.model.classify(images)
                loss = self.classification_loss(logits, labels)
                total_loss += loss.item()

                # Apply sigmoid for predictions
                predictions = torch.sigmoid(logits)
                pred_binary = (predictions > 0.5).float()
                all_predictions.extend(pred_binary.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        accuracy = (all_predictions == all_labels).mean()

        # Calculate F1 score manually to avoid sklearn dependency issues
        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()
        tn = ((all_predictions == 0) & (all_labels == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Calculate balanced accuracy (important for imbalanced data)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
        balanced_acc = (sensitivity + specificity) / 2

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": accuracy,
            "f1": f1,
            "balanced_accuracy": balanced_acc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    def full_training(self, train_loader, val_loader):
        """Complete two-stage training with improvements"""
        print("Starting Supervised Contrastive Learning Training...")
        print(f"Device: {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(
            f"Stage 1 epochs: {self.epochs_stage1}, Stage 2 epochs: {self.epochs_stage2}"
        )
        print(f"Early stopping patience: {self.patience}")

        # Stage 1: Contrastive learning
        self.train_stage1_contrastive(train_loader, val_loader)

        # Reset early stopping for stage 2
        self.best_val_acc = 0.0
        self.early_stop_counter = 0

        # Stage 2: Classification fine-tuning
        self.train_stage2_classification(train_loader, val_loader)

        # Save training history and generate reports
        self.save_training_history()
        self.generate_training_plots()

        print(f"\nTraining completed! Best model saved at: {self.save_path}")
        print(f"Best validation accuracy: {self.best_val_acc:.6f}")

    def save_training_history(self):
        """Save training history to JSON and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Create directories if they don't exist
        metrics_dir = Path("reports/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        history_path = metrics_dir / f"supcon_{timestamp}_training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # Save as CSV
        history_df = pd.DataFrame(
            dict([(k, pd.Series(v)) for k, v in self.history.items()])
        )
        csv_path = metrics_dir / f"supcon_{timestamp}_training_history.csv"
        history_df.to_csv(csv_path, index=False)

        print(f"Training history saved to: {history_path}")
        print(f"Training history CSV saved to: {csv_path}")

    def generate_training_plots(self):
        """Generate and save training plots"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        figures_dir = Path("reports/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Plot 1: Stage 1 losses
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(
            self.history["stage1_contrastive_loss"], label="Train Contrastive Loss"
        )
        plt.plot(self.history["stage1_val_loss"], label="Val Contrastive Loss")
        plt.title("Stage 1: Contrastive Learning")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(
            self.history["stage2_classification_loss"],
            label="Train Classification Loss",
        )
        plt.plot(self.history["stage2_val_loss"], label="Val Classification Loss")
        plt.title("Stage 2: Classification Fine-tuning")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            figures_dir / f"supcon_{timestamp}_training_losses.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Plot 2: Stage 2 metrics
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(
            self.history["stage2_val_accuracy"],
            label="Validation Accuracy",
            color="blue",
        )
        plt.title("Stage 2: Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(
            self.history["stage2_val_f1"], label="Validation F1-Score", color="green"
        )
        plt.title("Stage 2: Validation F1-Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1-Score")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            figures_dir / f"supcon_{timestamp}_training_metrics.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"Training plots saved to: {figures_dir}")


def run_supcon_training(
    model,
    train_loader,
    val_loader,
    device,
    save_path="models/best_supcon_model_aug.pth",
):
    """Main training function for SupCon model with improved hyperparameters"""

    trainer = SupConTrainer(
        model=model,
        device=device,
        learning_rate=0.0005,  # Reduced learning rate
        temperature=0.1,  # Better temperature for contrastive learning
        epochs_stage1=80,  # Increased contrastive learning epochs
        epochs_stage2=60,  # Increased fine-tuning epochs
        save_path=save_path,
        patience=20,  # More patience for medical data
    )

    trainer.full_training(train_loader, val_loader)

    return trainer
