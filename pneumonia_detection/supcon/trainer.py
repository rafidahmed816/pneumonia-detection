import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import os
from torchvision import transforms
from typing import List, Dict, Any, Tuple
import time

from pneumonia_detection.supcon.model import SupConModel, SupConLoss
from pneumonia_detection.config import MODEL_DIR


class CrossValidationSupConTrainer:
    def __init__(
        self,
        device,
        learning_rate=0.001,
        temperature=0.05,  # Lower temperature for harder negatives
        epochs_stage1=30,  # Reduced epochs
        epochs_stage2=20,  # Reduced epochs
        save_path="models/best_supcon_model_cv.pth",
        patience=8,  # Reduced patience
        k_folds=5,
        backbone="resnet18",
        feat_dim=128,  # Reduced feature dimension
        min_improvement=0.001,  # Minimum improvement for early stopping
    ):
        self.device = device
        self.temperature = temperature
        self.epochs_stage1 = epochs_stage1
        self.epochs_stage2 = epochs_stage2
        self.save_path = save_path
        self.patience = patience
        self.k_folds = k_folds
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.learning_rate = learning_rate
        self.min_improvement = min_improvement

        # Training history for cross-validation
        self.cv_history = {
            "fold_results": [],
            "avg_metrics": {},
        }

        # Stronger augmentation for contrastive learning
        self.strong_aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomAdjustSharpness(1.5, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _create_augmented_views(self, images):
        """Create two augmented views efficiently"""
        batch_size = images.size(0)
        view1_list = []
        view2_list = []

        for i in range(batch_size):
            img = images[i]
            # Convert to numpy for augmentation
            img_np = img.cpu().permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array(
                [0.485, 0.456, 0.406]
            )
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

            view1 = self.strong_aug(img_np).to(self.device)
            view2 = self.strong_aug(img_np).to(self.device)

            view1_list.append(view1)
            view2_list.append(view2)

        return torch.stack(view1_list), torch.stack(view2_list)

    def _train_single_fold(self, train_loader, val_loader, fold_idx):
        """Train a single fold with improved strategy"""
        print(f"\n=== Fold {fold_idx + 1}/{self.k_folds} ===")

        # Create model for this fold
        model = SupConModel(
            backbone=self.backbone,
            feat_dim=self.feat_dim,
            num_classes=1,
            dropout_rate=0.2,
        ).to(self.device)

        # Loss functions
        contrastive_loss = SupConLoss(temperature=self.temperature)
        pos_weight = torch.tensor([2.0]).to(self.device)  # Reduced class weight
        classification_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Single optimizer for both stages with cosine annealing
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        # Cosine annealing scheduler
        total_epochs = self.epochs_stage1 + self.epochs_stage2
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=self.learning_rate * 0.01
        )

        best_val_loss = float("inf")
        patience_counter = 0
        fold_history = {
            "stage1_losses": [],
            "stage2_losses": [],
            "val_losses": [],
            "val_accuracies": [],
        }

        # Stage 1: Contrastive Learning (shorter)
        print(f"Stage 1: Contrastive Learning ({self.epochs_stage1} epochs)")
        for epoch in range(self.epochs_stage1):
            model.train()
            total_loss = 0.0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                # Create augmented views
                view1, view2 = self._create_augmented_views(images)

                # Get features
                features1 = model(view1, mode="contrastive")
                features2 = model(view2, mode="contrastive")
                features = torch.stack([features1, features2], dim=1)

                # Contrastive loss
                loss = contrastive_loss(features, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=0.5
                )  # Reduced clipping
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            fold_history["stage1_losses"].append(avg_loss)

            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs_stage1}: Loss = {avg_loss:.4f}")

        # Stage 2: Classification (shorter)
        print(f"Stage 2: Classification ({self.epochs_stage2} epochs)")
        for epoch in range(self.epochs_stage2):
            model.train()
            total_loss = 0.0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)

                logits = model.classify(images)
                loss = classification_loss(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            fold_history["stage2_losses"].append(avg_loss)

            # Validation
            val_metrics = self._validate_fold(model, val_loader, classification_loss)
            fold_history["val_losses"].append(val_metrics["loss"])
            fold_history["val_accuracies"].append(val_metrics["accuracy"])

            # Early stopping based on validation loss
            if val_metrics["loss"] < best_val_loss - self.min_improvement:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                # Save best model for this fold
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                print(
                    f"  Epoch {epoch+1}/{self.epochs_stage2}: "
                    f"Loss = {avg_loss:.4f}, Val Acc = {val_metrics['accuracy']:.4f}"
                )

            # Early stopping
            if patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best state for final evaluation
        model.load_state_dict(best_state)
        final_metrics = self._validate_fold(model, val_loader, classification_loss)

        return model, final_metrics, fold_history

    def _validate_fold(self, model, val_loader, classification_loss):
        """Validate model on validation set"""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)

                logits = model.classify(images)
                loss = classification_loss(logits, labels)
                total_loss += loss.item()

                predictions = torch.sigmoid(logits)
                pred_binary = (predictions > 0.5).float()
                all_predictions.extend(pred_binary.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(all_labels, all_predictions)

        return {
            "loss": total_loss / len(val_loader),
            "accuracy": accuracy,
            "f1": f1,
            "mcc": mcc,
        }

    def cross_validate_training(self, full_dataset):
        """Perform k-fold cross-validation"""
        print(f"Starting {self.k_folds}-Fold Cross-Validation")
        print(f"Device: {self.device}")
        print(f"Epochs: Stage1={self.epochs_stage1}, Stage2={self.epochs_stage2}")

        # Get all labels for stratification
        all_labels = [full_dataset[i][1].item() for i in range(len(full_dataset))]

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        fold_results = []
        best_overall_acc = 0.0
        best_model_state = None

        for fold_idx, (train_indices, val_indices) in enumerate(
            skf.split(range(len(full_dataset)), all_labels)
        ):
            # Create fold datasets
            train_subset = Subset(full_dataset, train_indices)
            val_subset = Subset(full_dataset, val_indices)

            # Create data loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=32,  # Smaller batch size for stability
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_subset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
            )

            print(
                f"\nFold {fold_idx + 1}: Train={len(train_subset)}, Val={len(val_subset)}"
            )

            # Train fold
            start_time = time.time()
            model, metrics, history = self._train_single_fold(
                train_loader, val_loader, fold_idx
            )
            fold_time = time.time() - start_time

            # Store results
            fold_result = {
                "fold": fold_idx + 1,
                "train_size": len(train_subset),
                "val_size": len(val_subset),
                "final_metrics": metrics,
                "training_time": fold_time,
                "history": history,
            }
            fold_results.append(fold_result)

            # Save best model across all folds
            if metrics["accuracy"] > best_overall_acc:
                best_overall_acc = metrics["accuracy"]
                best_model_state = model.state_dict().copy()
                print(
                    f"*** New best model from fold {fold_idx + 1} (acc={metrics['accuracy']:.4f}) ***"
                )

            print(
                f"Fold {fold_idx + 1} completed in {fold_time:.1f}s - "
                f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, MCC: {metrics['mcc']:.4f}"
            )

        # Save best model
        if best_model_state is not None:
            final_model = SupConModel(
                backbone=self.backbone, feat_dim=self.feat_dim, num_classes=1
            ).to(self.device)
            final_model.load_state_dict(best_model_state)
            torch.save(best_model_state, self.save_path)
            print(f"\nBest model saved to: {self.save_path}")

        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics(fold_results)
        self.cv_history = {
            "fold_results": fold_results,
            "avg_metrics": avg_metrics,
            "best_accuracy": best_overall_acc,
        }

        self._print_cv_summary(fold_results, avg_metrics)
        self._save_cv_results()

        return final_model, avg_metrics

    def _calculate_average_metrics(self, fold_results):
        """Calculate average metrics across folds"""
        metrics = ["accuracy", "f1", "mcc", "loss"]
        avg_metrics = {}

        for metric in metrics:
            values = [fold["final_metrics"][metric] for fold in fold_results]
            avg_metrics[f"avg_{metric}"] = np.mean(values)
            avg_metrics[f"std_{metric}"] = np.std(values)

        return avg_metrics

    def _print_cv_summary(self, fold_results, avg_metrics):
        """Print cross-validation summary"""
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 80)

        print(
            f"{'Fold':<6} {'Accuracy':<10} {'F1-Score':<10} {'MCC':<10} {'Val Loss':<10} {'Time(s)':<10}"
        )
        print("-" * 70)

        for result in fold_results:
            metrics = result["final_metrics"]
            print(
                f"{result['fold']:<6} {metrics['accuracy']:<10.4f} "
                f"{metrics['f1']:<10.4f} {metrics['mcc']:<10.4f} "
                f"{metrics['loss']:<10.4f} {result['training_time']:<10.1f}"
            )

        print("-" * 70)
        print(
            f"{'Mean':<6} {avg_metrics['avg_accuracy']:<10.4f} "
            f"{avg_metrics['avg_f1']:<10.4f} {avg_metrics['avg_mcc']:<10.4f} "
            f"{avg_metrics['avg_loss']:<10.4f}"
        )
        print(
            f"{'Std':<6} {avg_metrics['std_accuracy']:<10.4f} "
            f"{avg_metrics['std_f1']:<10.4f} {avg_metrics['std_mcc']:<10.4f} "
            f"{avg_metrics['std_loss']:<10.4f}"
        )
        print("=" * 80)

    def _save_cv_results(self):
        """Save cross-validation results"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = Path("reports/cv_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_path = results_dir / f"supcon_cv_{timestamp}.json"
        with open(results_path, "w") as f:
            # Convert numpy types to Python types for JSON serialization
            json_compatible = self._make_json_compatible(self.cv_history)
            json.dump(json_compatible, f, indent=2)

        print(f"Cross-validation results saved to: {results_path}")

    def _make_json_compatible(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_json_compatible(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_compatible(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj


class FastSupConTrainer:
    """Faster single-split trainer with better validation strategy"""

    def __init__(
        self,
        model,
        device,
        learning_rate=0.001,
        temperature=0.05,
        epochs_stage1=25,  # Reduced
        epochs_stage2=15,  # Reduced
        save_path="models/best_supcon_model_fast.pth",
        patience=6,  # Reduced
        min_improvement=0.002,
    ):
        self.model = model.to(device)
        self.device = device
        self.temperature = temperature
        self.epochs_stage1 = epochs_stage1
        self.epochs_stage2 = epochs_stage2
        self.save_path = save_path
        self.patience = patience
        self.min_improvement = min_improvement

        # Loss functions
        self.contrastive_loss = SupConLoss(temperature=temperature)
        pos_weight = torch.tensor([2.0]).to(self.device)
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer and scheduler
        self.optimizer = None  # Will be created in train_fast
        self.scheduler = None

        # Training history
        self.history = {
            "stage1_loss": [],
            "stage2_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }

        self.best_val_acc = 0.0
        self.patience_counter = 0

        # Stronger augmentation for contrastive learning
        self.strong_aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomAdjustSharpness(1.5, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        pos_weight = torch.tensor([2.0]).to(device)
        self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Single optimizer for entire training
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        # Cosine annealing with warm restarts
        total_epochs = epochs_stage1 + epochs_stage2
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=total_epochs // 3, T_mult=1
        )

        self.history = {
            "stage1_losses": [],
            "stage2_losses": [],
            "val_losses": [],
            "val_accuracies": [],
        }

        # Efficient augmentation
        self.strong_aug = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomAffine(degrees=10, translate=(0.08, 0.08)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _create_augmented_views_fast(self, images):
        """Faster augmentation using tensor operations where possible"""
        batch_size = images.size(0)

        # Simple tensor-based augmentations
        view1 = images.clone()
        view2 = images.clone()

        # Random horizontal flip
        flip_mask = torch.rand(batch_size) > 0.5
        view1[flip_mask] = torch.flip(view1[flip_mask], dims=[3])
        view2[flip_mask] = torch.flip(view2[flip_mask], dims=[3])

        # Add noise for variation
        noise1 = torch.randn_like(view1) * 0.01
        noise2 = torch.randn_like(view2) * 0.01
        view1 = torch.clamp(view1 + noise1, 0, 1)
        view2 = torch.clamp(view2 + noise2, 0, 1)

        return view1, view2

    def train_fast(self, train_loader, val_loader):
        """Fast training with efficient strategies"""
        print("=== Fast SupCon Training ===")

        best_val_acc = 0.0
        patience_counter = 0
        best_state = None

        # Stage 1: Contrastive
        print(f"Stage 1: Contrastive Learning ({self.epochs_stage1} epochs)")
        for epoch in range(self.epochs_stage1):
            self.model.train()
            total_loss = 0.0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()

                # Fast augmentation
                view1, view2 = self._create_augmented_views_fast(images)

                features1 = self.model(view1, mode="contrastive")
                features2 = self.model(view2, mode="contrastive")
                features = torch.stack([features1, features2], dim=1)

                loss = self.contrastive_loss(features, labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.history["stage1_losses"].append(avg_loss)

            if epoch % 5 == 0:
                print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        # Stage 2: Classification with validation
        print(f"Stage 2: Classification ({self.epochs_stage2} epochs)")
        for epoch in range(self.epochs_stage2):
            self.model.train()
            total_loss = 0.0
            num_batches = len(train_loader)

            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)

                logits = self.model.classify(images)
                loss = self.classification_loss(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()

                # Progress logging every 25% of batches
                if batch_idx % max(1, num_batches // 4) == 0:
                    print(
                        f"    Batch {batch_idx+1}/{num_batches}: Loss = {loss.item():.4f}"
                    )

            avg_loss = total_loss / len(train_loader)
            self.history["stage2_losses"].append(avg_loss)

            # Validation every epoch in stage 2
            val_metrics = self._validate_fast(val_loader)
            self.history["val_losses"].append(val_metrics["loss"])
            self.history["val_accuracies"].append(val_metrics["accuracy"])

            print(
                f"  Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, "
                f"Val Loss = {val_metrics['loss']:.4f}, "
                f"Val Acc = {val_metrics['accuracy']:.4f}"
            )

            # Early stopping based on validation accuracy
            if val_metrics["accuracy"] > best_val_acc + self.min_improvement:
                best_val_acc = val_metrics["accuracy"]
                best_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"    *** New best accuracy: {best_val_acc:.4f} ***")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

            self.scheduler.step()

        # Save best model
        if best_state is not None:
            torch.save(best_state, self.save_path)
            print(f"\nTraining completed! Best model saved: {self.save_path}")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
        else:
            print("\nWarning: No improvement found during training")

    def _validate_fast(self, val_loader):
        """Fast validation for stage 2"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)

                logits = self.model.classify(images)
                loss = self.classification_loss(logits, labels)
                total_loss += loss.item()

                # Calculate accuracy
                predictions = torch.sigmoid(logits)
                pred_binary = (predictions > 0.5).float()
                correct += (pred_binary == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        return {"loss": total_loss / len(val_loader), "accuracy": accuracy}
