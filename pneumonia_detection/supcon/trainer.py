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

from pneumonia_detection.supcon.model import PneumoniaSupConModel, SupConLoss
from pneumonia_detection.config import MODEL_DIR


class SupConTrainer:
    def __init__(
        self,
        model,
        device,
        learning_rate=0.001,
        temperature=0.07,
        epochs_stage1=100,
        epochs_stage2=50,
        save_path="models/best_supcon_model_aug.pth"
    ):
        self.model = model.to(device)
        self.device = device
        self.temperature = temperature
        self.epochs_stage1 = epochs_stage1
        self.epochs_stage2 = epochs_stage2
        self.save_path = save_path
        
        # Loss functions
        self.contrastive_loss = SupConLoss(temperature=temperature)
        self.classification_loss = nn.BCELoss()
        
        # Optimizers
        self.optimizer_stage1 = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.optimizer_stage2 = optim.Adam(self.model.parameters(), lr=learning_rate * 0.1)
        
        # Schedulers
        self.scheduler_stage1 = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_stage1, T_max=epochs_stage1
        )
        self.scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_stage2, T_max=epochs_stage2
        )
        
        # Training history
        self.history = {
            'stage1_contrastive_loss': [],
            'stage1_val_loss': [],
            'stage2_classification_loss': [],
            'stage2_val_loss': [],
            'stage2_val_accuracy': [],
            'stage2_val_f1': []
        }
        
        self.best_val_loss = float('inf')
        
    def train_stage1_contrastive(self, train_loader, val_loader):
        """Stage 1: Train with contrastive loss only"""
        print("=== Stage 1: Contrastive Learning ===")
        
        for epoch in range(self.epochs_stage1):
            self.model.train()
            total_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                
                # Create two augmented views (in practice, we use the same augmented batch)
                # For simplicity, we'll use the batch as both views
                batch_size = images.size(0)
                
                # Get contrastive features
                features = self.model(images, mode='contrastive')
                
                # Create two views by reshaping
                features = features.unsqueeze(1)  # [bsz, 1, feat_dim]
                features = features.repeat(1, 2, 1)  # [bsz, 2, feat_dim] - same view twice
                
                # Compute contrastive loss
                loss = self.contrastive_loss(features, labels)
                
                self.optimizer_stage1.zero_grad()
                loss.backward()
                self.optimizer_stage1.step()
                
                total_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f'Stage 1 Epoch {epoch+1}/{self.epochs_stage1}, '
                          f'Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.6f}')
            
            avg_train_loss = total_loss / len(train_loader)
            val_loss = self._validate_stage1(val_loader)
            
            self.history['stage1_contrastive_loss'].append(avg_train_loss)
            self.history['stage1_val_loss'].append(val_loss)
            
            self.scheduler_stage1.step()
            
            print(f'Stage 1 Epoch {epoch+1}/{self.epochs_stage1}: '
                  f'Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        print("Stage 1 completed successfully!")
    
    def _validate_stage1(self, val_loader):
        """Validation for stage 1 (contrastive learning)"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                
                features = self.model(images, mode='contrastive')
                features = features.unsqueeze(1).repeat(1, 2, 1)
                
                loss = self.contrastive_loss(features, labels)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train_stage2_classification(self, train_loader, val_loader):
        """Stage 2: Fine-tune with classification loss"""
        print("\n=== Stage 2: Classification Fine-tuning ===")
        
        for epoch in range(self.epochs_stage2):
            self.model.train()
            total_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)
                
                # Get classification predictions
                predictions = self.model(images, mode='classify')
                
                # Compute classification loss
                loss = self.classification_loss(predictions, labels)
                
                self.optimizer_stage2.zero_grad()
                loss.backward()
                self.optimizer_stage2.step()
                
                total_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f'Stage 2 Epoch {epoch+1}/{self.epochs_stage2}, '
                          f'Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.6f}')
            
            avg_train_loss = total_loss / len(train_loader)
            val_metrics = self._validate_stage2(val_loader)
            
            self.history['stage2_classification_loss'].append(avg_train_loss)
            self.history['stage2_val_loss'].append(val_metrics['loss'])
            self.history['stage2_val_accuracy'].append(val_metrics['accuracy'])
            self.history['stage2_val_f1'].append(val_metrics['f1'])
            
            self.scheduler_stage2.step()
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), self.save_path)
                print(f"*** New best model saved with val_loss: {val_metrics['loss']:.6f} ***")
            
            print(f'Stage 2 Epoch {epoch+1}/{self.epochs_stage2}: '
                  f'Train Loss: {avg_train_loss:.6f}, '
                  f'Val Loss: {val_metrics["loss"]:.6f}, '
                  f'Val Acc: {val_metrics["accuracy"]:.4f}, '
                  f'Val F1: {val_metrics["f1"]:.4f}')
        
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
                
                predictions = self.model(images, mode='classify')
                loss = self.classification_loss(predictions, labels)
                total_loss += loss.item()
                
                # Collect predictions and labels for metrics
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
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'f1': f1
        }
    
    def full_training(self, train_loader, val_loader):
        """Complete two-stage training"""
        print("Starting Supervised Contrastive Learning Training...")
        print(f"Device: {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Stage 1 epochs: {self.epochs_stage1}, Stage 2 epochs: {self.epochs_stage2}")
        
        # Stage 1: Contrastive learning
        self.train_stage1_contrastive(train_loader, val_loader)
        
        # Stage 2: Classification fine-tuning
        self.train_stage2_classification(train_loader, val_loader)
        
        # Save training history and generate reports
        self.save_training_history()
        self.generate_training_plots()
        
        print(f"\nTraining completed! Best model saved at: {self.save_path}")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
    
    def save_training_history(self):
        """Save training history to JSON and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create directories if they don't exist
        metrics_dir = Path("reports/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        history_path = metrics_dir / f"supcon_{timestamp}_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Save as CSV
        history_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in self.history.items()]))
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
        plt.plot(self.history['stage1_contrastive_loss'], label='Train Contrastive Loss')
        plt.plot(self.history['stage1_val_loss'], label='Val Contrastive Loss')
        plt.title('Stage 1: Contrastive Learning')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['stage2_classification_loss'], label='Train Classification Loss')
        plt.plot(self.history['stage2_val_loss'], label='Val Classification Loss')
        plt.title('Stage 2: Classification Fine-tuning')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(figures_dir / f"supcon_{timestamp}_training_losses.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Stage 2 metrics
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['stage2_val_accuracy'], label='Validation Accuracy', color='blue')
        plt.title('Stage 2: Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['stage2_val_f1'], label='Validation F1-Score', color='green')
        plt.title('Stage 2: Validation F1-Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1-Score')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(figures_dir / f"supcon_{timestamp}_training_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved to: {figures_dir}")


def run_supcon_training(model, train_loader, val_loader, device, save_path="models/best_supcon_model_aug.pth"):
    """Main training function for SupCon model"""
    
    trainer = SupConTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        temperature=0.07,
        epochs_stage1=50,  # Reduced for faster training
        epochs_stage2=30,
        save_path=save_path
    )
    
    trainer.full_training(train_loader, val_loader)
    
    return trainer