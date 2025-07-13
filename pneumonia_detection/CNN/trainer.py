# pneumonia_detection/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path().resolve().parent.parent))
from pneumonia_detection.config import LEARNING_RATE, NUM_EPOCHS


def calculate_class_weights(dataset, device):
    labels = [label for _, label in dataset]
    class_counts = Counter(labels)
    total = sum(class_counts.values())
    weights = [total / class_counts[i] for i in range(2)]
    return torch.FloatTensor(weights).to(device)


def get_loss_fn(class_weights):
    return nn.BCELoss(reduction="none"), class_weights


def train(model, loader, optimizer, criterion, class_weights, device):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        weighted_loss = (
            loss * labels * class_weights[1] + loss * (1 - labels) * class_weights[0]
        ).mean()
        weighted_loss.backward()
        optimizer.step()

        total_loss += weighted_loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy


def validate(model, loader, criterion, class_weights, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            weighted_loss = (
                loss * labels * class_weights[1]
                + loss * (1 - labels) * class_weights[0]
            ).mean()

            total_loss += weighted_loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy


def run_training(
    model, train_loader, val_loader, device, save_path="best_cnn_model.pth"
):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    class_weights = calculate_class_weights(train_loader.dataset, device)
    criterion, class_weights = get_loss_fn(class_weights)

    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, class_weights, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, class_weights, device
        )

        scheduler.step(val_acc)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}  "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}  "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"🎯 Best Validation Accuracy: {best_val_acc:.4f}")
