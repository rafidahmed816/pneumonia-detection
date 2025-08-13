import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from pneumonia_detection.config import LEARNING_RATE, NUM_EPOCHS

def _class_weights_from_dataset(dataset, device):
    counts = Counter(dataset.labels)  # uses dataset.labels list
    total = counts[0] + counts[1]
    weights = torch.tensor([total / counts[0], total / counts[1]], dtype=torch.float32, device=device)
    return weights

def _bce_loss():
    # per-sample BCE so we can reweight
    return nn.BCELoss(reduction="none")

def _epoch(model, loader, device, criterion, class_w, train=True, optimizer=None):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)  # (N,1)

        if train:
            optimizer.zero_grad()

        outputs = model(images)                          # (N,1) sigmoid
        loss = criterion(outputs, labels)                # (N,1)
        # reweight positives/negatives
        weighted = loss * (labels * class_w[1] + (1 - labels) * class_w[0])
        batch_loss = weighted.mean()

        if train:
            batch_loss.backward()
            optimizer.step()

        total_loss += batch_loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()

    return total_loss / len(loader), correct / total

def run_training(model, train_loader, val_loader, device, save_path="best_cnn_model.pth"):
    criterion = _bce_loss()
    class_w = _class_weights_from_dataset(train_loader.dataset, device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        tr_loss, tr_acc = _epoch(model, train_loader, device, criterion, class_w, train=True, optimizer=optimizer)
        va_loss, va_acc = _epoch(model, val_loader,   device, criterion, class_w, train=False)

        scheduler.step(va_acc)
        print(f"Epoch {epoch}/{NUM_EPOCHS}  Train Loss {tr_loss:.4f} Acc {tr_acc:.4f}  "
              f"Val Loss {va_loss:.4f} Acc {va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), save_path)

    print(f"🎯 Best Validation Accuracy: {best_val:.4f}")
