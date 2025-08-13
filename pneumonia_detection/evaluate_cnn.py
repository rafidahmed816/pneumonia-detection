"""
🩺 Pneumonia Detection – Test Set Evaluation (with saving)

This script loads the best-trained CNN model (best_cnn_model.pth), evaluates it on the test set,
and SAVES:
    • classification report (CSV)
    • confusion matrix (PNG + CSV)
    • a summary JSON with key metrics & run metadata (timestamp, model path, sizes)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pneumonia_detection.dataset import get_dataloaders
from pneumonia_detection.CNN.model import PneumoniaCNN
from pneumonia_detection.config import MODEL_DIR 
# FYI: loaders use IMAGE_SIZE from config (224) and batch size from config:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}

def evaluate_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Paths (keep results organized under reports/) ----
    project_root = Path(__file__).resolve().parents[1]
    reports_dir = project_root / "reports"
    figures_dir = reports_dir / "figures"
    metrics_dir = reports_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")  # e.g., 20250814-221530

    # ---- Data ----
    _, _, test_loader = get_dataloaders()  
    # ---- Model ----
    model_path = MODEL_DIR / "best_cnn_model.pth"
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)  
            outputs = model(images)  
            all_probs.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # ---- Threshold & preds ----
    threshold = 0.5
    all_labels = np.array(all_labels, dtype=int)
    all_preds = (np.array(all_probs) > threshold).astype(int)

    # ---- Metrics ----
    report_dict = classification_report(
        all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"], output_dict=True
    )
    acc = accuracy_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    # ---- Print to console ----
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"]))
    print(f"Overall accuracy: {acc:.4f} | Weighted F1: {f1_weighted:.4f}")

    # ---- Save: report CSV ----
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = metrics_dir / f"{run_id}_classification_report.csv"
    report_df.to_csv(report_csv_path, index=True)

    # ---- Save: confusion matrix (PNG + CSV) ----
    cm_csv_path = metrics_dir / f"{run_id}_confusion_matrix.csv"
    pd.DataFrame(cm, index=["TRUE_NORMAL", "TRUE_PNEUMONIA"], columns=["PRED_NORMAL", "PRED_PNEUMONIA"]).to_csv(cm_csv_path)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NORMAL", "PNEUMONIA"],
                yticklabels=["NORMAL", "PNEUMONIA"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("CNN Confusion Matrix on Test Set")
    cm_png_path = figures_dir / f"{run_id}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=160)
    plt.close()

    # ---- Save: summary JSON (handy for comparisons later) ----
    summary = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(model_path),
        "threshold": threshold,
        "accuracy": acc,
        "f1_weighted": f1_weighted,
        "report_csv": str(report_csv_path),
        "confusion_matrix_csv": str(cm_csv_path),
        "confusion_matrix_png": str(cm_png_path),
        "notes": "Baseline CNN (no extra augmentation in loaders). To compare, rerun after enabling augmentations."
    }
    summary_path = metrics_dir / f"{run_id}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved:\n- {report_csv_path}\n- {cm_csv_path}\n- {cm_png_path}\n- {summary_path}")

if __name__ == "__main__":
    evaluate_cnn()
