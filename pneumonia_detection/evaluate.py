"""
🩺 Pneumonia Detection — Evaluation & Comparison

Evaluate trained CNNs on the TEST set:
  • --mode noaug  -> models/best_cnn_model_noaug.pth
  • --mode aug    -> models/best_cnn_model_aug.pth
  • --mode both   -> evaluates both and prints a side-by-side score table
Or pass an explicit --model_path.

All evaluations use the *same* clean (no-augmentation) test transform so results are comparable.

Examples:
  python -m pneumonia_detection.evaluate --plot
  python -m pneumonia_detection.evaluate --mode noaug --plot
  python -m pneumonia_detection.evaluate --model_path models/best_cnn_model_aug.pth --threshold 0.5 --plot
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import matplotlib.pyplot as plt

# seaborn is optional; used only for --plot
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

from pneumonia_detection.CNN.model import PneumoniaCNN
from pneumonia_detection.config import MODEL_DIR
# We always evaluate with the "no-aug" eval loader for fair comparison
from pneumonia_detection.dataset import get_dataloaders_cnn_no_aug


# ---------------------- helpers ---------------------- #
def _load_test_loader():
    _, _, test_loader = get_dataloaders_cnn_no_aug()
    return test_loader


def _load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    model = PneumoniaCNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _evaluate_one(model_path: Path, threshold: float, device: torch.device,
                  plot_cm: bool, title: str) -> Dict[str, Any]:
    """Run one evaluation and return metrics + confusion matrix."""
    test_loader = _load_test_loader()
    model = _load_model(model_path, device)

    all_labels: list[float] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            # labels may come as shape [N] or [N,1] depending on dataset impl → flatten safely
            labels = labels.to(device).float().view(-1)
            outputs = model(images)                      # sigmoid probs, shape [N,1]
            probs = outputs.squeeze(1).detach().cpu().numpy()  # [N]
            all_probs.extend(probs)
            all_labels.extend(labels.detach().cpu().numpy())

    y_true = np.asarray(all_labels, dtype=int)
    y_pred = (np.asarray(all_probs) >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted")
    report_txt = classification_report(y_true, y_pred,
                                       target_names=["NORMAL", "PNEUMONIA"])
    report_dict = classification_report(y_true, y_pred,
                                        target_names=["NORMAL", "PNEUMONIA"],
                                        output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== {title} ===")
    print(f"Model: {model_path}")
    print(f"Threshold: {threshold}")
    print("\nClassification Report:")
    print(report_txt)
    print(f"Overall accuracy: {acc:.4f} | Weighted F1: {f1w:.4f}")
    print("Confusion Matrix [rows=True labels, cols=Pred]:")
    print(cm)

    if plot_cm and _HAS_SNS:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["NORMAL", "PNEUMONIA"],
                    yticklabels=["NORMAL", "PNEUMONIA"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix — {title}")
        plt.tight_layout()
        plt.show()
    elif plot_cm and not _HAS_SNS:
        print("\n(seaborn not available; install it or rerun without --plot)")

    return {
        "title": title,
        "model_path": str(model_path),
        "threshold": threshold,
        "accuracy": acc,
        "f1_weighted": f1w,
        "report": report_dict,
        "cm": cm,
    }


def _resolve_model_path(mode: str | None, model_path_cli: str | None) -> Tuple[Path | None, Path | None]:
    """
    Returns (noaug_path, aug_path) depending on requested mode or explicit path.
    If model_path_cli is provided, returns (Path(model_path_cli), None) and ignores mode.
    """
    if model_path_cli:
        p = Path(model_path_cli)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        # Explicit path → evaluate only that one
        return p, None

    if mode == "noaug":
        return MODEL_DIR / "best_cnn_model_noaug.pth", None
    if mode == "aug":
        return MODEL_DIR / "best_cnn_model_aug.pth", None
    # both or None → use both defaults
    return MODEL_DIR / "best_cnn_model_noaug.pth", MODEL_DIR / "best_cnn_model_aug.pth"


def _print_comparison(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    """Pretty scoreboard of the two runs."""
    print("\n================ Comparison (Test) ================")
    print(f"{'Model':<18} {'Accuracy':>10} {'F1 (weighted)':>15} "
          f"{'NORMAL R':>10} {'PNEUMONIA R':>13}")
    print("-" * 66)

    for m in (a, b):
        rep = m["report"]
        normal_recall = rep["NORMAL"]["recall"]
        pneumonia_recall = rep["PNEUMONIA"]["recall"]
        print(f"{m['title']:<18} {m['accuracy']:>10.4f} {m['f1_weighted']:>15.4f} "
              f"{normal_recall:>10.2f} {pneumonia_recall:>13.2f}")
    print("=" * 66 + "\n")


# ---------------------- CLI ---------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["noaug", "aug", "both"], default="both",
                    help="Which trained model(s) to evaluate.")
    ap.add_argument("--model_path", default=None,
                    help="Explicit path to a single .pth file (overrides --mode).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold for pneumonia (>= thr → PNEUMONIA).")
    ap.add_argument("--plot", action="store_true",
                    help="Show confusion matrix heatmap(s).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noaug_path, aug_path = _resolve_model_path(args.mode, args.model_path)

    # single-file case (explicit --model_path or single mode)
    if aug_path is None and noaug_path is not None:
        if not Path(noaug_path).exists():
            raise FileNotFoundError(f"Model file not found: {noaug_path}")
        _ = _evaluate_one(Path(noaug_path), args.threshold, device, args.plot, title="Single Model")
        return

    # both case
    results = []
    if noaug_path is not None:
        if not Path(noaug_path).exists():
            raise FileNotFoundError(f"Model file not found: {noaug_path}")
        results.append(_evaluate_one(Path(noaug_path), args.threshold, device, args.plot, title="CNN (no aug)"))
    if aug_path is not None:
        if not Path(aug_path).exists():
            raise FileNotFoundError(f"Model file not found: {aug_path}")
        results.append(_evaluate_one(Path(aug_path), args.threshold, device, args.plot, title="CNN (with aug)"))

    if len(results) == 2:
        _print_comparison(results[0], results[1])


if __name__ == "__main__":
    main()
