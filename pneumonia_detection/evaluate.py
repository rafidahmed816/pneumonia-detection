"""
Pneumonia Detection — Evaluation & Comparison (CNN + ResNet, with MCC)

Evaluate trained models on the TEST set:
  • --mode noaug   -> models/best_cnn_model_noaug.pth   (CNN, grayscale)
  • --mode aug     -> models/best_cnn_model_aug.pth     (CNN, grayscale)
  • --mode both    -> compares CNN noaug vs CNN aug
  • --mode resnet  -> models/best_resnet18_aug.pth      (ResNet18, RGB/ImageNet)
  • --mode compare -> compares ResNet18 (aug) vs CNN (aug)
Or pass an explicit --model_path and --backbone {cnn|resnet}.


#First evaluate:
pneumonia_detection/evaluate.py

# CNN no-aug
python -m pneumonia_detection.evaluate --mode noaug --plot

# CNN aug
python -m pneumonia_detection.evaluate --mode aug --plot

# Compare CNN no-aug vs aug
python -m pneumonia_detection.evaluate --mode both --plot

# ResNet18 (aug only)
python -m pneumonia_detection.evaluate --mode resnet --plot

# Compare ResNet18 (aug) vs CNN (aug)
python -m pneumonia_detection.evaluate --mode compare --plot

# Evaluate any checkpoint (specify backbone to pick the right loader)
python -m pneumonia_detection.evaluate --model_path models/my.pth --backbone resnet --plot

All runs use backbone-appropriate *clean* test transforms (no augmentation) so results are comparable.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)

# Optional plotting
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

# ---------------------- Model imports ---------------------- #
# CNN (flat or under CNN/ package)
try:
    from pneumonia_detection.CNN.model import PneumoniaCNN  # if you structured it that way
except Exception:
    from pneumonia_detection.model import PneumoniaCNN      # fallback to flat layout

# ResNet wrapper (you created in pneumonia_detection/resnet.py)
try:
    from pneumonia_detection.resnet import ResNet18Binary
    _HAS_RESNET = True
except Exception:
    ResNet18Binary = None
    _HAS_RESNET = False

from pneumonia_detection.config import MODEL_DIR

# ---------------------- Loader imports ---------------------- #
# CNN eval loader (grayscale pipeline)
try:
    from pneumonia_detection.dataset import get_dataloaders_cnn_no_aug as _cnn_eval_loaders
    _HAS_CNN_EVAL = True
except Exception:
    _HAS_CNN_EVAL = False

# ResNet eval loader (RGB/ImageNet pipeline)
try:
    from pneumonia_detection.dataset import get_dataloaders_resnet_no_aug as _resnet_eval_loaders
    _HAS_RESNET_EVAL = True
except Exception:
    _HAS_RESNET_EVAL = False


@dataclass
class EvalPlan:
    title: str
    model_path: Path
    backbone: str  # "cnn" or "resnet"
    threshold: float


# ---------------------- helpers ---------------------- #
def _load_test_loader(backbone: str):
    """
    Returns the test loader appropriate for the backbone:
      - cnn     -> grayscale eval transforms
      - resnet  -> RGB ImageNet eval transforms
    """
    if backbone == "resnet":
        if not _HAS_RESNET_EVAL:
            print("[WARN] ResNet eval loader not found. Falling back to CNN eval loader.")
            return _load_test_loader("cnn")
        try:
            _, _, test_loader = _resnet_eval_loaders()
        except TypeError:
            _, _, test_loader = _resnet_eval_loaders()
        return test_loader
    else:
        if not _HAS_CNN_EVAL:
            raise RuntimeError("CNN eval loader not available in dataset.py")
        try:
            _, _, test_loader = _cnn_eval_loaders()
        except TypeError:
            _, _, test_loader = _cnn_eval_loaders()
        return test_loader


def _load_model(model_path: Path, backbone: str, device: torch.device) -> torch.nn.Module:
    if backbone == "resnet":
        if not _HAS_RESNET or ResNet18Binary is None:
            raise RuntimeError("ResNet18Binary not available. Did you create pneumonia_detection/resnet.py?")
        model = ResNet18Binary(pretrained=False).to(device)  # pretrained flag not needed for eval
    else:
        model = PneumoniaCNN().to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _evaluate_one(plan: EvalPlan, device: torch.device, plot_cm: bool) -> Dict[str, Any]:
    """Run one evaluation and return metrics + confusion matrix."""
    test_loader = _load_test_loader(plan.backbone)
    model = _load_model(plan.model_path, plan.backbone, device)

    all_labels: List[float] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1)  # [N] regardless of [N] or [N,1]
            outputs = model(images)                      # sigmoid probs, [N,1]
            probs = outputs.squeeze(1).detach().cpu().numpy()  # [N]
            all_probs.extend(probs)
            all_labels.extend(labels.detach().cpu().numpy())

    y_true = np.asarray(all_labels, dtype=int)
    y_pred = (np.asarray(all_probs) >= plan.threshold).astype(int)

    # Metrics
    acc  = accuracy_score(y_true, y_pred)
    f1w  = f1_score(y_true, y_pred, average="weighted")
    mcc  = matthews_corrcoef(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    report_txt  = classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"])
    report_dict = classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # Print
    print(f"\n=== {plan.title} ===")
    print(f"Model: {plan.model_path}")
    print(f"Backbone: {plan.backbone} | Threshold: {plan.threshold}")
    print("\nClassification Report:")
    print(report_txt)
    print(f"Overall accuracy: {acc:.4f} | Weighted F1: {f1w:.4f} | MCC: {mcc:.4f} | Balanced Acc: {bacc:.4f}")
    print("Confusion Matrix [rows=True labels, cols=Pred]:")
    print(cm)

    # Plot
    if plot_cm and _HAS_SNS:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["NORMAL", "PNEUMONIA"], yticklabels=["NORMAL", "PNEUMONIA"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix — {plan.title}")
        plt.tight_layout()
        plt.show()
    elif plot_cm and not _HAS_SNS:
        print("\n(seaborn not available; install it or rerun without --plot)")

    return {
        "title": plan.title,
        "model_path": str(plan.model_path),
        "backbone": plan.backbone,
        "threshold": plan.threshold,
        "accuracy": acc,
        "f1_weighted": f1w,
        "mcc": mcc,
        "balanced_accuracy": bacc,
        "report": report_dict,
        "cm": cm,
    }


def _print_comparison(results: List[Dict[str, Any]]) -> None:
    """Pretty scoreboard across multiple runs."""
    print("\n================ Comparison (Test) ================")
    print(f"{'Model':<24} {'Backbone':<8} {'Acc':>7} {'F1w':>8} {'MCC':>8} {'BalAcc':>8} "
          f"{'NORMAL R':>10} {'PNEUM R':>10}")
    print("-" * 100)
    for m in results:
        rep = m["report"]
        normal_recall = rep["NORMAL"]["recall"]
        pneumonia_recall = rep["PNEUMONIA"]["recall"]
        print(f"{m['title']:<24} {m['backbone']:<8} {m['accuracy']:>7.4f} {m['f1_weighted']:>8.4f} "
              f"{m['mcc']:>8.4f} {m['balanced_accuracy']:>8.4f} "
              f"{normal_recall:>10.2f} {pneumonia_recall:>10.2f}")
    print("=" * 100 + "\n")


def _plans_from_mode(mode: str, threshold: float) -> List[EvalPlan]:
    """
    Build evaluation plans for the selected mode.
    """
    plans: List[EvalPlan] = []
    if mode == "noaug":
        plans.append(EvalPlan("CNN (no aug)", MODEL_DIR / "best_cnn_model_noaug.pth", "cnn", threshold))
    elif mode == "aug":
        plans.append(EvalPlan("CNN (with aug)", MODEL_DIR / "best_cnn_model_aug.pth", "cnn", threshold))
    elif mode == "both":
        plans.append(EvalPlan("CNN (no aug)",  MODEL_DIR / "best_cnn_model_noaug.pth", "cnn", threshold))
        plans.append(EvalPlan("CNN (with aug)", MODEL_DIR / "best_cnn_model_aug.pth",   "cnn", threshold))
    elif mode == "resnet":
        plans.append(EvalPlan("ResNet18 (with aug)", MODEL_DIR / "best_resnet18_aug.pth", "resnet", threshold))
    elif mode == "compare":
        # Compare ResNet18 (aug) vs CNN (aug)
        plans.append(EvalPlan("CNN (with aug)",     MODEL_DIR / "best_cnn_model_aug.pth",    "cnn", threshold))
        plans.append(EvalPlan("ResNet18 (with aug)", MODEL_DIR / "best_resnet18_aug.pth", "resnet", threshold))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return plans


# ---------------------- CLI ---------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["noaug", "aug", "both", "resnet", "compare"],
                    default="both",
                    help="Which trained model(s) to evaluate.")
    ap.add_argument("--model_path", default=None,
                    help="Explicit path to a single .pth file (overrides --mode).")
    ap.add_argument("--backbone", choices=["cnn", "resnet"], default="cnn",
                    help="Backbone to use with --model_path (selects proper test transforms).")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Decision threshold for pneumonia (>= thr → PNEUMONIA).")
    ap.add_argument("--plot", action="store_true",
                    help="Show confusion matrix heatmap(s).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Single explicit checkpoint path
    if args.model_path:
        mp = Path(args.model_path)
        if not mp.exists():
            raise FileNotFoundError(f"Model file not found: {mp}")
        plan = EvalPlan("Single Model", mp, args.backbone, args.threshold)
        res = _evaluate_one(plan, device, args.plot)
        return

    # Plans by mode
    plans = _plans_from_mode(args.mode, args.threshold)
    # sanity check files exist
    for p in plans:
        if not p.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {p.model_path}")

    results = []
    for p in plans:
        results.append(_evaluate_one(p, device, args.plot))

    if len(results) >= 2:
        _print_comparison(results)


if __name__ == "__main__":
    main()
