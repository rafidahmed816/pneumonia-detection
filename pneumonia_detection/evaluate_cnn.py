import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from pneumonia_detection.dataset import get_dataloaders
from pneumonia_detection.model import PneumoniaCNN
from pneumonia_detection.config import MODEL_DIR

def evaluate_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test set
    _, _, test_loader = get_dataloaders()

    # Load trained model
    model_path = MODEL_DIR / "best_cnn_model.pth"
    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            preds = (outputs > 0.5).float()

            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())

    # Convert to numpy arrays
    all_labels = np.array(all_labels, dtype=int)
    all_preds = np.array(all_preds, dtype=int)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"]))

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["NORMAL", "PNEUMONIA"],
                yticklabels=["NORMAL", "PNEUMONIA"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("CNN Confusion Matrix on Test Set")
    plt.show()

if __name__ == "__main__":
    evaluate_cnn()
