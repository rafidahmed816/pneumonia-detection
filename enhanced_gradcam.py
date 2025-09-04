"""
Enhanced GradCAM visualization with side-by-side comparison
Usage: python enhanced_gradcam.py --model cnn --samples 3
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import cv2
import random

from pneumonia_detection.CNN.model import PneumoniaCNN
from pneumonia_detection.resnet.resnet import ResNet18Binary
from pneumonia_detection.densenet.densenet import DenseNet121Binary
from pneumonia_detection.supcon.model import SupConModel
from pneumonia_detection.dataset import ChestXRayDataset
from pneumonia_detection.augmentation.transformations import (
    build_eval_transform,
    build_resnet_transform,
    build_densenet_transform,
)


class EnhancedGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image):
        self.gradients = None
        self.activations = None

        input_image.requires_grad_(True)
        self.model.eval()

        # Forward pass
        output = self.model(input_image)
        if isinstance(output, tuple):
            output = output[0]

        # Use raw logit for better gradients
        target_score = output[0, 0] if output.dim() > 1 else output[0]

        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            print("Warning: No gradients captured!")
            h, w = input_image.shape[-2:]
            return np.ones((h, w)) * 0.5, torch.sigmoid(output).item()

        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Compute importance weights
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Weighted sum of activation maps
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU and normalize
        cam = torch.relu(cam)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = torch.ones_like(cam) * 0.5

        return cam.cpu().numpy(), torch.sigmoid(output).item()


def create_enhanced_visualization(model_name: str, n_samples: int = 3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    models_dir = Path("models")
    if model_name == "cnn":
        model_path = models_dir / "best_cnn_model_aug.pth"
        model = PneumoniaCNN().to(device)
        transform = build_eval_transform()
        target_layer = model.conv_block3[0]
    elif model_name == "resnet":
        model_path = models_dir / "best_resnet_model_aug.pth"
        model = ResNet18Binary(pretrained=False).to(device)
        transform = build_resnet_transform()
        target_layer = model.resnet.layer4[-1].conv2
    elif model_name == "densenet":
        model_path = models_dir / "best_densenet_model_aug.pth"
        model = DenseNet121Binary(pretrained=False).to(device)
        transform = build_densenet_transform()
        target_layer = model.densenet.features.denseblock4
    elif model_name == "supcon":
        model_path = models_dir / "best_supcon_model_aug.pth"
        model = SupConModel(backbone="resnet18", feat_dim=256, num_classes=1).to(device)
        transform = build_resnet_transform()
        target_layer = model.encoder.layer4[-1].conv2

    # Load weights
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Get test dataset
    test_dataset = ChestXRayDataset(split="test", transform=None)

    # Select mixed samples
    normal_indices = []
    pneumonia_indices = []

    for i in range(min(500, len(test_dataset))):  # Check first 500 to speed up
        _, label = test_dataset[i]
        if label.item() == 0:
            normal_indices.append(i)
        else:
            pneumonia_indices.append(i)

    random.seed(42)
    n_normal = n_samples // 2
    n_pneumonia = n_samples - n_normal

    selected_normal = random.sample(normal_indices, min(n_normal, len(normal_indices)))
    selected_pneumonia = random.sample(
        pneumonia_indices, min(n_pneumonia, len(pneumonia_indices))
    )

    selected_indices = selected_normal + selected_pneumonia
    random.shuffle(selected_indices)

    # Create GradCAM
    gradcam = EnhancedGradCAM(model, target_layer)

    # Create visualization with 3 columns: Original, Heatmap, Overlay
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(selected_indices):
        original_image, true_label = test_dataset[idx]
        true_label_str = "PNEUMONIA" if true_label.item() == 1 else "NORMAL"

        # Transform image
        input_tensor = transform(original_image).unsqueeze(0).to(device)

        # Generate CAM
        try:
            cam, pred_prob = gradcam.generate_cam(input_tensor)
            print(
                f"Sample {i+1}: CAM range [{cam.min():.3f}, {cam.max():.3f}], mean={cam.mean():.3f}"
            )
        except Exception as e:
            print(f"Error generating CAM: {e}")
            cam = np.ones((224, 224)) * 0.5
            pred_prob = 0.5

        pred_label_str = "PNEUMONIA" if pred_prob > 0.5 else "NORMAL"
        confidence = pred_prob if pred_prob > 0.5 else (1 - pred_prob)

        # Prepare display image
        with torch.no_grad():
            display_img = input_tensor[0].detach().cpu().numpy()
            if display_img.shape[0] == 3:
                display_img = np.transpose(display_img, (1, 2, 0))
                display_img = np.dot(display_img[..., :3], [0.299, 0.587, 0.114])
            else:
                display_img = display_img[0]

            display_img = (display_img - display_img.min()) / (
                display_img.max() - display_img.min()
            )

        # Resize CAM
        if cam.shape != display_img.shape:
            cam = cv2.resize(cam, (display_img.shape[1], display_img.shape[0]))

        # Column 1: Original image
        axes[i, 0].imshow(display_img, cmap="gray")
        axes[i, 0].set_title(f"Original\nTrue: {true_label_str}", fontsize=10)
        axes[i, 0].axis("off")

        # Column 2: Heatmap only
        axes[i, 1].imshow(cam, cmap="jet")
        axes[i, 1].set_title(f"Activation Heatmap\n(Focus regions)", fontsize=10)
        axes[i, 1].axis("off")

        # Column 3: Superimposed image
        # Enhanced heatmap processing
        cam_enhanced = cv2.equalizeHist(np.uint8(255 * cam)) / 255.0
        threshold = np.percentile(cam_enhanced, 60)
        cam_mask = cam_enhanced > threshold

        # Create colored heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_enhanced), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # Create overlay
        img_rgb = np.stack([display_img] * 3, axis=-1)
        alpha = 0.6
        overlay = alpha * heatmap + (1 - alpha) * img_rgb

        # Enhance the important regions
        overlay = np.where(
            np.stack([cam_mask] * 3, axis=-1),
            0.8 * heatmap + 0.2 * img_rgb,
            0.3 * heatmap + 0.7 * img_rgb,
        )

        overlay = np.clip(overlay, 0, 1)

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(
            f"Superimposed\nPred: {pred_label_str} ({confidence:.3f})", fontsize=10
        )
        axes[i, 2].axis("off")

    plt.suptitle(
        f"Enhanced GradCAM Visualization - {model_name.upper()}", fontsize=16, y=0.98
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    save_path = f"enhanced_gradcam_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Enhanced visualization saved as {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced GradCAM with side-by-side visualization"
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet", "densenet", "supcon"],
        default="cnn",
        help="Model to visualize",
    )
    parser.add_argument(
        "--samples", type=int, default=3, help="Number of samples to visualize"
    )

    args = parser.parse_args()
    create_enhanced_visualization(args.model, args.samples)


if __name__ == "__main__":
    main()
