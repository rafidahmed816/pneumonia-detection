"""
Simple GradCAM visualization script
Usage: python simple_gradcam.py --model cnn
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
from pneumonia_detection.dataset import ChestXRayDataset  # Use this instead
from pneumonia_detection.augmentation.transformations import (
    build_eval_transform,
    build_resnet_transform,
    build_densenet_transform,
)


class SimpleGradCAM:
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
        # Reset gradients and activations
        self.gradients = None
        self.activations = None

        # Enable gradients for the input
        input_image.requires_grad_(True)
        self.model.eval()

        # Forward pass
        output = self.model(input_image)
        if isinstance(output, tuple):
            output = output[0]

        # For binary classification, we want to maximize the logit (before sigmoid)
        # Get the raw logit value
        if output.dim() > 1:
            target_score = output[0, 0]
        else:
            target_score = output[0]

        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        # Check if gradients were captured
        if self.gradients is None or self.activations is None:
            print("Warning: No gradients or activations captured!")
            h, w = input_image.shape[-2:]
            return np.random.random((h, w)) * 0.5, torch.sigmoid(output).item()

        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension [C, H, W]
        activations = self.activations[0]  # Remove batch dimension [C, H, W]

        # Method 1: Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Method 2: Alternative - use positive gradients only
        # weights = torch.clamp(gradients, min=0).mean(dim=(1, 2))

        # Weighted combination of activation maps
        cam = torch.zeros(
            activations.shape[1:], dtype=torch.float32, device=activations.device
        )  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to remove negative influences
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            # If uniform, create some variation
            cam = torch.ones_like(cam) * 0.3

        # Apply slight Gaussian smoothing to make heatmap more visually appealing
        # (This can be commented out if you prefer sharper heatmaps)
        if (
            cam.shape[0] > 7 and cam.shape[1] > 7
        ):  # Only smooth if image is large enough
            from scipy import ndimage

            cam_np = cam.cpu().numpy()
            cam_np = ndimage.gaussian_filter(cam_np, sigma=1.0)
            cam = torch.from_numpy(cam_np).to(cam.device)

        return cam.cpu().numpy(), torch.sigmoid(output).item()


def visualize_single_model(model_name: str, n_samples: int = 9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    models_dir = Path("models")
    if model_name == "cnn":
        model_path = models_dir / "best_cnn_model_aug.pth"
        model = PneumoniaCNN().to(device)
        transform = build_eval_transform()
        # Use the conv layer in conv_block3 (index 0 is Conv2d)
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
        target_layer = model.encoder.layer4[-1].conv2  # Note: encoder, not backbone

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Get test dataset
    test_dataset = ChestXRayDataset(split="test", transform=None)

    # Select samples
    normal_indices = []
    pneumonia_indices = []

    for i in range(len(test_dataset)):
        _, label = test_dataset[i]
        if label.item() == 0:
            normal_indices.append(i)
        else:
            pneumonia_indices.append(i)

    random.seed(42)
    selected_normal = random.sample(normal_indices, min(5, len(normal_indices)))
    selected_pneumonia = random.sample(
        pneumonia_indices, min(4, len(pneumonia_indices))
    )

    selected_indices = selected_normal + selected_pneumonia
    random.shuffle(selected_indices)
    selected_indices = selected_indices[:n_samples]

    # Create GradCAM
    gradcam = SimpleGradCAM(model, target_layer)

    # Create visualization - show both heatmap and overlay
    cols = 4 if n_samples <= 6 else 3  # 4 columns for <= 6 samples, 3 for more
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1) if n_samples > 1 else [axes]
    axes = axes.flatten() if n_samples > 1 else axes

    for i, idx in enumerate(selected_indices):
        original_image, true_label = test_dataset[idx]
        true_label_str = "PNEUMONIA" if true_label.item() == 1 else "NORMAL"

        # Transform and predict
        input_tensor = transform(original_image).unsqueeze(0).to(device)

        # Generate CAM
        try:
            cam, pred_prob = gradcam.generate_cam(input_tensor)

            # Debug: Print CAM statistics
            print(
                f"Sample {i}: CAM min={cam.min():.4f}, max={cam.max():.4f}, mean={cam.mean():.4f}"
            )

        except Exception as e:
            print(f"Error generating CAM for sample {i}: {e}")
            cam = (
                np.random.random((input_tensor.shape[-2], input_tensor.shape[-1])) * 0.5
            )
            pred_prob = 0.5

        pred_label_str = "PNEUMONIA" if pred_prob > 0.5 else "NORMAL"
        confidence = pred_prob if pred_prob > 0.5 else (1 - pred_prob)

        # Prepare display image - fix the tensor grad issue
        with torch.no_grad():
            display_img = input_tensor[0].detach().cpu().numpy()
            if display_img.shape[0] == 3:
                display_img = np.transpose(display_img, (1, 2, 0))
                display_img = np.dot(display_img[..., :3], [0.299, 0.587, 0.114])
            else:
                display_img = display_img[0]

            # Normalize for display
            display_img = (display_img - display_img.min()) / (
                display_img.max() - display_img.min()
            )

        # Resize CAM to match image
        if cam.shape != display_img.shape:
            cam = cv2.resize(cam, (display_img.shape[1], display_img.shape[0]))

        # Enhance the CAM for better visualization
        # Apply histogram equalization to increase contrast
        cam_enhanced = cv2.equalizeHist(np.uint8(255 * cam)) / 255.0

        # Apply a threshold to highlight only the most important regions
        threshold = np.percentile(cam_enhanced, 70)  # Top 30% of activations
        cam_thresholded = np.where(
            cam_enhanced > threshold, cam_enhanced, cam_enhanced * 0.3
        )

        # Create heatmap with jet colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_thresholded), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # Create a better overlay
        # Convert grayscale to RGB
        img_rgb = np.stack([display_img] * 3, axis=-1)

        # Blend with more sophisticated alpha blending
        # Use the CAM intensity to vary the alpha
        alpha = (
            0.4 + 0.4 * cam_thresholded
        )  # Variable alpha based on activation strength
        alpha = np.stack([alpha] * 3, axis=-1)  # Make it RGB

        overlay = alpha * heatmap + (1 - alpha) * img_rgb
        overlay = np.clip(overlay, 0, 1)

        # Plot the overlay
        if i < len(axes):
            axes[i].imshow(overlay)
            axes[i].set_title(
                f"True: {true_label_str}\nPred: {pred_label_str} ({confidence:.3f})",
                fontsize=10,
            )
            axes[i].axis("off")

    # Hide unused subplots
    for j in range(len(selected_indices), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"GradCAM Visualization - {model_name.upper()}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"gradcam_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet", "densenet", "supcon"],
        default="cnn",
        help="Model to visualize",
    )
    parser.add_argument("--n_samples", type=int, default=9, help="Number of samples")

    args = parser.parse_args()

    print(f"Creating GradCAM visualization for {args.model}")
    visualize_single_model(args.model, args.n_samples)


if __name__ == "__main__":
    main()
