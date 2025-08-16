from torchvision import transforms
from pneumonia_detection.config import IMAGE_SIZE
from torchvision import transforms


# preprocessing for val/test (no-aug)
def build_eval_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


# augmentations for training
def build_train_augment_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomAffine(
                degrees=5,  # tiny rotations
                translate=(0.04, 0.04),  # <= 4% shifts
                scale=(0.99, 1.01),  # light zoom
                shear=(-2, 2),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAdjustSharpness(1.2, p=0.25),
            transforms.RandomAutocontrast(p=0.25),
            transforms.RandomEqualize(p=0.15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


# Modify existing function to handle RGB inputs for ResNet
from torchvision import transforms


def build_resnet_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to 224x224 for ResNet
            transforms.Grayscale(
                num_output_channels=3
            ),  # Convert grayscale to 3 channels (RGB)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet mean/std for ResNet
        ]
    )


def build_resnet_train_augment_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # 3 channels for ResNet
            transforms.RandomAffine(
                degrees=5,
                translate=(0.04, 0.04),
                scale=(0.99, 1.01),
                shear=(-2, 2),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAdjustSharpness(1.2, p=0.25),
            transforms.RandomAutocontrast(p=0.25),
            transforms.RandomEqualize(p=0.15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_densenet_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to 224x224 for DenseNet
            transforms.Grayscale(num_output_channels=3),  # 3 channels for DenseNet
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet mean/std for DenseNet
        ]
    )


# Augmentation for DenseNet with RGB input
def build_densenet_train_augment_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to 224x224 for DenseNet
            transforms.Grayscale(
                num_output_channels=3
            ),  # Convert grayscale to 3 channels (RGB)
            transforms.RandomAffine(
                degrees=5,  # Tiny rotations
                translate=(0.04, 0.04),  # <= 4% shifts
                scale=(0.99, 1.01),  # Light zoom
                shear=(-2, 2),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAdjustSharpness(1.2, p=0.25),
            transforms.RandomAutocontrast(p=0.25),
            transforms.RandomEqualize(p=0.15),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet mean/std for DenseNet
        ]
    )
