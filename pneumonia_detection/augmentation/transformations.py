from torchvision import transforms
from pneumonia_detection.config import IMAGE_SIZE


#preprocessing for val/test (no-aug)
def build_eval_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


#augmentations for training
def build_train_augment_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.RandomAffine(
            degrees=5,              # tiny rotations
            translate=(0.04, 0.04), # <= 4% shifts
            scale=(0.98, 1.02),     # light zoom
            shear=(-2, 2),
        ),
        transforms.RandomHorizontalFlip(p=0.5),

    
        transforms.RandomAdjustSharpness(1.2, p=0.25),
        transforms.RandomAutocontrast(p=0.25),
        transforms.RandomEqualize(p=0.15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
