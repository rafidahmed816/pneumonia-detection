from torchvision import transforms

# Constants
IMAGE_SIZE = 150

# ✅ Safer medical augmentation for training
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with 1 channel
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize images to the target size
    transforms.RandomHorizontalFlip(p=0.5),      # Randomly flip images horizontally (50% chance)
    transforms.RandomRotation(degrees=3),         # Slight random rotation for better generalization
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),  # Minor shifts in images
    transforms.ToTensor(),                       # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize to have values between [-1, 1]
])

# ✅ Clean pipeline for validation and testing (no augmentation)
test_val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale for consistency
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize images to the target size
    transforms.ToTensor(),                       # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize to have values between [-1, 1]
])
