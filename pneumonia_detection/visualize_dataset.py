import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from pneumonia_detection.config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE


def count_images():
    splits = ["train", "val", "test"]
    stats = {}

    for split in splits:
        split_dir = DATA_DIR / split
        normal_count = len(list((split_dir / "NORMAL").glob("*.jpeg")))
        pneumonia_count = len(list((split_dir / "PNEUMONIA").glob("*.jpeg")))

        stats[split] = {
            "NORMAL": normal_count,
            "PNEUMONIA": pneumonia_count,
            "Total": normal_count + pneumonia_count,
        }

    return stats


def plot_dataset_statistics(stats):
    splits = list(stats.keys())
    normal_counts = [stats[split]["NORMAL"] for split in splits]
    pneumonia_counts = [stats[split]["PNEUMONIA"] for split in splits]

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, normal_counts, width, label="Normal")
    ax.bar(x + width / 2, pneumonia_counts, width, label="Pneumonia")

    ax.set_ylabel("Number of Images")
    ax.set_title("Dataset Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()

    # Add value labels on bars
    for i, v in enumerate(normal_counts):
        ax.text(i - width / 2, v, str(v), ha="center", va="bottom")
    for i, v in enumerate(pneumonia_counts):
        ax.text(i + width / 2, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.show()  # <-- Show directly in notebook


def show_sample_images():
    splits = ["train"]
    classes = ["NORMAL", "PNEUMONIA"]
    samples_per_class = 3

    fig, axes = plt.subplots(
        len(splits), len(classes) * samples_per_class, figsize=(15, 5 * len(splits))
    )

    for i, split in enumerate(splits):
        split_dir = DATA_DIR / split
        row_axes = axes if len(splits) == 1 else axes[i]

        col = 0
        for class_name in classes:
            class_dir = split_dir / class_name
            image_paths = list(class_dir.glob("*.jpeg"))[:samples_per_class]

            for img_path in image_paths:
                # Load and resize image
                img = Image.open(img_path).convert("RGB")
                img = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(img)

                # Get image details
                original_img = Image.open(img_path)
                img_details = f"{class_name}\nSize: {original_img.size}\nMode: {original_img.mode}"

                # Display image
                row_axes[col].imshow(img)
                row_axes[col].set_title(img_details)
                row_axes[col].axis("off")
                col += 1

    plt.tight_layout()
    plt.show()  # <-- Show directly in notebook


def get_class_distribution(dataset):
    # Assumes dataset.labels exists
    from collections import Counter

    counts = Counter(dataset.labels)
    return dict(counts)


def plot_class_distribution(class_counts, class_names=None):
    if class_names is None:
        class_names = [str(k) for k in class_counts.keys()]
    values = [class_counts[k] for k in class_counts.keys()]
    plt.figure(figsize=(6, 4))
    plt.bar(class_names, values, color=["skyblue", "salmon"])
    plt.ylabel("Number of Images")
    plt.title("Class Distribution on training set")
    for i, v in enumerate(values):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.show()


def show_samples_from_dataset(dataset, class_names=None, samples_per_class=3):
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np

    if class_names is None:
        class_names = ["Normal", "Pneumonia"]
    fig, axes = plt.subplots(1, len(class_names) * samples_per_class, figsize=(15, 5))
    idx = 0
    for class_id, class_name in enumerate(class_names):
        indices = [i for i, label in enumerate(dataset.labels) if label == class_id][
            :samples_per_class
        ]
        for i in indices:
            img_path = dataset.images[i]
            img = Image.open(img_path).convert("RGB")
            axes[idx].imshow(img)
            axes[idx].set_title(class_name)
            axes[idx].axis("off")
            idx += 1
    plt.suptitle("Sample Images from Training Set", fontsize=14)
    plt.tight_layout()
    plt.show()
