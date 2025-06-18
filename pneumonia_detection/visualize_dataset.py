import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from .config import DATA_DIR, IMAGE_SIZE

def count_images():
    splits = ['train', 'val', 'test']
    stats = {}
    
    for split in splits:
        split_dir = DATA_DIR / split
        normal_count = len(list((split_dir / 'NORMAL').glob('*.jpeg')))
        pneumonia_count = len(list((split_dir / 'PNEUMONIA').glob('*.jpeg')))
        
        stats[split] = {
            'NORMAL': normal_count,
            'PNEUMONIA': pneumonia_count,
            'Total': normal_count + pneumonia_count
        }
    
    return stats

def plot_dataset_statistics(stats):
    splits = list(stats.keys())
    normal_counts = [stats[split]['NORMAL'] for split in splits]
    pneumonia_counts = [stats[split]['PNEUMONIA'] for split in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, normal_counts, width, label='Normal')
    ax.bar(x + width/2, pneumonia_counts, width, label='Pneumonia')
    
    ax.set_ylabel('Number of Images')
    ax.set_title('Dataset Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()
    
    # Add value labels on bars
    for i, v in enumerate(normal_counts):
        ax.text(i - width/2, v, str(v), ha='center', va='bottom')
    for i, v in enumerate(pneumonia_counts):
        ax.text(i + width/2, v, str(v), ha='center', va='bottom')
    
    plt.savefig('reports/figures/dataset_distribution.png')
    plt.close()

def show_sample_images():
    splits = ['train']
    classes = ['NORMAL', 'PNEUMONIA']
    samples_per_class = 3
    
    fig, axes = plt.subplots(len(splits), len(classes) * samples_per_class, 
                            figsize=(15, 5 * len(splits)))
    
    for i, split in enumerate(splits):
        split_dir = DATA_DIR / split
        row_axes = axes if len(splits) == 1 else axes[i]
        
        col = 0
        for class_name in classes:
            class_dir = split_dir / class_name
            image_paths = list(class_dir.glob('*.jpeg'))[:samples_per_class]
            
            for img_path in image_paths:
                # Load and resize image
                img = Image.open(img_path).convert('RGB')
                img = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(img)
                
                # Get image details
                original_img = Image.open(img_path)
                img_details = f'{class_name}\nSize: {original_img.size}\nMode: {original_img.mode}'
                
                # Display image
                row_axes[col].imshow(img)
                row_axes[col].set_title(img_details)
                row_axes[col].axis('off')
                col += 1
    
    plt.tight_layout()
    plt.savefig('reports/figures/sample_images.png')
    plt.close()

def main():
    # Create figures directory if it doesn't exist
    (Path('reports') / 'figures').mkdir(parents=True, exist_ok=True)
    
    # Get and display dataset statistics
    stats = count_images()
    print('\nDataset Statistics:')
    for split, counts in stats.items():
        print(f'\n{split.upper()} set:')
        print(f"Normal images: {counts['NORMAL']}")
        print(f"Pneumonia images: {counts['PNEUMONIA']}")
        print(f"Total images: {counts['Total']}")
    
    # Create visualizations
    plot_dataset_statistics(stats)
    show_sample_images()
    print('\nVisualizations saved in reports/figures/')

if __name__ == '__main__':
    main()