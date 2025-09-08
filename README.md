# 🩺 Pneumonia Detection from Chest X-Ray Images

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive deep learning solution for automated pneumonia detection from chest X-ray images using multiple CNN architectures. This project implements and compares four distinct approaches: custom CNN, CNN with augmentation, ResNet18, and DenseNet121.

## 🎯 Project Overview

This project addresses the critical medical imaging challenge of pneumonia detection through automated analysis of chest X-ray images. The system achieves clinically relevant performance with up to **89.10% accuracy** using DenseNet121 architecture.

### Key Features

- **Multiple Architecture Support**: Custom CNN, ResNet18, DenseNet121
- **Class Imbalance Handling**: Weighted sampling and loss functions
- **Medical-Specific Augmentation**: Radiology-appropriate transformations
- **Comprehensive Evaluation**: Multiple metrics including MCC and balanced accuracy
- **Production Ready**: Modular design with configurable parameters

## 📊 Performance Results

| Model               | Accuracy   | Weighted F1 | MCC        | Key Features                       |
| ------------------- | ---------- | ----------- | ---------- | ---------------------------------- |
| **DenseNet121**     | **89.10%** | **88.76%**  | **0.7702** | Dense connections, feature reuse   |
| **CNN (Augmented)** | 86.06%     | 85.94%      | 0.7108     | Custom architecture + augmentation |
| **ResNet18**        | 85.90%     | 85.77%      | 0.7075     | Skip connections, deep learning    |
| **CNN (Baseline)**  | 76.60%     | 75.48%      | 0.5064     | Baseline comparison                |

## 🏗 Architecture Overview

### Dataset Statistics

- **Total Images**: 5,856 chest X-rays
- **Class Distribution**: 73% Pneumonia, 27% Normal
- **Training Set**: 5,216 images
- **Validation Set**: 16 images
- **Test Set**: 624 images

### Technical Approach

- **Class Imbalance Solution**: Weighted random sampling + class-weighted loss functions
- **Data Augmentation**: Medical-specific transformations (rotation, brightness, contrast)
- **Transfer Learning**: Architecture comparison without pre-training for fair evaluation
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, MCC, Balanced Accuracy

## 📂 Project Structure

```
pneumonia-detection/
├── data/chest_xray/           # Kaggle pneumonia dataset
│   ├── train/                 # Training images (5,216)
│   ├── val/                   # Validation images (16)
│   └── test/                  # Test images (624)
├── models/                    # Trained model checkpoints
├── pneumonia_detection/       # Core implementation
│   ├── CNN/                   # Custom CNN implementation
│   ├── resnet/                # ResNet18 binary classifier
│   ├── densenet/              # DenseNet121 binary classifier
│   ├── augmentation/          # Data transformation pipelines
│   ├── dataset.py             # Dataset classes and data loaders
│   ├── evaluate.py            # Model evaluation and metrics
│   └── main.py                # Training orchestration
├── reports/                   # Generated results and visualizations
│   ├── figures/               # Confusion matrices and plots
│   └── metrics/               # Classification reports (CSV/JSON)
├── docs/                      # Documentation
└── notebooks/                 # Jupyter notebooks for analysis
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/rafidahmed816/pneumonia-detection.git
cd pneumonia-detection
```

2. **Set up environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Dataset Setup

Download the [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and extract to:

```
data/chest_xray/
```

## 🎓 Training Models

### Train Custom CNN with Augmentation (Recommended)

```bash
python -m pneumonia_detection.main cnn_aug
```

### Train ResNet18

```bash
python -m pneumonia_detection.main resnet_aug
```

### Train DenseNet121 (Best Performance)

```bash
python -m pneumonia_detection.main densenet_aug
```

### Train Baseline CNN

```bash
python -m pneumonia_detection.main cnn_noaug
```

## � Model Evaluation

### Evaluate All Models

```bash
python -m pneumonia_detection.evaluate
```

### Evaluate Specific Model

```bash
python -m pneumonia_detection.evaluate --model densenet --plot
```

This generates:

- Confusion matrices
- Classification reports
- Performance metrics (JSON/CSV)
- ROC curves and precision-recall plots

## � Technical Highlights

### Class Imbalance Mitigation

- **Weighted Random Sampling**: Balances training batches (Normal: 3.70x, Pneumonia: 1.37x weight)
- **Class-Weighted Loss**: Higher penalties for minority class misclassification
- **Evaluation Metrics**: Focus on balanced accuracy and MCC over simple accuracy

### Medical-Specific Augmentation

- **Rotation**: ±10 degrees (anatomical variation)
- **Brightness/Contrast**: Simulates different X-ray machine settings
- **Horizontal Flip**: Accounts for image orientation differences
- **Normalization**: ImageNet statistics for transfer learning compatibility

### Architecture Comparison

- **Custom CNN**: 25.8M parameters, designed for binary classification
- **ResNet18**: Skip connections for gradient flow, 11.2M parameters
- **DenseNet121**: Dense connections for feature reuse, 7.0M parameters

## 📋 Configuration

Key parameters in `pneumonia_detection/config.py`:

- **Image Size**: 224×224 pixels
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Epochs**: 30
- **Optimizer**: Adam with ReduceLROnPlateau scheduling

## 🧪 Validation Strategy

- **No Pre-training**: Fair comparison across architectures
- **Stratified Splits**: Maintains class distribution
- **Early Stopping**: Based on validation accuracy
- **Multiple Metrics**: Comprehensive performance assessment

## 📖 Clinical Relevance

### High Sensitivity Design

- **96-98% Pneumonia Detection**: Critical for medical screening
- **Balanced Precision**: Reduces false positives (143→71 cases for CNN)
- **Clinical Workflow Integration**: Suitable for diagnostic assistance

### Performance Benchmarks

- **DenseNet121**: Best overall performance (89.10% accuracy)
- **Augmentation Impact**: 9.46% improvement over baseline CNN
- **Class Balance**: Effective handling of 2.70:1 class imbalance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 🙏 Acknowledgments

- **Dataset**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Frameworks**: PyTorch, torchvision, scikit-learn
- **Inspiration**: Medical AI research community

---

⭐ **Star this repository if you find it helpful!**
