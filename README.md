# 🩺 Pneumonia Detection from Chest X-Ray Images

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2024.xxxxx-b31b1b.svg)](https://arxiv.org)

A comprehensive deep learning solution for automated pneumonia detection from chest X-ray images using multiple CNN architectures. This project implements and compares **five distinct approaches**: custom CNN, CNN with augmentation, ResNet18, DenseNet121, and Supervised Contrastive Learning (SupCon).

## 🎯 Project Overview

This project addresses the critical medical imaging challenge of pneumonia detection through automated analysis of chest X-ray images. Our comprehensive comparative study demonstrates that **DenseNet121 achieves the highest overall performance** with **89.10% accuracy**, while **SupCon excels in pneumonia sensitivity** with **99% recall**, making it ideal for screening applications where missing pneumonia cases is critical.

### 🔬 Research Contributions

- **Comprehensive Architecture Comparison**: Five different deep learning approaches evaluated on identical dataset
- **Medical-Specific Data Augmentation**: Radiology-appropriate transformations improving performance by 8.34%
- **Class Imbalance Solutions**: Weighted sampling and loss functions addressing 73%-27% class distribution
- **Clinical Validation**: High pneumonia sensitivity (96-99%) suitable for diagnostic assistance
- **Supervised Contrastive Learning**: Novel two-stage training approach achieving 99% pneumonia sensitivity

### 🏆 Key Features

- **Multiple Architecture Support**: Custom CNN, ResNet18, DenseNet121, SupCon
- **Advanced Training Protocols**: Single-stage and two-stage contrastive learning
- **Class Imbalance Handling**: Weighted sampling and class-weighted loss functions
- **Medical-Specific Augmentation**: Clinically validated transformations
- **Comprehensive Evaluation**: Multiple metrics including MCC and balanced accuracy
- **Production Ready**: Modular design with configurable parameters

## 📊 Performance Results

### Overall Model Comparison

| Model               | Accuracy   | Weighted F1 | MCC        | Balanced Acc | Normal Recall | Pneumonia Recall | Clinical Application           |
| ------------------- | ---------- | ----------- | ---------- | ------------ | ------------- | ---------------- | ------------------------------ |
| **DenseNet121**     | **89.10%** | **88.76%**  | **0.7702** | **86.15%**   | **74%**       | **98%**          | **Balanced screening**         |
| **SupCon**          | 85.58%     | 84.67%      | 0.7043     | 80.94%       | 62%           | **99%**          | **High-sensitivity screening** |
| **ResNet18**        | 85.90%     | 85.26%      | 0.7023     | 82.05%       | 67%           | 97%              | General purpose                |
| **CNN (Augmented)** | 84.94%     | 84.17%      | 0.6823     | 80.77%       | 64%           | 97%              | Resource-constrained           |
| **CNN (Baseline)**  | 76.60%     | 75.48%      | 0.5064     | 70.50%       | 39%           | 99%              | Baseline comparison            |

### Clinical Performance Analysis
#### 🔧 ResNet18 (Reliable Performance)

- **85.90% accuracy** with solid performance
- **97% pneumonia sensitivity** for reliable detection
- **Proven architecture** with residual connections
- **Recommended for**: General-purpose pneumonia detection

#### 🥇 DenseNet121 (Best Overall Performance)

- **89.10% accuracy** with excellent precision-recall balance
- **74% normal recall** reduces false alarms
- **98% pneumonia sensitivity** ensures minimal missed diagnoses
- **60 false positives** - lowest among all models
- **Recommended for**: Balanced clinical screening applications

#### 🎯 SupCon (Best Pneumonia Sensitivity)

- **99% pneumonia recall** - highest sensitivity
- **Only 2 false negatives** across entire test set
- **Ultra-conservative approach** ideal for initial screening
- **88 false positives** require workflow management
- **Recommended for**: High-sensitivity screening where missing pneumonia is critical

### Impact of Data Augmentation

The comparison between CNN with and without augmentation reveals critical improvements:

| Metric              | CNN Baseline | CNN Augmented | Improvement |
| ------------------- | ------------ | ------------- | ----------- |
| **Accuracy**        | 76.60%       | 84.94%        | **+8.34%**  |
| **Weighted F1**     | 75.48%       | 84.17%        | **+8.69%**  |
| **Normal Recall**   | 39%          | 64%           | **+25%**    |
| **False Positives** | 143          | 71            | **-50%**    |

## 🏗 Technical Architecture

### Dataset Characteristics

- **Total Images**: 5,856 chest X-rays from Guangzhou Women and Children's Medical Center
- **Class Distribution**: 4,275 Pneumonia (73%) vs 1,585 Normal (27%)
- **Imbalance Ratio**: 2.70:1 (Pneumonia:Normal)
- **Training Split**: 4,175 images (71.4%)
- **Validation Split**: 1,061 images (18.1%)
- **Test Split**: 624 images (10.7%)

### Class Imbalance Mitigation

Our comprehensive approach addresses the significant class imbalance:

#### 1. Weighted Random Sampling

- **Normal weight**: 3.70× (higher sampling probability)
- **Pneumonia weight**: 1.37× (lower sampling probability)
- **Result**: Balanced training batches

#### 2. Class-Weighted Loss Functions

- **Higher penalties** for misclassified minority class (Normal)
- **Binary Cross-Entropy** with sample-wise weighting
- **Reduces model bias** toward majority class

#### 3. Medical-Specific Data Augmentation

```python
Medical Transformations:
• Rotation: ±5° (anatomical variation)
• Translation: ±4% (positioning differences)
• Horizontal Flip: 50% (image orientation)
• Brightness/Contrast: Simulate equipment variations
• Gaussian Blur: Account for image quality differences
```

### Architecture Specifications

#### Custom CNN Architecture

- **25.8M parameters** optimized for grayscale chest X-rays
- **Progressive channel expansion**: 1→32→64→128
- **Three convolutional blocks** with batch normalization
- **Dropout regularization** (0.5, 0.3) preventing overfitting
- **Binary classification** with sigmoid activation

#### ResNet18 Implementation

- **11.2M parameters** with residual connections
- **Skip connections** solve vanishing gradient problems
- **18 layers deep** enabling complex feature learning
- **ImageNet architecture** adapted for medical imaging
- **Transfer learning compatible** (though trained from scratch)

#### DenseNet121 Implementation

- **7.0M parameters** with dense connectivity
- **Dense connections** preserve all previous layer features
- **Feature concatenation** (not addition) preserves information
- **Efficient parameter usage** through feature reuse
- **Excellent gradient flow** through skip connections

#### SupCon Architecture (Novel Approach)

- **Two-stage training protocol**:
  1. **Stage 1**: Contrastive representation learning (25 epochs)
  2. **Stage 2**: Classification fine-tuning (15 epochs)
- **ResNet18 backbone** with projection head (512→256→128)
- **Temperature parameter**: τ = 0.05 for concentration control
- **Strong augmentation** for contrastive learning
- **Class-balanced fine-tuning** with weighted BCE loss

## 📂 Project Structure

```
pneumonia-detection/
├── data/chest_xray/              # Kaggle pneumonia dataset
│   ├── train/                    # Training images (4,175)
│   ├── val/                      # Validation images (1,061)
│   └── test/                     # Test images (624)
├── models/                       # Trained model checkpoints
│   ├── best_cnn_model.pth        # CNN without augmentation
│   ├── best_cnn_model_aug.pth    # CNN with augmentation
│   ├── best_resnet_model_aug.pth # ResNet18 model
│   ├── best_densenet_model_aug.pth # DenseNet121 model
│   └── best_supcon_model.pth     # SupCon model
├── pneumonia_detection/          # Core implementation
│   ├── CNN/                      # Custom CNN implementation
│   │   ├── model.py              # CNN architecture
│   │   └── cnn_trainer.py        # Training with class weighting
│   ├── resnet/                   # ResNet18 binary classifier
│   │   └── resnet.py             # ResNet implementation
│   ├── densenet/                 # DenseNet121 binary classifier
│   │   └── densenet.py           # DenseNet implementation
│   ├── supcon/                   # Supervised Contrastive Learning
│   │   ├── model.py              # SupCon architecture
│   │   ├── trainer.py            # Two-stage training protocol
│   │   └── loss.py               # Contrastive loss function
│   ├── augmentation/             # Data transformation pipelines
│   │   └── transformations.py    # Medical-specific augmentations
│   ├── dataset.py                # Dataset classes and data loaders
│   ├── evaluate.py               # Model evaluation and metrics
│   ├── main.py                   # Training orchestration
│   └── config.py                 # Configuration parameters
├── reports/                      # Generated results and analysis
│   ├── figures/                  # Confusion matrices and visualizations
│   │   ├── confusion_matrices/   # Model-specific confusion matrices
│   │   ├── performance_plots/    # ROC curves and metrics plots
│   │   └── architecture_diagrams/ # Model architecture visualizations
│   └── metrics/                  # Classification reports (CSV/JSON)
├── docs/                         # Documentation and papers
│   ├── IEEE_Pneumonia_Paper.tex  # Academic paper LaTeX source
│   └── architecture_specs/       # Detailed architecture documentation
└── notebooks/                    # Jupyter notebooks for analysis
    ├── data_exploration.ipynb    # Dataset analysis and visualization
    ├── model_comparison.ipynb    # Performance comparison analysis
    └── supcon_analysis.ipynb     # SupCon approach deep dive
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

### Train Custom CNN with Augmentation

```bash
python -m pneumonia_detection.main cnn_aug
```

### Train ResNet18

```bash
python -m pneumonia_detection.main resnet_aug
```

### Train DenseNet121 (Best Overall Performance)

```bash
python -m pneumonia_detection.main densenet_aug
```

### Train SupCon (Best Pneumonia Sensitivity)

```bash
python -m pneumonia_detection.main supcon_aug
```

### Train Baseline CNN

```bash
python -m pneumonia_detection.main cnn_noaug
```

## 📈 Model Evaluation

### Evaluate All Models

```bash
python -m pneumonia_detection.evaluate --mode compare --plot
```

### Evaluate Specific Model

```bash
python -m pneumonia_detection.evaluate --model densenet --plot
python -m pneumonia_detection.evaluate --model supcon --plot
```

This generates:

- Confusion matrices
- Classification reports
- Performance metrics (JSON/CSV)
- ROC curves and precision-recall plots

## 🔬 Supervised Contrastive Learning (SupCon)

### Novel Two-Stage Training Protocol

#### Stage 1: Contrastive Representation Learning

- **Duration**: 25 epochs
- **Objective**: Learn discriminative feature representations
- **Loss Function**: Supervised contrastive loss with temperature τ=0.05
- **Augmentation**: Strong medical-specific transformations

#### Stage 2: Classification Fine-tuning

- **Duration**: 15 epochs
- **Objective**: Binary pneumonia classification
- **Loss Function**: Weighted Binary Cross-Entropy
- **Frozen Features**: Representations learned in Stage 1

### SupCon Architecture Components

```python
Components:
• Backbone: ResNet18 (512 features)
• Projection Head: 512 → 256 → 128 dimensions
• Classification Head: 128 → 64 → 1 output
• Temperature: τ = 0.05
• Feature Dimension: 128
```

### Clinical Advantages of SupCon

- **99% Pneumonia Sensitivity**: Highest among all models
- **Only 2 False Negatives**: Ultra-conservative for patient safety
- **Robust Feature Learning**: Two-stage protocol ensures quality representations
- **Screening Application**: Ideal for initial pneumonia screening

## 🔧 Technical Highlights

### Class Imbalance Mitigation

- **Weighted Random Sampling**: Balances training batches (Normal: 3.70×, Pneumonia: 1.37× weight)
- **Class-Weighted Loss**: Higher penalties for minority class misclassification
- **Evaluation Metrics**: Focus on balanced accuracy and MCC over simple accuracy

### Medical-Specific Augmentation

- **Rotation**: ±5 degrees (anatomical variation)
- **Translation**: ±4% (positioning differences)
- **Brightness/Contrast**: Simulates different X-ray machine settings
- **Horizontal Flip**: Accounts for image orientation differences
- **Gaussian Blur**: Models varying image quality

### Training Configuration

- **Optimizer**: Adam with learning rate 0.001
- **Batch Size**: 32
- **Maximum Epochs**: 30 (standard models), 40 (SupCon)
- **Early Stopping**: Patience of 5 epochs
- **Hardware**: CUDA-compatible GPU recommended

## 📋 Configuration

Key parameters in `pneumonia_detection/config.py`:

- **Image Size**: 224×224 pixels
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Epochs**: 30
- **Optimizer**: Adam with ReduceLROnPlateau scheduling

## 🧪 Validation Strategy

- **No Pre-training**: Fair comparison across architectures (all trained from scratch)
- **Stratified Splits**: Maintains class distribution across train/val/test
- **Early Stopping**: Based on validation accuracy to prevent overfitting
- **Multiple Metrics**: Comprehensive assessment including MCC and balanced accuracy

## 📖 Clinical Relevance

### High Sensitivity Design

- **96-99% Pneumonia Detection**: Critical for medical screening applications
- **Balanced Precision**: Reduces false positives (143→60 cases for DenseNet121)
- **Clinical Workflow Integration**: Suitable for diagnostic assistance
- **Patient Safety**: SupCon's 99% sensitivity minimizes missed diagnoses

### Performance Benchmarks

- **DenseNet121**: Best overall performance (89.10% accuracy)
- **SupCon**: Best pneumonia sensitivity (99% recall)
- **Augmentation Impact**: 8.34% improvement over baseline CNN
- **Class Balance**: Effective handling of 2.70:1 class imbalance

### Clinical Applications

#### Balanced Screening (DenseNet121)

- **Primary care settings** requiring balanced performance
- **General radiology workflow** integration
- **Cost-effective screening** with minimal false alarms

#### High-Sensitivity Screening (SupCon)

- **Emergency departments** where missing pneumonia is critical
- **Resource-limited settings** requiring conservative diagnosis
- **Initial screening** before radiologist review

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 🙏 Acknowledgments

- **Dataset**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **Medical Center**: Guangzhou Women and Children's Medical Center
- **Frameworks**: PyTorch, torchvision, scikit-learn
- **Inspiration**: Medical AI research community

---

⭐ **Star this repository if you find it helpful!**
