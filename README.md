# 🩺 Pneumonia Detection from Chest X-Rays

A deep learning project to automatically classify chest X-ray images as **NORMAL** or **PNEUMONIA** using a custom **CNN**.

We provide **two training modes**:

1. **CNN without augmentation** – baseline model.
2. **CNN with augmentation** – uses radiology-friendly image augmentations to improve generalization.

---

## 📂 Project Structure

```
pneumonia-detection/
│
├── data/                    # Dataset folder
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
│
├── models/                  # Saved trained models (.pth)
│
├── pneumonia_detection/
│   ├── dataset.py           # Dataset & dataloaders
│   ├── transformations.py   # Augmentation & preprocessing
│   ├── main.py             # Training entrypoint
│   ├── trainer.py          # Training loop & loss
│   ├── evaluate.py         # Evaluation script
│   └── model.py            # CNN architecture
│
├── reports/
│   ├── metrics/            # Saved classification reports
│   └── figures/            # Confusion matrices
│
└── requirements.txt        # Python dependencies
```

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/<yourusername>/pneumonia-detection.git
cd pneumonia-detection
```

2. **Create a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🗂 Dataset Setup

Download the dataset from [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and place it like this:

```
data/chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    val/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/
```

---

## 🚀 Training

We support two training routes:

### 1️⃣ Train **without augmentation** (baseline)

```bash
python -m pneumonia_detection.main --mode noaug
```

This will save:

```
models/best_cnn_model_noaug.pth
```

### 2️⃣ Train **with augmentation** (better generalization)

```bash
python -m pneumonia_detection.main --mode aug
```

This will save:

```
models/best_cnn_model_aug.pth
```

Both modes print training progress with:

```
Epoch 1/30  Train Loss ...  Val Loss ...  Acc ...
🎯 Best Validation Accuracy: ...
```

---

## 📊 Evaluation

You can evaluate either model on the **test set**:

### Evaluate baseline model:

```bash
python -m pneumonia_detection.evaluate --mode noaug --plot
```

### Evaluate augmented model:

```bash
python -m pneumonia_detection.evaluate --mode aug --plot
```

### Or specify an explicit model file:

```bash
python -m pneumonia_detection.evaluate --model_path models/best_cnn_model_aug.pth --threshold 0.5 --plot
```

---

## 📈 Example Results

### Baseline CNN (No Augmentation)

```
Overall accuracy: 0.81
Weighted F1: 0.80
NORMAL    – Precision: 0.97 | Recall: 0.52 | F1: 0.67
PNEUMONIA – Precision: 0.77 | Recall: 0.99 | F1: 0.87
```

Confusion matrix shows **high pneumonia recall**, but many NORMAL cases misclassified as PNEUMONIA.

---

### CNN with Augmentation

```
Overall accuracy: 0.88
Weighted F1: 0.88
NORMAL    – Precision: 0.93 | Recall: 0.80 | F1: 0.86
PNEUMONIA – Precision: 0.85 | Recall: 0.94 | F1: 0.89
```

Confusion matrix shows **better balance** between NORMAL and PNEUMONIA predictions.

---

## 🧠 Model Architecture

* Convolutional layers with ReLU activation
* MaxPooling for downsampling
* Dropout for regularization
* Fully connected layer for binary classification
* Sigmoid output for probability of pneumonia

---

## 🛠 Future Improvements

* Add threshold tuning on the validation set for optimal balance
* Try transfer learning with pretrained models (ResNet, DenseNet)
* Use Grad-CAM to visualize important lung regions


