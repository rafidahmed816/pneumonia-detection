# Pneumonia Detection from Chest X-Rays

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A deep learning project for detecting pneumonia from chest X-ray images.


### Installation

1. Clone the repository:
```bash
git clone https://github.com/rafidahmed816/pneumonia-detection.git
cd pneumonia-detection
```


2. Create and activate a virtual environment:

For Windows:
```bash
python -m venv venv
venv\Scripts\activate

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## for cuda:
1st uninstall torch:
pip uninstall torch torchvision torchaudio

then run:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



### Dataset Setup

1. Create the following directory structure:
```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

2. Place your chest X-ray images in the respective folders:
   - NORMAL: Contains normal chest X-ray images
   - PNEUMONIA: Contains pneumonia chest X-ray images

## Running the Project

### 1. Dataset Visualization

To visualize the dataset distribution and sample images:
```bash
python -m pneumonia_detection.visualize_dataset
```

This will generate:
- Dataset statistics in the console
- Distribution plots in `reports/figures/dataset_distribution.png`
- Sample images with details in `reports/figures/sample_images.png`

### 2. Training the Model

To train the pneumonia detection model:
```bash
python -m pneumonia_detection.main
```

The training process will:
- trained model on the training set
- Train for 30 epochs
- Apply data augmentation (random flips and rotations)
- Save the best model in `models/best_model.pth`
- Display training progress with accuracy and loss metrics

## Model Architecture

- trained model
- Modifications:
  - Custom classifier layer
  - Dropout for regularization
  - Binary classification (Normal vs Pneumonia)

## Project Organization
```
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers
├── data               <- Dataset directory
├── docs               <- Documentation directory
├── models             <- Trained and serialized models
├── notebooks          <- Jupyter notebooks
├── pneumonia_detection <- Source code directory
├── references         <- Data dictionaries, manuals, etc.
├── reports            <- Generated analysis reports and figures
├── requirements.txt   <- Project dependencies
└── pyproject.toml     <- Project configuration file
```


