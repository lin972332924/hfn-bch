# Breast Cancer Pathology Image Classification

Paper Implementation: Achieving Breast Cancer Pathology Image Classification Through Local and Global Approaches

## Overview

This repository implements a hybrid deep learning approach for breast cancer pathology image classification that combines both **local (patch-based)** and **global (whole-image)** feature extraction methods. The model leverages the strengths of both approaches to improve classification accuracy for distinguishing between benign and malignant breast cancer pathology images.

## Key Features

- **Local Feature Extraction**: Patch-based analysis using CNN backbones (ResNet) to capture fine-grained local patterns
- **Global Feature Extraction**: Whole-image analysis to capture overall tissue architecture and global patterns
- **Attention Mechanism**: Adaptive weighting of local patches based on their importance
- **Hybrid Fusion**: Intelligent fusion of local and global features for final classification
- **Flexible Architecture**: Support for multiple CNN backbones (ResNet18, ResNet34, ResNet50)
- **Comprehensive Training Pipeline**: Including data augmentation, learning rate scheduling, and early stopping

## Architecture

The model consists of three main components:

1. **Local Feature Extractor**: Extracts features from image patches using a pretrained CNN
2. **Global Feature Extractor**: Extracts features from the entire image using a pretrained CNN
3. **Attention-based Fusion Module**: Combines local and global features with attention weighting

```
Input Image
    ├── Local Branch (Patches) → CNN → Attention Aggregation ┐
    └── Global Branch (Whole)  → CNN ────────────────────────┤
                                                              ├→ Fusion → Classification
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lin972332924/hfn-bch.git
cd hfn-bch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Structure

Organize your data in the following structure:

```
data/
├── benign/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── malignant/
    ├── image1.png
    ├── image2.png
    └── ...
```

## Configuration

Edit `config.yaml` to customize model parameters:

- **Model Parameters**: Patch size, number of patches, backbone architecture
- **Training Parameters**: Batch size, learning rate, number of epochs
- **Data Parameters**: Train/val/test split ratios, augmentation settings
- **Paths**: Data directory, checkpoint directory, results directory

## Usage

### Training

Train the model on your dataset:

```bash
python train.py --config config.yaml
```

The training script will:
- Load and preprocess the data
- Train the model with the specified configuration
- Save checkpoints of the best model
- Log training metrics to TensorBoard

### Evaluation

Evaluate the trained model on test data:

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

This will generate:
- Test metrics (accuracy, precision, recall, F1-score, AUC)
- Confusion matrix visualization
- ROC curve
- Classification report

### Prediction

Make predictions on new images:

```bash
# Single image
python predict.py --config config.yaml --checkpoint checkpoints/best_model.pth --image path/to/image.png

# Batch prediction
python predict.py --config config.yaml --checkpoint checkpoints/best_model.pth --image path/to/images/ --output results.txt
```

## Model Details

### Local Feature Extraction

- Divides input image into N×N grid of patches
- Each patch is processed through a CNN backbone
- Features from all patches are aggregated using attention mechanism
- Allows the model to focus on the most discriminative regions

### Global Feature Extraction

- Processes the entire image through a CNN backbone
- Captures overall tissue architecture and spatial relationships
- Provides context that may be lost in patch-based analysis

### Fusion Strategy

- Concatenates attention-weighted local features with global features
- Passes through fully connected layers with dropout for regularization
- Final classification layer produces probability distribution over classes

## Performance Metrics

The model is evaluated using multiple metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Proportion of positive identifications that are correct
- **Recall**: Proportion of actual positives that are identified correctly
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

## Requirements

- Python >= 3.7
- PyTorch >= 1.12.0
- torchvision >= 0.13.0
- OpenCV >= 4.6.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- Pillow >= 9.0.0
- tqdm >= 4.62.0
- PyYAML >= 6.0

## Project Structure

```
hfn-bch/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py          # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── hybrid_model.py     # Model architecture
│   └── utils/
│       └── __init__.py
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── predict.py                   # Inference script
├── config.yaml                  # Configuration file
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## Citation

If you use this implementation in your research, please cite:

```
Breast Cancer Pathology Image Classification Through Local and Global Approaches
```

## License

This project is open-source and available for research and educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or issues, please open an issue on GitHub.
