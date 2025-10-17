# Quick Start Guide

## Installation

1. Clone the repository:
```bash
git clone https://github.com/lin972332924/hfn-bch.git
cd hfn-bch
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

## Data Preparation

Organize your breast cancer pathology images as follows:

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

Edit `config.yaml` to customize:
- Model parameters (patch size, backbone)
- Training hyperparameters (batch size, learning rate)
- Data splits (train/val/test ratios)
- Paths (data directory, checkpoint directory)

## Training

Train the model:
```bash
python train.py --config config.yaml
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs/
```

## Evaluation

Evaluate the trained model:
```bash
python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth
```

This generates:
- Test metrics (accuracy, precision, recall, F1, AUC)
- Confusion matrix (saved as PNG)
- ROC curve (saved as PNG)
- Classification report (saved as TXT)

## Prediction

Make predictions on new images:

Single image:
```bash
python predict.py --config config.yaml --checkpoint checkpoints/best_model.pth --image path/to/image.png
```

Batch prediction:
```bash
python predict.py --config config.yaml --checkpoint checkpoints/best_model.pth --image path/to/images/ --output results.txt
```

## Example

Run the example script to see model architecture and usage:
```bash
python example.py
```

## Tips

1. **Data Augmentation**: Enable in `config.yaml` to improve generalization
2. **Patch Size**: Smaller patches capture fine details; larger patches provide more context
3. **Number of Patches**: More patches increase computational cost but may improve accuracy
4. **Backbone Selection**: ResNet50 is balanced; ResNet18 is faster; ResNet34 is in-between
5. **Early Stopping**: Prevents overfitting by monitoring validation accuracy

## Troubleshooting

**Out of Memory**: Reduce batch size or image size in config.yaml

**Poor Performance**: 
- Increase training epochs
- Try different learning rates
- Adjust data augmentation settings
- Use a different backbone architecture

**Data Loading Issues**: Verify data directory structure matches the expected format

## Next Steps

- Experiment with different model configurations
- Try transfer learning with different pretrained models
- Implement additional evaluation metrics
- Visualize attention weights to understand model decisions
