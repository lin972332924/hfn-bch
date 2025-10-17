"""
Example script demonstrating how to use the breast cancer classification system.
"""

import os
import sys
sys.path.append('src')

from models.hybrid_model import HybridFusionModel
from utils.visualization import print_model_summary


def create_example_model():
    """Create and display an example model."""
    print("Creating example breast cancer classification model...")
    print()
    
    # Create model
    model = HybridFusionModel(
        local_backbone='resnet50',
        global_backbone='resnet50',
        pretrained=True,
        hidden_dim=512,
        num_classes=2,
        dropout=0.5
    )
    
    # Print model summary
    print_model_summary(model)
    
    print("\nModel Architecture:")
    print("-" * 60)
    print("1. Local Feature Extractor (Patch-based)")
    print("   - Backbone: ResNet50")
    print("   - Input: Multiple patches from the image")
    print("   - Output: Feature vector for each patch")
    print()
    print("2. Global Feature Extractor (Whole-image)")
    print("   - Backbone: ResNet50")
    print("   - Input: Entire image")
    print("   - Output: Global feature vector")
    print()
    print("3. Attention Aggregation")
    print("   - Aggregates local patch features using attention weights")
    print("   - Focuses on the most discriminative regions")
    print()
    print("4. Fusion Layer")
    print("   - Combines local and global features")
    print("   - Final classification layer")
    print("-" * 60)
    
    return model


def print_usage_examples():
    """Print usage examples."""
    print("\n" + "="*60)
    print("Usage Examples")
    print("="*60)
    
    print("\n1. Training:")
    print("-" * 60)
    print("python train.py --config config.yaml")
    print()
    print("This will:")
    print("  - Load training data from the configured data directory")
    print("  - Train the hybrid model with local and global features")
    print("  - Save checkpoints of the best model")
    print("  - Log metrics to TensorBoard")
    
    print("\n2. Evaluation:")
    print("-" * 60)
    print("python evaluate.py --config config.yaml --checkpoint checkpoints/best_model.pth")
    print()
    print("This will:")
    print("  - Load the trained model")
    print("  - Evaluate on test data")
    print("  - Generate confusion matrix and ROC curve")
    print("  - Save results to the results directory")
    
    print("\n3. Prediction (Single Image):")
    print("-" * 60)
    print("python predict.py --config config.yaml --checkpoint checkpoints/best_model.pth --image path/to/image.png")
    print()
    print("This will:")
    print("  - Load the trained model")
    print("  - Make prediction on the input image")
    print("  - Display predicted class and confidence")
    
    print("\n4. Prediction (Batch):")
    print("-" * 60)
    print("python predict.py --config config.yaml --checkpoint checkpoints/best_model.pth --image path/to/images/ --output results.txt")
    print()
    print("This will:")
    print("  - Load the trained model")
    print("  - Make predictions on all images in the directory")
    print("  - Save results to the specified output file")
    
    print("\n" + "="*60)


def print_data_structure():
    """Print expected data structure."""
    print("\n" + "="*60)
    print("Expected Data Structure")
    print("="*60)
    print("""
data/
├── benign/
│   ├── image1.png
│   ├── image2.png
│   ├── image3.png
│   └── ...
└── malignant/
    ├── image1.png
    ├── image2.png
    ├── image3.png
    └── ...
    
Notes:
- Place benign images in the 'benign' subdirectory
- Place malignant images in the 'malignant' subdirectory
- Supported formats: .png, .jpg, .jpeg
- Images will be automatically split into train/val/test sets
""")
    print("="*60)


def main():
    """Main function."""
    print("\n" + "="*60)
    print("Breast Cancer Pathology Image Classification")
    print("Local and Global Approaches")
    print("="*60)
    
    # Create example model
    model = create_example_model()
    
    # Print usage examples
    print_usage_examples()
    
    # Print data structure
    print_data_structure()
    
    print("\nFor more information, see README.md")
    print()


if __name__ == '__main__':
    main()
