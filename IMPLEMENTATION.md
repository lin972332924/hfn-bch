# Implementation Summary

## Breast Cancer Pathology Image Classification: Local and Global Approaches

This document provides a technical overview of the implementation.

## Architecture Overview

The implementation follows a hybrid approach that combines:

1. **Local Feature Extraction**: Analyzes fine-grained details from image patches
2. **Global Feature Extraction**: Captures overall tissue structure and patterns
3. **Attention-based Fusion**: Intelligently combines features for classification

## Key Components

### 1. Data Module (`src/data/dataset.py`)

**BreastCancerDataset Class**:
- Loads and preprocesses pathology images
- Extracts local patches using grid-based sampling
- Applies data augmentation (random flips, rotations, color jitter)
- Normalizes images using ImageNet statistics
- Returns both local patches and global image representations

**Features**:
- Configurable patch size and number of patches
- Separate transforms for training (augmented) and validation/testing
- Automatic data splitting into train/validation/test sets
- Support for standard image formats (PNG, JPEG)

### 2. Model Architecture (`src/models/hybrid_model.py`)

**LocalFeatureExtractor**:
- Uses pretrained CNN (ResNet) as backbone
- Processes multiple patches in parallel
- Extracts feature vectors for each patch
- Output: (batch_size, num_patches, feature_dim)

**GlobalFeatureExtractor**:
- Uses pretrained CNN (ResNet) as backbone
- Processes entire image
- Extracts global feature representation
- Output: (batch_size, feature_dim)

**AttentionAggregation**:
- Computes attention weights for each patch
- Uses two-layer MLP with Tanh activation
- Applies softmax to obtain normalized weights
- Aggregates patch features via weighted sum
- Output: (batch_size, feature_dim)

**HybridFusionModel**:
- Combines local and global extractors
- Fuses features through concatenation
- Uses multi-layer perceptron for classification
- Dropout for regularization
- Output: Class logits

### 3. Training Script (`train.py`)

**Trainer Class**:
- Manages training loop and validation
- Supports multiple optimizers (Adam, SGD)
- Learning rate scheduling (Cosine, Step)
- Early stopping based on validation accuracy
- Checkpoint saving for best model
- TensorBoard logging for metrics

**Metrics Tracked**:
- Loss (training and validation)
- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

### 4. Evaluation Script (`evaluate.py`)

**Features**:
- Loads trained model checkpoint
- Evaluates on test set
- Generates comprehensive metrics
- Creates visualizations:
  - Confusion matrix heatmap
  - ROC curve with AUC score
- Saves classification report

### 5. Prediction Script (`predict.py`)

**BreastCancerPredictor Class**:
- Loads trained model
- Preprocesses input images
- Makes predictions with confidence scores
- Supports single image or batch prediction
- Returns class probabilities

### 6. Utilities (`src/utils/visualization.py`)

**Visualization Functions**:
- `visualize_patches()`: Shows how images are divided into patches
- `visualize_attention_weights()`: Displays attention heatmap overlay
- `plot_training_history()`: Plots training curves
- `plot_confusion_matrix()`: Creates confusion matrix visualization
- `plot_roc_curve()`: Generates ROC curve

**Model Analysis**:
- `calculate_model_size()`: Computes model size in MB
- `count_parameters()`: Counts trainable parameters
- `print_model_summary()`: Displays detailed model information

## Technical Details

### Patch Extraction Strategy

Images are divided into a regular grid of patches:
- Grid size = √(num_patches)
- For 16 patches: 4×4 grid
- Preserves spatial relationships
- Captures local details at multiple locations

### Attention Mechanism

The attention module learns to weight patches based on their importance:
```
attention_score = MLP(patch_features)
attention_weights = softmax(attention_scores)
aggregated_features = sum(patch_features × attention_weights)
```

Benefits:
- Focuses on discriminative regions
- Handles variable patch importance
- Interpretable via attention weights

### Feature Fusion

Local and global features are fused via concatenation:
```
fused = concatenate([local_features, global_features])
output = MLP(fused)
```

This allows the model to leverage:
- Fine-grained local patterns (from patches)
- Overall structural information (from global view)

### Transfer Learning

Uses pretrained ImageNet models:
- ResNet18: 11M parameters, faster inference
- ResNet34: 21M parameters, balanced
- ResNet50: 25M parameters, higher capacity

Benefits:
- Reduces training time
- Improves generalization
- Works well with limited medical data

## Configuration Options

### Model Configuration
```yaml
model:
  local:
    patch_size: 224          # Size of each patch
    num_patches: 16          # Number of patches to extract
    backbone: 'resnet50'     # CNN architecture
    pretrained: true         # Use ImageNet weights
  global:
    input_size: 512          # Size for global image
    backbone: 'resnet50'     # CNN architecture
    pretrained: true         # Use ImageNet weights
  fusion:
    hidden_dim: 512          # Dimension of fusion layers
    dropout: 0.5             # Dropout rate
    num_classes: 2           # Number of output classes
```

### Training Configuration
```yaml
training:
  batch_size: 16             # Batch size
  num_epochs: 100            # Maximum epochs
  learning_rate: 0.0001      # Initial learning rate
  weight_decay: 0.0001       # L2 regularization
  optimizer: 'adam'          # Optimizer type
  scheduler: 'cosine'        # LR scheduler
  early_stopping_patience: 10 # Early stopping patience
```

## Performance Considerations

### Memory Usage
- Batch size affects GPU memory
- Larger patches increase memory requirements
- More patches per image increases memory

### Training Time
- ResNet50 is slower than ResNet18/34
- More patches increase training time
- Data augmentation adds overhead

### Inference Speed
- Batch prediction is more efficient
- GPU acceleration recommended
- Patch extraction can be parallelized

## Best Practices

1. **Data Preparation**:
   - Balance classes to avoid bias
   - Use consistent image quality
   - Normalize staining if possible

2. **Hyperparameter Tuning**:
   - Start with default configuration
   - Adjust learning rate if not converging
   - Increase patience for early stopping

3. **Model Selection**:
   - Use ResNet50 for best accuracy
   - Use ResNet18 for faster training
   - Consider data size when choosing

4. **Evaluation**:
   - Use multiple metrics (not just accuracy)
   - Check confusion matrix for biases
   - Monitor both precision and recall

## Future Enhancements

Potential improvements:
- Multi-scale patch extraction
- Pyramid pooling for global features
- Class activation mapping (CAM) for interpretability
- Ensemble of multiple models
- Support for multi-class classification
- Integration with whole-slide imaging (WSI) formats

## References

Implementation based on concepts from:
- Deep learning for digital pathology
- Attention mechanisms in computer vision
- Transfer learning for medical imaging
- Hybrid local-global feature fusion

## License

MIT License - See LICENSE file for details
