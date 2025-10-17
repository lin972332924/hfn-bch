# Project Files Reference

## Documentation Files

1. **README.md** - Main project documentation
   - Overview of the project
   - Installation instructions
   - Usage examples
   - Architecture description
   - Requirements list

2. **QUICKSTART.md** - Quick start guide
   - Fast installation
   - Basic usage examples
   - Troubleshooting tips

3. **IMPLEMENTATION.md** - Technical implementation details
   - Architecture deep dive
   - Module descriptions
   - Configuration options
   - Best practices

4. **CONTRIBUTING.md** - Contribution guidelines
   - How to contribute
   - Code style guidelines
   - Development setup
   - Pull request process

5. **LICENSE** - MIT License file

## Configuration Files

6. **config.yaml** - Model and training configuration
   - Model parameters
   - Training hyperparameters
   - Data settings
   - Path configurations

7. **requirements.txt** - Python dependencies
   - All required packages
   - Version specifications

8. **setup.py** - Package installation script
   - Package metadata
   - Dependencies
   - Entry points

9. **.gitignore** - Git ignore rules
   - Excludes data files
   - Excludes model checkpoints
   - Excludes temporary files

## Main Scripts

10. **train.py** - Training script (294 lines)
    - Trainer class
    - Training loop
    - Validation
    - Checkpoint management
    - TensorBoard logging

11. **evaluate.py** - Evaluation script (236 lines)
    - Model evaluation
    - Metrics calculation
    - Confusion matrix
    - ROC curve generation

12. **predict.py** - Inference script (274 lines)
    - BreastCancerPredictor class
    - Single image prediction
    - Batch prediction
    - Confidence scores

13. **example.py** - Example and demo script (150 lines)
    - Model demonstration
    - Usage examples
    - Documentation display

14. **verify.py** - Verification script (192 lines)
    - File existence checks
    - Syntax validation
    - Configuration validation
    - Documentation completeness

## Source Code Modules

### Data Module (src/data/)

15. **src/data/__init__.py** - Data module initializer
16. **src/data/dataset.py** - Dataset implementation (243 lines)
    - BreastCancerDataset class
    - Patch extraction
    - Data augmentation
    - Data loaders

### Models Module (src/models/)

17. **src/models/__init__.py** - Models module initializer
18. **src/models/hybrid_model.py** - Model architecture (240 lines)
    - LocalFeatureExtractor
    - GlobalFeatureExtractor
    - AttentionAggregation
    - HybridFusionModel

### Utilities Module (src/utils/)

19. **src/utils/__init__.py** - Utils module initializer
20. **src/utils/visualization.py** - Visualization tools (258 lines)
    - Patch visualization
    - Attention visualization
    - Training history plots
    - Model summary functions

## Directory Structure

```
hfn-bch/
├── Documentation (5 files)
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── IMPLEMENTATION.md
│   ├── CONTRIBUTING.md
│   └── LICENSE
├── Configuration (4 files)
│   ├── config.yaml
│   ├── requirements.txt
│   ├── setup.py
│   └── .gitignore
├── Scripts (5 files)
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── example.py
│   └── verify.py
└── Source Code (6 files)
    └── src/
        ├── __init__.py
        ├── data/
        │   ├── __init__.py
        │   └── dataset.py
        ├── models/
        │   ├── __init__.py
        │   └── hybrid_model.py
        └── utils/
            ├── __init__.py
            └── visualization.py
```

## File Statistics

- Total Files: 20
- Documentation: 5 files
- Configuration: 4 files
- Scripts: 5 files
- Source Code: 6 files
- Total Lines of Code: ~1,545 lines

## File Purposes Summary

### For Users
- README.md - Learn about the project
- QUICKSTART.md - Get started quickly
- config.yaml - Configure the model
- train.py - Train your model
- evaluate.py - Test your model
- predict.py - Make predictions

### For Developers
- IMPLEMENTATION.md - Understand the architecture
- CONTRIBUTING.md - Contribute to the project
- verify.py - Verify your setup
- src/ - Browse the source code

### For Reference
- LICENSE - Check the license
- requirements.txt - Install dependencies
- setup.py - Install as a package
