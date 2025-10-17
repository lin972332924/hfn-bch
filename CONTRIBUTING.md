# Contributing to Breast Cancer Classification

Thank you for your interest in contributing to this project! This guide will help you get started.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:
1. Check if the issue already exists in the issue tracker
2. Create a new issue with a clear title and description
3. Include steps to reproduce (for bugs)
4. Add relevant labels

### Code Contributions

1. **Fork the repository**
   ```bash
   git clone https://github.com/lin972332924/hfn-bch.git
   cd hfn-bch
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation as needed

4. **Test your changes**
   - Ensure code runs without errors
   - Test on sample data if possible
   - Verify backward compatibility

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Describe your changes clearly
   - Reference related issues
   - Request review from maintainers

## Code Style Guidelines

### Python Code
- Follow PEP 8 style guide
- Use descriptive variable names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Maximum line length: 100 characters

### Documentation
- Update README.md for significant changes
- Add inline comments for complex logic
- Include docstrings with:
  - Function description
  - Args with types
  - Returns with types
  - Examples (if helpful)

### Example Docstring Format
```python
def function_name(arg1: type1, arg2: type2) -> return_type:
    """
    Brief description of function.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
    pass
```

## Areas for Contribution

### High Priority
- [ ] Add unit tests for core modules
- [ ] Improve data augmentation strategies
- [ ] Add support for multi-class classification
- [ ] Implement cross-validation
- [ ] Add model interpretability features

### Medium Priority
- [ ] Support for whole-slide images (WSI)
- [ ] Integration with medical image formats (DICOM)
- [ ] Ensemble methods
- [ ] Hyperparameter optimization
- [ ] Additional visualization tools

### Documentation
- [ ] Tutorial notebooks
- [ ] API documentation
- [ ] Video tutorials
- [ ] Use case examples
- [ ] Troubleshooting guide

## Testing

Before submitting a pull request:

1. **Syntax Check**
   ```bash
   python -m py_compile src/models/*.py src/data/*.py
   ```

2. **Manual Testing**
   - Test with sample data
   - Verify all scripts run without errors
   - Check that outputs are as expected

3. **Documentation**
   - Ensure all new functions have docstrings
   - Update README if adding new features
   - Add examples for new functionality

## Development Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install in development mode**
   ```bash
   pip install -e .
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Code Review Process

1. Maintainers will review your pull request
2. Address any feedback or requested changes
3. Once approved, your code will be merged
4. Your contribution will be credited

## Questions?

- Open an issue for discussion
- Contact maintainers
- Check existing documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for helping improve breast cancer classification research!
