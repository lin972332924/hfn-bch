#!/usr/bin/env python
"""
Verification script to check the implementation without running it.
This script verifies:
1. All required files exist
2. Python files have valid syntax
3. Configuration is properly formatted
4. Documentation is complete
"""

import os
import sys
import ast
import yaml


def check_file_exists(filepath):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✓ {filepath}")
        return True
    else:
        print(f"✗ {filepath} - MISSING")
        return False


def check_python_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print(f"✗ {filepath} - SYNTAX ERROR: {e}")
        return False


def check_yaml_syntax(filepath):
    """Check if a YAML file is valid."""
    try:
        with open(filepath, 'r') as f:
            yaml.safe_load(f)
        return True
    except yaml.YAMLError as e:
        print(f"✗ {filepath} - YAML ERROR: {e}")
        return False


def main():
    """Main verification function."""
    print("="*60)
    print("Breast Cancer Classification - Implementation Verification")
    print("="*60)
    
    all_passed = True
    
    # Check essential files
    print("\n1. Checking essential files...")
    essential_files = [
        'README.md',
        'QUICKSTART.md',
        'IMPLEMENTATION.md',
        'CONTRIBUTING.md',
        'LICENSE',
        'requirements.txt',
        'config.yaml',
        'setup.py',
        '.gitignore',
    ]
    
    for file in essential_files:
        if not check_file_exists(file):
            all_passed = False
    
    # Check Python files
    print("\n2. Checking Python files...")
    python_files = [
        'train.py',
        'evaluate.py',
        'predict.py',
        'example.py',
        'src/__init__.py',
        'src/data/__init__.py',
        'src/data/dataset.py',
        'src/models/__init__.py',
        'src/models/hybrid_model.py',
        'src/utils/__init__.py',
        'src/utils/visualization.py',
    ]
    
    for file in python_files:
        if check_file_exists(file):
            if not check_python_syntax(file):
                all_passed = False
        else:
            all_passed = False
    
    # Check YAML files
    print("\n3. Checking configuration files...")
    if check_file_exists('config.yaml'):
        if not check_yaml_syntax('config.yaml'):
            all_passed = False
    else:
        all_passed = False
    
    # Check directory structure
    print("\n4. Checking directory structure...")
    directories = [
        'src',
        'src/data',
        'src/models',
        'src/utils',
        'src/config',
    ]
    
    for directory in directories:
        if os.path.isdir(directory):
            print(f"✓ {directory}/")
        else:
            print(f"✗ {directory}/ - MISSING")
            all_passed = False
    
    # Check documentation completeness
    print("\n5. Checking documentation...")
    
    with open('README.md', 'r') as f:
        readme = f.read()
        required_sections = [
            'Overview',
            'Installation',
            'Usage',
            'Architecture',
            'Requirements',
        ]
        
        for section in required_sections:
            if section.lower() in readme.lower():
                print(f"✓ README contains {section} section")
            else:
                print(f"✗ README missing {section} section")
                all_passed = False
    
    # Check configuration parameters
    print("\n6. Checking configuration parameters...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
        required_params = [
            ['model', 'local', 'patch_size'],
            ['model', 'local', 'num_patches'],
            ['model', 'local', 'backbone'],
            ['model', 'global', 'input_size'],
            ['model', 'fusion', 'hidden_dim'],
            ['model', 'fusion', 'num_classes'],
            ['training', 'batch_size'],
            ['training', 'num_epochs'],
            ['training', 'learning_rate'],
        ]
        
        for param_path in required_params:
            current = config
            valid = True
            for key in param_path:
                if key in current:
                    current = current[key]
                else:
                    print(f"✗ Missing config parameter: {'.'.join(param_path)}")
                    valid = False
                    all_passed = False
                    break
            
            if valid:
                print(f"✓ Config parameter: {'.'.join(param_path)}")
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("="*60)
        print("\nThe implementation is complete and ready to use!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Prepare your data in the data/ directory")
        print("3. Run training: python train.py --config config.yaml")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
