"""
Evaluation script for breast cancer pathology image classification.
"""

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.hybrid_model import create_model
from data.dataset import get_data_loaders


def evaluate_model(model, test_loader, device, save_dir):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        save_dir: Directory to save results
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for local_input, global_input, labels in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            local_input = local_input.to(device)
            global_input = global_input.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits, _, _ = model(local_input, global_input)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    # Print results
    print("\n" + "="*50)
    print("Test Results")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("="*50)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=['Benign', 'Malignant']
    ))
    
    # Save metrics to file
    metrics_path = os.path.join(save_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Test Results\n")
        f.write("="*50 + "\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n")
        f.write(f"AUC:       {auc:.4f}\n")
        f.write("="*50 + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(
            all_labels, all_preds,
            target_names=['Benign', 'Malignant']
        ))
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, save_dir)
    
    # Plot ROC curve
    if auc > 0:
        plot_roc_curve(all_labels, all_probs, auc, save_dir)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def plot_confusion_matrix(labels, preds, save_dir):
    """
    Plot and save confusion matrix.
    
    Args:
        labels: True labels
        preds: Predicted labels
        save_dir: Directory to save the plot
    """
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curve(labels, probs, auc, save_dir):
    """
    Plot and save ROC curve.
    
    Args:
        labels: True labels
        probs: Predicted probabilities
        auc: AUC score
        save_dir: Directory to save the plot
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to {save_path}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate breast cancer classification model')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                      help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    _, _, test_loader = get_data_loaders(
        data_dir=config['paths']['data_dir'],
        batch_size=config['training']['batch_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        num_workers=config['data']['num_workers'],
        patch_size=config['model']['local']['patch_size'],
        num_patches=config['model']['local']['num_patches'],
        global_size=config['model']['global']['input_size']
    )
    
    if test_loader is None:
        print("Error: Could not load data. Please check the data directory.")
        return
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model
    print("Loading model...")
    model = create_model(config).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create results directory
    results_dir = config['paths']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, device, results_dir)
    
    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
