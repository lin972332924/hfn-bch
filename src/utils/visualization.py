"""
Utility functions for visualization and analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import cv2


def visualize_patches(image_path: str, num_patches: int = 16, save_path: str = None):
    """
    Visualize how an image is divided into patches.
    
    Args:
        image_path: Path to image
        num_patches: Number of patches to extract
        save_path: Path to save visualization (optional)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Calculate grid dimensions
    grid_size = int(np.sqrt(num_patches))
    height, width = image_np.shape[:2]
    patch_height = height // grid_size
    patch_width = width // grid_size
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle('Image Patches', fontsize=16)
    
    # Extract and display patches
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            top = i * patch_height
            left = j * patch_width
            bottom = top + patch_height
            right = left + patch_width
            
            patch = image_np[top:bottom, left:right]
            
            if grid_size > 1:
                ax = axes[i, j]
            else:
                ax = axes
            
            ax.imshow(patch)
            ax.axis('off')
            ax.set_title(f'Patch {idx}', fontsize=8)
            idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Patch visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_attention_weights(
    image_path: str,
    attention_weights: np.ndarray,
    num_patches: int = 16,
    save_path: str = None
):
    """
    Visualize attention weights on the image.
    
    Args:
        image_path: Path to image
        attention_weights: Attention weights array of shape (num_patches,)
        num_patches: Number of patches
        save_path: Path to save visualization (optional)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Calculate grid dimensions
    grid_size = int(np.sqrt(num_patches))
    height, width = image_np.shape[:2]
    patch_height = height // grid_size
    patch_width = width // grid_size
    
    # Create heatmap
    heatmap = np.zeros((height, width))
    
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            top = i * patch_height
            left = j * patch_width
            bottom = top + patch_height
            right = left + patch_width
            
            if idx < len(attention_weights):
                heatmap[top:bottom, left:right] = attention_weights[idx]
            idx += 1
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention heatmap
    im = axes[1].imshow(heatmap, cmap='hot', interpolation='bilinear')
    axes[1].set_title('Attention Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    overlay = image_np.copy()
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    if 'val_f1' in history:
        axes[1, 0].plot(history['val_f1'], label='Validation F1')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # AUC
    if 'val_auc' in history:
        axes[1, 1].plot(history['val_auc'], label='Validation AUC')
        axes[1, 1].set_title('AUC-ROC')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_model_size(model: torch.nn.Module):
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def count_parameters(model: torch.nn.Module):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module):
    """
    Print a summary of the model.
    
    Args:
        model: PyTorch model
    """
    print("="*60)
    print("Model Summary")
    print("="*60)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Model size: {calculate_model_size(model):.2f} MB")
    print("="*60)
    
    # Print layer-wise information
    print("\nLayer-wise Information:")
    print("-"*60)
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                print(f"{name:40s} {num_params:>15,}")
    print("="*60)
