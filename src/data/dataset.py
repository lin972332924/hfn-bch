"""
Data loading and preprocessing module for breast cancer pathology images.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional


class BreastCancerDataset(Dataset):
    """
    Dataset class for breast cancer pathology images.
    Supports both local (patch-based) and global (whole-image) analysis.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        patch_size: int = 224,
        num_patches: int = 16,
        global_size: int = 512,
        augment: bool = False
    ):
        """
        Args:
            image_paths: List of paths to images
            labels: List of labels (0: benign, 1: malignant)
            patch_size: Size of local patches
            num_patches: Number of patches to extract per image
            global_size: Size for global image representation
            augment: Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.global_size = global_size
        self.augment = augment
        
        # Transforms for local patches
        if augment:
            self.patch_transform = transforms.Compose([
                transforms.Resize((patch_size, patch_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.patch_transform = transforms.Compose([
                transforms.Resize((patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Transforms for global image
        if augment:
            self.global_transform = transforms.Compose([
                transforms.Resize((global_size, global_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.global_transform = transforms.Compose([
                transforms.Resize((global_size, global_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def extract_patches(self, image: Image.Image) -> List[Image.Image]:
        """
        Extract patches from an image using a grid-based approach.
        
        Args:
            image: PIL Image
            
        Returns:
            List of patch images
        """
        width, height = image.size
        patches = []
        
        # Calculate grid dimensions
        grid_size = int(np.sqrt(self.num_patches))
        patch_width = width // grid_size
        patch_height = height // grid_size
        
        # Extract patches from grid
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * patch_width
                top = i * patch_height
                right = left + patch_width
                bottom = top + patch_height
                
                patch = image.crop((left, top, right, bottom))
                patches.append(patch)
        
        return patches[:self.num_patches]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (local_features, global_features, label)
            - local_features: Tensor of shape (num_patches, 3, patch_size, patch_size)
            - global_features: Tensor of shape (3, global_size, global_size)
            - label: Integer label
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        # Extract local patches
        patches = self.extract_patches(image)
        local_features = torch.stack([self.patch_transform(patch) for patch in patches])
        
        # Get global representation
        global_features = self.global_transform(image)
        
        return local_features, global_features, label


def get_data_loaders(
    data_dir: str,
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    patch_size: int = 224,
    num_patches: int = 16,
    global_size: int = 512
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing the data
        batch_size: Batch size
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        num_workers: Number of data loading workers
        patch_size: Size of local patches
        num_patches: Number of patches per image
        global_size: Size for global image representation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # This is a placeholder implementation
    # In practice, you would scan the data_dir for images and labels
    # For now, we'll return empty loaders
    
    # Example structure:
    # data_dir/
    #   benign/
    #     image1.png
    #     image2.png
    #   malignant/
    #     image1.png
    #     image2.png
    
    image_paths = []
    labels = []
    
    # Scan for benign images
    benign_dir = os.path.join(data_dir, 'benign')
    if os.path.exists(benign_dir):
        for img_file in os.listdir(benign_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(benign_dir, img_file))
                labels.append(0)
    
    # Scan for malignant images
    malignant_dir = os.path.join(data_dir, 'malignant')
    if os.path.exists(malignant_dir):
        for img_file in os.listdir(malignant_dir):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(malignant_dir, img_file))
                labels.append(1)
    
    if len(image_paths) == 0:
        print("Warning: No images found in data directory")
        return None, None, None
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=(1 - train_ratio), random_state=42, stratify=labels
    )
    
    val_size = val_ratio / (1 - train_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        test_paths, test_labels, test_size=(1 - val_size), random_state=42, stratify=test_labels
    )
    
    # Create datasets
    train_dataset = BreastCancerDataset(
        train_paths, train_labels, patch_size, num_patches, global_size, augment=True
    )
    val_dataset = BreastCancerDataset(
        val_paths, val_labels, patch_size, num_patches, global_size, augment=False
    )
    test_dataset = BreastCancerDataset(
        test_paths, test_labels, patch_size, num_patches, global_size, augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
