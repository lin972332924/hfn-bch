"""
Model architecture for breast cancer pathology image classification.
Implements local (patch-based) and global (whole-image) feature extraction.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class LocalFeatureExtractor(nn.Module):
    """
    Extracts features from local patches using a CNN backbone.
    """
    
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True):
        """
        Args:
            backbone: CNN backbone architecture
            pretrained: Whether to use pretrained weights
        """
        super(LocalFeatureExtractor, self).__init__()
        
        # Load backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for local feature extraction.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches, 3, H, W)
            
        Returns:
            Features tensor of shape (batch_size, num_patches, feature_dim)
        """
        batch_size, num_patches, c, h, w = x.size()
        
        # Reshape to process all patches at once
        x = x.view(batch_size * num_patches, c, h, w)
        
        # Extract features
        features = self.features(x)
        features = features.view(batch_size, num_patches, -1)
        
        return features


class GlobalFeatureExtractor(nn.Module):
    """
    Extracts features from the whole image using a CNN backbone.
    """
    
    def __init__(self, backbone: str = 'resnet50', pretrained: bool = True):
        """
        Args:
            backbone: CNN backbone architecture
            pretrained: Whether to use pretrained weights
        """
        super(GlobalFeatureExtractor, self).__init__()
        
        # Load backbone
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for global feature extraction.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Features tensor of shape (batch_size, feature_dim)
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features


class AttentionAggregation(nn.Module):
    """
    Attention-based aggregation of local patch features.
    """
    
    def __init__(self, feature_dim: int):
        """
        Args:
            feature_dim: Dimension of input features
        """
        super(AttentionAggregation, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for attention aggregation.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches, feature_dim)
            
        Returns:
            Aggregated features of shape (batch_size, feature_dim)
        """
        # Compute attention weights
        attention_weights = self.attention(x)  # (batch_size, num_patches, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        aggregated = torch.sum(x * attention_weights, dim=1)  # (batch_size, feature_dim)
        
        return aggregated


class HybridFusionModel(nn.Module):
    """
    Hybrid model that fuses local and global features for classification.
    """
    
    def __init__(
        self,
        local_backbone: str = 'resnet50',
        global_backbone: str = 'resnet50',
        pretrained: bool = True,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        """
        Args:
            local_backbone: Backbone for local feature extraction
            global_backbone: Backbone for global feature extraction
            pretrained: Whether to use pretrained weights
            hidden_dim: Hidden dimension for fusion layer
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(HybridFusionModel, self).__init__()
        
        # Local and global feature extractors
        self.local_extractor = LocalFeatureExtractor(local_backbone, pretrained)
        self.global_extractor = GlobalFeatureExtractor(global_backbone, pretrained)
        
        # Attention aggregation for local features
        self.attention_aggregation = AttentionAggregation(self.local_extractor.feature_dim)
        
        # Fusion layer
        fusion_input_dim = self.local_extractor.feature_dim + self.global_extractor.feature_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        local_input: torch.Tensor,
        global_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            local_input: Local patches tensor of shape (batch_size, num_patches, 3, H, W)
            global_input: Global image tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Tuple of (logits, local_features, global_features)
        """
        # Extract local features
        local_features = self.local_extractor(local_input)
        local_features_aggregated = self.attention_aggregation(local_features)
        
        # Extract global features
        global_features = self.global_extractor(global_input)
        
        # Fuse features
        fused_features = torch.cat([local_features_aggregated, global_features], dim=1)
        
        # Classification
        logits = self.fusion(fused_features)
        
        return logits, local_features_aggregated, global_features


def create_model(config: dict) -> HybridFusionModel:
    """
    Create model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HybridFusionModel instance
    """
    model = HybridFusionModel(
        local_backbone=config['model']['local']['backbone'],
        global_backbone=config['model']['global']['backbone'],
        pretrained=config['model']['local']['pretrained'],
        hidden_dim=config['model']['fusion']['hidden_dim'],
        num_classes=config['model']['fusion']['num_classes'],
        dropout=config['model']['fusion']['dropout']
    )
    
    return model
