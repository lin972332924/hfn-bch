"""
Training script for breast cancer pathology image classification.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.hybrid_model import create_model
from data.dataset import get_data_loaders


class Trainer:
    """
    Trainer class for the hybrid breast cancer classification model.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        os.makedirs(self.config['paths']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['results_dir'], exist_ok=True)
        os.makedirs(self.config['paths']['log_dir'], exist_ok=True)
        
        # Initialize model
        self.model = create_model(self.config).to(self.device)
        print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(self.config['paths']['log_dir'])
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def _create_optimizer(self):
        """Create optimizer based on configuration."""
        optimizer_name = self.config['training']['optimizer'].lower()
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_name = self.config['training']['scheduler'].lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            return None
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for batch_idx, (local_input, global_input, labels) in enumerate(pbar):
            # Move to device
            local_input = local_input.to(self.device)
            global_input = global_input.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits, _, _ = self.model(local_input, global_input)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
            
            for local_input, global_input, labels in pbar:
                # Move to device
                local_input = local_input.to(self.device)
                global_input = global_input.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits, _, _ = self.model(local_input, global_input)
                loss = self.criterion(logits, labels)
                
                # Statistics
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        return avg_loss, accuracy, precision, recall, f1, auc
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        scheduler = self._create_scheduler()
        
        for epoch in range(self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = self.validate(val_loader)
            
            # Log metrics
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Precision/val', val_prec, epoch)
            self.writer.add_scalar('Recall/val', val_rec, epoch)
            self.writer.add_scalar('F1/val', val_f1, epoch)
            self.writer.add_scalar('AUC/val', val_auc, epoch)
            
            print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}")
            print(f"Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step()
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                
                checkpoint_path = os.path.join(
                    self.config['paths']['checkpoint_dir'],
                    'best_model.pth'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, checkpoint_path)
                
                print(f"Saved best model with validation accuracy: {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        
        self.writer.close()
        print(f"Training completed. Best validation accuracy: {self.best_val_acc:.4f}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train breast cancer classification model')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=config['paths']['data_dir'],
        batch_size=config['training']['batch_size'],
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        num_workers=config['data']['num_workers'],
        patch_size=config['model']['local']['patch_size'],
        num_patches=config['model']['local']['num_patches'],
        global_size=config['model']['global']['input_size']
    )
    
    if train_loader is None:
        print("Error: Could not load data. Please check the data directory.")
        return
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Train model
    trainer = Trainer(args.config)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
