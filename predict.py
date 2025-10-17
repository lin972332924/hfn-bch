"""
Inference script for breast cancer pathology image classification.
"""

import os
import yaml
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.hybrid_model import create_model


class BreastCancerPredictor:
    """
    Predictor class for breast cancer pathology image classification.
    """
    
    def __init__(self, config_path: str, checkpoint_path: str):
        """
        Initialize predictor.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = create_model(self.config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Get model parameters
        self.patch_size = self.config['model']['local']['patch_size']
        self.num_patches = self.config['model']['local']['num_patches']
        self.global_size = self.config['model']['global']['input_size']
        
        # Setup transforms
        self.patch_transform = transforms.Compose([
            transforms.Resize((self.patch_size, self.patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.global_transform = transforms.Compose([
            transforms.Resize((self.global_size, self.global_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Class names
        self.class_names = ['Benign', 'Malignant']
    
    def extract_patches(self, image: Image.Image):
        """
        Extract patches from an image.
        
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
    
    def preprocess_image(self, image_path: str):
        """
        Preprocess image for prediction.
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (local_features, global_features)
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Extract local patches
        patches = self.extract_patches(image)
        local_features = torch.stack([self.patch_transform(patch) for patch in patches])
        local_features = local_features.unsqueeze(0)  # Add batch dimension
        
        # Get global representation
        global_features = self.global_transform(image)
        global_features = global_features.unsqueeze(0)  # Add batch dimension
        
        return local_features, global_features
    
    def predict(self, image_path: str):
        """
        Predict class for an image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        local_features, global_features = self.preprocess_image(image_path)
        
        # Move to device
        local_features = local_features.to(self.device)
        global_features = global_features.to(self.device)
        
        # Predict
        with torch.no_grad():
            logits, _, _ = self.model(local_features, global_features)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(logits, dim=1).item()
        
        # Get probabilities
        probs = probs.cpu().numpy()[0]
        
        return {
            'predicted_class': self.class_names[pred_class],
            'predicted_class_idx': pred_class,
            'confidence': float(probs[pred_class]),
            'probabilities': {
                'Benign': float(probs[0]),
                'Malignant': float(probs[1])
            }
        }
    
    def predict_batch(self, image_paths: list):
        """
        Predict classes for a batch of images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of prediction results
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            result['image_path'] = image_path
            results.append(result)
        
        return results


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict breast cancer classification')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to image file or directory')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to output file (optional)')
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = BreastCancerPredictor(args.config, args.checkpoint)
    
    # Check if input is file or directory
    if os.path.isfile(args.image):
        # Single image prediction
        print(f"Processing image: {args.image}")
        result = predictor.predict(args.image)
        
        print("\n" + "="*50)
        print("Prediction Results")
        print("="*50)
        print(f"Image: {args.image}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nProbabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        print("="*50)
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                f.write("Prediction Results\n")
                f.write("="*50 + "\n")
                f.write(f"Image: {args.image}\n")
                f.write(f"Predicted Class: {result['predicted_class']}\n")
                f.write(f"Confidence: {result['confidence']:.4f}\n")
                f.write("\nProbabilities:\n")
                for class_name, prob in result['probabilities'].items():
                    f.write(f"  {class_name}: {prob:.4f}\n")
                f.write("="*50 + "\n")
            print(f"\nResults saved to {args.output}")
    
    elif os.path.isdir(args.image):
        # Batch prediction
        image_files = [
            os.path.join(args.image, f)
            for f in os.listdir(args.image)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if len(image_files) == 0:
            print(f"No images found in {args.image}")
            return
        
        print(f"Processing {len(image_files)} images...")
        results = predictor.predict_batch(image_files)
        
        # Print results
        print("\n" + "="*50)
        print("Batch Prediction Results")
        print("="*50)
        for result in results:
            print(f"\nImage: {result['image_path']}")
            print(f"Predicted Class: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.4f}")
        print("="*50)
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                f.write("Batch Prediction Results\n")
                f.write("="*50 + "\n")
                for result in results:
                    f.write(f"\nImage: {result['image_path']}\n")
                    f.write(f"Predicted Class: {result['predicted_class']}\n")
                    f.write(f"Confidence: {result['confidence']:.4f}\n")
                    f.write("Probabilities:\n")
                    for class_name, prob in result['probabilities'].items():
                        f.write(f"  {class_name}: {prob:.4f}\n")
                f.write("="*50 + "\n")
            print(f"\nResults saved to {args.output}")
    
    else:
        print(f"Error: {args.image} is not a valid file or directory")


if __name__ == '__main__':
    main()
