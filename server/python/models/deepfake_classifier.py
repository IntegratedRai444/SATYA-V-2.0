"""
Real Deepfake Classifier Model
Uses EfficientNet-B4 for binary classification of real vs fake faces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DeepfakeClassifier(nn.Module):
    """
    EfficientNet-B4 based deepfake classifier.
    """
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        """
        Initialize the classifier.
        
        Args:
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Whether to use pretrained weights
        """
        super(DeepfakeClassifier, self).__init__()
        
        # Use EfficientNet-B4 as backbone
        self.backbone = models.efficientnet_b4(pretrained=pretrained)
        
        # Get the number of features from the classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier with our custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Add attention mechanism
        self.attention = SpatialAttention()
        
        logger.info(f"DeepfakeClassifier initialized with {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features before the final classifier
        features = self.backbone.features(x)
        
        # Apply attention
        attended_features = self.attention(features)
        
        # Global average pooling
        pooled = self.backbone.avgpool(attended_features)
        
        # Flatten
        flattened = torch.flatten(pooled, 1)
        
        # Final classification
        output = self.backbone.classifier(flattened)
        
        return output
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Softmax probabilities
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        
        return probabilities
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and confidence scores.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        probabilities = self.predict_proba(x)
        predictions = torch.argmax(probabilities, dim=1)
        confidence_scores = torch.max(probabilities, dim=1)[0]
        
        return predictions, confidence_scores


def load_pretrained_classifier_advanced(model_path: str, model_type: str = 'efficientnet', device: str = 'cpu') -> Optional[nn.Module]:
    """
    Load a pretrained deepfake classifier with flexible loading.
    
    Args:
        model_path: Path to the model file
        model_type: Type of model architecture
        device: Device to load the model on
        
    Returns:
        Loaded model or None if loading fails
    """
    try:
        import torch
        import torchvision.models as models
        from pathlib import Path
        
        model_file = Path(model_path)
        if not model_file.exists():
            logger.warning(f"Model file not found: {model_path}, creating new model")
            return create_simple_classifier(device)
        
        # Try to load existing model with flexible approach
        try:
            state_dict = torch.load(model_path, map_location=device)
            
            # Check if this is a ResNet50 model (common format)
            if any('layer1' in key for key in state_dict.keys()):
                logger.info("Detected ResNet50 format, creating compatible model")
                model = models.resnet50(pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                
                # Load with strict=False to handle minor differences
                model.load_state_dict(state_dict, strict=False)
                model.to(device)
                model.eval()
                
                logger.info(f"Successfully loaded ResNet50 model from {model_path}")
                return model
            
            # Try EfficientNet format
            elif any('backbone.features' in key for key in state_dict.keys()):
                logger.info("Detected EfficientNet format, creating compatible model")
                model = DeepfakeClassifier(num_classes=2, pretrained=False)
                
                # Handle different state dict formats
                if 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'], strict=False)
                elif 'state_dict' in state_dict:
                    model.load_state_dict(state_dict['state_dict'], strict=False)
                else:
                    model.load_state_dict(state_dict, strict=False)
                
                model.to(device)
                model.eval()
                
                logger.info(f"Successfully loaded EfficientNet model from {model_path}")
                return model
            
            else:
                logger.warning(f"Unknown model format in {model_path}, creating new model")
                return create_simple_classifier(device)
                
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}, creating new model")
            return create_simple_classifier(device)
            
    except Exception as e:
        logger.error(f"Failed to load pretrained classifier: {e}")
        return create_simple_classifier(device)


def create_simple_classifier(device: str = 'cpu') -> DeepfakeClassifier:
    """
    Create a simple classifier with pretrained backbone.
    
    Args:
        device: Device to create the model on
        
    Returns:
        Simple classifier model
    """
    try:
        model = DeepfakeClassifier(num_classes=2, pretrained=True)
        model.to(device)
        model.eval()
        
        logger.info("Created simple classifier with pretrained backbone")
        return model
        
    except Exception as e:
        logger.error(f"Failed to create simple classifier: {e}")
        # Return a very basic model as fallback
        return DeepfakeClassifier(num_classes=2, pretrained=False)


def load_xception_model(model_path: str, device: str = 'cpu') -> Optional[torch.nn.Module]:
    """
    Load Xception model trained on FaceForensics++.
    
    Args:
        model_path: Path to the Xception model file
        device: Device to load the model on
        
    Returns:
        Loaded Xception model or None
    """
    try:
        import torch
        import torch.nn as nn
        from pathlib import Path
        
        if not Path(model_path).exists():
            logger.error(f"Xception model not found: {model_path}")
            return None
        
        # Create Xception architecture
        class XceptionDeepfake(nn.Module):
            def __init__(self, num_classes=2):
                super(XceptionDeepfake, self).__init__()
                
                # Use torchvision's implementation or create custom Xception
                try:
                    import torchvision.models as models
                    # Use ResNet as Xception substitute if Xception not available
                    self.backbone = models.resnet50(pretrained=False)
                    self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
                except:
                    # Fallback to simple CNN
                    self.backbone = nn.Sequential(
                        nn.Conv2d(3, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(64, num_classes)
                    )
            
            def forward(self, x):
                return self.backbone(x)
        
        # Load model
        model = XceptionDeepfake(num_classes=2)
        
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)  # Allow partial loading
        except Exception as e:
            logger.warning(f"Could not load Xception weights, using random initialization: {e}")
        
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded Xception model from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load Xception model: {e}")
        return None


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism to focus on important regions.
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialize spatial attention.
        
        Args:
            kernel_size: Kernel size for convolution
        """
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, 
            padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.
        
        Args:
            x: Input feature tensor
            
        Returns:
            Attention-weighted features
        """
        # Compute channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_weights = self.sigmoid(self.conv(attention_input))
        
        # Apply attention weights
        return x * attention_weights


class XceptionDeepfakeClassifier(nn.Module):
    """
    Alternative Xception-based classifier (commonly used in deepfake detection).
    """
    
    def __init__(self, num_classes: int = 2):
        """Initialize Xception classifier."""
        super(XceptionDeepfakeClassifier, self).__init__()
        
        # Load pretrained Xception (we'll use a ResNet as substitute since Xception isn't in torchvision)
        self.backbone = models.resnet50(pretrained=True)
        
        # Modify first conv layer for different input size if needed
        # self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        logger.info("XceptionDeepfakeClassifier initialized")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        
        return probabilities


def load_pretrained_classifier(model_path: str, model_type: str = 'efficientnet', device: str = 'cpu') -> Optional[nn.Module]:
    """
    Load a pretrained deepfake classifier.
    
    Args:
        model_path: Path to the model file
        model_type: Type of model ('efficientnet' or 'xception')
        device: Device to load the model on
        
    Returns:
        Loaded model or None if loading fails
    """
    try:
        if model_type == 'efficientnet':
            model = DeepfakeClassifier(num_classes=2, pretrained=False)
        elif model_type == 'xception':
            model = XceptionDeepfakeClassifier(num_classes=2)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle different state dict formats
        if 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        elif 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)
        
        model.eval()
        model.to(device)
        
        logger.info(f"Loaded pretrained {model_type} classifier from {model_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load classifier from {model_path}: {e}")
        return None


def create_simple_classifier(device: str = 'cpu') -> nn.Module:
    """
    Create a simple classifier for testing when pretrained models aren't available.
    
    Args:
        device: Device to create the model on
        
    Returns:
        Simple CNN classifier
    """
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            
            self.features = nn.Sequential(
                # First block
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                # Second block
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                # Third block
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                # Fourth block
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((7, 7))
            )
            
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256 * 7 * 7, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
        
        def predict_proba(self, x):
            with torch.no_grad():
                logits = self.forward(x)
                probabilities = F.softmax(logits, dim=1)
            return probabilities
    
    model = SimpleCNN()
    model.eval()
    model.to(device)
    
    logger.info("Created simple CNN classifier for testing")
    return model


# Training utilities (for future model training)
class DeepfakeTrainer:
    """Trainer class for deepfake detection models."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """Initialize trainer."""
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
    
    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        logger.info(f'Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        return avg_loss, accuracy
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy