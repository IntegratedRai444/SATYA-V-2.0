"""
Ensemble Deepfake Detection System
Combines multiple state-of-the-art models for superior accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from PIL import Image
import cv2

from .deepfake_classifier import DeepfakeClassifier, XceptionDeepfakeClassifier
from ..utils.error_handler import error_handler, ModelLoadError

logger = logging.getLogger(__name__)


class EnsembleDeepfakeDetector(nn.Module):
    """
    Ensemble detector combining multiple specialized models for maximum accuracy.
    """
    
    def __init__(self, model_path: Path, device: str = 'cpu'):
        """
        Initialize ensemble detector.
        
        Args:
            model_path: Path to model directory
            device: Device to run models on
        """
        super(EnsembleDeepfakeDetector, self).__init__()
        
        self.model_path = Path(model_path)
        self.device = device
        self.models = {}
        self.model_weights = {}
        self.is_loaded = False
        
        # Model configurations
        self.model_configs = {
            'efficientnet_b7': {
                'class': DeepfakeClassifier,
                'input_size': (224, 224),
                'weight': 0.35,
                'specialty': 'general_detection'
            },
            'xception': {
                'class': XceptionDeepfakeClassifier,
                'input_size': (299, 299),
                'weight': 0.25,
                'specialty': 'face_manipulation'
            },
            'face_xray': {
                'class': FaceXRayDetector,
                'input_size': (224, 224),
                'weight': 0.25,
                'specialty': 'blending_artifacts'
            },
            'capsule_forensics': {
                'class': CapsuleForensicsDetector,
                'input_size': (256, 256),
                'weight': 0.15,
                'specialty': 'temporal_consistency'
            }
        }
        
        logger.info(f"Ensemble detector initialized with {len(self.model_configs)} models")
    
    def load_models(self) -> bool:
        """
        Load all available models in the ensemble.
        
        Returns:
            True if at least one model loaded successfully
        """
        loaded_count = 0
        
        for model_name, config in self.model_configs.items():
            try:
                model_file = self.model_path / f"{model_name}.pth"
                
                if model_file.exists():
                    # Load model
                    model = config['class']()
                    state_dict = torch.load(model_file, map_location=self.device)
                    
                    # Handle different state dict formats
                    if 'model_state_dict' in state_dict:
                        model.load_state_dict(state_dict['model_state_dict'])
                    elif 'state_dict' in state_dict:
                        model.load_state_dict(state_dict['state_dict'])
                    else:
                        model.load_state_dict(state_dict)
                    
                    model.eval()
                    model.to(self.device)
                    
                    self.models[model_name] = model
                    self.model_weights[model_name] = config['weight']
                    loaded_count += 1
                    
                    logger.info(f"✓ Loaded {model_name} model")
                else:
                    logger.warning(f"Model file not found: {model_file}")
                    
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
        
        if loaded_count > 0:
            # Normalize weights for loaded models only
            total_weight = sum(self.model_weights.values())
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
            
            self.is_loaded = True
            logger.info(f"Ensemble loaded with {loaded_count}/{len(self.model_configs)} models")
            return True
        else:
            logger.error("No models could be loaded for ensemble")
            return False
    
    @error_handler
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict using ensemble of models.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Ensemble prediction results
        """
        if not self.is_loaded:
            raise ModelLoadError("Ensemble models not loaded")
        
        predictions = {}
        confidences = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                config = self.model_configs[model_name]
                
                # Preprocess image for this model
                processed_image = self._preprocess_for_model(image, config['input_size'])
                
                # Get prediction
                with torch.no_grad():
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(processed_image)
                    else:
                        logits = model(processed_image)
                        probs = F.softmax(logits, dim=1)
                    
                    # Extract probabilities (assuming binary classification)
                    fake_prob = probs[0][0].item()  # Fake probability
                    real_prob = probs[0][1].item()  # Real probability
                    
                    predictions[model_name] = {
                        'fake_prob': fake_prob,
                        'real_prob': real_prob,
                        'prediction': 'fake' if fake_prob > real_prob else 'real',
                        'confidence': max(fake_prob, real_prob),
                        'specialty': config['specialty']
                    }
                    
                    confidences[model_name] = max(fake_prob, real_prob)
                    
                    logger.debug(f"{model_name}: {predictions[model_name]['prediction']} "
                               f"({predictions[model_name]['confidence']:.3f})")
                    
            except Exception as e:
                logger.error(f"Prediction failed for {model_name}: {e}")
                predictions[model_name] = {
                    'fake_prob': 0.5,
                    'real_prob': 0.5,
                    'prediction': 'uncertain',
                    'confidence': 0.5,
                    'error': str(e)
                }
        
        # Combine predictions using weighted voting
        ensemble_result = self._combine_predictions(predictions)
        
        return {
            'ensemble_prediction': ensemble_result,
            'individual_predictions': predictions,
            'model_weights': self.model_weights,
            'models_used': list(self.models.keys())
        }
    
    def _preprocess_for_model(self, image: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Preprocess image for specific model requirements.
        
        Args:
            image: Input image (RGB)
            target_size: Target size (width, height)
            
        Returns:
            Preprocessed tensor
        """
        # Convert to PIL for resizing
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize(target_size, Image.BILINEAR)
        
        # Convert to tensor and normalize
        image_array = np.array(pil_image).astype(np.float32) / 255.0
        
        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        image_array = (image_array - mean) / std
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _combine_predictions(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Combine individual model predictions using weighted ensemble.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Combined ensemble prediction
        """
        weighted_fake_prob = 0.0
        weighted_real_prob = 0.0
        total_weight = 0.0
        
        confidence_scores = []
        individual_votes = []
        
        for model_name, pred in predictions.items():
            if 'error' in pred:
                continue
                
            weight = self.model_weights.get(model_name, 0.0)
            
            weighted_fake_prob += pred['fake_prob'] * weight
            weighted_real_prob += pred['real_prob'] * weight
            total_weight += weight
            
            confidence_scores.append(pred['confidence'])
            individual_votes.append(1 if pred['prediction'] == 'fake' else 0)
        
        if total_weight == 0:
            return {
                'prediction': 'uncertain',
                'confidence': 0.5,
                'fake_probability': 0.5,
                'real_probability': 0.5,
                'ensemble_method': 'fallback'
            }
        
        # Normalize probabilities
        weighted_fake_prob /= total_weight
        weighted_real_prob /= total_weight
        
        # Final prediction
        final_prediction = 'fake' if weighted_fake_prob > weighted_real_prob else 'real'
        final_confidence = max(weighted_fake_prob, weighted_real_prob)
        
        # Calculate consensus metrics
        vote_consensus = np.mean(individual_votes)
        confidence_std = np.std(confidence_scores) if confidence_scores else 0.0
        
        return {
            'prediction': final_prediction,
            'confidence': float(final_confidence),
            'fake_probability': float(weighted_fake_prob),
            'real_probability': float(weighted_real_prob),
            'vote_consensus': float(vote_consensus),
            'confidence_std': float(confidence_std),
            'ensemble_method': 'weighted_average',
            'agreement_level': 'high' if confidence_std < 0.1 else 'medium' if confidence_std < 0.2 else 'low'
        }


class FaceXRayDetector(nn.Module):
    """
    Face X-Ray detector for blending artifact detection.
    Specialized in detecting face swapping artifacts.
    """
    
    def __init__(self, num_classes: int = 2):
        """Initialize Face X-Ray detector."""
        super(FaceXRayDetector, self).__init__()
        
        # Use ResNet-50 as backbone
        import torchvision.models as models
        self.backbone = models.resnet50(pretrained=True)
        
        # Modify for binary classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Add attention mechanism for blending artifacts
        self.attention = BlendingAttention()
        
        logger.info("Face X-Ray detector initialized")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention on blending regions."""
        # Extract features
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)
        
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        
        # Apply attention
        attended_features = self.attention(features)
        
        # Global average pooling and classification
        pooled = self.backbone.avgpool(attended_features)
        flattened = torch.flatten(pooled, 1)
        output = self.backbone.fc(flattened)
        
        return output
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


class CapsuleForensicsDetector(nn.Module):
    """
    Capsule Network for forensics detection.
    Specialized in temporal consistency analysis.
    """
    
    def __init__(self, num_classes: int = 2):
        """Initialize Capsule Forensics detector."""
        super(CapsuleForensicsDetector, self).__init__()
        
        # Primary capsules
        self.primary_caps = PrimaryCapsules(256, 32, 8)
        
        # Digital forensics capsules
        self.forensics_caps = ForensicsCapsules(32 * 6 * 6, num_classes, 16, 8)
        
        # Reconstruction network
        self.reconstruction = ReconstructionNet()
        
        logger.info("Capsule Forensics detector initialized")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through capsule network."""
        # Extract primary capsules
        primary_caps = self.primary_caps(x)
        
        # Forensics analysis
        forensics_caps = self.forensics_caps(primary_caps)
        
        # Get class predictions
        class_probs = torch.sqrt((forensics_caps ** 2).sum(dim=2))
        
        return class_probs
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        with torch.no_grad():
            class_probs = self.forward(x)
            # Normalize to get probabilities
            probabilities = F.softmax(class_probs, dim=1)
        return probabilities


class BlendingAttention(nn.Module):
    """Attention mechanism for detecting blending artifacts."""
    
    def __init__(self, in_channels: int = 2048):
        """Initialize blending attention."""
        super(BlendingAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention to focus on blending regions."""
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


class PrimaryCapsules(nn.Module):
    """Primary capsules layer."""
    
    def __init__(self, in_channels: int, out_channels: int, dim_caps: int):
        """Initialize primary capsules."""
        super(PrimaryCapsules, self).__init__()
        
        self.dim_caps = dim_caps
        self.num_caps = out_channels
        
        self.conv = nn.Conv2d(in_channels, out_channels * dim_caps, 
                             kernel_size=9, stride=2, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through primary capsules."""
        outputs = self.conv(x)
        outputs = outputs.view(outputs.size(0), self.num_caps, self.dim_caps, 
                              outputs.size(2), outputs.size(3))
        
        # Squash function
        outputs = self.squash(outputs)
        
        return outputs.view(outputs.size(0), -1, self.dim_caps)
    
    def squash(self, tensor: torch.Tensor) -> torch.Tensor:
        """Squash function for capsules."""
        squared_norm = (tensor ** 2).sum(dim=2, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        unit_vector = tensor / torch.sqrt(squared_norm + 1e-8)
        
        return scale * unit_vector


class ForensicsCapsules(nn.Module):
    """Forensics capsules for deepfake detection."""
    
    def __init__(self, in_caps: int, out_caps: int, dim_caps: int, iterations: int = 3):
        """Initialize forensics capsules."""
        super(ForensicsCapsules, self).__init__()
        
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.dim_caps = dim_caps
        self.iterations = iterations
        
        self.W = nn.Parameter(torch.randn(1, in_caps, out_caps, dim_caps, 8))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic routing."""
        batch_size = x.size(0)
        
        # Expand weights for batch
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        
        # Compute predictions
        u_hat = torch.matmul(W, x.unsqueeze(2).unsqueeze(4))
        u_hat = u_hat.squeeze(4)
        
        # Dynamic routing
        b = torch.zeros(batch_size, self.in_caps, self.out_caps, 1).to(x.device)
        
        for i in range(self.iterations):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1, keepdim=True)
            v = self.squash(s)
            
            if i < self.iterations - 1:
                agreement = torch.matmul(u_hat, v.transpose(2, 3))
                b = b + agreement
        
        return v.squeeze(1)
    
    def squash(self, tensor: torch.Tensor) -> torch.Tensor:
        """Squash function for capsules."""
        squared_norm = (tensor ** 2).sum(dim=3, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        unit_vector = tensor / torch.sqrt(squared_norm + 1e-8)
        
        return scale * unit_vector


class ReconstructionNet(nn.Module):
    """Reconstruction network for capsule regularization."""
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 512):
        """Initialize reconstruction network."""
        super(ReconstructionNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, 28 * 28)  # Adjust based on input size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from capsule representation."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x.view(-1, 1, 28, 28)  # Adjust based on input size


def create_ensemble_detector(model_path: Path, device: str = 'cpu') -> EnsembleDeepfakeDetector:
    """
    Factory function to create and load ensemble detector.
    
    Args:
        model_path: Path to model directory
        device: Device to run on
        
    Returns:
        Loaded ensemble detector
    """
    detector = EnsembleDeepfakeDetector(model_path, device)
    
    if detector.load_models():
        logger.info("Ensemble detector created successfully")
        return detector
    else:
        logger.warning("Ensemble detector created but no models loaded")
        return detector