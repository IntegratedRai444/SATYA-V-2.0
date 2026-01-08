"""
Ensemble Deepfake Detection System
Combines multiple state-of-the-art models for superior accuracy
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..utils.error_handler import ModelLoadError, error_handler
from .deepfake_classifier import DeepfakeClassifier, XceptionDeepfakeClassifier

logger = logging.getLogger(__name__)


class EnsembleDeepfakeDetector(nn.Module):
    """
    Advanced ensemble detector combining multiple specialized models with dynamic weighting
    and adaptive processing for optimal deepfake detection.

    Features:
    - Dynamic model weighting based on input characteristics
    - Batch processing for improved throughput
    - Comprehensive error handling and logging
    - Performance monitoring and metrics
    - Fallback mechanisms for model failures
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize ensemble detector with dynamic capabilities.

        Args:
            model_path: Path to model directory
            device: Device to run models on ('cuda' or 'cpu')
        """
        super(EnsembleDeepfakeDetector, self).__init__()

        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.models = {}
        self.model_weights = {}
        self.is_loaded = False
        self.performance_metrics = {}
        self.fallback_models = []

        # Enhanced model configurations with dynamic weight adjustment parameters
        self.model_configs = {
            "efficientnet_b7": {
                "class": DeepfakeClassifier,
                "input_size": (224, 224),
                "base_weight": 0.35,
                "weight": 0.35,
                "specialty": "general_detection",
                "min_weight": 0.2,
                "max_weight": 0.5,
                "confidence_threshold": 0.7,
            },
            "xception": {
                "class": XceptionDeepfakeClassifier,
                "input_size": (299, 299),
                "base_weight": 0.25,
                "weight": 0.25,
                "specialty": "face_manipulation",
                "min_weight": 0.15,
                "max_weight": 0.4,
                "confidence_threshold": 0.7,
            },
            "face_xray": {
                "class": FaceXRayDetector,
                "input_size": (224, 224),
                "base_weight": 0.25,
                "weight": 0.25,
                "specialty": "blending_artifacts",
                "min_weight": 0.15,
                "max_weight": 0.4,
                "confidence_threshold": 0.65,
            },
            "capsule_forensics": {
                "class": CapsuleForensicsDetector,
                "input_size": (256, 256),
                "base_weight": 0.15,
                "weight": 0.15,
                "specialty": "temporal_consistency",
                "min_weight": 0.1,
                "max_weight": 0.3,
                "confidence_threshold": 0.6,
            },
        }

        # Initialize performance metrics
        self._init_performance_metrics()
        logger.info(
            f"Ensemble detector initialized with {len(self.model_configs)} models on device: {self.device}"
        )

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
                    model = config["class"]()
                    state_dict = torch.load(model_file, map_location=self.device)

                    # Handle different state dict formats
                    if "model_state_dict" in state_dict:
                        model.load_state_dict(state_dict["model_state_dict"])
                    elif "state_dict" in state_dict:
                        model.load_state_dict(state_dict["state_dict"])
                    else:
                        model.load_state_dict(state_dict)

                    model.eval()
                    model.to(self.device)

                    self.models[model_name] = model
                    self.model_weights[model_name] = config["weight"]
                    loaded_count += 1

                    logger.info(f"Loaded {model_name} model")
                else:
                    logger.warning(f"Model file not found: {model_file}")

            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")

        if loaded_count > 0:
            # Normalize weights for loaded models only
            total_weight = sum(self.model_weights.values())
            self.model_weights = {
                k: v / total_weight for k, v in self.model_weights.items()
            }

            self.is_loaded = True
            logger.info(
                f"Ensemble loaded with {loaded_count}/{len(self.model_configs)} models"
            )
            return True
        else:
            logger.error("No models could be loaded for ensemble")
            return False

    def _update_model_weights(self, model_name: str, confidence: float) -> None:
        """
        Dynamically adjust model weights based on prediction confidence.

        Args:
            model_name: Name of the model
            confidence: Prediction confidence (0-1)
        """
        if model_name not in self.model_configs:
            return

        config = self.model_configs[model_name]
        confidence_ratio = min(1.0, confidence / config["confidence_threshold"])

        # Adjust weight based on confidence
        if confidence > config["confidence_threshold"]:
            # Increase weight if confidence is high
            new_weight = min(
                config["base_weight"] * (1 + 0.5 * (confidence_ratio - 1)),
                config["max_weight"],
            )
        else:
            # Decrease weight if confidence is low
            new_weight = max(
                config["base_weight"] * confidence_ratio, config["min_weight"]
            )

        # Smooth weight update
        config["weight"] = 0.9 * config["weight"] + 0.1 * new_weight

    def _init_performance_metrics(self) -> None:
        """Initialize performance tracking metrics."""
        for model_name in self.model_configs:
            self.performance_metrics[model_name] = {
                "total_predictions": 0,
                "correct_predictions": 0,
                "total_confidence": 0.0,
                "last_10_confidences": [],
                "avg_processing_time": 0.0,
                "last_used": None,
            }

    def _update_performance_metrics(
        self,
        model_name: str,
        is_correct: bool,
        confidence: float,
        processing_time: float,
    ) -> None:
        """Update performance metrics for a model."""
        if model_name not in self.performance_metrics:
            return

        metrics = self.performance_metrics[model_name]
        metrics["total_predictions"] += 1
        metrics["correct_predictions"] += int(is_correct)
        metrics["total_confidence"] += confidence
        metrics["last_10_confidences"].append(confidence)
        metrics["last_10_confidences"] = metrics["last_10_confidences"][-10:]
        metrics["avg_processing_time"] = (
            0.9 * metrics["avg_processing_time"] + 0.1 * processing_time
        )
        metrics["last_used"] = datetime.now()

    def _get_fallback_models(self) -> List[str]:
        """Get list of models to use as fallbacks."""
        if not self.fallback_models:
            # Sort models by accuracy (descending)
            return sorted(
                self.model_configs.keys(),
                key=lambda x: (
                    self.performance_metrics[x]["correct_predictions"]
                    / max(1, self.performance_metrics[x]["total_predictions"]),
                    -self.performance_metrics[x]["avg_processing_time"],
                ),
                reverse=True,
            )
        return self.fallback_models

    @error_handler
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process a batch of images using the ensemble.

        Args:
            images: List of input images as numpy arrays (RGB)

        Returns:
            List of prediction results for each image
        """
        if not self.is_loaded:
            raise ModelLoadError("Ensemble models not loaded")

        if not images:
            return []

        # Process each image in the batch
        results = []
        for img in images:
            try:
                results.append(self.predict_single(img))
            except Exception as e:
                logger.error(f"Error processing image in batch: {str(e)}")
                results.append(
                    {"error": str(e), "prediction": "error", "confidence": 0.0}
                )

        return results

    @error_handler
    def predict_single(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict using ensemble of models with dynamic weighting.

        Args:
            image: Input image as numpy array (RGB)

        Returns:
            Ensemble prediction results with confidence scores
        """
        if not self.is_loaded:
            raise ModelLoadError("Ensemble models not loaded")

        start_time = time.time()
        predictions = {}

        # Get predictions from each model with error handling
        for model_name, model in self.models.items():
            try:
                model_start = time.time()
                config = self.model_configs[model_name]

                # Preprocess image for this model
                processed_image = self._preprocess_for_model(
                    image, config["input_size"]
                )

                # Get prediction with timing
                with torch.no_grad():
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(processed_image)
                    else:
                        logits = model(processed_image)
                        probs = F.softmax(logits, dim=1)

                # Extract probabilities
                fake_prob = probs[0][0].item()
                real_prob = probs[0][1].item()
                confidence = max(fake_prob, real_prob)
                prediction = "fake" if fake_prob > real_prob else "real"

                # Update model performance metrics
                processing_time = time.time() - model_start
                self._update_performance_metrics(
                    model_name=model_name,
                    is_correct=True,  # Ground truth would be needed for actual correctness
                    confidence=confidence,
                    processing_time=processing_time,
                )

                # Store prediction
                predictions[model_name] = {
                    "fake_prob": fake_prob,
                    "real_prob": real_prob,
                    "prediction": prediction,
                    "confidence": confidence,
                    "specialty": config["specialty"],
                    "processing_time": processing_time,
                }

                # Update model weights based on confidence
                self._update_model_weights(model_name, confidence)

            except Exception as e:
                logger.error(f"Error in model {model_name}: {str(e)}", exc_info=True)
                # Reduce weight for failed models
                if model_name in self.model_configs:
                    self.model_configs[model_name]["weight"] = max(
                        self.model_configs[model_name]["min_weight"],
                        self.model_configs[model_name]["weight"] * 0.8,
                    )

        # If no predictions were successful, try fallback models
        if not predictions and self.fallback_models:
            logger.warning("Primary models failed, trying fallback models")
            return self._predict_with_fallback(image)
        elif not predictions:
            raise RuntimeError("All models failed to make predictions")

        # Combine predictions using dynamic weights
        ensemble_result = self._combine_predictions(predictions)

        # Log performance metrics
        total_time = time.time() - start_time
        logger.info(
            f"Prediction completed in {total_time:.4f}s with {len(predictions)} models"
        )

        return {
            **ensemble_result,
            "models_used": list(predictions.keys()),
            "processing_time": total_time,
            "model_metrics": {
                name: {
                    "weight": self.model_configs[name]["weight"],
                    "avg_confidence": self.performance_metrics[name]["total_confidence"]
                    / max(1, self.performance_metrics[name]["total_predictions"]),
                    "accuracy": self.performance_metrics[name]["correct_predictions"]
                    / max(1, self.performance_metrics[name]["total_predictions"]),
                    "avg_processing_time": self.performance_metrics[name][
                        "avg_processing_time"
                    ],
                }
                for name in predictions.keys()
            },
        }

    # Alias for backward compatibility
    predict = predict_single

    def _preprocess_for_model(
        self, image: np.ndarray, target_size: Tuple[int, int]
    ) -> torch.Tensor:
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

    def _predict_with_fallback(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Attempt prediction using fallback models when primary models fail.

        Args:
            image: Input image as numpy array (RGB)

        Returns:
            Prediction from fallback models
        """
        fallback_models = self._get_fallback_models()
        for model_name in fallback_models:
            try:
                if model_name not in self.models:
                    continue

                model = self.models[model_name]
                config = self.model_configs[model_name]

                # Preprocess and predict
                processed_image = self._preprocess_for_model(
                    image, config["input_size"]
                )

                with torch.no_grad():
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(processed_image)
                    else:
                        logits = model(processed_image)
                        probs = F.softmax(logits, dim=1)

                fake_prob = probs[0][0].item()
                real_prob = probs[0][1].item()
                confidence = max(fake_prob, real_prob)

                return {
                    "fake_prob": fake_prob,
                    "real_prob": real_prob,
                    "prediction": "fake" if fake_prob > real_prob else "real",
                    "confidence": confidence,
                    "model_used": f"fallback_{model_name}",
                    "warning": "Used fallback model due to primary model failures",
                }

            except Exception as e:
                logger.warning(f"Fallback model {model_name} failed: {str(e)}")

        raise RuntimeError(
            "All models, including fallbacks, failed to make predictions"
        )

    def _combine_predictions(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Combine individual model predictions using dynamic weighted ensemble.

        Args:
            predictions: Dictionary of model predictions with their confidences

        Returns:
            Combined ensemble prediction with confidence and model contributions
        """
        weighted_fake_prob = 0.0
        weighted_real_prob = 0.0
        total_weight = 0.0

        confidence_scores = []
        # Calculate dynamic weights based on model confidence and historical performance
        weighted_predictions = []
        total_weight = 0.0

        for name, pred in predictions.items():
            if name not in self.model_configs:
                continue

            # Base weight from config
            base_weight = self.model_configs[name]["weight"]

            # Adjust weight based on confidence
            confidence_factor = min(
                1.0, pred["confidence"] / 0.9
            )  # Normalize confidence

            # Consider model's historical performance
            metrics = self.performance_metrics.get(name, {})
            accuracy_factor = metrics.get(
                "accuracy", 0.8
            )  # Default to 0.8 if not available

            # Calculate final weight with decay for frequently wrong models
            weight = base_weight * confidence_factor * accuracy_factor

            # Apply minimum weight threshold
            min_weight = self.model_configs[name].get("min_weight", 0.1)
            weight = max(weight, min_weight)

            weighted_predictions.append({**pred, "weight": weight, "model_name": name})
            total_weight += weight

        if not weighted_predictions:
            raise ValueError("No valid predictions to combine")

        # Normalize weights
        for wp in weighted_predictions:
            wp["normalized_weight"] = wp["weight"] / total_weight
        weighted_fake_prob /= total_weight
        weighted_real_prob /= total_weight

        # Final prediction
        final_prediction = "fake" if weighted_fake_prob > weighted_real_prob else "real"
        final_confidence = max(weighted_fake_prob, weighted_real_prob)

        # Calculate consensus metrics
        vote_consensus = np.mean(individual_votes)
        confidence_std = np.std(confidence_scores) if confidence_scores else 0.0

        return {
            "prediction": final_prediction,
            "confidence": float(final_confidence),
            "fake_probability": float(weighted_fake_prob),
            "real_probability": float(weighted_real_prob),
            "vote_consensus": float(vote_consensus),
            "confidence_std": float(confidence_std),
            "ensemble_method": "weighted_average",
            "agreement_level": "high"
            if confidence_std < 0.1
            else "medium"
            if confidence_std < 0.2
            else "low",
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
            nn.Linear(256, num_classes),
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
        class_probs = torch.sqrt((forensics_caps**2).sum(dim=2))

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

        self.conv = nn.Conv2d(
            in_channels, out_channels * dim_caps, kernel_size=9, stride=2, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through primary capsules."""
        outputs = self.conv(x)
        outputs = outputs.view(
            outputs.size(0),
            self.num_caps,
            self.dim_caps,
            outputs.size(2),
            outputs.size(3),
        )

        # Squash function
        outputs = self.squash(outputs)

        return outputs.view(outputs.size(0), -1, self.dim_caps)

    def squash(self, tensor: torch.Tensor) -> torch.Tensor:
        """Squash function for capsules."""
        squared_norm = (tensor**2).sum(dim=2, keepdim=True)
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
        squared_norm = (tensor**2).sum(dim=3, keepdim=True)
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


def create_ensemble_detector(
    model_path: Path, device: str = "cpu"
) -> EnsembleDeepfakeDetector:
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
