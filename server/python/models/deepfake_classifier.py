"""
Deepfake Classifier - SINGLE SOURCE OF TRUTH for ML Inference

This is the ONLY file that should load ML models and perform inference.
All other files MUST call methods from this module for ML operations.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
# PyTorch imports (only in this file!)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path(__file__).parent / ".." / ".." / "models"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Custom exceptions
class ModelLoadError(Exception):
    """Raised when a model fails to load"""

    pass


class InferenceError(Exception):
    """Raised when inference fails"""

    pass


class DeepfakeClassifier(nn.Module):
    """
    Centralized Deepfake Classifier - The SINGLE point of ML inference.

    This class is responsible for ALL ML model loading and inference.
    No other file should directly load models or perform inference.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DeepfakeClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_type: str = "efficientnet", device: str = None):
        if DeepfakeClassifier._initialized:
            return

        super(DeepfakeClassifier, self).__init__()
        self.device = device or DEFAULT_DEVICE
        self.model_type = model_type
        self.model = None
        self.transform = None
        self._load_model(model_type)

        DeepfakeClassifier._initialized = True
        logger.info(
            f"DeepfakeClassifier initialized with {model_type} on {self.device}"
        )

    def _load_model(self, model_type: str) -> None:
        """Load the specified model type"""
        try:
            if model_type == "efficientnet":
                self._load_efficientnet()
            elif model_type == "xception":
                self._load_xception()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            self.model = self.model.to(self.device)
            self.model.eval()

            # Set up transforms
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {str(e)}")
            raise ModelLoadError(f"Failed to load model: {str(e)}")

    def _load_efficientnet(self) -> None:
        """Load EfficientNet-B4 model"""
        self.model = models.efficientnet_b4(pretrained=False)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(num_features, 2)  # Binary classification
        )

        # Load weights if available
        model_path = MODEL_DIR / "efficientnet_b4_deepfake.pth"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

    def _load_xception(self) -> None:
        """Load Xception model"""
        self.model = models.xception(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)  # Binary classification

        # Load weights if available
        model_path = MODEL_DIR / "xception_deepfake.pth"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def predict_image(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Predict if an image is real or fake.

        Args:
            image: Input image as numpy array or torch.Tensor

        Returns:
            Dict containing:
                - prediction: 'real' or 'fake'
                - confidence: float between 0 and 1
                - logits: raw model outputs
                - features: extracted features (optional)
        """
        if not self.model or not self.transform:
            raise InferenceError("Model not loaded. Call load_models() first.")

        try:
            # Convert input to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif torch.is_tensor(image):
                image = transforms.ToPILImage()(image.cpu())

            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Run inference
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)

            return {
                "prediction": "fake" if pred.item() == 1 else "real",
                "confidence": confidence.item(),
                "logits": logits.cpu().numpy(),
                "class_probs": {"real": probs[0][0].item(), "fake": probs[0][1].item()},
            }

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise InferenceError(f"Inference failed: {str(e)}")

    @torch.no_grad()
    def predict_batch(
        self, images: List[Union[np.ndarray, torch.Tensor]]
    ) -> List[Dict[str, Any]]:
        """
        Predict on a batch of images.

        Args:
            images: List of PIL Images, numpy arrays, or torch.Tensors

        Returns:
            List of prediction dicts (same format as predict_image)
        """
        if not self.model or not self.transform:
            raise InferenceError("Model not loaded. Call load_models() first.")

        try:
            # Convert all images to tensors
            input_tensors = []
            for img in images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                elif torch.is_tensor(img):
                    img = transforms.ToPILImage()(img.cpu())
                input_tensors.append(self.transform(img))

            # Stack into batch
            batch = torch.stack(input_tensors).to(self.device)

            # Run batch inference
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            # Convert to list of dicts
            results = []
            for i in range(len(images)):
                results.append(
                    {
                        "prediction": "fake" if preds[i].item() == 1 else "real",
                        "confidence": confidences[i].item(),
                        "logits": logits[i].cpu().numpy(),
                        "class_probs": {
                            "real": probs[i][0].item(),
                            "fake": probs[i][1].item(),
                        },
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Batch inference failed: {str(e)}")
            raise InferenceError(f"Batch inference failed: {str(e)}")

    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference"""
        return self.model is not None and self.transform is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        if not self.model:
            raise InferenceError("Model not loaded. Call load_models() first.")

        return self.model(x)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dict containing model metadata
        """
        if not self.model:
            return {
                "status": "not_loaded",
                "device": self.device,
                "model_type": self.model_type,
            }

        return {
            "status": "loaded",
            "device": self.device,
            "model_type": self.model_type,
            "parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "device_memory": torch.cuda.memory_allocated(self.device)
            if "cuda" in str(self.device)
            else 0,
        }


# Global instance of the classifier
_classifier_instance = None


def get_classifier(
    model_type: str = "efficientnet", device: str = None
) -> "DeepfakeClassifier":
    """
    Get or create the global classifier instance.

    Args:
        model_type: Type of model to use ('efficientnet' or 'xception')
        device: Device to run the model on ('cuda' or 'cpu')

    Returns:
        DeepfakeClassifier instance

    Raises:
        ModelLoadError: If the model fails to load
    """
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = DeepfakeClassifier(model_type=model_type, device=device)

    if not _classifier_instance.is_ready():
        raise ModelLoadError("Failed to initialize deepfake classifier")

    return _classifier_instance


def predict_image(
    image: Union[np.ndarray, torch.Tensor],
    model_type: str = "efficientnet",
    device: str = None,
) -> Dict[str, Any]:
    """
    Predict if an image is a deepfake using the centralized ML classifier.

    This is the ONLY function that should be called for ML inference.
    All other files must use this function for predictions.

    Args:
        image: Input image (PIL.Image, numpy array, or torch.Tensor)
        images: List of input images
        model_type: Type of model to use
        device: Device to run inference on

    Returns:
        List of prediction dictionaries

    Raises:
        InferenceError: If prediction fails
    """
    classifier = get_classifier(model_type=model_type, device=device)
    return classifier.predict_batch(images)


def is_model_available() -> bool:
    """
    Check if the deepfake classification model is available.

    Returns:
        bool: True if model is available, False otherwise
    """
    try:
        # Check if PyTorch is available
        import torch

        # Check if required models exist
        required_models = [
            MODEL_DIR / "efficientnet_b4_deepfake.pth",
            MODEL_DIR / "xception_deepfake.pth",
        ]

        return any(path.exists() for path in required_models)
    except ImportError:
        return False


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded model.

    Returns:
        Dict containing model metadata
    """
    if _classifier_instance is None:
        return {"status": "not_initialized"}
    return _classifier_instance.get_model_info()


def ensure_models_downloaded() -> bool:
    """
    Ensure that required model files are downloaded.

    Returns:
        bool: True if all models are available, False otherwise
    """
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # List of required model files and their download URLs
        models_to_download = {
            "efficientnet_b4_deepfake.pth": "https://example.com/models/efficientnet_b4_deepfake.pth",
            "xception_deepfake.pth": "https://example.com/models/xception_deepfake.pth",
        }

        all_downloaded = True

        for filename, url in models_to_download.items():
            model_path = MODEL_DIR / filename
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                all_downloaded = False

                # Here you would implement the actual download logic
                # For now, we'll just log a warning
                logger.warning(
                    f"Please download {filename} from {url} and save it to {MODEL_DIR}"
                )

        return all_downloaded
    except Exception as e:
        logger.error(f"Error checking/downloading models: {str(e)}")
        return False


def _cleanup():
    """Clean up resources when the module is unloaded"""
    global _classifier_instance
    if _classifier_instance is not None:
        del _classifier_instance
        _classifier_instance = None

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Register cleanup for when the module is unloaded
import atexit

atexit.register(_cleanup)
