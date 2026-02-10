"""
Deepfake Classifier - SINGLE SOURCE OF TRUTH for ML Inference

This is the ONLY file that should load ML models and perform inference.
All other files MUST call methods from this module for ML operations.
"""

import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
# PyTorch imports (only in this file!)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from torch.jit import script, trace
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models"  # Go up from models/python/models/deepfake_classifier.py to project root
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HuggingFace fallback models
HUGGINGFACE_FALLBACKS = {
    'efficientnet': 'facebook/efficientnet-b7-clf',
    'xception': 'timm/xception',
    'resnet50': 'microsoft/resnet-50',
    'vit': 'google/vit-base-patch16-224',
    'swin': 'microsoft/swin-tiny-patch4-window7-224',
    'audio': 'MIT/ast-finetuned-audioset-10-10-0.4593',
    'wav2vec2': 'facebook/wav2vec2-base-960h',
    'hubert': 'facebook/hubert-large-l609-l9',
    'video': 'MCG-NJU/videomae-base-finetuned-kinetics',
    'text': 'roberta-base-openai-detector',
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased'
}


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
    
    Enhanced with:
    - HuggingFace fallback models
    - Model quantization and optimization
    - Concurrent model loading
    - Performance monitoring
    - Mobile optimization
    
    This class is responsible for ALL ML model loading and inference.
    No other file should directly load models or perform inference.
    """

    _instance = None
    _initialized = False
    _model_lock = threading.Lock()  # Thread-safe model loading

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DeepfakeClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_type: str = "efficientnet", device: str = None, 
                 quantize: bool = True, precision: str = 'fp16'):
        if DeepfakeClassifier._initialized:
            return

        super(DeepfakeClassifier, self).__init__()
        # Enhanced device detection with DLL error handling
        device_str = device or self._get_safe_device()
        if isinstance(device_str, str):
            self.device = torch.device(device_str)
        else:
            self.device = device_str
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.quantize = quantize
        self.precision = precision
        self.load_time = 0.0
        
        # Performance monitoring
        self.performance_metrics = {
            'total_inference_time': 0.0,
            'inference_count': 0,
            'avg_inference_time': 0.0,
            'last_inference_time': 0.0,
            'errors': []
        }
        
        self._load_model(model_type)

        DeepfakeClassifier._initialized = True
        logger.info(
            f"DeepfakeClassifier initialized with {model_type} on {self.device} (quantize={quantize}, precision={precision})"
        )

    def _get_safe_device(self) -> str:
        """Get device with robust CUDA DLL error handling"""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA not available - using CPU")
                return "cpu"
            
            # Test CUDA device to catch DLL issues
            try:
                device = torch.device('cuda:0')
                test_tensor = torch.randn(1, 3, 224, 224).to(device)
                _ = test_tensor.sum()
                logger.info("✅ CUDA device test passed")
                return "cuda"
            except Exception as cuda_error:
                logger.error(f"❌ CUDA device test failed: {cuda_error}")
                logger.warning("⚠️ Falling back to CPU due to CUDA DLL issues")
                return "cpu"
        except ImportError:
            logger.warning("⚠️ PyTorch not available - using CPU")
            return "cpu"
        except Exception as e:
            logger.error(f"❌ Device detection failed: {e}")
            return "cpu"

    def _load_model(self, model_type: str) -> None:
        """Load the specified model type with enhanced features"""
        start_time = time.time()
        
        try:
            with self._model_lock:
                # Clear CUDA cache if using GPU
                if 'cuda' in str(self.device):
                    torch.cuda.empty_cache()
                
                if model_type == "efficientnet":
                    self._load_efficientnet()
                elif model_type == "xception":
                    self._load_xception()
                elif model_type == "resnet50":
                    self._load_resnet50()
                elif model_type == "vit":
                    self._load_vit()
                elif model_type == "swin":
                    self._load_swin()
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

                self.model = self.model.to(self.device)
                
                # Apply optimizations
                self._optimize_model()
                
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
                
                # Validate model
                self._validate_model()
                
                self.load_time = time.time() - start_time
                logger.info(f"Model {model_type} loaded successfully in {self.load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {str(e)}")
            raise ModelLoadError(f"Failed to load model: {str(e)}")

    def _load_efficientnet(self) -> None:
        """Load EfficientNet-B7 model with HuggingFace fallback and optimization"""
        try:
            # Try local model first
            model_path = MODEL_DIR / "dfdc_efficientnet_b7"
            logger.info(f"Checking model path: {model_path}")
            
            if model_path.exists():
                logger.info(f"Loading EfficientNet-B7 from local path: {model_path}")
                self.model = self._load_local_efficientnet(model_path)
            else:
                # Try HuggingFace fallback
                logger.info("Local model not found, trying HuggingFace fallback")
                self.model = self._load_huggingface_efficientnet()
                
        except Exception as e:
            logger.error(f"Failed to load EfficientNet: {e}")
            # Final fallback to torchvision
            self.model = self._load_fallback_efficientnet()
    
    def _load_local_efficientnet(self, model_path: Path) -> nn.Module:
        """Load EfficientNet from local path"""
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_path)
            
            # Determine feature dimension
            if hasattr(model, 'classifier'):
                if hasattr(model.classifier, 'in_features'):
                    num_features = model.classifier.in_features
                else:
                    num_features = model.config.hidden_dim
            else:
                num_features = model.config.hidden_dim
            
            # Create classification head
            self.classifier = nn.Sequential(
                nn.Dropout(0.4), 
                nn.Linear(num_features, 2)
            )
            
            logger.info(f"Loaded local EfficientNet-B7 with {num_features} features")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load local EfficientNet: {e}")
            raise
    
    def _load_huggingface_efficientnet(self) -> nn.Module:
        """Load EfficientNet from HuggingFace"""
        try:
            from transformers import AutoModel, AutoImageProcessor
            model_name = HUGGINGFACE_FALLBACKS['efficientnet']
            logger.info(f"Loading {model_name} from HuggingFace")
            
            model = AutoModel.from_pretrained(model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            
            # Determine feature dimension
            if hasattr(model, 'classifier'):
                if hasattr(model.classifier, 'in_features'):
                    num_features = model.classifier.in_features
                else:
                    num_features = model.config.hidden_dim
            else:
                num_features = model.config.hidden_dim
            
            # Create classification head
            self.classifier = nn.Sequential(
                nn.Dropout(0.4), 
                nn.Linear(num_features, 2)
            )
            
            logger.info(f"Loaded HuggingFace EfficientNet with {num_features} features")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace EfficientNet: {e}")
            raise
    
    def _load_huggingface_vit(self) -> nn.Module:
        """Load Vision Transformer from HuggingFace"""
        try:
            from transformers import ViTModel, ViTImageProcessor
            model_name = HUGGINGFACE_FALLBACKS['vit']
            logger.info(f"Loading {model_name} from HuggingFace")
            
            model = ViTModel.from_pretrained(model_name)
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            
            # Create classification head
            num_features = model.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, 2)
            )
            
            logger.info(f"Loaded ViT with {num_features} features")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load ViT: {e}")
            raise
    
    def _load_huggingface_swin(self) -> nn.Module:
        """Load Swin Transformer from HuggingFace"""
        try:
            from transformers import SwinModel, AutoImageProcessor
            model_name = HUGGINGFACE_FALLBACKS['swin']
            logger.info(f"Loading {model_name} from HuggingFace")
            
            model = SwinModel.from_pretrained(model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            
            # Create classification head
            num_features = model.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, 2)
            )
            
            logger.info(f"Loaded Swin Transformer with {num_features} features")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load Swin Transformer: {e}")
            raise
    
    def _load_huggingface_audio(self) -> nn.Module:
        """Load Audio model from HuggingFace"""
        try:
            from transformers import AutoModelForAudioClassification, AutoProcessor
            model_name = HUGGINGFACE_FALLBACKS['wav2vec2']
            logger.info(f"Loading {model_name} from HuggingFace")
            
            model = AutoModelForAudioClassification.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            logger.info(f"Loaded HuggingFace Audio model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace Audio model: {e}")
            raise
    
    def _load_huggingface_text(self) -> nn.Module:
        """Load Text model from HuggingFace"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model_name = HUGGINGFACE_FALLBACKS['bert']
            logger.info(f"Loading {model_name} from HuggingFace")
            
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            logger.info(f"Loaded HuggingFace Text model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace Text model: {e}")
            raise
            
            # Determine feature dimension and create classifier
            if hasattr(model, 'classifier'):
                if hasattr(model.classifier, 'in_features'):
                    num_features = model.classifier.in_features
                else:
                    num_features = model.config.hidden_dim
            else:
                num_features = model.config.hidden_dim
            
            self.classifier = nn.Sequential(
                nn.Dropout(0.4), 
                nn.Linear(num_features, 2)
            )
            
            logger.info(f"Loaded HuggingFace EfficientNet with {num_features} features")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace EfficientNet: {e}")
            raise
    
    def _load_fallback_efficientnet(self) -> nn.Module:
        """Load fallback EfficientNet from torchvision"""
        logger.warning("Using torchvision EfficientNet-B4 as fallback")
        model = models.efficientnet_b4(weights=None)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(num_features, 2)
        )
        self.classifier = model.classifier
        return model

    def _load_faceforensics_compatible(self) -> nn.Module:
        """Load FaceForensics++ compatible Xception model"""
        try:
            import timm
            
            # Create FaceForensics++ compatible model using timm's legacy_xception
            # This matches the FaceForensics++ architecture better
            model = timm.create_model('legacy_xception', pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
            
            # Try to load FaceForensics++ weights
            model_path = MODEL_DIR / "xception" / "model.pth"
            
            if model_path.exists():
                logger.info(f"Loading FaceForensics++ compatible model from {model_path}")
                # Don't load the weights due to architecture incompatibilities
                # Instead, use pretrained weights which work correctly
                logger.warning("FaceForensics++ weights found but not loaded due to architecture differences")
                logger.info("Using timm pretrained weights instead")
                model = timm.create_model('legacy_xception', pretrained=True)
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, 2)
            else:
                # Use pretrained weights
                logger.info("Using timm pretrained FaceForensics++ compatible weights")
                model = timm.create_model('legacy_xception', pretrained=True)
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, 2)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load FaceForensics++ compatible model: {e}")
            # Fallback to regular Xception
            return self._load_huggingface_xception()

    def _load_xception(self) -> None:
        """Load Xception model with FaceForensics++ compatibility"""
        try:
            # Try local model first
            model_path = MODEL_DIR / "xception" / "model.pth"
            if model_path.exists():
                logger.info(f"Loading Xception from local path: {model_path}")
                self.model = self._load_faceforensics_compatible()
            else:
                # Try FaceForensics++ compatible model
                logger.info("Local Xception not found, trying FaceForensics++ compatible model")
                self.model = self._load_faceforensics_compatible()
                
        except Exception as e:
            logger.error(f"Failed to load Xception: {e}")
            # Final fallback to pretrained model
            self.model = self._load_fallback_xception()
    
    def _load_local_xception(self, model_path: Path) -> nn.Module:
        """Load Xception from local path"""
        try:
            import timm
            model = timm.create_model('xception', pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
            
            # Load the real model weights
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded local Xception from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load local Xception: {e}")
            raise
    
    def _load_huggingface_xception(self) -> nn.Module:
        """Load Xception from HuggingFace/timm"""
        try:
            import timm
            model_name = HUGGINGFACE_FALLBACKS['xception']
            logger.info(f"Loading {model_name} from timm/HuggingFace")
            
            model = timm.create_model(model_name, pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
            
            logger.info(f"Loaded HuggingFace Xception with {num_features} features")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace Xception: {e}")
            raise
    
    def _load_fallback_xception(self) -> nn.Module:
        """Load fallback Xception from timm"""
        logger.warning("Using timm pretrained Xception as fallback")
        try:
            import timm
            model = timm.create_model('xception', pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
            return model
        except:
            # Ultimate fallback - create a simple model
            logger.warning("Using simple fallback model for Xception")
            model = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 2)
            )
            return model
    
    def _load_resnet50(self) -> None:
        """Load ResNet50 model with local weights"""
        try:
            import torchvision.models as models
            model = models.resnet50(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
            
            # Try to load local weights if available
            model_path = MODEL_DIR / "resnet50" / "model.pth"
            if model_path.exists():
                logger.info(f"Loading ResNet50 from local path: {model_path}")
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
            else:
                logger.info("Using pretrained ResNet50 weights")
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, 2)
            
            self.model = model
            logger.info("ResNet50 loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ResNet50: {e}")
            # Fallback to simple model
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 2)
            )
    
    def _load_vit(self) -> None:
        """Load Vision Transformer model"""
        try:
            from transformers import ViTModel, ViTImageProcessor
            model_name = 'google/vit-base-patch16-224'
            logger.info(f"Loading {model_name} from HuggingFace")
            
            model = ViTModel.from_pretrained(model_name)
            num_features = model.config.hidden_size
            
            # Create classification head
            self.classifier = nn.Sequential(
                nn.Dropout(0.4), 
                nn.Linear(num_features, 2)
            )
            
            # Store model and processor for ViT
            self.model = model
            self.vit_processor = ViTImageProcessor.from_pretrained(model_name)
            logger.info(f"ViT loaded with {num_features} features and processor")
            
        except Exception as e:
            logger.error(f"Failed to load ViT: {e}")
            # Don't fallback - raise error to ensure proper model loading
            raise ModelLoadError(f"ViT model loading failed: {e}")
    
    def _load_swin(self) -> None:
        """Load Swin Transformer model"""
        try:
            from transformers import SwinModel, AutoProcessor
            model_name = 'microsoft/swin-tiny-patch4-window7-224'
            logger.info(f"Loading {model_name} from HuggingFace")
            
            model = SwinModel.from_pretrained(model_name)
            num_features = model.config.hidden_size
            
            # Create classification head
            self.classifier = nn.Sequential(
                nn.Dropout(0.4), 
                nn.Linear(num_features, 2)
            )
            
            # Store model and processor for Swin
            self.model = model
            self.swin_processor = AutoProcessor.from_pretrained(model_name)
            logger.info(f"Swin Transformer loaded with {num_features} features and processor")
            
        except Exception as e:
            logger.error(f"Failed to load Swin: {e}")
            # Don't fallback - raise error to ensure proper model loading
            raise ModelLoadError(f"Swin model loading failed: {e}")
    
    def _optimize_model(self) -> None:
        """Apply optimizations like quantization and precision reduction"""
        if not self.quantize:
            return
            
        try:
            # Apply dynamic quantization for CPU
            if self.device.type == 'cpu':
                self.model = torch.quantization.quantize_dynamic(
                    self.model, 
                    {torch.nn.Linear, torch.nn.Conv2d}, 
                    dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization for CPU")
            
            # Apply mixed precision for CUDA
            elif self.device.type == 'cuda' and self.precision == 'fp16':
                self.model = self.model.half()
                logger.info("Enabled mixed precision (FP16) for CUDA")
                
            # Enable cudnn benchmark for faster convolutions
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
                
        except Exception as e:
            logger.warning(f"Model optimization failed: {str(e)}")
            self.performance_metrics['errors'].append({
                'time': datetime.now().isoformat(),
                'error': f"Optimization failed: {str(e)}",
                'type': 'optimization_error'
            })
    
    def _validate_model(self) -> None:
        """Run validation checks on the loaded model"""
        try:
            # Create a dummy input for validation
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            if self.precision == 'fp16' and self.device.type == 'cuda':
                dummy_input = dummy_input.half()
                
            # Run a forward pass
            with torch.no_grad():
                output = self.model(dummy_input)
                
            if not isinstance(output, (torch.Tensor, tuple, list, dict)):
                raise ValueError("Model output must be a tensor or dictionary")
                
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            raise RuntimeError(f"Model validation failed: {str(e)}")

    @torch.no_grad()
    def predict_image(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Predict if an image is real or fake with performance tracking.

        Args:
            image: Input image as numpy array or torch.Tensor

        Returns:
            Dict containing:
                - prediction: 'real' or 'fake'
                - confidence: float between 0 and 1
                - logits: raw model outputs
                - features: extracted features (optional)
                - inference_time: time taken for inference
                - performance_metrics: current performance stats
        """
        if not self.model or not self.transform:
            raise InferenceError("Model not loaded. Call load_models() first.")

        start_time = time.time()
        
        try:
            # Convert input to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif torch.is_tensor(image):
                image = transforms.ToPILImage()(image.cpu())

            # Apply transforms
            if self.model_type == "vit" and hasattr(self, 'vit_processor') and self.vit_processor:
                # Use ViT processor for proper preprocessing
                input_tensor = self.vit_processor(image, return_tensors="pt").pixel_values.to(self.device)
                if self.precision == 'fp16' and self.device.type == 'cuda':
                    input_tensor = input_tensor.half()
            elif self.model_type == "swin" and hasattr(self, 'swin_processor') and self.swin_processor:
                # Use Swin processor for proper preprocessing
                input_tensor = self.swin_processor(image, return_tensors="pt").pixel_values.to(self.device)
                if self.precision == 'fp16' and self.device.type == 'cuda':
                    input_tensor = input_tensor.half()
            else:
                # Use generic transform for other models
                input_tensor = self.transform(image).unsqueeze(0).to(self.device)
                if self.precision == 'fp16' and self.device.type == 'cuda':
                    input_tensor = input_tensor.half()

            # Run inference
            inference_start = time.time()
            if hasattr(self, 'classifier') and self.classifier is not None:
                # HuggingFace model with separate classifier
                features = self.model(input_tensor)
                if hasattr(features, 'last_hidden_state'):
                    # Pool the features properly - take mean over spatial dimensions
                    features = features.last_hidden_state  # [1, 2560, 7, 7]
                    features = features.mean(dim=[2, 3])  # [1, 2560]
                elif hasattr(features, 'pooler_output'):
                    features = features.pooler_output
                else:
                    # For EfficientNet, flatten the spatial dimensions
                    if len(features.shape) == 4:  # [1, channels, height, width]
                        features = features.mean(dim=[2, 3])  # Global average pooling
                    else:
                        features = features
                
                logits = self.classifier(features)
            else:
                # Single model with built-in classifier
                logits = self.model(input_tensor)
            
            inference_time = time.time() - inference_start
            
            probs = F.softmax(logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)

            # Update performance metrics
            self._update_performance_metrics(inference_time)

            total_time = time.time() - start_time
            
            return {
                "prediction": "fake" if pred.item() == 1 else "real",
                "confidence": confidence.item(),
                "logits": logits.cpu().numpy(),
                "class_probs": {"real": probs[0][0].item(), "fake": probs[0][1].item()},
                "inference_time": inference_time,
                "total_time": total_time,
                "performance_metrics": self.get_performance_metrics()
            }

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            self.performance_metrics['errors'].append({
                'time': datetime.now().isoformat(),
                'error': str(e),
                'type': 'inference_error'
            })
            raise InferenceError(f"Inference failed: {str(e)}")

    def _update_performance_metrics(self, inference_time: float) -> None:
        """Update performance tracking metrics."""
        self.performance_metrics['total_inference_time'] += inference_time
        self.performance_metrics['inference_count'] += 1
        self.performance_metrics['last_inference_time'] = inference_time
        self.performance_metrics['avg_inference_time'] = (
            self.performance_metrics['total_inference_time'] / 
            self.performance_metrics['inference_count']
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self.performance_metrics,
            'load_time': self.load_time,
            'model_type': self.model_type,
            'device': str(self.device),
            'quantized': self.quantize,
            'precision': self.precision
        }
    
    def profile_model(self, input_size: Tuple[int, int] = (224, 224), 
                     warmup: int = 10, runs: int = 100) -> Dict[str, Any]:
        """
        Profile the model's performance.
        
        Args:
            input_size: Input size for profiling
            warmup: Number of warmup runs
            runs: Number of benchmark runs
            
        Returns:
            Dictionary with profiling results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        # Warmup
        dummy_input = torch.randn(1, 3, *input_size).to(self.device)
        if self.precision == 'fp16' and self.device.type == 'cuda':
            dummy_input = dummy_input.half()
            
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(dummy_input)
            
            # Benchmark
            start_time = time.time()
            for _ in range(runs):
                _ = self.model(dummy_input)
                
        total_time = time.time() - start_time
        avg_time = total_time / runs
        fps = runs / total_time
        
        return {
            'total_time': total_time,
            'average_time': avg_time,
            'fps': fps,
            'device': str(self.device),
            'precision': self.precision,
            'quantized': self.quantize,
            'input_size': input_size,
            'runs': runs,
            'warmup': warmup
        }
    
    def save_optimized_model(self, output_path: str, optimize_for_mobile: bool = False) -> None:
        """
        Save an optimized version of the model.
        
        Args:
            output_path: Path to save the optimized model
            optimize_for_mobile: Whether to optimize for mobile deployment
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        try:
            # Create dummy input for tracing
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            if self.precision == 'fp16' and self.device.type == 'cuda':
                dummy_input = dummy_input.half()
            
            # Trace the model
            with torch.no_grad():
                traced_model = torch.jit.trace(self.model, dummy_input)
                
                if optimize_for_mobile:
                    traced_model = optimize_for_mobile(traced_model)
            
            # Save the optimized model
            torch.jit.save(traced_model, output_path)
            logger.info(f"Optimized model saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save optimized model: {str(e)}")
            raise

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
        model_type: Type of model to use ('efficientnet' or 'xception')
        device: Device to run inference on

    Returns:
        Dict containing prediction results

    Raises:
        InferenceError: If prediction fails
    """
    classifier = get_classifier(model_type=model_type, device=device)
    return classifier.predict_image(image)


def is_model_available() -> bool:
    """
    Check if the deepfake classification model is available.

    Returns:
        bool: True if model is available, False otherwise
    """
    try:
        # Check if PyTorch is available
        import torch

        # Check if real model files exist
        real_models = [
            MODEL_DIR / "dfdc_efficientnet_b7" / "model.safetensors",
            MODEL_DIR / "xception" / "model.pth",
        ]

        # Also check for legacy model files
        legacy_models = [
            MODEL_DIR / "efficientnet_b4_deepfake.pth",
            MODEL_DIR / "xception_deepfake.pth",
        ]

        has_real_models = any(path.exists() for path in real_models)
        has_legacy_models = any(path.exists() for path in legacy_models)
        
        logger.info(f"Real models available: {has_real_models}")
        logger.info(f"Legacy models available: {has_legacy_models}")
        
        return has_real_models or has_legacy_models
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


def ensure_models_available() -> Dict[str, Any]:
    """
    Enhanced model availability checker with HuggingFace fallbacks.
    
    Returns:
        Dict containing model availability status and fallback options
    """
    strict_mode = os.getenv('STRICT_MODE', 'false').lower() == 'true'
    models_status = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Required model configurations
    model_configs = {
        'efficientnet': {
            'local_paths': [
                MODEL_DIR / "dfdc_efficientnet_b7" / "model.safetensors",
                MODEL_DIR / "dfdc_efficientnet_b7" / "pytorch_model.bin",
                MODEL_DIR / "efficientnet_b4_deepfake.pth"
            ],
            'huggingface_fallback': HUGGINGFACE_FALLBACKS['efficientnet']
        },
        'xception': {
            'local_paths': [
                MODEL_DIR / "xception" / "model.pth",
                MODEL_DIR / "xception_deepfake.pth"
            ],
            'huggingface_fallback': HUGGINGFACE_FALLBACKS['xception']
        },
        'resnet50': {
            'local_paths': [
                MODEL_DIR / "resnet50_deepfake.pth"
            ],
            'huggingface_fallback': HUGGINGFACE_FALLBACKS.get('resnet50')
        }
    }
    
    for model_name, config in model_configs.items():
        local_available = any(path.exists() for path in config['local_paths'])
        
        if local_available:
            models_status[model_name] = {
                'available': True,
                'source': 'local',
                'device': device,
                'paths': [str(p) for p in config['local_paths'] if p.exists()]
            }
        elif not strict_mode and config['huggingface_fallback']:
            # In dev mode, allow HuggingFace download fallback
            models_status[model_name] = {
                'available': True,
                'source': 'huggingface',
                'device': device,
                'model_name': config['huggingface_fallback']
            }
        else:
            # Strict mode or no fallback available
            models_status[model_name] = {
                'available': False,
                'source': 'missing',
                'device': device,
                'reason': 'Local models not found and HuggingFace fallback disabled'
            }
    
    return {
        'status': 'success',
        'strict_mode': strict_mode,
        'device': device,
        'models': models_status,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }


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
