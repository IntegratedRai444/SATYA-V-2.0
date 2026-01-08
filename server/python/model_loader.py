"""
Model Loader for SatyaAI
Handles loading and managing all deepfake detection models.
"""
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.quantization
import torchaudio
import torchvision.models as models
from torch.jit import ScriptModule, script, trace
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision import transforms

logger = logging.getLogger(__name__)

class BaseModelLoader:
    """
    Base class for model loaders with enhanced features:
    - Model quantization for inference optimization
    - Thread-safe model loading
    - Performance monitoring
    - Automatic device selection
    - Model validation
    """
    
    def __init__(self, model_path: Path, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 quantize: bool = True,
                 precision: str = 'fp16'):
        """Initialize the model loader with advanced options.
        
        Args:
            model_path: Path to the model file or directory
            device: Device to load the model on ('cuda', 'cpu', or 'auto')
            quantize: Whether to apply quantization for faster inference
            precision: Model precision ('fp32', 'fp16', 'int8')
        """
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        self.model = None
        self.transform = None
        self.quantize = quantize
        self.precision = precision
        self.load_time = 0.0
        self.inference_time = 0.0
        self.inference_count = 0
        self._model_lock = threading.Lock()  # For thread-safe operations
        
        # Initialize transforms
        self._init_transforms()
        
        # Performance monitoring
        self.performance_metrics = {
            'total_inference_time': 0.0,
            'inference_count': 0,
            'avg_inference_time': 0.0,
            'last_inference_time': 0.0,
            'errors': []
        }
        
    def _get_device(self, device_str: str) -> torch.device:
        """Get the appropriate device for model execution."""
        if device_str == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_str)
        
    def _init_transforms(self) -> None:
        """Initialize data transformation pipeline."""
        # Default transforms - can be overridden by subclasses
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def load(self, force_reload: bool = False) -> bool:
        """
        Load the model from disk with error handling and performance tracking.
        
        Args:
            force_reload: Whether to force reloading the model even if already loaded
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        if self.model is not None and not force_reload:
            return True
            
        start_time = time.time()
        
        try:
            with self._model_lock:
                # Clear CUDA cache if using GPU
                if 'cuda' in str(self.device):
                    torch.cuda.empty_cache()
                
                # Load the model
                self._load_model_impl()
                
                # Move to device
                self.model = self.model.to(self.device)
                
                # Apply optimizations
                self._optimize_model()
                
                # Set to evaluation mode
                self.model.eval()
                
                # Verify model is working
                self._validate_model()
                
                self.load_time = time.time() - start_time
                logger.info(f"Model loaded successfully in {self.load_time:.2f}s")
                return True
                
        except Exception as e:
            self.performance_metrics['errors'].append({
                'time': datetime.now().isoformat(),
                'error': str(e),
                'type': 'load_error'
            })
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            self.model = None
            return False
    
    def _load_model_impl(self) -> None:
        """Implementation of model loading - to be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _load_model_impl()")
    
    def _optimize_model(self) -> None:
        """Apply optimizations like quantization and script/trace."""
        if self.quantize:
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
                    
            except Exception as e:
                logger.warning(f"Model optimization failed: {str(e)}")
                self.performance_metrics['errors'].append({
                    'time': datetime.now().isoformat(),
                    'error': f"Optimization failed: {str(e)}",
                    'type': 'optimization_error'
                })
    
    def _validate_model(self) -> None:
        """Run validation checks on the loaded model."""
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
    
    def preprocess(self, input_data: Any) -> torch.Tensor:
        """
        Preprocess input data for the model with validation.
        
        Args:
            input_data: Input data (numpy array, PIL Image, file path, etc.)
            
        Returns:
            Preprocessed tensor ready for model input
            
        Raises:
            ValueError: If input data is invalid
        """
        try:
            if isinstance(input_data, str):
                # Load from file
                if not os.path.exists(input_data):
                    raise FileNotFoundError(f"Input file not found: {input_data}")
                input_data = Image.open(input_data).convert('RGB')
                
            elif isinstance(input_data, np.ndarray):
                # Convert numpy array to PIL Image
                if input_data.dtype != np.uint8:
                    input_data = (input_data * 255).astype(np.uint8)
                input_data = Image.fromarray(input_data)
                
            # Apply transforms
            if self.transform is not None:
                input_data = self.transform(input_data)
                
            # Add batch dimension if needed
            if len(input_data.shape) == 3:
                input_data = input_data.unsqueeze(0)
                
            # Move to device
            input_data = input_data.to(self.device)
            
            # Convert precision if needed
            if self.precision == 'fp16' and self.device.type == 'cuda':
                input_data = input_data.half()
                
            return input_data
            
        except Exception as e:
            logger.error(f"Input preprocessing failed: {str(e)}")
            self.performance_metrics['errors'].append({
                'time': datetime.now().isoformat(),
                'error': f"Preprocessing failed: {str(e)}",
                'type': 'preprocessing_error'
            })
            raise
    
    def predict(self, input_data: Any, return_confidence: bool = True) -> Dict[str, Any]:
        """
        Run inference on the input data with performance tracking.
        
        Args:
            input_data: Input data to process
            return_confidence: Whether to include confidence scores
            
        Returns:
            Dictionary containing predictions and metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        start_time = time.time()
        
        try:
            with torch.no_grad():
                # Preprocess input
                input_tensor = self.preprocess(input_data)
                
                # Run inference
                inference_start = time.time()
                outputs = self.model(input_tensor)
                inference_time = time.time() - inference_start
                
                # Process outputs
                result = self._process_outputs(outputs, return_confidence)
                
                # Update performance metrics
                self._update_metrics(inference_time)
                
                return {
                    **result,
                    'inference_time': inference_time,
                    'device': str(self.device),
                    'model_loaded_in': self.load_time,
                    'total_inferences': self.performance_metrics['inference_count']
                }
                
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.performance_metrics['errors'].append({
                'time': datetime.now().isoformat(),
                'error': error_msg,
                'type': 'inference_error'
            })
            raise RuntimeError(error_msg) from e
    
    def _process_outputs(self, outputs: Any, return_confidence: bool) -> Dict[str, Any]:
        """Process model outputs into a standardized format."""
        if isinstance(outputs, torch.Tensor):
            # Standard classification output
            probs = F.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, 1)
            
            result = {
                'prediction': int(preds[0]),
                'probabilities': probs[0].cpu().numpy().tolist()
            }
            
            if return_confidence:
                result['confidence'] = float(confidence[0])
                
            return result
            
        elif isinstance(outputs, dict):
            # Handle dictionary outputs
            result = {k: v.cpu().numpy().tolist() for k, v in outputs.items()}
            return result
            
        else:
            raise ValueError(f"Unsupported model output type: {type(outputs)}")
    
    def _update_metrics(self, inference_time: float) -> None:
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
        return self.performance_metrics
    
    def profile(self, input_size: Tuple[int, int] = (224, 224), warmup: int = 10, runs: int = 100) -> Dict[str, Any]:
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
    
    def save_optimized(self, output_path: str, optimize_for_mobile: bool = False) -> None:
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


class EfficientNetB7Loader(BaseModelLoader):
    """
    Loader for EfficientNet-B7 model with optimizations for deepfake detection.
    
    Features:
    - Optimized for both CPU and GPU
    - Dynamic quantization support
    - Mixed precision training support
    - Batch processing
    """
    
    def __init__(self, model_path: Path, **kwargs):
        """
        Initialize the EfficientNet-B7 model loader.
        
        Args:
            model_path: Path to the model file or directory
            **kwargs: Additional arguments for BaseModelLoader
        """
        super().__init__(model_path, **kwargs)
        self.input_size = 224
        self.model_name = "efficientnet_b7"
        self.supported_precisions = ['fp32', 'fp16']
        self.min_input_size = (32, 32)
        self.max_input_size = (1024, 1024)
        
        # Update transforms for this specific model
        self._init_transforms()
    
    def _init_transforms(self) -> None:
        """Initialize data transformation pipeline for EfficientNet."""
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model_impl(self) -> None:
        """Load the EfficientNet-B7 model with pretrained weights."""
        # Initialize the model
        self.model = models.efficientnet_b7(pretrained=True)
        
        # Modify the classifier for binary classification
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, 2)  # Binary classification
        
        # Load custom weights if provided
        if self.model_path.is_file():
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # Handle different state dict formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Load state dict with strict=False to handle partial matches
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded custom weights from {self.model_path}")
        
        # Apply optimizations
        self._apply_optimizations()
    
    def _apply_optimizations(self) -> None:
        """Apply model-specific optimizations."""
        # Enable cudnn benchmark for faster convolutions
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
        
        # Enable channels last memory format for better performance on some GPUs
        if self.device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)
    
    def preprocess_batch(self, batch: List[Any]) -> torch.Tensor:
        """
        Preprocess a batch of images efficiently.
        
        Args:
            batch: List of input images (file paths, numpy arrays, or PIL Images)
            
        Returns:
            Batch tensor ready for model input
        """
        processed = []
        
        with ThreadPoolExecutor() as executor:
            # Process images in parallel
            futures = [executor.submit(self.preprocess, img) for img in batch]
            for future in as_completed(futures):
                try:
                    processed.append(future.result())
                except Exception as e:
                    logger.warning(f"Failed to preprocess image: {str(e)}")
        
        if not processed:
            raise ValueError("No valid images in batch")
            
        return torch.stack(processed)
    
    def predict_batch(self, batch: List[Any]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.
        
        Args:
            batch: List of input images
            
        Returns:
            List of prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        start_time = time.time()
        
        try:
            # Preprocess batch
            input_batch = self.preprocess_batch(batch)
            
            # Run inference
            with torch.no_grad():
                if self.precision == 'fp16' and self.device.type == 'cuda':
                    input_batch = input_batch.half()
                    
                outputs = self.model(input_batch)
                
                # Process outputs
                results = []
                probs = F.softmax(outputs, dim=1)
                
                for i in range(probs.size(0)):
                    confidence, pred = torch.max(probs[i], 0)
                    results.append({
                        'prediction': int(pred.item()),
                        'confidence': float(confidence.item()),
                        'probabilities': probs[i].cpu().numpy().tolist()
                    })
                
                # Update metrics
                batch_time = time.time() - start_time
                self._update_metrics(batch_time / len(batch))
                
                return results
                
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
            
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        


class XceptionLoader(BaseModelLoader):
    """Loader for Xception model with optimizations."""
    
    def _init_model(self) -> None:
        from torchvision.models import xception
        
        self.model = xception(pretrained=False, num_classes=2)
        
        # Optimize first layer
        self.model.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
    
    def _init_transforms(self) -> None:
        self.transforms = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _optimize_model(self) -> None:
        super()._optimize_model()
        # Additional Xception-specific optimizations
        if self.device.type == 'cuda':
            # Enable TF32 for Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


class FaceXRayLoader(BaseModelLoader):
    """Loader for Face X-Ray model with optimizations."""
    
    def _init_model(self) -> None:
        from torchvision.models import resnet18

        # Using ResNet18 as base for Face X-Ray
        self.model = resnet18(pretrained=False, num_classes=2)
        
        # Optimize first layer
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    
    def _init_transforms(self) -> None:
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_batch(self, batch: List[Any]) -> torch.Tensor:
        """
        Preprocess a batch of face images efficiently.
        
        Args:
            batch: List of input face images (file paths, numpy arrays, or PIL Images)
            
        Returns:
            Batch tensor ready for model input
        """
        processed = []
        
        with ThreadPoolExecutor() as executor:
            # Process images in parallel
            futures = [executor.submit(self.preprocess, img) for img in batch]
            for future in as_completed(futures):
                try:
                    processed.append(future.result())
                except Exception as e:
                    logger.warning(f"Failed to preprocess face image: {str(e)}")
        
        if not processed:
            raise ValueError("No valid face images in batch")
            
        return torch.stack(processed)
    
    def predict_batch(self, batch: List[Any]) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of face images.
        
        Args:
            batch: List of input face images
            
        Returns:
            List of prediction results with confidence scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        start_time = time.time()
        
        try:
            # Preprocess batch
            input_batch = self.preprocess_batch(batch)
            
            # Run inference
            with torch.no_grad():
                if self.precision == 'fp16' and self.device.type == 'cuda':
                    input_batch = input_batch.half()
                    
                outputs = self.model(input_batch)
                
                # Process outputs
                results = []
                probs = F.softmax(outputs, dim=1)
                
                for i in range(probs.size(0)):
                    confidence, pred = torch.max(probs[i], 0)
                    results.append({
                        'prediction': int(pred.item()),
                        'confidence': float(confidence.item()),
                        'probabilities': probs[i].cpu().numpy().tolist(),
                        'model': self.model_name
                    })
                
                # Update metrics
                batch_time = time.time() - start_time
                self._update_metrics(batch_time / len(batch))
                
                return results
                
        except Exception as e:
            logger.error(f"Face X-Ray prediction failed: {str(e)}", exc_info=True)
            raise
    
    def detect_manipulation(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Specialized method to detect face manipulation artifacts.
        
        Args:
            image: Input face image
            
        Returns:
            Dictionary with manipulation detection results
        """
        if self.model is None:
            self.load()
            
        # Preprocess the image
        input_tensor = self.preprocess(image)
        
        # Add batch dimension
        input_tensor = input_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            if self.precision == 'fp16' and self.device.type == 'cuda':
                input_tensor = input_tensor.half()
                
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)
            
            return {
                'is_manipulated': bool(pred.item() == 1),  # Assuming 1 is the manipulated class
                'confidence': float(confidence.item()),
                'probabilities': probs[0].cpu().numpy().tolist(),
                'model': self.model_name
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
            
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "name": self.model_name,
            "input_size": self.input_size,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "precision": self.precision,
            "quantized": self.quantize,
            "status": "loaded"
        }
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded EfficientNet-B7 model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load EfficientNet-B7 model: {e}")
            raise
    
    def preprocess(self, image):
        """Preprocess image for EfficientNet-B7."""
        if self.transform is not None:
            image = self.transform(image)
            image = image.unsqueeze(0).to(self.device)
        return image
    
    def predict(self, image) -> Dict[str, float]:
        """Predict if image is real or fake."""
        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            fake_prob = probs[0][1].item()
            
        return {
            'is_fake': fake_prob > 0.5,
            'fake_confidence': float(fake_prob),
            'real_confidence': float(1 - fake_prob)
        }


class XceptionLoader(BaseModelLoader):
    """Loader for Xception model."""
    
    def __init__(self, model_path: Path, **kwargs):
        super().__init__(model_path, **kwargs)
        self.input_size = 299
        self._init_transforms()
    
    def _init_transforms(self):
        """Initialize image transformations."""
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def load(self) -> None:
        """Load the Xception model."""
        try:
            # Xception model implementation would go here
            # This is a placeholder - you'll need to implement the actual model loading
            logger.info(f"Xception model loading from {self.model_path}")
            # Load model implementation here
            raise NotImplementedError("Xception model loading not implemented")
            
        except Exception as e:
            logger.error(f"Failed to load Xception model: {e}")
            raise


class ModelManager:
    """Manages all deepfake detection models."""
    
    def __init__(self, models_dir: Union[str, Path] = "models"):
        """Initialize the model manager.
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_models(self):
        """Load all available models."""
        try:
            # Load EfficientNet-B7
            effnet_path = self.models_dir / "dfdc_efficientnet_b7" / "model.pth"
            if effnet_path.exists():
                self.models['efficientnet_b7'] = EfficientNetB7Loader(effnet_path, device=self.device)
                self.models['efficientnet_b7'].load()
            
            # Load Xception
            xception_path = self.models_dir / "xception" / "model.pth"
            if xception_path.exists():
                self.models['xception'] = XceptionLoader(xception_path, device=self.device)
                self.models['xception'].load()
                
            logger.info(f"Loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def get_model(self, model_name: str) -> BaseModelLoader:
        """Get a loaded model by name."""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found or not loaded")
        return model
    
    def predict_image(self, image, model_name: str = 'efficientnet_b7') -> Dict[str, float]:
        """Predict if an image is real or fake."""
        model = self.get_model(model_name)
        processed_image = model.preprocess(image)
        return model.predict(processed_image)


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize model manager
    manager = ModelManager("../models")
    manager.load_models()
    
    # Example: Load and predict on an image
    from PIL import Image
    
    try:
        # Replace with actual image path
        image_path = "path_to_test_image.jpg"
        image = Image.open(image_path).convert('RGB')
        
        # Get prediction
        result = manager.predict_image(image)
        print(f"Prediction result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
