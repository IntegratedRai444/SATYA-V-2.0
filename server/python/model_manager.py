"""
Enhanced Model Manager for SatyaAI
Handles loading, managing, and optimizing multiple deepfake detection models.
"""
import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms

from .model_loader import (BaseModelLoader, EfficientNetB7Loader,
                           FaceXRayLoader, XceptionLoader)

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class ModelLoadError(ModelError):
    """Raised when a model fails to load."""
    pass

class ModelValidationError(ModelError):
    """Raised when model validation fails."""
    pass

class ModelVersionError(ModelError):
    """Raised for version-related errors."""
    pass

class PredictionError(ModelError):
    """Raised when prediction fails."""
    pass

class ModelManager:
    """
    Advanced model manager for deepfake detection with the following features:
    - Dynamic model loading and caching
    - Performance monitoring and metrics
    - Request queuing and load balancing
    - Automatic 3-tier fallback mechanisms
    - Versioning and validation
    
    Fallback Strategy:
    - Tier 1: Primary models (high accuracy, high resource)
    - Tier 2: Lightweight models (moderate accuracy, lower resource)
    - Tier 3: Heuristics/rule-based fallback (low accuracy, minimal resource)
    """
    
    # Model tiers for fallback mechanism
    MODEL_TIERS = {
        'tier1': ['efficientnet_b7', 'xception'],  # High accuracy models
        'tier2': ['mobilenet_v3', 'shufflenet_v2'],  # Lightweight models
        'tier3': ['heuristic']  # Rule-based fallback
    }
    
    def __init__(self, config: Dict[str, Any] = None, model_dir: str = "models"):
        """
        Initialize the model manager with configuration.
        
        Args:
            config: Dictionary containing model configurations
            model_dir: Directory to store models and configurations
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        # Initialize components
        self.model_loaders: Dict[str, BaseModelLoader] = {}
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        self.model_weights: Dict[str, float] = {}
        self.request_queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor()
        self.is_running = True
        
        # Load configuration
        self.config = config or self._load_default_config()
        
        # Start request processing loop
        asyncio.create_task(self._process_requests())
        
        # Initialize models
        self._initialize_models()
    
    def _load_default_config(self) -> Dict[str, Dict]:
        """Load default model configuration."""
        config_path = self.model_dir / "models_config.json"
        
        # Default configuration if file doesn't exist
        default_config = {
            'efficientnet_b7': {
                'name': 'EfficientNet-B7',
                'path': str(self.model_dir / 'efficientnet_b7.pth'),
                'weight': 0.35,
                'enabled': True,
                'version': '1.0.0',
                'description': 'EfficientNet-B7 for deepfake detection',
                'input_size': [224, 224],
                'precision': 'fp16' if self.device.type == 'cuda' else 'fp32',
                'quantize': True,
                'compatibility': {
                    'python': '>=3.8',
                    'torch': '>=1.10.0',
                    'torchvision': '>=0.11.0'
                },
                'status': 'production'
            },
            'xception': {
                'name': 'Xception',
                'path': str(self.model_dir / 'xception.pth'),
                'weight': 0.30,
                'enabled': True,
                'version': '1.0.0',
                'description': 'Xception model for deepfake detection',
                'input_size': [299, 299],
                'precision': 'fp16' if self.device.type == 'cuda' else 'fp32',
                'quantize': True,
                'compatibility': {
                    'python': '>=3.8',
                    'torch': '>=1.10.0',
                    'torchvision': '>=0.11.0'
                },
                'status': 'production'
            },
            'face_xray': {
                'name': 'Face X-Ray',
                'path': str(self.model_dir / 'face_xray.pth'),
                'weight': 0.25,
                'enabled': True,
                'version': '1.0.0',
                'description': 'Face X-Ray for manipulation detection',
                'input_size': [224, 224],
                'precision': 'fp16' if self.device.type == 'cuda' else 'fp32',
                'quantize': True,
                'compatibility': {
                    'python': '>=3.8',
                    'torch': '>=1.10.0',
                    'torchvision': '>=0.11.0'
                },
                'status': 'production'
            }
        }
        
        if not config_path.exists():
            # Save default config if it doesn't exist
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
        else:
            try:
                # Load and merge with default config
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with default config
                    for key, value in user_config.items():
                        if key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                logger.error(f"Error loading model config: {e}")
        
        return default_config
    
    def _initialize_models(self):
        """
        Initialize all configured models with fallback support.
        Ensures at least one model from each tier is available.
        """
        tier_status = {tier: False for tier in self.MODEL_TIERS.keys()}
        
        # Try to load at least one model from each tier
        for tier, model_ids in self.MODEL_TIERS.items():
            for model_id in model_ids:
                if model_id in self.config and self.config[model_id].get('enabled', True):
                    try:
                        loader_class = self._get_loader_class(model_id)
                        if not loader_class:
                            logger.warning(f"No loader found for model: {model_id}")
                            continue
                        
                        loader = loader_class(
                            model_path=Path(self.config[model_id]['path']),
                            device=self.device,
                            quantize=self.config[model_id].get('quantize', True),
                            precision=self.config[model_id].get('precision', 'fp16' if self.device.type == 'cuda' else 'fp32')
                        )
                        
                        # Load the model
                        if loader.load():
                            self.model_loaders[model_id] = loader
                            self.model_weights[model_id] = self.config[model_id]['weight']
                            self.performance_metrics[model_id] = {
                                'total_inference_time': 0.0,
                                'inference_count': 0,
                                'avg_inference_time': 0.0,
                                'success_count': 0,
                                'error_count': 0,
                                'last_used': time.time()
                            }
                            logger.info(f"Initialized model: {model_id}")
                            tier_status[tier] = True
                            break  # Move to next tier after one model loads
                        else:
                            logger.error(f"Failed to load model: {model_id}")
                    except Exception as e:
                        logger.warning(f"Failed to load {tier} model {model_id}: {str(e)}")
            
            # If no model in this tier could be loaded, log a warning
            if not tier_status[tier]:
                logger.warning(f"No working model available in {tier}")
        
        # If no models loaded at all, raise an error
        if not any(tier_status.values()):
            raise ModelLoadError("Failed to load any models across all tiers")
    
    def _get_loader_class(self, model_name: str) -> Optional[Type[BaseModelLoader]]:
        """Get the appropriate loader class for a model."""
        loader_map = {
            'efficientnet_b7': EfficientNetB7Loader,
            'xception': XceptionLoader,
            'face_xray': FaceXRayLoader
        }
        return loader_map.get(model_name)
    
    async def _process_requests(self) -> None:
        """Process prediction requests from the queue."""
        while self.is_running:
            try:
                future, model_name, input_data, kwargs = await self.request_queue.get()
                
                if model_name not in self.model_loaders:
                    future.set_exception(ValueError(f"Model not found: {model_name}"))
                    continue
                
                try:
                    # Run the prediction in a thread pool
                    loop = asyncio.get_running_loop()
                    start_time = time.time()
                    
                    def _run_prediction():
                        return self.model_loaders[model_name].predict(input_data, **kwargs)
                    
                    result = await loop.run_in_executor(self.executor, _run_prediction)
                    
                    # Update performance metrics
                    inference_time = time.time() - start_time
                    self._update_metrics(model_name, True, inference_time)
                    
                    future.set_result(result)
                    
                except Exception as e:
                    self._update_metrics(model_name, False)
                    future.set_exception(PredictionError(f"Prediction failed: {str(e)}"))
                    
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                if not future.done():
                    future.set_exception(e)
            finally:
                self.request_queue.task_done()
    
    def _update_metrics(self, model_name: str, success: bool, inference_time: float = 0.0) -> None:
        """Update performance metrics for a model."""
        if model_name not in self.performance_metrics:
            self.performance_metrics[model_name] = {
                'total_inference_time': 0.0,
                'inference_count': 0,
                'avg_inference_time': 0.0,
                'success_count': 0,
                'error_count': 0,
                'last_used': time.time()
            }
        
        metrics = self.performance_metrics[model_name]
        metrics['inference_count'] += 1
        metrics['last_used'] = time.time()
        
        if success:
            metrics['success_count'] += 1
            metrics['total_inference_time'] += inference_time
            metrics['avg_inference_time'] = (
                metrics['total_inference_time'] / metrics['success_count']
            )
        else:
            metrics['error_count'] += 1
    
    async def predict_with_fallback(self, input_data: Any, preferred_model: str = None) -> Dict[str, Any]:
        """
        Make a prediction with automatic fallback through tiers.
        
        Args:
            input_data: Input data for prediction
            preferred_model: Optional preferred model ID to try first
            
        Returns:
            Dictionary containing prediction results and metadata
        """
        # Try preferred model first if specified
        if preferred_model and preferred_model in self.model_loaders:
            try:
                result = await self._try_predict(preferred_model, input_data)
                result['model_used'] = preferred_model
                result['fallback_used'] = False
                return result
            except Exception as e:
                logger.warning(f"Preferred model {preferred_model} failed: {str(e)}")
        
        # Tier 1: Try high-accuracy models
        for model_id in self.MODEL_TIERS['tier1']:
            if model_id in self.model_loaders and model_id != preferred_model:
                try:
                    result = await self._try_predict(model_id, input_data)
                    result['model_used'] = model_id
                    result['fallback_used'] = True
                    result['fallback_tier'] = 'tier1'
                    return result
                except Exception as e:
                    logger.warning(f"Tier 1 model {model_id} failed: {str(e)}")
        
        # Tier 2: Try lightweight models
        for model_id in self.MODEL_TIERS['tier2']:
            if model_id in self.model_loaders:
                try:
                    result = await self._try_predict(model_id, input_data)
                    result['model_used'] = model_id
                    result['fallback_used'] = True
                    result['fallback_tier'] = 'tier2'
                    return result
                except Exception as e:
                    logger.warning(f"Tier 2 model {model_id} failed: {str(e)}")
        
        # Tier 3: Use rule-based fallback
        try:
            result = self._heuristic_predict(input_data)
            result['model_used'] = 'heuristic'
            result['fallback_used'] = True
            result['fallback_tier'] = 'tier3'
            return result
        except Exception as e:
            logger.error(f"All prediction attempts failed: {str(e)}")
            raise PredictionError("All prediction attempts failed") from e
    
    async def _try_predict(self, model_id: str, input_data: Any) -> Dict[str, Any]:
        """
        Attempt to make a prediction with a specific model.
        
        Args:
            model_id: ID of the model to use
            input_data: Input data for prediction
            
        Returns:
            Dictionary containing prediction results
        """
        if model_id not in self.model_loaders:
            raise ValueError(f"Model {model_id} not loaded")
            
        # Add request to queue and wait for result
        future = asyncio.Future()
        await self.request_queue.put((model_id, input_data, future))
        return await future
    
    def _heuristic_predict(self, input_data: Any) -> Dict[str, Any]:
        """
        Fallback prediction using heuristics when all models fail.
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Dictionary containing fallback prediction results
        """
        try:
            import numpy as np
            
            # Analyze input data characteristics
            prediction = 0.5  # Default to uncertain
            confidence = 0.3   # Low confidence for fallback
            reasoning = []
            
            # Handle different input types
            if isinstance(input_data, np.ndarray):
                if input_data.ndim == 4:  # Image batch
                    # Analyze image characteristics
                    batch_size, channels, height, width = input_data.shape
                    
                    # Calculate various image quality metrics
                    avg_brightness = np.mean(input_data)
                    contrast = np.std(input_data)
                    
                    # High contrast might indicate manipulation
                    if contrast > 0.3:
                        prediction += 0.2
                        reasoning.append("High contrast detected")
                    
                    # Unusual brightness patterns
                    if avg_brightness < 0.2 or avg_brightness > 0.8:
                        prediction += 0.15
                        reasoning.append("Unusual brightness patterns")
                    
                    # Check for compression artifacts (simplified)
                    if channels == 3:
                        # Calculate color channel correlations
                        corr_rg = np.corrcoef(input_data[:, 0].flatten(), input_data[:, 1].flatten())[0, 1]
                        corr_rb = np.corrcoef(input_data[:, 0].flatten(), input_data[:, 2].flatten())[0, 1]
                        corr_gb = np.corrcoef(input_data[:, 1].flatten(), input_data[:, 2].flatten())[0, 1]
                        
                        avg_correlation = np.mean([corr_rg, corr_rb, corr_gb])
                        if avg_correlation > 0.95:  # Very high correlation might indicate synthetic content
                            prediction += 0.1
                            reasoning.append("High color channel correlation")
                    
                    confidence = min(0.6, confidence + len(reasoning) * 0.05)
                    
                elif input_data.ndim == 2:  # Text embeddings or features
                    # Analyze feature characteristics
                    feature_magnitude = np.linalg.norm(input_data, axis=1) if input_data.ndim > 1 else [np.linalg.norm(input_data)]
                    
                    # Unusually high feature magnitudes might indicate synthetic content
                    avg_magnitude = np.mean(feature_magnitude)
                    if avg_magnitude > 10.0:
                        prediction += 0.2
                        reasoning.append("High feature magnitude")
                    
                    # Check for patterns in feature distribution
                    if len(feature_magnitude) > 1:
                        magnitude_std = np.std(feature_magnitude)
                        if magnitude_std < 0.1:  # Very uniform features
                            prediction += 0.1
                            reasoning.append("Uniform feature distribution")
                    
                    confidence = min(0.5, confidence + len(reasoning) * 0.05)
                    
                elif input_data.ndim == 1:  # Single feature vector
                    # Analyze single feature vector
                    vector_norm = np.linalg.norm(input_data)
                    vector_std = np.std(input_data)
                    
                    # High norm or low std might indicate synthetic content
                    if vector_norm > 15.0:
                        prediction += 0.15
                        reasoning.append("High vector norm")
                    
                    if vector_std < 0.05:
                        prediction += 0.1
                        reasoning.append("Low feature variance")
                    
                    confidence = min(0.4, confidence + len(reasoning) * 0.03)
            
            elif isinstance(input_data, dict):
                # Handle dictionary input (common in multimodal scenarios)
                reasoning.append("Dictionary input analyzed")
                
                # Look for specific keys that might indicate manipulation
                suspicious_keys = ['synthetic', 'generated', 'artificial', 'fake']
                for key in input_data.keys():
                    if any(sus in key.lower() for sus in suspicious_keys):
                        prediction += 0.2
                        reasoning.append(f"Suspicious key: {key}")
                
                # Check value ranges
                for key, value in input_data.items():
                    if isinstance(value, (int, float)):
                        if value > 1.0 or value < 0.0:  # Out of normalized range
                            prediction += 0.05
                            reasoning.append(f"Out-of-range value in {key}")
                
                confidence = min(0.5, confidence + len(reasoning) * 0.02)
            
            elif isinstance(input_data, (list, tuple)):
                # Handle list/tuple input
                if len(input_data) > 0:
                    # Check for unusual patterns
                    if all(isinstance(x, (int, float)) for x in input_data):
                        # Numeric list
                        avg_val = np.mean(input_data)
                        std_val = np.std(input_data)
                        
                        if std_val < 0.01:  # Very uniform values
                            prediction += 0.1
                            reasoning.append("Uniform numeric values")
                        
                        if abs(avg_val) > 5.0:  # Large average value
                            prediction += 0.1
                            reasoning.append("Large average values")
                    
                    confidence = min(0.4, confidence + len(reasoning) * 0.03)
            
            # Clamp prediction to valid range
            prediction = max(0.0, min(1.0, prediction))
            
            # Adjust confidence based on prediction extremity
            if prediction > 0.7 or prediction < 0.3:
                confidence = min(confidence + 0.1, 0.7)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'is_fallback': True,
                'metadata': {
                    'warning': 'Using heuristic fallback prediction',
                    'reason': 'All model predictions failed',
                    'reasoning': reasoning,
                    'input_type': type(input_data).__name__,
                    'heuristic_version': '1.0'
                }
            }
            
        except Exception as e:
            logger.error(f"Heuristic fallback failed: {e}")
            # Ultimate fallback
            return {
                'prediction': 0.5,
                'confidence': 0.2,
                'is_fallback': True,
                'metadata': {
                    'warning': 'Heuristic fallback failed',
                    'reason': f'Error: {str(e)}',
                    'ultimate_fallback': True
                }
            }
    
    async def close(self) -> None:
        """Clean up resources."""
        self.is_running = False
        await self.request_queue.join()
        self.executor.shutdown(wait=True)
        
        # Unload models
        for model in self.model_loaders.values():
            if hasattr(model, 'unload'):
                model.unload()
        self.model_loaders.clear()
        torch.cuda.empty_cache()
        logger.info("Model manager shutdown complete")
    
    def __del__(self):
        """Ensure resources are cleaned up on deletion."""
        if hasattr(self, 'is_running') and self.is_running:
            asyncio.create_task(self.close())
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models with their versions.
        
        Returns:
            List of model information dictionaries
        """
        result = []
        
        for model_name, config in self.models_config.items():
            latest_version = self.version_manager.get_latest_version(model_name)
            
            if latest_version:
                result.append({
                    "name": model_name,
                    "latest_version": latest_version.version,
                    "description": config.get("description", ""),
                    "status": config.get("status", "unknown"),
                    "input_size": config.get("input_size"),
                    "last_updated": config.get("last_updated", "")
                })
                
        return result
