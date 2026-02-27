"""
AI Hardened Inference Wrapper for SatyaAI
Provides deterministic, calibrated, and safe inference for all models
"""

import logging
import time
import os
import psutil
import gc
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

from models.deepfake_classifier import (
    DeepfakeClassifier, 
    DeterminismConfig, 
    ModelSafetyFlags,
    UnifiedPreprocessing
)

logger = logging.getLogger(__name__)

# Environment stability guards
class EnvironmentStabilityGuards:
    @staticmethod
    def check_cuda_availability() -> Dict[str, Any]:
        """Check CUDA availability and fallback safely to CPU"""
        hardware_profile = {
            'cuda_available': False,
            'cuda_version': None,
            'gpu_count': 0,
            'gpu_memory': [],
            'fallback_used': False,
            'cpu_cores': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        try:
            if torch.cuda.is_available():
                hardware_profile['cuda_available'] = True
                hardware_profile['cuda_version'] = torch.version.cuda
                hardware_profile['gpu_count'] = torch.cuda.device_count()
                
                for i in range(hardware_profile['gpu_count']):
                    props = torch.cuda.get_device_properties(i)
                    hardware_profile['gpu_memory'].append({
                        'name': props.name,
                        'total_memory_gb': props.total_memory / (1024**3)
                    })
                    
                logger.info(f"🔒 CUDA available: {hardware_profile['gpu_count']} GPUs")
            else:
                logger.warning("🔒 CUDA not available, falling back to CPU")
                hardware_profile['fallback_used'] = True
        except Exception as e:
            logger.error(f"🔒 CUDA detection failed: {e}, forcing CPU fallback")
            hardware_profile['fallback_used'] = True
            
        return hardware_profile
    
    @staticmethod
    def log_hardware_profile():
        """Log hardware profile at startup"""
        profile = EnvironmentStabilityGuards.check_cuda_availability()
        logger.info("🔒 HARDWARE PROFILE", extra={
            'hardware_profile': profile,
            'pytorch_version': torch.__version__,
            'numpy_version': np.__version__
        })

# Model warmup safety
class ModelWarmupSafety:
    _model_cache = {}
    _loading_locks = {}
    
    @classmethod
    def get_cached_model(cls, model_type: str, device: str = None):
        """Lazy load with caching and duplicate load prevention"""
        cache_key = f"{model_type}_{device or 'auto'}"
        
        if cache_key in cls._model_cache:
            logger.debug(f"🔒 Using cached model: {cache_key}")
            return cls._model_cache[cache_key]
            
        if cache_key in cls._loading_locks:
            logger.warning(f"🔒 Model already loading: {cache_key}, waiting...")
            # Simple wait for load completion
            import time
            for _ in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if cache_key in cls._model_cache:
                    return cls._model_cache[cache_key]
            raise RuntimeError(f"🔒 Model loading timeout: {cache_key}")
            
        cls._loading_locks[cache_key] = True
        
        try:
            logger.info(f"🔒 Loading model: {cache_key}")
            model = DeepfakeClassifier(model_type=model_type, device=device)
            cls._model_cache[cache_key] = model
            logger.info(f"🔒 Model cached: {cache_key}")
            return model
        except Exception as e:
            logger.error(f"🔒 Model loading failed: {cache_key}, error: {e}")
            raise
        finally:
            cls._loading_locks.pop(cache_key, None)

# Memory pressure protection
class MemoryPressureProtection:
    @staticmethod
    def check_memory_usage() -> Dict[str, Any]:
        """Check current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        gpu_memory = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                gpu_memory.append({
                    'device': i,
                    'allocated_gb': memory_allocated,
                    'reserved_gb': memory_reserved
                })
        
        return {
            'cpu_memory_gb': memory_info.rss / (1024**3),
            'cpu_memory_percent': process.memory_percent(),
            'gpu_memory': gpu_memory
        }
    
    @staticmethod
    def limit_concurrent_inferences(max_concurrent: int = 2):
        """Limit concurrent inferences to prevent OOM"""
        if not hasattr(MemoryPressureProtection, '_inference_count'):
            MemoryPressureProtection._inference_count = 0
            
        if MemoryPressureProtection._inference_count >= max_concurrent:
            raise RuntimeError(f"🔒 Max concurrent inferences reached: {max_concurrent}")
            
        MemoryPressureProtection._inference_count += 1
        
    @staticmethod
    def release_inference_slot():
        """Release inference slot"""
        if hasattr(MemoryPressureProtection, '_inference_count'):
            MemoryPressureProtection._inference_count = max(0, MemoryPressureProtection._inference_count - 1)
            
    @staticmethod
    def cleanup_after_inference():
        """Aggressive cleanup after inference"""
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.empty_cache()
                    
            # Force Python garbage collection
            gc.collect()
            
            logger.debug("🔒 Memory cleanup completed")
        except Exception as e:
            logger.error(f"🔒 Memory cleanup failed: {e}")

class AIHardenedInference:
    """
    Production-safe inference wrapper with:
    - Deterministic behavior
    - Calibrated confidence scores
    - Model safety warnings
    - Hardware consistency checks
    """
    
    def __init__(self, model_type: str = "efficientnet", device: str = None):
        """Initialize hardened inference wrapper"""
        
        # Environment stability checks
        EnvironmentStabilityGuards.log_hardware_profile()
        
        # AI HARDENING: Ensure deterministic behavior
        DeterminismConfig.set_deterministic()
        
        # Model warmup safety
        self.classifier = ModelWarmupSafety.get_cached_model(model_type, device)
        self.model_type = model_type
        
        # Memory pressure monitoring
        self.memory_before = MemoryPressureProtection.check_memory_usage()
        
        logger.info(f"🔒 AI Hardened Inference initialized for {model_type}")
    
    def predict_with_safety(self, image: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Safe prediction with comprehensive AI hardening
        
        Args:
            image: Input image as numpy array or torch.Tensor
            
        Returns:
            Dictionary with hardened prediction results
        """
        # Memory pressure protection
        MemoryPressureProtection.limit_concurrent_inferences(max_concurrent=2)
        
        try:
            # Run prediction with built-in safety
            result = self.classifier.predict_image(image)
            
            # AI HARDENING: Add confidence bands
            confidence = result["confidence"]
            confidence_band = self._get_confidence_band(confidence)
            
            # AI HARDENING: Add uncertainty estimation
            uncertainty = self._estimate_uncertainty(result)
            
            # AI HARDENING: Add hardware consistency check
            hardware_check = self._check_hardware_consistency(result)
            
            # AI HARDENING: Add safety warnings
            safety_warnings = self._generate_safety_warnings(result)
            
            # AI HARDENING: Add memory usage tracking
            memory_after = MemoryPressureProtection.check_memory_usage()
            
            # Enhance result with hardening information
            result.update({
                "confidence_band": confidence_band,
                "uncertainty_estimate": uncertainty,
                "hardware_consistency": hardware_check,
                "safety_warnings": safety_warnings,
                "inference_hardened": True,
                "deterministic_seed": DeterminismConfig.INFERENCE_SEED,
                "memory_usage": {
                    "before": self.memory_before,
                    "after": memory_after,
                    "delta_gb": memory_after['cpu_memory_gb'] - self.memory_before['cpu_memory_gb']
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"🔒 Hardened inference failed: {e}")
            raise
        finally:
            # Always release inference slot and cleanup
            MemoryPressureProtection.release_inference_slot()
            MemoryPressureProtection.cleanup_after_inference()
    
    def _get_confidence_band(self, confidence: float) -> str:
        """Convert confidence to interpretable bands"""
        if confidence >= 0.8:
            return "HIGH"
        elif confidence >= 0.6:
            return "MEDIUM"
        elif confidence >= 0.4:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _estimate_uncertainty(self, result: Dict[str, Any]) -> float:
        """
        Estimate prediction uncertainty using confidence gap
        
        Args:
            result: Prediction result dictionary
            
        Returns:
            Uncertainty score (0-1, higher = more uncertain)
        """
        class_probs = result["class_probs"]
        real_prob = class_probs["real"]
        fake_prob = class_probs["fake"]
        
        # Uncertainty = 1 - max_probability (entropy approximation)
        uncertainty = 1 - max(real_prob, fake_prob)
        return uncertainty
    
    def _check_hardware_consistency(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for hardware-related consistency issues
        
        Args:
            result: Prediction result dictionary
            
        Returns:
            Hardware consistency report
        """
        device = str(self.classifier.device)
        precision = self.classifier.precision
        
        return {
            "device": device,
            "precision": precision,
            "consistency_risk": "low" if device == "cpu" else "medium",
            "recommendation": "CPU inference is most consistent" if device != "cpu" else "Consistent inference"
        }
    
    def _generate_safety_warnings(self, result: Dict[str, Any]) -> list:
        """
        Generate safety warnings based on model state and confidence
        
        Args:
            result: Prediction result dictionary
            
        Returns:
            List of safety warnings
        """
        warnings = []
        model_info = result.get("model_info", {})
        confidence = result["confidence"]
        
        # Fallback model warning
        if model_info.get("is_fallback_model", False):
            warnings.append(f"⚠️ USING FALLBACK MODEL: {model_info.get('model_source', 'unknown')}")
            warnings.append("⚠️ This model is NOT trained for deepfake detection")
        
        # Low confidence warning
        if confidence < 0.6:
            warnings.append(f"⚠️ LOW CONFIDENCE: {confidence:.2f} - prediction may be unreliable")
        
        # High uncertainty warning
        uncertainty = self._estimate_uncertainty(result)
        if uncertainty > 0.4:
            warnings.append(f"⚠️ HIGH UNCERTAINTY: {uncertainty:.2f} - model is uncertain")
        
        # Hardware consistency warning
        if result["hardware_consistency"]["consistency_risk"] == "medium":
            warnings.append("⚠️ GPU inference may have slight variability vs CPU")
        
        return warnings if warnings else ["✅ No safety concerns"]
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get calibration information for the current model"""
        return {
            "model_type": self.model_type,
            "temperature": DeterminismConfig.TEMPERATURE_SCALING.get(self.model_type, 1.0),
            "deterministic_seed": DeterminismConfig.INFERENCE_SEED,
            "preprocessing": "Unified ImageNet normalization",
            "safety_flags": {
                "is_deepfake_model": self.classifier.safety_flags.is_deepfake_model,
                "is_fallback_model": self.classifier.safety_flags.is_fallback_model,
                "model_source": self.classifier.safety_flags.model_source
            }
        }

# Convenience function for drop-in replacement
def create_hardened_classifier(model_type: str = "efficientnet", device: str = None) -> AIHardenedInference:
    """
    Create a hardened classifier for production use
    
    Args:
        model_type: Type of model to use
        device: Device to run inference on
        
    Returns:
        Hardened inference wrapper
    """
    return AIHardenedInference(model_type=model_type, device=device)

# Example usage:
"""
# OLD WAY (unsafe):
classifier = DeepfakeClassifier("efficientnet")
result = classifier.predict_image(image)

# NEW WAY (hardened):
hardened_classifier = create_hardened_classifier("efficientnet")
result = hardened_classifier.predict_with_safety(image)

# Result now includes:
# - confidence_band: "HIGH" | "MEDIUM" | "LOW" | "VERY_LOW"
# - uncertainty_estimate: float (0-1)
# - hardware_consistency: dict
# - safety_warnings: list[str]
# - model_info: dict with safety flags
# - deterministic_seed: int for reproducibility
"""
