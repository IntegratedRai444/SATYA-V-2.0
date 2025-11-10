"""
Enhanced Model Manager for SatyaAI
Extends the base ModelManager with versioning and validation.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import torch
import json
from datetime import datetime

from .validation.model_validator import ModelValidator
from .validation.model_versioning import ModelVersionManager, ModelVersion

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

class EnhancedModelManager:
    """
    Enhanced model manager with versioning and validation support.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the enhanced model manager.
        
        Args:
            model_dir: Directory to store models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        # Initialize components
        self.validator = ModelValidator(str(self.model_dir))
        self.version_manager = ModelVersionManager(str(self.model_dir))
        
        # Loaded models cache
        self.loaded_models: Dict[str, Any] = {}
        
        # Model configurations with extended metadata
        self.models_config = self._load_model_config()
    
    def _load_model_config(self) -> Dict[str, Dict]:
        """Load model configuration with versioning support."""
        config_path = self.model_dir / "models_config.json"
        
        # Default configuration if file doesn't exist
        default_config = {
            'face_detector': {
                'name': 'MTCNN',
                'version': '1.0.0',
                'source': 'facenet_pytorch',
                'local_path': None,
                'description': 'Multi-task CNN for face detection and alignment',
                'input_size': (160, 160),
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
                'min_confidence': 0.8,
                'compatibility': {
                    'python': '>=3.8',
                    'torch': '>=1.9.0'
                },
                'expected_md5': None,
                'last_updated': '2023-10-01',
                'status': 'production'
            }
        }
        
        if not config_path.exists():
            # Save default config if it doesn't exist
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading model config: {e}")
            return default_config
    
    def get_model(self, model_name: str, version: str = None) -> Any:
        """
        Get a model instance, loading it if necessary.
        
        Args:
            model_name: Name of the model to load
            version: Specific version to load (defaults to latest)
            
        Returns:
            Loaded model instance
            
        Raises:
            ModelLoadError: If the model fails to load
            ModelVersionError: If the specified version is not found
        """
        cache_key = f"{model_name}:{version or 'latest'}"
        
        # Check if model is already loaded
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        # Get model config
        if model_name not in self.models_config:
            raise ModelLoadError(f"Unknown model: {model_name}")
            
        config = self.models_config[model_name]
        
        try:
            # Get model version
            model_version = self._get_model_version(model_name, version)
            if not model_version:
                raise ModelVersionError(f"No compatible version found for {model_name}:{version or 'latest'}")
            
            # Load the model
            model = self._load_model_impl(model_name, model_version, config)
            
            # Cache the loaded model
            self.loaded_models[cache_key] = model
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}:{version or 'latest'}: {e}")
            if isinstance(e, ModelError):
                raise
            raise ModelLoadError(f"Failed to load model {model_name}: {str(e)}")
    
    def _get_model_version(self, model_name: str, version: str = None) -> Optional[ModelVersion]:
        """Get a specific or the latest compatible version of a model."""
        if version:
            # Get specific version
            model_version = self.version_manager.get_version(model_name, version)
            if not model_version:
                return None
                
            # Check compatibility
            compatible, details = self.version_manager.check_compatibility(
                model_name,
                version,
                {
                    'python': '>=3.8',  # Current Python version
                    'torch': torch.__version__
                }
            )
            
            if not compatible:
                logger.warning(f"Version {version} of {model_name} has compatibility issues: {details}")
                return None
                
            return model_version
        else:
            # Get latest compatible version
            return self.version_manager.get_latest_version(model_name)
    
    def _load_model_impl(self, model_name: str, version: ModelVersion, config: Dict) -> Any:
        """
        Implementation of model loading.
        
        This is a placeholder - implement actual model loading logic here.
        """
        model_path = self.model_dir / version.path
        
        if not model_path.exists():
            raise ModelLoadError(f"Model file not found: {model_path}")
            
        try:
            # Example for PyTorch models:
            # model = YourModelClass()
            # model.load_state_dict(torch.load(model_path, map_location=self.device))
            # model.to(self.device)
            # model.eval()
            # return model
            
            # For now, return a dummy model for illustration
            class DummyModel:
                def __init__(self, name, version):
                    self.name = name
                    self.version = version
                    self.device = torch.device('cpu')
                
                def predict(self, x):
                    return torch.rand(len(x))  # Random predictions for illustration
            
            return DummyModel(model_name, version.version)
            
        except Exception as e:
            raise ModelLoadError(f"Error loading model from {model_path}: {str(e)}")
    
    def validate_model(
        self, 
        model_name: str, 
        test_data: Any, 
        test_labels: Any,
        version: str = None,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        Validate a model's performance.
        
        Args:
            model_name: Name of the model to validate
            test_data: Test data for validation
            test_labels: Ground truth labels
            version: Specific version to validate (defaults to latest)
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of metric names and values
            
        Raises:
            ModelValidationError: If validation fails
        """
        try:
            # Get the model
            model = self.get_model(model_name, version)
            
            # Perform validation
            results = self.validator.validate_model(
                model_name=model_name,
                test_data=test_data,
                test_labels=test_labels,
                metrics=metrics or ['accuracy', 'precision', 'recall', 'f1']
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Validation failed for {model_name}:{version or 'latest'}: {str(e)}"
            logger.error(error_msg)
            raise ModelValidationError(error_msg) from e
    
    def get_model_info(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """
        Get information about a model version.
        
        Args:
            model_name: Name of the model
            version: Specific version (defaults to latest)
            
        Returns:
            Dictionary with model information
        """
        if model_name not in self.models_config:
            return {"error": f"Unknown model: {model_name}"}
            
        model_version = self._get_model_version(model_name, version)
        if not model_version:
            return {"error": f"Version {version or 'latest'} not found for {model_name}"}
            
        # Get basic config
        info = {
            "name": model_name,
            "version": model_version.version,
            "description": self.models_config[model_name].get("description", ""),
            "status": self.models_config[model_name].get("status", "unknown"),
            "created_at": model_version.created_at,
            "is_active": model_version.is_active,
            "compatibility": model_version.compatibility or {},
            "metadata": model_version.metadata or {}
        }
        
        # Add validation results if available
        validation_history = self.validator.get_validation_history(model_name, limit=1)
        if validation_history:
            info["last_validation"] = validation_history[0]
            
        return info
    
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
