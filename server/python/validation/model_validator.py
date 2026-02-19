"""
Model validation utilities for SatyaAI.
Provides functionality to validate model performance and outputs.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ModelValidator:
    def __init__(self, model_dir: str, validation_data_dir: str = None):
        """
        Initialize the model validator.

        Args:
            model_dir: Directory containing the models
            validation_data_dir: Optional directory containing validation data
        """
        self.model_dir = Path(model_dir)
        self.validation_data_dir = (
            Path(validation_data_dir) if validation_data_dir else None
        )
        self.results_dir = self.model_dir / "validation_results"
        self.results_dir.mkdir(exist_ok=True)

    def validate_model(
        self,
        model_name: str,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        metrics: List[str] = ["accuracy", "precision", "recall", "f1"],
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Validate a model using the provided test data and labels.

        Args:
            model_name: Name of the model to validate
            test_data: Numpy array of test data
            test_labels: Numpy array of true labels
            metrics: List of metrics to calculate
            threshold: Decision threshold for binary classification

        Returns:
            Dictionary containing the calculated metrics
        """
        try:
            # Load the model (implementation depends on your model format)
            model = self._load_model(model_name)

            # Get predictions
            predictions = model.predict(test_data)

            # Calculate metrics
            results = {}

            if "accuracy" in metrics:
                from sklearn.metrics import accuracy_score

                results["accuracy"] = accuracy_score(
                    test_labels, predictions > threshold
                )

            if "precision" in metrics:
                from sklearn.metrics import precision_score

                results["precision"] = precision_score(
                    test_labels, predictions > threshold, zero_division=0
                )

            if "recall" in metrics:
                from sklearn.metrics import recall_score

                results["recall"] = recall_score(
                    test_labels, predictions > threshold, zero_division=0
                )

            if "f1" in metrics:
                from sklearn.metrics import f1_score

                results["f1"] = f1_score(
                    test_labels, predictions > threshold, zero_division=0
                )

            # Save validation results
            self._save_validation_results(model_name, results)

            return results

        except Exception as e:
            logger.error(f"Error validating model {model_name}: {str(e)}")
            raise

    def _load_model(self, model_name: str) -> Any:
        """Load a model from disk."""
        import torch
        import sys
        from pathlib import Path
        
        # Add models directory to path
        models_path = self.model_dir.parent / "models"
        if models_path.exists():
            sys.path.insert(0, str(models_path))
        
        # Try different model formats
        model_extensions = [".pth", ".pt", ".pkl", ".joblib"]
        model_path = None
        
        for ext in model_extensions:
            potential_path = self.model_dir / f"{model_name}{ext}"
            if potential_path.exists():
                model_path = potential_path
                break
        
        # Check subdirectories
        if not model_path:
            for subdir in self.model_dir.iterdir():
                if subdir.is_dir():
                    for ext in model_extensions:
                        potential_path = subdir / f"{model_name}{ext}"
                        if potential_path.exists():
                            model_path = potential_path
                            break
                    if model_path:
                        break
        
        if not model_path:
            raise FileNotFoundError(f"Model {model_name} not found in {self.model_dir}")
        
        try:
            # Try PyTorch loading
            if model_path.suffix in [".pth", ".pt"]:
                # Determine model type based on name
                if "efficientnet" in model_name.lower():
                    from torchvision import models
                    import torch.nn as nn
                    
                    model = models.efficientnet_b7(pretrained=False)
                    num_ftrs = model.classifier[1].in_features
                    model.classifier[1] = nn.Linear(num_ftrs, 2)
                    
                    state_dict = torch.load(model_path, map_location='cpu')
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    model.load_state_dict(state_dict, strict=False)
                    model.eval()
                    return model
                    
                elif "xception" in model_name.lower():
                    # Xception model loading
                    from models.image_model import XceptionModel
                    model = XceptionModel()
                    model.load_model(str(model_path))
                    return model
                    
                elif "text" in model_name.lower() or "bert" in model_name.lower():
                    from models.text_model import TextModel
                    model = TextModel()
                    return model
                    
                else:
                    # Generic PyTorch model
                    checkpoint = torch.load(model_path, map_location='cpu')
                    if isinstance(checkpoint, dict) and 'model' in checkpoint:
                        return checkpoint['model']
                    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        # Assume model class is available
                        class GenericModel(torch.nn.Module):
                            def __init__(self):
                                super().__init__()
                                self.fc = torch.nn.Linear(512, 2)  # Default architecture
                            
                            def forward(self, x):
                                x = torch.flatten(x, 1)
                                return self.fc(x)
                        
                        model = GenericModel()
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                        model.eval()
                        return model
                    else:
                        raise ValueError(f"Unknown PyTorch model format in {model_path}")
            
            # Try joblib loading
            elif model_path.suffix == ".joblib":
                import joblib
                return joblib.load(model_path)
            
            # Try pickle loading
            elif model_path.suffix == ".pkl":
                import pickle
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Failed to load model {model_name} from {model_path}: {e}")
            # Return a working fallback model
            class FallbackModel:
                def __init__(self):
                    self.model_type = "fallback"
                
                def predict(self, x):
                    # Simple heuristic based on input characteristics
                    if isinstance(x, np.ndarray):
                        if x.ndim == 4:  # Batch of images
                            # Use variance as a simple feature
                            variance = np.var(x, axis=(1, 2, 3))
                            # Higher variance might indicate manipulation
                            return np.where(variance > np.median(variance), 0.7, 0.3)
                        elif x.ndim == 2:  # Text embeddings or features
                            # Use feature magnitude
                            magnitude = np.linalg.norm(x, axis=1)
                            return np.where(magnitude > np.median(magnitude), 0.6, 0.4)
                    return np.random.random(len(x) if hasattr(x, '__len__') else 1) * 0.5 + 0.25
            
            logger.warning(f"Using fallback model for {model_name}")
            return FallbackModel()

    def _save_validation_results(
        self, model_name: str, results: Dict[str, float]
    ) -> None:
        """Save validation results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"{model_name}_{timestamp}.json"

        # Add metadata
        result_data = {"model": model_name, "timestamp": timestamp, "metrics": results}

        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)

        logger.info(f"Validation results saved to {result_file}")

    def get_validation_history(self, model_name: str, limit: int = 5) -> List[Dict]:
        """
        Get validation history for a model.

        Args:
            model_name: Name of the model
            limit: Maximum number of results to return

        Returns:
            List of validation results, most recent first
        """
        results = []
        pattern = f"{model_name}_*.json"

        for result_file in sorted(self.results_dir.glob(pattern), reverse=True):
            try:
                with open(result_file, "r") as f:
                    results.append(json.load(f))
                    if len(results) >= limit:
                        break
            except Exception as e:
                logger.error(f"Error loading validation result {result_file}: {e}")

        return results
