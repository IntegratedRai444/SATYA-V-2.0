"""
Model validation utilities for SatyaAI.
Provides functionality to validate model performance and outputs.
"""
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime

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
        self.validation_data_dir = Path(validation_data_dir) if validation_data_dir else None
        self.results_dir = self.model_dir / "validation_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def validate_model(
        self, 
        model_name: str, 
        test_data: np.ndarray, 
        test_labels: np.ndarray,
        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
        threshold: float = 0.5
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
            
            if 'accuracy' in metrics:
                from sklearn.metrics import accuracy_score
                results['accuracy'] = accuracy_score(test_labels, predictions > threshold)
                
            if 'precision' in metrics:
                from sklearn.metrics import precision_score
                results['precision'] = precision_score(test_labels, predictions > threshold, zero_division=0)
                
            if 'recall' in metrics:
                from sklearn.metrics import recall_score
                results['recall'] = recall_score(test_labels, predictions > threshold, zero_division=0)
                
            if 'f1' in metrics:
                from sklearn.metrics import f1_score
                results['f1'] = f1_score(test_labels, predictions > threshold, zero_division=0)
                
            # Save validation results
            self._save_validation_results(model_name, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating model {model_name}: {str(e)}")
            raise
    
    def _load_model(self, model_name: str) -> Any:
        """Load a model from disk."""
        # This is a placeholder - implement based on your model format
        model_path = self.model_dir / f"{model_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
            
        # Example for PyTorch models:
        # import torch
        # model = YourModelClass()
        # model.load_state_dict(torch.load(model_path))
        # model.eval()
        # return model
        
        # For now, just return a dummy model for illustration
        class DummyModel:
            def predict(self, x):
                return np.random.random(len(x))  # Random predictions for illustration
                
        return DummyModel()
    
    def _save_validation_results(self, model_name: str, results: Dict[str, float]) -> None:
        """Save validation results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"{model_name}_{timestamp}.json"
        
        # Add metadata
        result_data = {
            "model": model_name,
            "timestamp": timestamp,
            "metrics": results
        }
        
        with open(result_file, 'w') as f:
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
                with open(result_file, 'r') as f:
                    results.append(json.load(f))
                    if len(results) >= limit:
                        break
            except Exception as e:
                logger.error(f"Error loading validation result {result_file}: {e}")
                
        return results
