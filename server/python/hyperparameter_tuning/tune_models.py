#!/usr/bin/env python3
"""
Hyperparameter Tuning for Deepfake Detection Models
Uses grid search and random search for optimal hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import numpy as np
import json
from pathlib import Path
from datetime import datetime

class HyperparameterTuner:
    """
    Hyperparameter tuning for deepfake detection models
    """
    
    def __init__(self, model_class, device='cpu'):
        self.model_class = model_class
        self.device = device
        self.results = []
    
    def grid_search(self, param_grid, train_loader, val_loader, epochs=10):
        """
        Perform grid search over hyperparameters
        
        Args:
            param_grid: Dictionary of hyperparameter ranges
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs per configuration
        """
        print("\n" + "="*70)
        print("GRID SEARCH HYPERPARAMETER TUNING")
        print("="*70 + "\n")
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(product(*values))
        
        print(f"Testing {len(combinations)} hyperparameter combinations\n")
        
        best_accuracy = 0
        best_params = None
        
        for i, combo in enumerate(combinations, 1):
            params = dict(zip(keys, combo))
            
            print(f"[{i}/{len(combinations)}] Testing: {params}")
            
            # Train with these parameters
            accuracy = self._train_and_evaluate(
                params, train_loader, val_loader, epochs
            )
            
            # Store results
            result = {
                'params': params,
                'val_accuracy': accuracy,
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            
            # Update best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                print(f"  ✅ New best accuracy: {accuracy:.2f}%\n")
            else:
                print(f"  Accuracy: {accuracy:.2f}%\n")
        
        print("="*70)
        print("GRID SEARCH COMPLETE")
        print("="*70)
        print(f"\nBest Parameters: {best_params}")
        print(f"Best Validation Accuracy: {best_accuracy:.2f}%\n")
        
        return best_params, best_accuracy
    
    def random_search(self, param_distributions, n_iter=20, 
                     train_loader=None, val_loader=None, epochs=10):
        """
        Perform random search over hyperparameters
        
        Args:
            param_distributions: Dictionary of parameter distributions
            n_iter: Number of random samples
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs per configuration
        """
        print("\n" + "="*70)
        print("RANDOM SEARCH HYPERPARAMETER TUNING")
        print("="*70 + "\n")
        
        print(f"Testing {n_iter} random hyperparameter configurations\n")
        
        best_accuracy = 0
        best_params = None
        
        for i in range(n_iter):
            # Sample random parameters
            params = {}
            for key, distribution in param_distributions.items():
                if isinstance(distribution, list):
                    params[key] = np.random.choice(distribution)
                elif isinstance(distribution, tuple) and len(distribution) == 2:
                    # Assume (min, max) for continuous values
                    params[key] = np.random.uniform(distribution[0], distribution[1])
            
            print(f"[{i+1}/{n_iter}] Testing: {params}")
            
            # Train with these parameters
            accuracy = self._train_and_evaluate(
                params, train_loader, val_loader, epochs
            )
            
            # Store results
            result = {
                'params': params,
                'val_accuracy': accuracy,
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(result)
            
            # Update best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                print(f"  ✅ New best accuracy: {accuracy:.2f}%\n")
            else:
                print(f"  Accuracy: {accuracy:.2f}%\n")
        
        print("="*70)
        print("RANDOM SEARCH COMPLETE")
        print("="*70)
        print(f"\nBest Parameters: {best_params}")
        print(f"Best Validation Accuracy: {best_accuracy:.2f}%\n")
        
        return best_params, best_accuracy
    
    def _train_and_evaluate(self, params, train_loader, val_loader, epochs):
        """
        Train model with given parameters and return validation accuracy
        """
        # Simulated training (replace with actual training in production)
        # This simulates different accuracies based on hyperparameters
        
        lr = params.get('learning_rate', 0.0001)
        batch_size = params.get('batch_size', 32)
        dropout = params.get('dropout', 0.5)
        weight_decay = params.get('weight_decay', 1e-5)
        
        # Simulate accuracy based on parameters
        # Optimal: lr=0.0001, batch_size=32, dropout=0.5, weight_decay=1e-5
        lr_score = 1.0 - abs(np.log10(lr) + 4) * 0.1
        batch_score = 1.0 - abs(batch_size - 32) / 32 * 0.2
        dropout_score = 1.0 - abs(dropout - 0.5) * 0.3
        wd_score = 1.0 - abs(np.log10(weight_decay) + 5) * 0.1
        
        base_accuracy = 90.0
        accuracy = base_accuracy + (lr_score + batch_score + dropout_score + wd_score) * 1.5
        accuracy = min(95.0, max(85.0, accuracy + np.random.normal(0, 0.5)))
        
        return accuracy
    
    def save_results(self, output_path):
        """Save tuning results to JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✅ Results saved to: {output_path}")
    
    def get_best_params(self):
        """Get best parameters from all runs"""
        if not self.results:
            return None
        
        best_result = max(self.results, key=lambda x: x['val_accuracy'])
        return best_result['params'], best_result['val_accuracy']


def demo_hyperparameter_tuning():
    """Demo hyperparameter tuning"""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING DEMO")
    print("="*70 + "\n")
    
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.001, 0.0001, 0.00001],
        'batch_size': [16, 32, 64],
        'dropout': [0.3, 0.5, 0.7],
        'weight_decay': [1e-6, 1e-5, 1e-4]
    }
    
    # Initialize tuner
    tuner = HyperparameterTuner(model_class=None, device='cpu')
    
    # Run grid search (simulated)
    best_params, best_acc = tuner.grid_search(
        param_grid=param_grid,
        train_loader=None,
        val_loader=None,
        epochs=5
    )
    
    # Save results
    tuner.save_results('results/hyperparameter_tuning_results.json')
    
    # Print summary
    print("\n" + "="*70)
    print("TUNING SUMMARY")
    print("="*70)
    print(f"\nTotal configurations tested: {len(tuner.results)}")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"\nBest hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    demo_hyperparameter_tuning()
