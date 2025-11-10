#!/usr/bin/env python3
"""
Optimized Model Verification Script
Verifies that optimized models produce the same outputs as the original models.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Any, Tuple, List, Optional
import onnxruntime as ort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_verification.log')
    ]
)
logger = logging.getLogger(__name__)

class ModelVerifier:
    """Verifies that optimized models match original model outputs."""
    
    def __init__(self, atol: float = 1e-3, rtol: float = 1e-3):
        """
        Initialize the model verifier.
        
        Args:
            atol: Absolute tolerance for numerical comparison
            rtol: Relative tolerance for numerical comparison
        """
        self.atol = atol
        self.rtol = rtol
    
    def verify_pytorch_models(
        self,
        original_model: torch.nn.Module,
        optimized_model: torch.nn.Module,
        input_tensor: torch.Tensor,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Verify that two PyTorch models produce the same outputs.
        
        Args:
            original_model: The original PyTorch model
            optimized_model: The optimized PyTorch model
            input_tensor: Input tensor for the models
            device: Device to run the models on ('cpu' or 'cuda')
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Move models and input to device
            original_model = original_model.to(device).eval()
            optimized_model = optimized_model.to(device).eval()
            input_tensor = input_tensor.to(device)
            
            # Run inference
            with torch.no_grad():
                original_output = original_model(input_tensor)
                optimized_output = optimized_model(input_tensor)
            
            # Convert outputs to numpy for comparison
            if isinstance(original_output, (list, tuple)):
                original_output = [o.cpu().numpy() for o in original_output]
                optimized_output = [o.cpu().numpy() for o in optimized_output]
                all_close = all(
                    np.allclose(o1, o2, atol=self.atol, rtol=self.rtol)
                    for o1, o2 in zip(original_output, optimized_output)
                )
            elif isinstance(original_output, dict):
                original_output = {k: v.cpu().numpy() for k, v in original_output.items()}
                optimized_output = {k: v.cpu().numpy() for k, v in optimized_output.items()}
                all_close = all(
                    np.allclose(original_output[k], v, atol=self.atol, rtol=self.rtol)
                    for k, v in optimized_output.items()
                    if k in original_output
                )
            else:
                original_output = original_output.cpu().numpy()
                optimized_output = optimized_output.cpu().numpy()
                all_close = np.allclose(original_output, optimized_output, 
                                      atol=self.atol, rtol=self.rtol)
            
            # Calculate statistics
            if not isinstance(original_output, (dict, list, tuple)):
                abs_diff = np.abs(original_output - optimized_output)
                max_diff = np.max(abs_diff)
                mean_diff = np.mean(abs_diff)
                std_diff = np.std(abs_diff)
            else:
                max_diff = mean_diff = std_diff = float('nan')
            
            return {
                'status': 'success' if all_close else 'mismatch',
                'all_close': bool(all_close),
                'max_difference': float(max_diff) if not np.isnan(max_diff) else None,
                'mean_difference': float(mean_diff) if not np.isnan(mean_diff) else None,
                'std_difference': float(std_diff) if not np.isnan(std_diff) else None,
                'original_shape': str(original_output.shape) if hasattr(original_output, 'shape') else 'N/A',
                'optimized_shape': str(optimized_output.shape) if hasattr(optimized_output, 'shape') else 'N/A',
                'device': device
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'device': device
            }
    
    def verify_onnx_model(
        self,
        original_model: torch.nn.Module,
        onnx_model_path: str,
        input_tensor: torch.Tensor,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Verify that an ONNX model matches the original PyTorch model.
        
        Args:
            original_model: The original PyTorch model
            onnx_model_path: Path to the ONNX model
            input_tensor: Input tensor for the models
            device: Device to run the ONNX model on ('cpu' or 'cuda')
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Prepare ONNX runtime session
            providers = ['CPUExecutionProvider']
            if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            session = ort.InferenceSession(onnx_model_path, providers=providers)
            input_name = session.get_inputs()[0].name
            
            # Prepare input
            input_data = input_tensor.cpu().numpy()
            
            # Run ONNX inference
            onnx_output = session.run(None, {input_name: input_data})
            
            # Run PyTorch inference
            with torch.no_grad():
                original_model = original_model.eval().to(device)
                original_output = original_model(
                    torch.from_numpy(input_data).to(device)
                )
                
                # Convert PyTorch output to numpy
                if isinstance(original_output, (list, tuple)):
                    original_output = [o.cpu().numpy() for o in original_output]
                    all_close = all(
                        np.allclose(o1, o2, atol=self.atol, rtol=self.rtol)
                        for o1, o2 in zip(original_output, onnx_output)
                    )
                    max_diff = max(
                        np.max(np.abs(o1 - o2))
                        for o1, o2 in zip(original_output, onnx_output)
                    )
                    mean_diff = np.mean([
                        np.mean(np.abs(o1 - o2))
                        for o1, o2 in zip(original_output, onnx_output)
                    ])
                elif isinstance(original_output, dict):
                    original_output = {k: v.cpu().numpy() for k, v in original_output.items()}
                    all_close = all(
                        np.allclose(original_output[k], v, atol=self.atol, rtol=self.rtol)
                        for k, v in enumerate(onnx_output)
                        if k in original_output
                    )
                    max_diff = max(
                        np.max(np.abs(original_output[k] - v))
                        for k, v in enumerate(onnx_output)
                        if k in original_output
                    )
                    mean_diff = np.mean([
                        np.mean(np.abs(original_output[k] - v))
                        for k, v in enumerate(onnx_output)
                        if k in original_output
                    ])
                else:
                    original_output = original_output.cpu().numpy()
                    all_close = np.allclose(original_output, onnx_output[0], 
                                          atol=self.atol, rtol=self.rtol)
                    max_diff = np.max(np.abs(original_output - onnx_output[0]))
                    mean_diff = np.mean(np.abs(original_output - onnx_output[0]))
            
            return {
                'status': 'success' if all_close else 'mismatch',
                'all_close': bool(all_close),
                'max_difference': float(max_diff) if not np.isnan(max_diff) else None,
                'mean_difference': float(mean_diff) if not np.isnan(mean_diff) else None,
                'onnx_output_shape': str(onnx_output[0].shape) if onnx_output else 'N/A',
                'original_output_shape': str(original_output.shape) if hasattr(original_output, 'shape') else 'N/A',
                'device': device,
                'onnx_providers': providers
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'device': device,
                'onnx_providers': providers if 'providers' in locals() else []
            }
    
    def verify_all_models(
        self,
        original_model_path: str,
        optimized_models: Dict[str, str],
        input_tensor: torch.Tensor,
        device: str = 'cpu'
    ) -> Dict[str, Any]:
        """
        Verify all optimized models against the original model.
        
        Args:
            original_model_path: Path to the original PyTorch model
            optimized_models: Dictionary of {model_type: model_path}
            input_tensor: Input tensor for the models
            device: Device to run the models on ('cpu' or 'cuda')
            
        Returns:
            Dictionary with verification results for each model
        """
        results = {}
        
        try:
            # Load original model
            original_model = torch.load(original_model_path, map_location='cpu')
            if isinstance(original_model, dict):
                if 'state_dict' in original_model:
                    original_model = original_model['state_dict']
                elif 'model' in original_model:
                    original_model = original_model['model']
            
            # Verify each optimized model
            for model_type, model_path in optimized_models.items():
                logger.info(f"Verifying {model_type} model: {model_path}")
                
                if not os.path.exists(model_path):
                    results[model_type] = {
                        'status': 'error',
                        'error': f'Model file not found: {model_path}'
                    }
                    continue
                
                try:
                    if model_path.endswith('.onnx'):
                        # Verify ONNX model
                        results[model_type] = self.verify_onnx_model(
                            original_model=original_model,
                            onnx_model_path=model_path,
                            input_tensor=input_tensor,
                            device=device
                        )
                    elif model_path.endswith(('.pt', '.pth')):
                        # Verify PyTorch model
                        optimized_model = torch.load(model_path, map_location='cpu')
                        if isinstance(optimized_model, dict):
                            if 'state_dict' in optimized_model:
                                optimized_model = optimized_model['state_dict']
                            elif 'model' in optimized_model:
                                optimized_model = optimized_model['model']
                        
                        results[model_type] = self.verify_pytorch_models(
                            original_model=original_model,
                            optimized_model=optimized_model,
                            input_tensor=input_tensor,
                            device=device
                        )
                    else:
                        results[model_type] = {
                            'status': 'error',
                            'error': f'Unsupported model format: {model_path}'
                        }
                    
                    logger.info(f"{model_type} verification: {results[model_type]['status']}")
                    
                except Exception as e:
                    results[model_type] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    logger.error(f"Error verifying {model_type}: {e}", exc_info=True)
            
            return results
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Failed to load original model: {str(e)}'
            }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify optimized models')
    parser.add_argument('--original-model', type=str, required=True,
                       help='Path to the original PyTorch model')
    parser.add_argument('--optimized-models', type=str, nargs='+', required=True,
                       help='Paths to optimized models (format: type:path)')
    parser.add_argument('--input-shape', type=int, nargs='+', default=[1, 3, 224, 224],
                       help='Input tensor shape (default: 1 3 224 224)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run verification on')
    parser.add_argument('--atol', type=float, default=1e-3,
                       help='Absolute tolerance for numerical comparison')
    parser.add_argument('--rtol', type=float, default=1e-3,
                       help='Relative tolerance for numerical comparison')
    parser.add_argument('--output', type=str, default='verification_results.json',
                       help='Output file for verification results')
    
    args = parser.parse_args()
    
    # Parse optimized models
    optimized_models = {}
    for model_spec in args.optimized_models:
        if ':' in model_spec:
            model_type, model_path = model_spec.split(':', 1)
            optimized_models[model_type] = model_path
        else:
            model_type = os.path.splitext(os.path.basename(model_spec))[0]
            optimized_models[model_type] = model_spec
    
    # Create input tensor
    input_tensor = torch.randn(*args.input_shape)
    
    # Initialize verifier
    verifier = ModelVerifier(atol=args.atol, rtol=args.rtol)
    
    # Run verification
    results = verifier.verify_all_models(
        original_model_path=args.original_model,
        optimized_models=optimized_models,
        input_tensor=input_tensor,
        device=args.device
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            'original_model': args.original_model,
            'input_shape': args.input_shape,
            'device': args.device,
            'tolerance': {
                'absolute': args.atol,
                'relative': args.rtol
            },
            'results': results
        }, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("Verification Complete")
    print("="*50)
    print(f"\nOriginal model: {args.original_model}")
    print(f"Input shape: {args.input_shape}")
    print(f"Device: {args.device}")
    print(f"Tolerance: atol={args.atol}, rtol={args.rtol}")
    
    print("\nResults:")
    for model_type, result in results.items():
        status = result.get('status', 'unknown')
        if status == 'success':
            status_str = f"\033[92m{status.upper()}\033[0m"  # Green
        elif status == 'mismatch':
            status_str = f"\033[93m{status.upper()}\033[0m"  # Yellow
        else:
            status_str = f"\033[91m{status.upper()}\033[0m"  # Red
        
        print(f"- {model_type}: {status_str}")
        if 'all_close' in result:
            print(f"  All close: {result['all_close']}")
        if 'max_difference' in result and result['max_difference'] is not None:
            print(f"  Max difference: {result['max_difference']:.2e}")
        if 'mean_difference' in result and result['mean_difference'] is not None:
            print(f"  Mean difference: {result['mean_difference']:.2e}")
        if 'error' in result:
            print(f"  Error: {result['error']}")
    
    print(f"\nDetailed results saved to: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
