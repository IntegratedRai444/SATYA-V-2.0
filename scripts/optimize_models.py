#!/usr/bin/env python3
"""
Model Optimization Script
Optimizes all deepfake detection models for deployment.
"""
import os
import sys
import torch
import torch.nn as nn
import logging
from pathlib import Path
import json
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import model optimizer
from server.python.optimization.model_optimizer import (
    ModelOptimizer,
    optimize_all_models,
    OptimizationLevel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_optimization.log')
    ]
)
logger = logging.getLogger(__name__)

def get_example_input(model_type: str = "image"):
    """Get example input tensor based on model type."""
    if model_type == "image":
        return torch.randn(1, 3, 224, 224)  # Standard image input
    elif model_type == "video":
        return torch.randn(1, 3, 16, 224, 224)  # 16-frame video clip
    elif model_type == "audio":
        return torch.randn(1, 1, 16000)  # 1 second of audio at 16kHz
    else:
        return torch.randn(1, 3, 224, 224)  # Default to image input

def optimize_single_model(
    model_path: str,
    output_dir: str,
    model_type: str = "image",
    optimization_level: str = "balanced"
) -> dict:
    """
    Optimize a single model.
    
    Args:
        model_path: Path to the model file
        output_dir: Directory to save optimized models
        model_type: Type of model ('image', 'video', or 'audio')
        optimization_level: Optimization level ('size', 'speed', or 'balanced')
        
    Returns:
        Dictionary with optimization results
    """
    try:
        logger.info(f"Optimizing model: {model_path}")
        
        # Load the model
        model = torch.load(model_path, map_location='cpu')
        
        # Handle different model formats
        if isinstance(model, dict):
            if 'state_dict' in model:
                model = model['state_dict']
            elif 'model' in model:
                model = model['model']
        
        # Get example input
        example_input = get_example_input(model_type)
        
        # Initialize optimizer
        optimizer = ModelOptimizer(
            model=model,
            example_input=example_input,
            model_name=Path(model_path).stem,
            output_dir=output_dir
        )
        
        # Apply optimizations based on level
        if optimization_level == "size":
            level = OptimizationLevel.SIZE
            pruned_model = optimizer.prune_model(amount=0.4)
        elif optimization_level == "speed":
            level = OptimizationLevel.PERFORMANCE
            quantized_model = optimizer.quantize("dynamic")
        else:  # balanced
            level = OptimizationLevel.BALANCED
            pruned_model = optimizer.prune_model(amount=0.2)
            quantized_model = optimizer.quantize("dynamic")
        
        # Optimize for mobile
        mobile_paths = optimizer.optimize_for_mobile(level)
        
        # Convert to ONNX
        onnx_paths = optimizer._save_model(
            quantized_model if 'quantized_model' in locals() else model,
            f"{Path(model_path).stem}_optimized",
            format="onnx"
        )
        
        # Optimize ONNX model
        if "onnx" in onnx_paths:
            optimized_onnx = optimizer.optimize_onnx_model(onnx_paths["onnx"])
            onnx_paths["onnx_optimized"] = optimized_onnx
        
        # Collect results
        result = {
            "status": "success",
            "original_size": f"{optimizer.original_size/1e6:.2f} MB",
            "optimized_models": {
                **onnx_paths,
                **mobile_paths
            }
        }
        
        logger.info(f"Successfully optimized {model_path}")
        return result
        
    except Exception as e:
        error_msg = f"Failed to optimize {model_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "error",
            "error": error_msg
        }

def main():
    parser = argparse.ArgumentParser(description='Optimize deepfake detection models')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing models to optimize')
    parser.add_argument('--output-dir', type=str, default='optimized_models',
                       help='Directory to save optimized models')
    parser.add_argument('--model-type', type=str, default='image',
                       choices=['image', 'video', 'audio'],
                       help='Type of models being optimized')
    parser.add_argument('--optimization-level', type=str, default='balanced',
                       choices=['size', 'speed', 'balanced'],
                       help='Optimization level')
    parser.add_argument('--model-file', type=str, default=None,
                       help='Optimize a single model file')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    try:
        if args.model_file:
            # Optimize single model
            model_name = os.path.basename(args.model_file)
            results[model_name] = optimize_single_model(
                args.model_file,
                args.output_dir,
                args.model_type,
                args.optimization_level
            )
        else:
            # Optimize all models in directory
            model_files = []
            for ext in ['*.pt', '*.pth']:
                model_files.extend(Path(args.models_dir).rglob(ext))
            
            if not model_files:
                logger.warning(f"No model files found in {args.models_dir}")
                return
            
            for model_file in model_files:
                model_name = model_file.name
                results[model_name] = optimize_single_model(
                    str(model_file),
                    args.output_dir,
                    args.model_type,
                    args.optimization_level
                )
        
        # Save results
        report_path = os.path.join(args.output_dir, 'optimization_report.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("Optimization Complete")
        print("="*50)
        
        success_count = sum(1 for r in results.values() if r.get('status') == 'success')
        total_count = len(results)
        
        print(f"\nProcessed {total_count} models")
        print(f"Successfully optimized: {success_count}/{total_count}")
        print(f"\nDetailed report saved to: {os.path.abspath(report_path)}")
        print("\nOptimized models saved to:", os.path.abspath(args.output_dir))
        
    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
