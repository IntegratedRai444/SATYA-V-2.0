"""
Model Optimization Module
Handles model quantization, pruning, and optimization for edge deployment.
"""
import os
import torch
import torch.nn as nn
import torch.quantization
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, QuantStub, DeQuantStub
from torch.quantization import fuse_modules
from torch.jit import script, trace
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
import logging
import numpy as np
from enum import Enum
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic as onnx_quantize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Optimization levels for model deployment."""
    DEFAULT = "default"
    PERFORMANCE = "performance"
    SIZE = "size"
    BALANCED = "balanced"

class ModelOptimizer:
    """Handles model optimization techniques for deployment."""
    
    def __init__(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        model_name: str = "model",
        output_dir: str = "optimized_models"
    ):
        """
        Initialize the model optimizer.
        
        Args:
            model: PyTorch model to optimize
            example_input: Example input tensor for tracing/quantization
            model_name: Name for saving optimized models
            output_dir: Directory to save optimized models
        """
        self.model = model.eval()
        self.example_input = example_input
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect device
        self.device = next(model.parameters()).device
        
        # Store original model for comparison
        self.original_size = self._get_model_size(model)
        logger.info(f"Original model size: {self.original_size/1e6:.2f} MB")
    
    def _get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes."""
        # Save to a temporary file to get size
        temp_path = self.output_dir / "temp.pt"
        torch.save(model.state_dict(), temp_path)
        size = temp_path.stat().st_size
        temp_path.unlink()
        return size
    
    def _save_model(
        self,
        model: nn.Module,
        name: str,
        format: str = "pt",
        onnx_opset: int = 13
    ) -> Dict[str, str]:
        """Save model in specified format."""
        save_paths = {}
        
        # PyTorch format
        if format in ["pt", "all"]:
            pt_path = self.output_dir / f"{name}.pt"
            torch.save(model.state_dict(), pt_path)
            save_paths["pytorch"] = str(pt_path)
        
        # TorchScript format
        if format in ["torchscript", "all"]:
            try:
                scripted_model = torch.jit.script(model)
                ts_path = self.output_dir / f"{name}_scripted.pt"
                torch.jit.save(scripted_model, ts_path)
                save_paths["torchscript"] = str(ts_path)
            except Exception as e:
                logger.warning(f"Failed to create TorchScript model: {e}")
        
        # ONNX format
        if format in ["onnx", "all"]:
            try:
                onnx_path = self.output_dir / f"{name}.onnx"
                torch.onnx.export(
                    model,
                    self.example_input,
                    onnx_path,
                    opset_version=onnx_opset,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                )
                save_paths["onnx"] = str(onnx_path)
            except Exception as e:
                logger.warning(f"Failed to export to ONNX: {e}")
        
        return save_paths
    
    def quantize(
        self,
        quant_type: str = "dynamic",
        qconfig_spec: Optional[Dict[Any, Any]] = None,
        **kwargs
    ) -> nn.Module:
        """
        Quantize the model to reduce size and improve inference speed.
        
        Args:
            quant_type: Type of quantization ('dynamic', 'static', 'qat')
            qconfig_spec: Custom quantization configuration
            
        Returns:
            Quantized model
        """
        logger.info(f"Applying {quant_type} quantization...")
        
        # Default quantization configuration
        if qconfig_spec is None:
            qconfig_spec = {
                # Default quantization for common layer types
                nn.Linear: torch.quantization.default_dynamic_qconfig,
                nn.LSTM: torch.quantization.default_dynamic_qconfig,
                nn.LSTMCell: torch.quantization.default_dynamic_qconfig,
                nn.GRU: torch.quantization.default_dynamic_qconfig,
                nn.GRUCell: torch.quantization.default_dynamic_qconfig,
                nn.Conv1d: torch.quantization.default_dynamic_qconfig,
                nn.Conv2d: torch.quantization.default_dynamic_qconfig,
                nn.Conv3d: torch.quantization.default_dynamic_qconfig,
            }
        
        quantized_model = self.model
        
        try:
            if quant_type == "dynamic":
                # Dynamic quantization (post-training)
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model,
                    qconfig_spec=qconfig_spec,
                    dtype=torch.qint8,
                    **kwargs
                )
                
            elif quant_type == "static":
                # Static quantization requires calibration
                logger.warning("Static quantization requires calibration data. "
                             "Using dynamic quantization instead.")
                quantized_model = self.quantize("dynamic", qconfig_spec, **kwargs)
                
            elif quant_type == "qat":
                # Quantization-Aware Training (requires training loop)
                logger.warning("QAT requires retraining. Using dynamic quantization instead.")
                quantized_model = self.quantize("dynamic", qconfig_spec, **kwargs)
                
            else:
                raise ValueError(f"Unsupported quantization type: {quant_type}")
            
            # Save quantized model
            save_paths = self._save_model(
                quantized_model,
                f"{self.model_name}_quantized_{quant_type}",
                format="all"
            )
            
            # Log size reduction
            quantized_size = self._get_model_size(quantized_model)
            reduction = (1 - quantized_size / self.original_size) * 100
            logger.info(f"Quantized model size: {quantized_size/1e6:.2f} MB "
                      f"({reduction:.1f}% reduction)")
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return self.model
    
    def prune_model(
        self,
        amount: float = 0.2,
        prune_type: str = "l1_unstructured",
        **kwargs
    ) -> nn.Module:
        """
        Apply pruning to the model to reduce its size.
        
        Args:
            amount: Fraction of connections to prune (0-1)
            prune_type: Type of pruning ('l1_unstructured', 'random_unstructured', 'ln_structured')
            
        Returns:
            Pruned model
        """
        logger.info(f"Applying {prune_type} pruning (amount={amount})...")
        
        # Select pruning function
        if prune_type == "l1_unstructured":
            prune_func = prune.l1_unstructured
        elif prune_type == "random_unstructured":
            prune_func = prune.random_unstructured
        elif prune_type == "ln_structured":
            prune_func = prune.ln_structured
        else:
            raise ValueError(f"Unsupported pruning type: {prune_type}")
        
        # Apply pruning to all Conv and Linear layers
        pruned_model = self.model
        parameters_to_prune = []
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune_func,
            amount=amount,
            **kwargs
        )
        
        # Remove pruning reparameterization for inference
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
        
        # Save pruned model
        save_paths = self._save_model(
            pruned_model,
            f"{self.model_name}_pruned_{prune_type}_{int(amount*100)}pct",
            format="all"
        )
        
        # Log size reduction
        pruned_size = self._get_model_size(pruned_model)
        reduction = (1 - pruned_size / self.original_size) * 100
        logger.info(f"Pruned model size: {pruned_size/1e6:.2f} MB "
                   f"({reduction:.1f}% reduction)")
        
        return pruned_model
    
    def optimize_for_mobile(
        self,
        optimization_level: OptimizationLevel = OptimizationLevel.DEFAULT
    ) -> Dict[str, str]:
        """
        Optimize the model for mobile deployment.
        
        Args:
            optimization_level: Level of optimization to apply
            
        Returns:
            Dictionary with paths to optimized models
        """
        logger.info(f"Optimizing for mobile with level: {optimization_level.value}")
        
        # First, quantize the model
        quantized_model = self.quantize("dynamic")
        
        # Then apply additional optimizations based on the level
        if optimization_level == OptimizationLevel.SIZE:
            # Focus on size reduction
            optimized_model = self.prune_model(amount=0.4)
        elif optimization_level == OptimizationLevel.PERFORMANCE:
            # Focus on inference speed
            optimized_model = quantized_model
        else:  # DEFAULT or BALANCED
            # Balanced approach
            optimized_model = self.prune_model(amount=0.2)
        
        # Convert to TorchScript for mobile
        try:
            scripted_model = torch.jit.script(optimized_model)
            
            # Additional optimization passes
            if optimization_level != OptimizationLevel.SIZE:
                scripted_model = torch.jit.optimize_for_inference(scripted_model)
            
            # Save the final model
            mobile_path = self.output_dir / f"{self.model_name}_mobile.pt"
            torch.jit.save(scripted_model, mobile_path)
            
            # Log final size
            final_size = mobile_path.stat().st_size
            reduction = (1 - final_size / self.original_size) * 100
            logger.info(f"Mobile-optimized model size: {final_size/1e6:.2f} MB "
                      f"({reduction:.1f}% reduction)")
            
            return {"mobile": str(mobile_path)}
            
        except Exception as e:
            logger.error(f"Mobile optimization failed: {e}")
            return {}
    
    def optimize_onnx_model(
        self,
        onnx_model_path: str,
        quantize: bool = True,
        optimization_level: str = "all"
    ) -> str:
        """
        Optimize an ONNX model.
        
        Args:
            onnx_model_path: Path to the ONNX model
            quantize: Whether to apply quantization
            optimization_level: ONNX optimization level ('all', 'basic', 'extended')
            
        Returns:
            Path to the optimized ONNX model
        ""
        try:
            import onnxruntime as ort
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Load the ONNX model
            onnx_model = onnx.load(onnx_model_path)
            
            # Optimize the model
            optimized_model_path = str(Path(onnx_model_path).with_stem(
                f"{Path(onnx_model_path).stem}_optimized"))
            
            # Apply ONNX Runtime optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            if optimization_level == 'all':
                sess_options.add_session_config_entry('session.optimization.enable_mem_pattern', '1')
                sess_options.add_session_config_entry('session.optimization.enable_cpu_mem_arena', '1')
            
            # Save the optimized model
            ort_session = ort.InferenceSession(onnx_model_path, sess_options)
            
            # Apply quantization if requested
            if quantize:
                quantized_model_path = str(Path(onnx_model_path).with_stem(
                    f"{Path(onnx_model_path).stem}_quantized"))
                
                onnx_quantize(
                    onnx_model_path,
                    quantized_model_path,
                    weight_type=QuantType.QUInt8
                )
                
                optimized_model_path = quantized_model_path
                logger.info(f"Quantized ONNX model saved to {optimized_model_path}")
            
            return optimized_model_path
            
        except Exception as e:
            logger.error(f"ONNX optimization failed: {e}")
            return onnx_model_path

def optimize_all_models(
    models_dir: str = "models",
    output_dir: str = "optimized_models",
    example_input: Optional[torch.Tensor] = None
) -> Dict[str, Dict[str, str]]:
    """
    Optimize all models in the specified directory.
    
    Args:
        models_dir: Directory containing PyTorch models
        output_dir: Directory to save optimized models
        example_input: Example input tensor for the models
        
    Returns:
        Dictionary mapping model names to their optimization results
    """
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default example input if not provided
    if example_input is None:
        example_input = torch.randn(1, 3, 224, 224)  # Adjust based on your model input
    
    results = {}
    
    # Find all model files
    model_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth"))
    
    for model_file in model_files:
        try:
            logger.info(f"Optimizing model: {model_file.name}")
            
            # Load the model
            model = torch.load(model_file, map_location='cpu')
            if isinstance(model, dict):
                # Handle state dict
                model = model.get('model', model.get('state_dict', model))
            
            # Initialize optimizer
            optimizer = ModelOptimizer(
                model=model,
                example_input=example_input,
                model_name=model_file.stem,
                output_dir=output_dir
            )
            
            # Apply optimizations
            optimized_paths = {}
            
            # 1. Quantize
            quantized_model = optimizer.quantize("dynamic")
            
            # 2. Prune
            pruned_model = optimizer.prune_model(amount=0.2)
            
            # 3. Optimize for mobile
            mobile_paths = optimizer.optimize_for_mobile(OptimizationLevel.BALANCED)
            
            # 4. Convert to ONNX and optimize
            onnx_paths = optimizer._save_model(model, f"{model_file.stem}_onnx", format="onnx")
            if "onnx" in onnx_paths:
                optimized_onnx = optimizer.optimize_onnx_model(onnx_paths["onnx"])
                onnx_paths["onnx_optimized"] = optimized_onnx
            
            # Collect all paths
            optimized_paths.update({
                "original": str(model_file),
                **onnx_paths,
                **mobile_paths
            })
            
            results[model_file.name] = optimized_paths
            
        except Exception as e:
            logger.error(f"Failed to optimize {model_file.name}: {e}")
    
    # Save optimization report
    report_path = output_dir / "optimization_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Optimization complete! Report saved to {report_path}")
    return results
