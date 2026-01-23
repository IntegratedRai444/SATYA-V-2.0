#!/usr/bin/env python3
"""
SatyaAI Model Optimization Script
Optimizes ML models for better performance and accuracy
"""

import os
import sys
import torch
import torch.nn as nn
import torch.quantization
from pathlib import Path
import logging
import time
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.image_model import EfficientNetDetector, XceptionDetector
from models.audio_model import AudioDetector
from models.video_model import VideoDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Optimize ML models for performance and accuracy"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.optimized_dir = self.models_dir / "optimized"
        self.optimized_dir.mkdir(exist_ok=True)
        
    def optimize_model_performance(self, model: nn.Module, model_name: str) -> nn.Module:
        """Optimize model for better performance"""
        logger.info(f"🚀 Optimizing {model_name} for performance...")
        
        # 1. Quantization (FP16/INT8)
        if torch.cuda.is_available():
            logger.info("  🔧 Applying FP16 quantization...")
            model = model.half()
            
        # 2. Dynamic Quantization for CPU inference
        if not torch.cuda.is_available():
            logger.info("  🔧 Applying INT8 dynamic quantization...")
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        
        # 3. TorchScript compilation
        logger.info("  🔧 Compiling with TorchScript...")
        try:
            model = torch.jit.script(model)
            logger.info("  ✅ TorchScript compilation successful")
        except Exception as e:
            logger.warning(f"  ⚠️ TorchScript compilation failed: {e}")
            try:
                # Fallback to tracing
                dummy_input = torch.randn(1, 3, 224, 224)
                model = torch.jit.trace(model, dummy_input)
                logger.info("  ✅ TorchScript tracing successful")
            except Exception as e2:
                logger.warning(f"  ⚠️ TorchScript tracing failed: {e2}")
        
        # 4. Model pruning (optional)
        logger.info("  🔧 Applying model pruning...")
        try:
            self._apply_pruning(model)
        except Exception as e:
            logger.warning(f"  ⚠️ Pruning failed: {e}")
        
        return model
    
    def _apply_pruning(self, model: nn.Module, pruning_ratio: float = 0.2):
        """Apply structured pruning to reduce model size"""
        import torch.nn.utils.prune as prune
        
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        if parameters_to_prune:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
            
            # Remove pruning masks to make it permanent
            for module, param_name in parameters_to_prune:
                prune.remove(module, param_name)
            
            logger.info(f"  ✅ Applied {pruning_ratio*100:.0f}% pruning to {len(parameters_to_prune)} layers")
    
    def fine_tune_model(self, model: nn.Module, model_name: str, 
                       train_data_loader=None, epochs: int = 5) -> nn.Module:
        """Fine-tune model for better accuracy"""
        logger.info(f"🎯 Fine-tuning {model_name} for better accuracy...")
        
        if train_data_loader is None:
            logger.warning("  ⚠️ No training data provided. Skipping fine-tuning.")
            return model
        
        # Set model to training mode
        model.train()
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info(f"  🏋️ Training for {epochs} epochs on {device}...")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(train_data_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"    Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.info(f"  📊 Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
            
            scheduler.step()
        
        # Set back to evaluation mode
        model.eval()
        logger.info(f"  ✅ Fine-tuning completed for {model_name}")
        
        return model
    
    def benchmark_model(self, model: nn.Module, model_name: str, 
                      input_shape: tuple = (1, 3, 224, 224), 
                      num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark model performance"""
        logger.info(f"⏱️ Benchmarking {model_name}...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / num_runs * 1000  # ms
        throughput = num_runs / (end_time - start_time)  # samples/sec
        
        # Model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024  # MB
        
        results = {
            'model_name': model_name,
            'avg_inference_time_ms': avg_inference_time,
            'throughput_samples_per_sec': throughput,
            'model_size_mb': model_size,
            'device': str(device),
            'num_runs': num_runs
        }
        
        logger.info(f"  📊 Results: {results}")
        return results
    
    def optimize_all_models(self):
        """Optimize all available models"""
        logger.info("🚀 Starting comprehensive model optimization...")
        
        results = {}
        
        # Optimize Image Models
        try:
            logger.info("🖼️ Optimizing Image Models...")
            
            # EfficientNet
            efficientnet = EfficientNetDetector()
            efficientnet_optimized = self.optimize_model_performance(efficientnet.model, "EfficientNet-B7")
            results['efficientnet'] = self.benchmark_model(efficientnet_optimized, "EfficientNet-B7")
            
            # Save optimized model
            torch.save(efficientnet_optimized.state_dict(), 
                       self.optimized_dir / "efficientnet_b7_optimized.pth")
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize EfficientNet: {e}")
        
        # Optimize Audio Models
        try:
            logger.info("🎵 Optimizing Audio Models...")
            audio_detector = AudioDetector()
            audio_optimized = self.optimize_model_performance(audio_detector.model, "AudioDetector")
            
            # Create dummy audio input for benchmarking
            dummy_audio = torch.randn(1, 1, 16000)  # 1 second of audio
            results['audio'] = self.benchmark_model(audio_optimized, "AudioDetector", 
                                                 input_shape=(1, 1, 16000), num_runs=50)
            
            torch.save(audio_optimized.state_dict(), 
                       self.optimized_dir / "audio_detector_optimized.pth")
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize Audio model: {e}")
        
        # Optimize Video Models
        try:
            logger.info("🎥 Optimizing Video Models...")
            video_detector = VideoDetector()
            video_optimized = self.optimize_model_performance(video_detector.model, "VideoDetector")
            
            # Create dummy video input for benchmarking
            dummy_video = torch.randn(1, 3, 16, 224, 224)  # 16 frames
            results['video'] = self.benchmark_model(video_optimized, "VideoDetector",
                                                 input_shape=(1, 3, 16, 224, 224), num_runs=20)
            
            torch.save(video_optimized.state_dict(), 
                       self.optimized_dir / "video_detector_optimized.pth")
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize Video model: {e}")
        
        # Save benchmark results
        import json
        with open(self.optimized_dir / "benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("✅ Model optimization completed!")
        logger.info(f"📁 Optimized models saved to: {self.optimized_dir}")
        logger.info(f"📊 Benchmark results saved to: {self.optimized_dir / 'benchmark_results.json'}")
        
        return results

def main():
    """Main optimization script"""
    logger.info("🎯 SatyaAI Model Optimization Script")
    logger.info("=" * 50)
    
    # Check if models directory exists
    models_dir = os.getenv('MODEL_DIR', 'models')
    if not Path(models_dir).exists():
        logger.error(f"❌ Models directory not found: {models_dir}")
        logger.info("💡 Please ensure models are downloaded first")
        return
    
    # Initialize optimizer
    optimizer = ModelOptimizer(models_dir)
    
    # Run optimization
    results = optimizer.optimize_all_models()
    
    # Print summary
    logger.info("\n🎉 OPTIMIZATION SUMMARY")
    logger.info("=" * 50)
    for model_name, metrics in results.items():
        logger.info(f"📊 {model_name.upper()}:")
        logger.info(f"   ⏱️  Inference Time: {metrics['avg_inference_time_ms']:.2f}ms")
        logger.info(f"   🚀 Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")
        logger.info(f"   💾 Model Size: {metrics['model_size_mb']:.2f}MB")
        logger.info(f"   🔧 Device: {metrics['device']}")
        logger.info("")

if __name__ == "__main__":
    main()
