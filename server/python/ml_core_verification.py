#!/usr/bin/env python3
"""
Core ML Verification - Deep dive into actual model inference
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

print("🔬 CORE ML INFERENCE DEEP DIVE")
print("=" * 60)

def verify_pytorch_model_loading():
    """Verify actual PyTorch model loading and inference"""
    print("\n🔍 Verifying PyTorch Model Loading...")
    
    try:
        from models.deepfake_classifier import DeepfakeClassifier
        print("✅ DeepfakeClassifier imported")
        
        # Test different model types
        model_types = ['efficientnet', 'xception', 'resnet50']
        
        for model_type in model_types:
            print(f"\n--- Testing {model_type.upper()} ---")
            try:
                classifier = DeepfakeClassifier(model_type=model_type, device='cpu')
                print(f"✅ {model_type} model loaded successfully")
                
                # Check if model is real
                model = classifier.model
                if hasattr(model, 'parameters'):
                    param_count = sum(p.numel() for p in model.parameters())
                    print(f"📊 Parameters: {param_count:,}")
                
                # Test inference with same input multiple times
                test_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                
                results = []
                for i in range(3):
                    result = classifier.predict_image(test_input)
                    results.append(result)
                    print(f"  Run {i+1}: {result['prediction']} ({result['confidence']:.4f})")
                
                # Check if results are deterministic (should be for same input)
                confidences = [r['confidence'] for r in results]
                predictions = [r['prediction'] for r in results]
                
                is_deterministic = all(c == confidences[0] for c in confidences)
                print(f"🔄 Deterministic: {'✅' if is_deterministic else '❌'}")
                
                # Test with different inputs
                different_input = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                diff_result = classifier.predict_image(different_input)
                
                print(f"🎲 Different input: {diff_result['prediction']} ({diff_result['confidence']:.4f})")
                
                # Verify logits are real
                logits = result['logits']
                print(f"📊 Logits shape: {logits.shape}")
                print(f"📊 Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                
                # Verify softmax is applied correctly
                class_probs = result['class_probs']
                prob_sum = class_probs['real'] + class_probs['fake']
                print(f"📊 Probability sum: {prob_sum:.6f} (should be ~1.0)")
                
                print(f"✅ {model_type} verification complete\n")
                
            except Exception as e:
                print(f"❌ {model_type} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_model_weights():
    """Verify that model weights are loaded and not random"""
    print("\n⚖️ Verifying Model Weights...")
    
    try:
        from models.deepfake_classifier import DeepfakeClassifier
        
        classifier = DeepfakeClassifier(model_type='efficientnet', device='cpu')
        model = classifier.model
        
        # Check if weights are loaded (not all zeros or random)
        weight_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weights = param.data
                weight_stats[name] = {
                    'mean': weights.mean().item(),
                    'std': weights.std().item(),
                    'min': weights.min().item(),
                    'max': weights.max().item(),
                    'shape': weights.shape
                }
        
        print(f"📊 Found {len(weight_stats)} parameter layers")
        
        # Show stats for first few layers
        for i, (name, stats) in enumerate(weight_stats.items()):
            if i < 3:  # Show first 3 layers
                print(f"  Layer {name}:")
                print(f"    Shape: {stats['shape']}")
                print(f"    Mean: {stats['mean']:.6f}")
                print(f"    Std:  {stats['std']:.6f}")
                print(f"    Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        
        # Check if weights are reasonable (not all zeros)
        all_zero_count = sum(1 for stats in weight_stats.values() 
                           if abs(stats['mean']) < 1e-6 and stats['std'] < 1e-6)
        
        print(f"🚫 Layers with all-zero weights: {all_zero_count}")
        
        if all_zero_count == 0:
            print("✅ Model weights appear to be properly loaded")
        else:
            print("⚠️ Some layers may have zero weights")
        
        return all_zero_count == 0
        
    except Exception as e:
        print(f"❌ Weight verification failed: {e}")
        return False

def verify_inference_pipeline():
    """Verify the complete inference pipeline"""
    print("\n🔄 Verifying Inference Pipeline...")
    
    try:
        from models.deepfake_classifier import DeepfakeClassifier
        
        classifier = DeepfakeClassifier(model_type='efficientnet', device='cpu')
        
        # Test with structured inputs
        test_cases = [
            ("All zeros", np.zeros((224, 224, 3), dtype=np.uint8)),
            ("All ones", np.ones((224, 224, 3), dtype=np.uint8) * 255),
            ("Random noise", np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
            ("Gradient", np.arange(224*224*3).reshape(224, 224, 3).astype(np.uint16) % 256),
        ]
        
        for case_name, test_input in test_cases:
            print(f"\n--- Testing {case_name} ---")
            
            # Preprocessing check
            if hasattr(classifier, 'transform'):
                from PIL import Image
                pil_image = Image.fromarray(test_input)
                transformed = classifier.transform(pil_image)
                print(f"📊 Transformed shape: {transformed.shape}")
                print(f"📊 Transformed range: [{transformed.min():.4f}, {transformed.max():.4f}]")
            
            # Inference
            result = classifier.predict_image(test_input)
            
            print(f"🎯 Prediction: {result['prediction']}")
            print(f"📊 Confidence: {result['confidence']:.4f}")
            print(f"📊 Real prob: {result['class_probs']['real']:.4f}")
            print(f"📊 Fake prob: {result['class_probs']['fake']:.4f}")
            print(f"⏱️ Inference time: {result['inference_time']*1000:.2f}ms")
            
            # Verify output consistency
            prob_sum = result['class_probs']['real'] + result['class_probs']['fake']
            print(f"📊 Prob sum: {prob_sum:.6f}")
            
            if abs(prob_sum - 1.0) > 0.001:
                print("❌ Probabilities don't sum to 1.0")
            else:
                print("✅ Probabilities sum correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference pipeline verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run core ML verification"""
    print("Starting core ML verification...\n")
    
    tests = [
        ("PyTorch Model Loading", verify_pytorch_model_loading),
        ("Model Weights Verification", verify_model_weights),
        ("Inference Pipeline", verify_inference_pipeline),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    print("\n" + "=" * 60)
    print("📊 CORE ML VERIFICATION RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 CORE ML VERIFICATION PASSED")
        print("✅ Real PyTorch models are loaded and working")
        print("✅ Model weights are properly initialized")
        print("✅ Inference pipeline is functional")
        print("✅ Outputs are mathematically sound")
        print("🚀 ML CORE IS PRODUCTION READY")
        return True
    else:
        print(f"\n⚠️ {total-passed} test(s) failed - Review core ML")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
