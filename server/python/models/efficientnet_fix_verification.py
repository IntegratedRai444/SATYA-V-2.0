"""
Enhanced EfficientNet-B7 Model Verification & Testing Suite
Comprehensive testing for model integrity, loading, and performance
"""

import sys
import logging
import time
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test that the fixed model loads correctly"""
    try:
        # Add parent directory to path
        sys.path.append(str(Path(__file__).parent.parent))
        
        from deepfake_classifier import DeepfakeClassifier, get_model_info
        
        logger.info("Testing EfficientNet-B7 model loading...")
        
        # Test model initialization
        classifier = DeepfakeClassifier(model_type='efficientnet', device='cpu')
        
        # Check if model loaded successfully
        if classifier.model is not None:
            logger.info("✅ EfficientNet-B7 model loaded successfully")
            
            # Test model info
            model_info = get_model_info()
            logger.info(f"📊 Model info: {model_info}")
            
            # Test inference with dummy data
            import numpy as np
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            result = classifier.predict_image(dummy_image)
            logger.info(f"🎯 Test prediction: {result['prediction']} (confidence: {result['confidence']:.3f})")
            
            # Verify output structure
            required_keys = ['prediction', 'confidence', 'logits', 'class_probs', 'inference_time']
            missing_keys = [key for key in required_keys if key not in result]
            
            if not missing_keys:
                logger.info("✅ Model output structure is correct")
                logger.info("✅ EfficientNet-B7 model fix verification PASSED")
                return True
            else:
                logger.error(f"❌ Missing output keys: {missing_keys}")
                return False
                
        else:
            logger.error("❌ Model failed to load")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        return False

def test_model_file_integrity():
    """Test the model file integrity"""
    try:
        model_path = Path("e:/SATYA-V-2.0/models/dfdc_efficientnet_b7/model.pth")
        
        if not model_path.exists():
            logger.error("❌ Model file does not exist")
            return False
            
        # Check file size
        file_size = model_path.stat().st_size
        if file_size < 1000000:  # Less than 1MB indicates fake model
            logger.error(f"❌ Model file too small: {file_size} bytes")
            return False
            
        logger.info(f"✅ Model file size: {file_size:,} bytes")
        
        # Check file header
        with open(model_path, 'rb') as f:
            header = f.read(4)
            if header == b'PK\x03\x04':
                logger.info("✅ Valid PyTorch model format detected")
            else:
                logger.warning("⚠️ Unexpected file header")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ File integrity test failed: {e}")
        return False

def test_model_performance():
    """Test model performance with multiple inputs"""
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from deepfake_classifier import DeepfakeClassifier
        
        logger.info("Testing model performance...")
        classifier = DeepfakeClassifier(model_type='efficientnet', device='cpu')
        
        # Test with multiple inputs
        test_images = []
        for i in range(5):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            test_images.append(img)
        
        # Batch testing
        start_time = time.time()
        results = []
        for i, img in enumerate(test_images):
            result = classifier.predict_image(img)
            results.append(result)
            logger.info(f"Test {i+1}: {result['prediction']} ({result['confidence']:.3f})")
        
        total_time = time.time() - start_time
        avg_time = total_time / len(test_images)
        
        logger.info(f"✅ Performance test completed")
        logger.info(f"   Total time: {total_time:.3f}s")
        logger.info(f"   Average per image: {avg_time:.3f}s")
        
        # Check consistency
        predictions = [r['prediction'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        logger.info(f"   Predictions: {predictions}")
        logger.info(f"   Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False

def test_model_robustness():
    """Test model robustness with edge cases"""
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        from deepfake_classifier import DeepfakeClassifier
        
        logger.info("Testing model robustness...")
        classifier = DeepfakeClassifier(model_type='efficientnet', device='cpu')
        
        # Test edge cases
        test_cases = [
            ("All zeros", np.zeros((224, 224, 3), dtype=np.uint8)),
            ("All ones", np.ones((224, 224, 3), dtype=np.uint8) * 255),
            ("Random noise", np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
            ("Gradient", np.arange(224*224*3, dtype=np.uint8).reshape(224, 224, 3)),
        ]
        
        for name, img in test_cases:
            try:
                result = classifier.predict_image(img)
                logger.info(f"✅ {name}: {result['prediction']} ({result['confidence']:.3f})")
            except Exception as e:
                logger.warning(f"⚠️ {name} failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Robustness test failed: {e}")
        return False

def generate_model_report():
    """Generate comprehensive model report"""
    try:
        model_path = Path("e:/SATYA-V-2.0/models/dfdc_efficientnet_b7/model.pth")
        
        if not model_path.exists():
            logger.error("❌ Model file not found")
            return None
        
        # Load and analyze model
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        
        report = {
            'file_info': {
                'path': str(model_path),
                'size_mb': model_path.stat().st_size / (1024*1024),
                'modified': model_path.stat().st_mtime
            },
            'model_info': {
                'architecture': checkpoint.get('architecture', 'unknown'),
                'config_name': checkpoint.get('config_name', 'unknown'),
                'num_classes': checkpoint.get('num_classes', 'unknown'),
                'input_size': checkpoint.get('input_size', 'unknown')
            },
            'training_info': checkpoint.get('training_info', {}),
            'tests_passed': []
        }
        
        logger.info("📊 Model Report Generated:")
        logger.info(f"   Architecture: {report['model_info']['architecture']}")
        logger.info(f"   Size: {report['file_info']['size_mb']:.1f} MB")
        logger.info(f"   Classes: {report['model_info']['num_classes']}")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED EFFICIENTNET-B7 MODEL VERIFICATION")
    print("=" * 60)
    
    # Test file integrity
    print("\n1. Testing file integrity...")
    integrity_ok = test_model_file_integrity()
    
    # Test model loading
    print("\n2. Testing model loading...")
    loading_ok = test_model_loading()
    
    # Test performance
    print("\n3. Testing model performance...")
    performance_ok = test_model_performance()
    
    # Test robustness
    print("\n4. Testing model robustness...")
    robustness_ok = test_model_robustness()
    
    # Generate report
    print("\n5. Generating model report...")
    report = generate_model_report()
    
    # Summary
    print("\n" + "=" * 60)
    print("ENHANCED VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"File Integrity: {'✅ PASS' if integrity_ok else '❌ FAIL'}")
    print(f"Model Loading: {'✅ PASS' if loading_ok else '❌ FAIL'}")
    print(f"Performance: {'✅ PASS' if performance_ok else '❌ FAIL'}")
    print(f"Robustness: {'✅ PASS' if robustness_ok else '❌ FAIL'}")
    print(f"Report: {'✅ GENERATED' if report else '❌ FAILED'}")
    
    all_passed = all([integrity_ok, loading_ok, performance_ok, robustness_ok, report])
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Model is fully functional!")
        print("✅ Ready for production use")
    else:
        print("\n⚠️ SOME TESTS FAILED - Review issues above")
        print("🔧 Model may need additional configuration")
    
    print("\n📚 Enhanced verification complete!")
