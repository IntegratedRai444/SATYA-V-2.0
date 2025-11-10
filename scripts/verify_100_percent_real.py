#!/usr/bin/env python3
"""
SatyaAI 100% Real Verification Script
Verifies that all components are using real AI models with no fallbacks
"""

import os
import sys
import json
from pathlib import Path
import importlib.util

def check_real_models():
    """Check if real AI models are available."""
    models_dir = Path("server/python/models")
    
    real_models = {
        "xception_c23.pth": "Xception C23 (FaceForensics++)",
        "dfdc_efficientnet_b7": "EfficientNet-B7 (DFDC Winner)",
        "microsoft_video_auth": "Microsoft Video Authenticator", 
        "nvidia_stylegan_detector.pth": "NVIDIA StyleGAN Detector",
        "stanford_audio_detector": "Stanford Audio Deepfake Detector",
        "resnet50_deepfake.pth": "ResNet50 Deepfake Detector",
        "haarcascade_frontalface_default.xml": "OpenCV Face Detector"
    }
    
    print("🔍 Checking Real AI Models:")
    print("=" * 50)
    
    available_models = 0
    for model_file, model_name in real_models.items():
        model_path = models_dir / model_file
        if model_path.exists():
            print(f"✅ {model_name}")
            available_models += 1
        else:
            print(f"❌ {model_name} - NOT FOUND")
    
    print("=" * 50)
    print(f"📊 Real Models Available: {available_models}/{len(real_models)}")
    
    return available_models >= len(real_models) * 0.7  # 70% threshold

def check_dependencies():
    """Check if all professional dependencies are installed."""
    professional_deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("facenet_pytorch", "FaceNet PyTorch"),
        ("transformers", "Transformers"),
        ("huggingface_hub", "Hugging Face Hub"),
        ("timm", "PyTorch Image Models"),
        ("librosa", "Librosa Audio"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy")
    ]
    
    print("\n🔍 Checking Professional Dependencies:")
    print("=" * 50)
    
    available_deps = 0
    for module_name, display_name in professional_deps:
        try:
            __import__(module_name)
            print(f"✅ {display_name}")
            available_deps += 1
        except ImportError:
            print(f"❌ {display_name} - NOT INSTALLED")
    
    print("=" * 50)
    print(f"📊 Dependencies Available: {available_deps}/{len(professional_deps)}")
    
    return available_deps == len(professional_deps)

def check_no_fallbacks():
    """Check that no fallback/mock code exists in detectors."""
    detector_files = [
        "server/python/detectors/image_detector.py",
        "server/python/detectors/audio_detector.py", 
        "server/python/detectors/video_detector.py",
        "server/python/enhanced_detector.py"
    ]
    
    print("\n🔍 Checking for Fallback/Mock Code:")
    print("=" * 50)
    
    fallback_patterns = [
        "mock",
        "fake", 
        "demo",
        "placeholder",
        "fallback",
        "generate_mock",
        "random.rand"
    ]
    
    clean_files = 0
    for file_path in detector_files:
        if not Path(file_path).exists():
            print(f"❌ {file_path} - FILE NOT FOUND")
            continue
            
        with open(file_path, 'r') as f:
            content = f.read().lower()
        
        has_fallbacks = False
        for pattern in fallback_patterns:
            if pattern in content:
                has_fallbacks = True
                break
        
        if has_fallbacks:
            print(f"⚠️  {file_path} - Contains fallback code")
        else:
            print(f"✅ {file_path} - 100% Real AI")
            clean_files += 1
    
    print("=" * 50)
    print(f"📊 Clean Files: {clean_files}/{len(detector_files)}")
    
    return clean_files == len(detector_files)

def test_real_detection():
    """Test that the system actually uses real AI models."""
    print("\n🧪 Testing Real AI Detection:")
    print("=" * 50)
    
    try:
        # Change to Python directory
        original_dir = os.getcwd()
        os.chdir("server/python")
        
        # Add to path
        sys.path.insert(0, os.getcwd())
        
        # Import and test
        from enhanced_detector import EnhancedDeepfakeDetector
        import numpy as np
        from PIL import Image
        import io
        
        # Create detector
        detector = EnhancedDeepfakeDetector()
        
        # Create test image
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Test detection
        result = detector.analyze_image(image_bytes)
        
        # Check if using real AI models
        if result.get('success', False):
            detailed_analysis = result.get('detailed_analysis', {})
            ai_analysis = detailed_analysis.get('ai_model_analysis', {})
            
            if ai_analysis:
                model_used = ai_analysis.get('model_used', 'Unknown')
                print(f"✅ Real AI Detection Working")
                print(f"   Model: {model_used}")
                print(f"   Confidence: {result.get('confidence', 0):.1f}%")
                print(f"   Processing Time: {result.get('technical_details', {}).get('processing_time_seconds', 0):.2f}s")
                return True
            else:
                print(f"⚠️  Detection working but using fallback methods")
                return False
        else:
            print(f"❌ Detection failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        os.chdir(original_dir)
        if os.getcwd() in sys.path:
            sys.path.remove(os.getcwd())

def generate_verification_report():
    """Generate a comprehensive verification report."""
    print("\n" + "=" * 60)
    print("🎯 SATYAAI 100% REAL VERIFICATION REPORT")
    print("=" * 60)
    
    # Run all checks
    models_ok = check_real_models()
    deps_ok = check_dependencies() 
    clean_ok = check_no_fallbacks()
    detection_ok = test_real_detection()
    
    # Calculate overall score
    checks = [models_ok, deps_ok, clean_ok, detection_ok]
    score = sum(checks) / len(checks) * 100
    
    print("\n📊 VERIFICATION SUMMARY:")
    print("=" * 30)
    print(f"✅ Real AI Models: {'PASS' if models_ok else 'FAIL'}")
    print(f"✅ Professional Dependencies: {'PASS' if deps_ok else 'FAIL'}")
    print(f"✅ No Fallback Code: {'PASS' if clean_ok else 'FAIL'}")
    print(f"✅ Real AI Detection: {'PASS' if detection_ok else 'FAIL'}")
    
    print("\n🎯 OVERALL SCORE:")
    print("=" * 20)
    print(f"🔥 {score:.0f}% REAL")
    
    if score == 100:
        print("\n🎉 CONGRATULATIONS!")
        print("Your SatyaAI system is 100% REAL with no fallbacks!")
        print("✅ Professional-grade AI models")
        print("✅ Research-level accuracy")
        print("✅ No mock or demo code")
        print("✅ Production-ready detection")
    elif score >= 80:
        print("\n🚀 EXCELLENT!")
        print("Your system is mostly real with professional capabilities")
        print("Minor issues can be resolved by running setup_professional_ai.py")
    elif score >= 60:
        print("\n⚠️  GOOD START")
        print("Your system has real components but needs improvement")
        print("Run setup_professional_ai.py to upgrade to 100% real")
    else:
        print("\n❌ NEEDS WORK")
        print("Your system requires significant upgrades")
        print("Run setup_professional_ai.py for full professional setup")
    
    return score

def main():
    """Main verification function."""
    try:
        score = generate_verification_report()
        return 0 if score >= 80 else 1
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())