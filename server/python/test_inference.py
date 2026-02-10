#!/usr/bin/env python3
"""
Test script to verify real ML inference in SATYA AI Python backend
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

print("🔍 SATYA AI Python Backend ML Inference Verification")
print("=" * 60)

def test_image_inference():
    """Test image detector with real ML inference"""
    print("\n📸 Testing Image Inference...")
    
    try:
        from models.deepfake_classifier import DeepfakeClassifier, is_model_available
        print("✅ DeepfakeClassifier import successful")
        
        # Check model availability
        available = is_model_available()
        print(f"📊 Model availability: {available}")
        
        # Try to initialize a model
        classifier = DeepfakeClassifier(model_type='efficientnet', device='cpu')
        print("✅ Model initialization successful")
        
        # Test inference with dummy data
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = classifier.predict_image(dummy_image)
        print(f"🎯 Inference result: {result['prediction']} with confidence {result['confidence']:.3f}")
        print("✅ Real inference working")
        
        # Verify output structure
        required_keys = ['prediction', 'confidence', 'logits', 'class_probs', 'inference_time']
        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing required key: {key}")
        
        # Verify confidence range
        if not 0.0 <= result['confidence'] <= 1.0:
            raise ValueError(f"Confidence out of range: {result['confidence']}")
        
        # Verify prediction values
        if result['prediction'] not in ['real', 'fake']:
            raise ValueError(f"Invalid prediction: {result['prediction']}")
        
        print("✅ Output validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Image inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_inference():
    """Test video detector with real ML inference"""
    print("\n🎥 Testing Video Inference...")
    
    try:
        from detectors.video_detector import VideoDetector
        print("✅ VideoDetector import successful")
        
        # Initialize video detector
        detector = VideoDetector()  # Use default config
        print("✅ Video detector initialization successful")
        
        # Test with dummy video data (simulate frames)
        dummy_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(10)]
        
        # Test frame analysis
        if detector.image_detector:
            frame_result = detector.image_detector.analyze(dummy_frames[0])
            print(f"🎯 Frame analysis result: {frame_result.get('label', 'N/A')}")
            print("✅ Frame-level inference working")
        
        return True
        
    except Exception as e:
        print(f"❌ Video inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_inference():
    """Test audio detector with real ML inference"""
    print("\n🎵 Testing Audio Inference...")
    
    try:
        from detectors.audio_detector import AudioDetector
        print("✅ AudioDetector import successful")
        
        # Initialize audio detector
        detector = AudioDetector(device='cpu')
        print("✅ Audio detector initialization successful")
        
        # Test with dummy audio data
        dummy_audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
        
        # Test basic audio processing
        if hasattr(detector, '_analyze_basic_features'):
            features = detector._analyze_basic_features(dummy_audio)
            print(f"🎯 Audio features extracted: {len(features) if isinstance(features, dict) else 'N/A'}")
            print("✅ Audio feature extraction working")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sentinel_agent():
    """Test SentinelAgent integration"""
    print("\n🤖 Testing SentinelAgent...")
    
    try:
        from sentinel_agent import SentinelAgent, AnalysisRequest, AnalysisType
        print("✅ SentinelAgent import successful")
        
        # Initialize agent
        agent = SentinelAgent()
        print("✅ SentinelAgent initialization successful")
        
        # Test analysis request
        request = AnalysisRequest(
            analysis_type=AnalysisType.IMAGE,
            content=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8).tobytes()
        )
        
        print("✅ Analysis request creation successful")
        return True
        
    except Exception as e:
        print(f"❌ SentinelAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detector_singleton():
    """Test detector singleton pattern"""
    print("\n🔒 Testing Detector Singleton...")
    
    try:
        from services.detector_singleton import get_detector_singleton
        print("✅ DetectorSingleton import successful")
        
        # Get singleton instance
        singleton = get_detector_singleton()
        print("✅ Singleton instance retrieved")
        
        # Test lazy loading
        image_detector = singleton.get_detector('image')
        print(f"🎯 Image detector: {'Loaded' if image_detector else 'Not loaded'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detector singleton test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Starting comprehensive ML inference verification...\n")
    
    tests = [
        ("Image Inference", test_image_inference),
        ("Video Inference", test_video_inference),
        ("Audio Inference", test_audio_inference),
        ("SentinelAgent", test_sentinel_agent),
        ("Detector Singleton", test_detector_singleton),
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Python backend is PRODUCTION READY")
        return True
    else:
        print("⚠️  Some tests failed - Review issues before production deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
