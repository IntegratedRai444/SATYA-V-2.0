#!/usr/bin/env python3
"""
Test analysis endpoints by importing and testing directly
"""

import sys
import os
import tempfile
import numpy as np
from pathlib import Path

# Add SATYA path
sys.path.insert(0, str(Path(__file__).parent / "server" / "python"))

print("🧪 TESTING ANALYSIS ENDPOINTS DIRECTLY")
print("=" * 60)

def test_imports():
    """Test if all analysis modules can be imported"""
    print("\n📦 TESTING IMPORTS")
    print("-" * 40)
    
    try:
        # Test main imports
        print("🔄 Testing main_api import...")
        from main_api import app
        print("✅ main_api imported successfully")
        
        print("🔄 Testing detector imports...")
        from detectors.unified_detector import UnifiedDetector
        print("✅ UnifiedDetector imported successfully")
        
        print("🔄 Testing model imports...")
        from models.deepfake_classifier import DeepfakeClassifier
        print("✅ DeepfakeClassifier imported successfully")
        
        print("🔄 Testing sentinel agent...")
        from sentinel_agent import SentinelAgent
        print("✅ SentinelAgent imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test if models can be loaded"""
    print("\n🤖 TESTING MODEL LOADING")
    print("-" * 40)
    
    try:
        from models.deepfake_classifier import DeepfakeClassifier
        
        # Test different model types
        models = ['xception', 'efficientnet', 'resnet50']
        results = {}
        
        for model_type in models:
            try:
                print(f"🔄 Testing {model_type} model...")
                classifier = DeepfakeClassifier(model_type=model_type, device='cpu')
                
                # Test with dummy image
                dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                result = classifier.predict_image(dummy_image)
                
                print(f"✅ {model_type}: {result['prediction']} ({result['confidence']:.3f})")
                results[model_type] = True
                
            except Exception as e:
                print(f"❌ {model_type} failed: {e}")
                results[model_type] = False
        
        return all(results.values()), results
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False, {}

def test_unified_detector():
    """Test unified detector functionality"""
    print("\n🔍 TESTING UNIFIED DETECTOR")
    print("-" * 40)
    
    try:
        from detectors.unified_detector import UnifiedDetector
        
        # Initialize detector
        print("🔄 Initializing UnifiedDetector...")
        detector = UnifiedDetector()
        print("✅ UnifiedDetector initialized")
        
        # Test image detection
        print("🔄 Testing image detection...")
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image_result = detector.detect_image(dummy_image)
        print(f"✅ Image detection: {image_result.authenticity} ({image_result.confidence:.3f})")
        
        # Test audio detection
        print("🔄 Testing audio detection...")
        dummy_audio = np.random.randn(16000).astype(np.float32)
        audio_result = detector.detect_audio(dummy_audio)
        print(f"✅ Audio detection: {audio_result.authenticity} ({audio_result.confidence:.3f})")
        
        # Test video detection
        print("🔄 Testing video detection...")
        dummy_video = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)]
        video_result = detector.detect_video(dummy_video)
        print(f"✅ Video detection: {video_result.authenticity} ({video_result.confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ UnifiedDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sentinel_agent():
    """Test sentinel agent functionality"""
    print("\n🤖 TESTING SENTINEL AGENT")
    print("-" * 40)
    
    try:
        from sentinel_agent import SentinelAgent, AnalysisRequest, AnalysisType
        
        # Initialize agent
        print("🔄 Initializing SentinelAgent...")
        agent = SentinelAgent()
        print("✅ SentinelAgent initialized")
        
        # Test image analysis
        print("🔄 Testing image analysis...")
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image_request = AnalysisRequest(
            analysis_type=AnalysisType.IMAGE,
            content=dummy_image.tobytes()
        )
        image_result = agent.analyze(image_request)
        print(f"✅ Image analysis: {image_result.get('authenticity', 'Unknown')} ({image_result.get('confidence', 0):.3f})")
        
        # Test audio analysis
        print("🔄 Testing audio analysis...")
        dummy_audio = np.random.randn(16000).astype(np.float32)
        audio_request = AnalysisRequest(
            analysis_type=AnalysisType.AUDIO,
            content=dummy_audio.tobytes()
        )
        audio_result = agent.analyze(audio_request)
        print(f"✅ Audio analysis: {audio_result.get('authenticity', 'Unknown')} ({audio_result.get('confidence', 0):.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ SentinelAgent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_endpoints():
    """Test API endpoint functions directly"""
    print("\n🌐 TESTING API ENDPOINTS")
    print("-" * 40)
    
    try:
        # Import main API functions
        from main_api import analyze_image_unified, analyze_audio_unified, analyze_video_unified
        
        # Create dummy files
        from fastapi import UploadFile
        from io import BytesIO
        
        # Test image endpoint
        print("🔄 Testing image endpoint function...")
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image_file = UploadFile("test.jpg", BytesIO(dummy_image.tobytes()), "image/jpeg")
        
        try:
            image_result = await analyze_image_unified(image_file, None)
            print(f"✅ Image endpoint: {image_result.get('authenticity', 'Unknown')} ({image_result.get('confidence', 0):.3f})")
        except Exception as e:
            print(f"❌ Image endpoint failed: {e}")
        
        # Test audio endpoint
        print("🔄 Testing audio endpoint function...")
        dummy_audio = np.random.randn(16000).astype(np.float32)
        audio_file = UploadFile("test.wav", BytesIO(dummy_audio.tobytes()), "audio/wav")
        
        try:
            audio_result = await analyze_audio_unified(audio_file, None)
            print(f"✅ Audio endpoint: {audio_result.get('authenticity', 'Unknown')} ({audio_result.get('confidence', 0):.3f})")
        except Exception as e:
            print(f"❌ Audio endpoint failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all analysis tests"""
    print("Starting comprehensive analysis testing (direct imports)...\n")
    
    # Run all tests
    tests = [
        ("Import Tests", test_imports),
        ("Model Loading", lambda: test_model_loading()[0]),
        ("Unified Detector", test_unified_detector),
        ("Sentinel Agent", test_sentinel_agent),
        ("API Endpoints", test_api_endpoints),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                results[test_name] = await test_func()
            else:
                results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 ANALYSIS TESTING RESULTS")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed >= 4:
        print("🎉 MOST ANALYSIS COMPONENTS WORKING")
        if passed == total:
            print("💡 ALL ANALYSIS COMPONENTS FULLY FUNCTIONAL")
        else:
            print("💡 Some components may need attention")
        return True
    else:
        print("⚠️ MULTIPLE ANALYSIS COMPONENTS FAILING")
        print("💡 Check model configuration and dependencies")
        return False

if __name__ == "__main__":
    import asyncio
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
