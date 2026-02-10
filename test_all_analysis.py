#!/usr/bin/env python3
"""
Test all analysis endpoints in SATYA AI system
"""

import sys
import os
import requests
import json
import time
from pathlib import Path

# Add SATYA path
sys.path.insert(0, str(Path(__file__).parent / "server" / "python"))

print("🧪 TESTING ALL ANALYSIS ENDPOINTS")
print("=" * 60)

def test_server_health():
    """Test if server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        return False

def test_image_analysis():
    """Test image analysis endpoint"""
    print("\n🖼️ TESTING IMAGE ANALYSIS")
    print("-" * 40)
    
    try:
        # Create a dummy image file
        import numpy as np
        
        # Generate a simple test image (224x224 RGB)
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Save as temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            from PIL import Image
            img = Image.fromarray(test_image)
            img.save(tmp.name)
            
            # Test the endpoint
            with open(tmp.name, 'rb') as f:
                files = {'file': ('test.jpg', f, 'image/jpeg')}
                response = requests.post(
                    "http://localhost:8000/api/v2/analysis/unified/image",
                    files=files,
                    timeout=30
                )
        
        # Clean up
        os.unlink(tmp.name)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Image analysis successful")
            print(f"  📊 Authenticity: {result.get('authenticity', 'Unknown')}")
            print(f"  📊 Confidence: {result.get('confidence', 0):.3f}")
            print(f"  📊 Model: {result.get('model_used', 'Unknown')}")
            return True
        else:
            print(f"❌ Image analysis failed: {response.status_code}")
            print(f"  📄 Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Image analysis error: {e}")
        return False

def test_video_analysis():
    """Test video analysis endpoint"""
    print("\n🎥 TESTING VIDEO ANALYSIS")
    print("-" * 40)
    
    try:
        # Create a dummy video file (simplified)
        import tempfile
        
        # Create a simple test video file (just dummy data)
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            # Write some dummy video data
            tmp.write(b'FAKE_VIDEO_DATA_FOR_TESTING')
            tmp.flush()
            
            # Test the endpoint
            with open(tmp.name, 'rb') as f:
                files = {'file': ('test.mp4', f, 'video/mp4')}
                response = requests.post(
                    "http://localhost:8000/api/v2/analysis/unified/video",
                    files=files,
                    timeout=30
                )
        
        # Clean up
        os.unlink(tmp.name)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Video analysis successful")
            print(f"  📊 Authenticity: {result.get('authenticity', 'Unknown')}")
            print(f"  📊 Confidence: {result.get('confidence', 0):.3f}")
            print(f"  📊 Model: {result.get('model_used', 'Unknown')}")
            return True
        else:
            print(f"❌ Video analysis failed: {response.status_code}")
            print(f"  📄 Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Video analysis error: {e}")
        return False

def test_audio_analysis():
    """Test audio analysis endpoint"""
    print("\n🎵 TESTING AUDIO ANALYSIS")
    print("-" * 40)
    
    try:
        # Create a dummy audio file
        import tempfile
        import numpy as np
        
        # Generate a simple test audio (1 second of random audio)
        sample_rate = 16000
        duration = 1.0
        test_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        # Save as temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import wave
            
            with wave.open(tmp.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((test_audio * 32767).astype(np.int16).tobytes())
            
            # Test the endpoint
            with open(tmp.name, 'rb') as f:
                files = {'file': ('test.wav', f, 'audio/wav')}
                response = requests.post(
                    "http://localhost:8000/api/v2/analysis/unified/audio",
                    files=files,
                    timeout=30
                )
        
        # Clean up
        os.unlink(tmp.name)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Audio analysis successful")
            print(f"  📊 Authenticity: {result.get('authenticity', 'Unknown')}")
            print(f"  📊 Confidence: {result.get('confidence', 0):.3f}")
            print(f"  📊 Model: {result.get('model_used', 'Unknown')}")
            return True
        else:
            print(f"❌ Audio analysis failed: {response.status_code}")
            print(f"  📄 Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Audio analysis error: {e}")
        return False

def test_multimodal_analysis():
    """Test multimodal analysis endpoint"""
    print("\n🔄 TESTING MULTIMODAL ANALYSIS")
    print("-" * 40)
    
    try:
        # Create a dummy image for multimodal test
        import numpy as np
        import tempfile
        from PIL import Image
        
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img = Image.fromarray(test_image)
            img.save(tmp.name)
            
            # Test the multimodal endpoint
            with open(tmp.name, 'rb') as f:
                files = {'file': ('test.jpg', f, 'image/jpeg')}
                data = {
                    'analysis_type': 'multimodal',
                    'include_forensic': 'true'
                }
                response = requests.post(
                    "http://localhost:8000/api/v2/analysis/multimodal",
                    files=files,
                    data=data,
                    timeout=30
                )
        
        # Clean up
        os.unlink(tmp.name)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Multimodal analysis successful")
            print(f"  📊 Authenticity: {result.get('authenticity', 'Unknown')}")
            print(f"  📊 Confidence: {result.get('confidence', 0):.3f}")
            print(f"  📊 Modalities: {result.get('modalities_used', [])}")
            return True
        else:
            print(f"❌ Multimodal analysis failed: {response.status_code}")
            print(f"  📄 Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Multimodal analysis error: {e}")
        return False

def test_analysis_info():
    """Test analysis info endpoint"""
    print("\n📋 TESTING ANALYSIS INFO")
    print("-" * 40)
    
    try:
        response = requests.get("http://localhost:8000/analyze/info", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis info successful")
            print(f"  📊 Endpoint: {result.get('endpoint', 'Unknown')}")
            print(f"  📊 Method: {result.get('method', 'Unknown')}")
            print(f"  📊 Supported formats: {len(result.get('supported_formats', []))}")
            return True
        else:
            print(f"❌ Analysis info failed: {response.status_code}")
            print(f"  📄 Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Analysis info error: {e}")
        return False

def test_model_status():
    """Test model status endpoint"""
    print("\n🤖 TESTING MODEL STATUS")
    print("-" * 40)
    
    try:
        response = requests.get("http://localhost:8000/api/v2/models/status", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Model status successful")
            print(f"  📊 Models loaded: {len(result.get('models', {}))}")
            for model_name, model_info in result.get('models', {}).items():
                status = model_info.get('status', 'Unknown')
                print(f"    • {model_name}: {status}")
            return True
        else:
            print(f"❌ Model status failed: {response.status_code}")
            print(f"  📄 Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Model status error: {e}")
        return False

def main():
    """Run all analysis tests"""
    print("Starting comprehensive analysis endpoint testing...\n")
    
    # Test server health first
    if not test_server_health():
        print("\n❌ SERVER NOT RUNNING - START THE SERVER FIRST")
        print("💡 Run: cd server/python && python main_api.py")
        return False
    
    # Run all tests
    tests = [
        ("Model Status", test_model_status),
        ("Analysis Info", test_analysis_info),
        ("Image Analysis", test_image_analysis),
        ("Video Analysis", test_video_analysis),
        ("Audio Analysis", test_audio_analysis),
        ("Multimodal Analysis", test_multimodal_analysis),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
        time.sleep(1)  # Brief pause between tests
    
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
    
    if passed >= 5:
        print("🎉 MOST ANALYSIS ENDPOINTS WORKING")
        if passed == total:
            print("💡 ALL ANALYSIS ENDPOINTS FULLY FUNCTIONAL")
        else:
            print("💡 Some endpoints may need attention")
        return True
    else:
        print("⚠️ MULTIPLE ANALYSIS ENDPOINTS FAILING")
        print("💡 Check server logs and model configuration")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
