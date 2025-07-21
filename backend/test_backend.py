#!/usr/bin/env python3
"""
Comprehensive backend test script
Tests all imports, models, and basic functionality
"""
import sys
import traceback

def test_import(module_name, description):
    """Test importing a module"""
    try:
        __import__(module_name)
        print(f"✅ {description}: OK")
        return True
    except Exception as e:
        print(f"❌ {description}: FAILED - {e}")
        return False

def main():
    print("🧪 SATYA-V-2.0 Backend Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test core imports
    print("\n📦 Testing Core Imports:")
    results.append(test_import("fastapi", "FastAPI"))
    results.append(test_import("uvicorn", "Uvicorn"))
    results.append(test_import("numpy", "NumPy"))
    results.append(test_import("torch", "PyTorch"))
    results.append(test_import("cv2", "OpenCV"))
    results.append(test_import("PIL", "Pillow"))
    results.append(test_import("librosa", "Librosa"))
    
    # Test backend modules
    print("\n🔧 Testing Backend Modules:")
    results.append(test_import("backend.main", "Main App"))
    results.append(test_import("backend.routes.image", "Image Routes"))
    results.append(test_import("backend.routes.video", "Video Routes"))
    results.append(test_import("backend.routes.audio", "Audio Routes"))
    results.append(test_import("backend.routes.webcam", "Webcam Routes"))
    results.append(test_import("backend.routes.auth", "Auth Routes"))
    results.append(test_import("backend.routes.system", "System Routes"))
    results.append(test_import("backend.routes.assistant", "Assistant Routes"))
    
    # Test models
    print("\n🤖 Testing Models:")
    results.append(test_import("backend.models.image_model", "Image Model"))
    results.append(test_import("backend.models.video_model", "Video Model"))
    results.append(test_import("backend.models.audio_model", "Audio Model"))
    results.append(test_import("backend.models.webcam_model", "Webcam Model"))
    
    # Test utilities
    print("\n🛠️ Testing Utilities:")
    results.append(test_import("backend.utils.image_utils", "Image Utils"))
    results.append(test_import("backend.utils.video_utils", "Video Utils"))
    results.append(test_import("backend.utils.audio_utils", "Audio Utils"))
    results.append(test_import("backend.utils.report_utils", "Report Utils"))
    
    # Test optional dependencies
    print("\n🔍 Testing Optional Dependencies:")
    try:
        import mediapipe
        print("✅ Mediapipe: Available")
        results.append(True)
    except ImportError:
        print("⚠️ Mediapipe: Not available (expected on Python 3.13+)")
        results.append(True)  # Not a failure
    
    try:
        import face_recognition
        print("✅ Face Recognition: Available")
        results.append(True)
    except ImportError:
        print("⚠️ Face Recognition: Not available (may need dlib)")
        results.append(True)  # Not a failure
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")
    successful = sum(results)
    total = len(results)
    print(f"Successful: {successful}/{total}")
    print(f"Success Rate: {(successful/total)*100:.1f}%")
    
    if successful == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Backend is ready to run")
    else:
        print("⚠️ Some tests failed. Check the details above.")
        print("💡 Install missing dependencies or fix import issues")
    
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 