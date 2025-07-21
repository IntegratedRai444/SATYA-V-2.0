#!/usr/bin/env python3
"""
SatyaAI System Test Script
Tests all components to ensure they're working correctly
"""

import requests
import time
import json
import sys
from pathlib import Path

def test_backend_health():
    """Test if the main backend server is running"""
    try:
        response = requests.get('http://localhost:3000/health', timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running")
            return True
        else:
            print(f"❌ Backend server returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend server is not running: {e}")
        return False

def test_ai_system_health():
    """Test if the AI system is running"""
    try:
        response = requests.get('http://localhost:5002/health', timeout=5)
        if response.status_code == 200:
            print("✅ AI system is running")
            return True
        else:
            print(f"❌ AI system returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ AI system is not running: {e}")
        return False

def test_frontend():
    """Test if the frontend is accessible"""
    try:
        response = requests.get('http://localhost:5173', timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            return True
        else:
            print(f"❌ Frontend returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend is not accessible: {e}")
        return False

def test_image_analysis():
    """Test image analysis endpoint"""
    try:
        # Create a simple test image (1x1 pixel)
        test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\x00\x00\x00\x00IEND\xaeB`\x82'
        
        files = {'image': ('test.png', test_image_data, 'image/png')}
        data = {
            'analysis_type': 'quick',
            'confidence_threshold': '80',
            'enable_advanced_models': 'true'
        }
        
        response = requests.post('http://localhost:3000/api/ai/analyze/image', 
                               files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Image analysis working - Result: {result.get('authenticity', 'Unknown')}")
            return True
        else:
            print(f"❌ Image analysis failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Image analysis test failed: {e}")
        return False

def test_webcam_analysis():
    """Test webcam analysis endpoint"""
    try:
        # Create a simple test image data URL
        test_image_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        
        data = {
            'imageData': test_image_data,
            'analysis_type': 'quick',
            'confidence_threshold': '80',
            'enable_advanced_models': 'true'
        }
        
        response = requests.post('http://localhost:3000/api/ai/analyze/image', 
                               data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Webcam analysis working - Result: {result.get('authenticity', 'Unknown')}")
            return True
        else:
            print(f"❌ Webcam analysis failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Webcam analysis test failed: {e}")
        return False

def test_history_endpoint():
    """Test history endpoint"""
    try:
        response = requests.get('http://localhost:3000/api/scans', timeout=10)
        if response.status_code == 200:
            print("✅ History endpoint working")
            return True
        else:
            print(f"❌ History endpoint failed with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ History endpoint test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing SatyaAI Deepfake Detection System")
    print("=" * 50)
    
    tests = [
        ("Backend Health", test_backend_health),
        ("AI System Health", test_ai_system_health),
        ("Frontend Access", test_frontend),
        ("Image Analysis", test_image_analysis),
        ("Webcam Analysis", test_webcam_analysis),
        ("History Endpoint", test_history_endpoint),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        print("\n🌐 Access your application at:")
        print("   Frontend: http://localhost:5173")
        print("   Backend:  http://localhost:3000")
        print("   AI System: http://localhost:5002")
        return 0
    else:
        print("❌ Some tests failed. Please check the server logs.")
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure all servers are running")
        print("   2. Check if ports 3000, 5002, and 5173 are available")
        print("   3. Verify Python and Node.js dependencies are installed")
        print("   4. Check the README_RUNNING.md for setup instructions")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 