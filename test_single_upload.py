#!/usr/bin/env python3
"""
Single upload test to debug authentication
"""

import requests
import json

API_BASE_URL = "http://localhost:5001/api/v2"
CREDENTIALS = {
    "email": "Rishabhkapoor@atomicmail.io",
    "password": "Rishabhkapoor@0444"
}

def test_single_upload():
    print("🧪 Single Upload Test")
    print("=" * 40)
    
    session = requests.Session()
    
    # Login
    print("1. Logging in...")
    response = session.post(
        f"{API_BASE_URL}/auth/login",
        json=CREDENTIALS,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Login status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Login success: {data.get('success')}")
        cookies = session.cookies.get_dict()
        print(f"Cookies: {list(cookies.keys())}")
        
        # Test upload
        print("\n2. Testing upload...")
        test_image = r"E:\SATYA-V-2.0\real_and_fake_face\training_real\real_00001.jpg"
        
        with open(test_image, 'rb') as f:
            files = {'file': (test_image.split('\\')[-1], f, 'image/jpeg')}
            print(f"Uploading: {test_image.split('\\')[-1]}")
            print(f"Session cookies: {dict(session.cookies)}")
            
            response = session.post(
                f"{API_BASE_URL}/analysis/image",
                files=files
            )
            
            print(f"Upload status: {response.status_code}")
            print(f"Upload response: {response.text[:200]}...")
            
            if response.status_code == 202:
                print("✅ Upload successful!")
                return True
            else:
                print(f"❌ Upload failed: {response.status_code}")
                return False
    else:
        print(f"❌ Login failed: {response.status_code}")
        return False

if __name__ == "__main__":
    test_single_upload()
