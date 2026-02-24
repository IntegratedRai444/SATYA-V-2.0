#!/usr/bin/env python3
"""
Simple authentication test for SatyaAI
"""

import requests
import json

API_BASE_URL = "http://localhost:5001/api/v2"
CREDENTIALS = {
    "email": "Rishabhkapoor@atomicmail.io",
    "password": "Rishabhkapoor@0444"
}

def test_auth():
    print("🧪 Testing Authentication Flow")
    print("=" * 40)
    
    session = requests.Session()
    
    # Test login
    print("1. Testing login...")
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
        print(f"Cookies received: {list(cookies.keys())}")
        
        # Test a protected endpoint
        print("\n2. Testing protected endpoint...")
        response = session.get(f"{API_BASE_URL}/health")
        print(f"Health check status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Authentication working!")
            return True
        else:
            print(f"❌ Protected endpoint failed: {response.status_code}")
            return False
    else:
        print(f"❌ Login failed: {response.status_code}")
        return False

if __name__ == "__main__":
    test_auth()
