#!/usr/bin/env python3
"""
Comprehensive endpoint test for SatyaAI Python server
"""
import pytest
import requests
import json
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Example test for root endpoint
# def test_root():
#     response = client.get("/")
#     assert response.status_code == 200

def test_endpoint(url, method='GET', data=None, description=""):
    """Test an endpoint and return detailed results"""
    try:
        print(f"\n{'='*60}")
        print(f"Testing: {description}")
        print(f"URL: {method} {url}")
        print(f"{'='*60}")
        
        if method == 'GET':
            response = requests.get(url, timeout=10)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'Not specified')}")
        print(f"CORS Header: {response.headers.get('Access-Control-Allow-Origin', 'Not found')}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Response (first 200 chars): {json.dumps(result, indent=2)[:200]}...")
                print("✅ SUCCESS")
                return True
            except json.JSONDecodeError:
                print(f"Response (text): {response.text[:200]}...")
                print("✅ SUCCESS (non-JSON response)")
                return True
        else:
            print(f"Response: {response.text}")
            print("❌ FAILED")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        print("❌ FAILED")
        return False

def main():
    base_url = "http://localhost:5002"
    
    print("SatyaAI Python Server - Comprehensive Endpoint Test")
    print("=" * 80)
    
    # Test results tracking
    results = []
    
    # 1. Health and Status endpoints
    results.append(test_endpoint(f"{base_url}/health", description="Health Check"))
    results.append(test_endpoint(f"{base_url}/status", description="Server Status"))
    
    # 2. Authentication endpoints
    login_data = {"username": "test", "password": "test"}
    results.append(test_endpoint(f"{base_url}/api/auth/login", 'POST', login_data, "User Login"))
    
    logout_data = {"token": "test-token"}
    results.append(test_endpoint(f"{base_url}/api/auth/logout", 'POST', logout_data, "User Logout"))
    
    validate_data = {"token": "test-token"}
    results.append(test_endpoint(f"{base_url}/api/auth/validate", 'POST', validate_data, "Session Validation"))
    
    # 3. Scan endpoints
    results.append(test_endpoint(f"{base_url}/api/scans/recent", description="Recent Scans"))
    results.append(test_endpoint(f"{base_url}/api/scans", description="All Scans"))
    results.append(test_endpoint(f"{base_url}/api/scans/1", description="Specific Scan (ID: 1)"))
    results.append(test_endpoint(f"{base_url}/api/scans/1/report", description="Scan Report (ID: 1)"))
    
    # 4. Analysis endpoints
    webcam_data = {"imageData": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"}
    results.append(test_endpoint(f"{base_url}/api/analyze/webcam", 'POST', webcam_data, "Webcam Analysis"))
    
    # 5. Settings endpoints
    preferences_data = {"theme": "dark", "language": "english"}
    results.append(test_endpoint(f"{base_url}/api/settings/preferences", 'POST', preferences_data, "Save Preferences"))
    
    profile_data = {"email": "test@example.com", "name": "Test User"}
    results.append(test_endpoint(f"{base_url}/api/settings/profile", 'POST', profile_data, "Save Profile"))
    
    # 6. Test non-existent endpoint
    results.append(test_endpoint(f"{base_url}/api/nonexistent", description="Non-existent Endpoint (should return 404)"))
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    successful = sum(results)
    total = len(results)
    print(f"Successful: {successful}/{total}")
    print(f"Success Rate: {(successful/total)*100:.1f}%")
    
    if successful == total:
        print("🎉 ALL ENDPOINTS WORKING CORRECTLY!")
    else:
        print("⚠️  Some endpoints failed. Check the details above.")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 