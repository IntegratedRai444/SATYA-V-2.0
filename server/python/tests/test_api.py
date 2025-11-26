"""
API Integration Tests
Tests for the main API endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from main_api import app

client = TestClient(app)

def test_read_main():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_system_info():
    """Test system info endpoint"""
    response = client.get("/api/system/info")
    assert response.status_code == 200
    assert "platform" in response.json()

def test_auth_flow():
    """Test authentication flow"""
    # 1. Register
    register_data = {
        "username": "test_api_user",
        "email": "test_api@example.com",
        "password": "password123",
        "full_name": "Test API User"
    }
    # Note: This might fail if user already exists, so we handle 400
    response = client.post("/api/auth/register", json=register_data)
    assert response.status_code in [200, 400]
    
    # 2. Login
    login_data = {
        "username": "test_api_user",
        "password": "password123"
    }
    response = client.post("/api/auth/login", data=login_data)
    assert response.status_code == 200
    tokens = response.json()
    assert "access_token" in tokens
    
    # 3. Get Me
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = client.get("/api/auth/me", headers=headers)
    assert response.status_code == 200
    assert response.json()["username"] == "test_api_user"

def test_dashboard_stats():
    """Test dashboard stats"""
    response = client.get("/api/dashboard/stats")
    assert response.status_code == 200
    assert "total_scans" in response.json()
