"""
Unit tests for the server API
"""
import pytest
import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient
from fastapi import status
from unittest.mock import patch, MagicMock
import json

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set up environment variables needed for the app
os.environ['ENVIRONMENT'] = 'test'

# Import your FastAPI app
from server.main import app

@pytest.fixture
def client():
    return TestClient(app)

class TestAPI:
    """Test cases for the API endpoints"""
    
    def test_health_check(self, client):
        """Test the health check endpoint"""
        with patch('server.main.datetime') as mock_datetime:
            # Mock the current time
            mock_datetime.utcnow.return_value.isoformat.return_value = "2023-01-01T00:00:00"
            
            response = client.get("/health")
            
            assert response.status_code == status.HTTP_200_OK
            assert response.json() == {
                "status": "ok",
                "timestamp": "2023-01-01T00:00:00",
                "version": "2.0.0"
            }
    
    def test_root_endpoint(self, client):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        assert "SatyaAI" in response.json()["message"]
    
    @patch('server.main.ModelManager')
    def test_analyze_image(self, mock_model_manager, client):
        """Test the image analysis endpoint"""
        # Setup mock
        mock_manager = MagicMock()
        # Mock the return value for analyze_image
        mock_manager.analyze_image.return_value = {
            "is_fake": False,
            "confidence": 0.95,
            "details": {}
        }
        mock_model_manager.return_value = mock_manager
        
        # Create a test file
        test_file = {
            "file": ("test.jpg", b"test image data", "image/jpeg")
        }
        
        # Make the request
        response = client.post("/api/analyze/image", files=test_file)
        
        # Check the response
        assert response.status_code == 200
        assert "analysis_id" in response.json()
        assert "status" in response.json()
        assert response.json()["status"] == "completed"
    
    def test_invalid_file_type(self):
        """Test with an invalid file type"""
        # Create a test file with invalid type
        test_file = {
            "file": ("test.txt", b"test data", "text/plain")
        }
        
        # Make the request
        response = client.post("/api/analyze/image", files=test_file)
        
        # Check the response
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

class TestAuthentication:
    """Test cases for authentication endpoints"""
    
    @patch('server.services.auth_service.authenticate_user')
    def test_login_success(self, mock_auth):
        """Test successful login"""
        # Mock the authentication
        mock_auth.return_value = {"access_token": "test_token", "token_type": "bearer"}
        
        # Test data
        login_data = {
            "username": "testuser",
            "password": "testpass123"
        }
        
        # Make the request
        response = client.post("/api/auth/login", json=login_data)
        
        # Check the response
        assert response.status_code == 200
        assert "access_token" in response.json()
        assert response.json()["token_type"] == "bearer"
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        # Test data with invalid credentials
        login_data = {
            "username": "wronguser",
            "password": "wrongpass"
        }
        
        # Make the request
        response = client.post("/api/auth/login", json=login_data)
        
        # Check the response
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]
