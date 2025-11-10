"""
API Endpoint Tests

Comprehensive tests for all API endpoints.
"""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
from PIL import Image
import io

# Import test data factory
from tests.test_factory import test_data

# Import the FastAPI app
from server.main import app

# Create test client
client = TestClient(app)

class TestAuthEndpoints:
    """Tests for authentication endpoints"""
    
    def test_register_user(self):
        """Test user registration"""
        user_data = test_data.create_test_user()
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": user_data["username"],
                "email": user_data["email"],
                "password": "testpass123",
                "first_name": user_data["first_name"],
                "last_name": user_data["last_name"]
            }
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["email"] == user_data["email"]
        assert "password" not in data
    
    def test_login(self):
        """Test user login"""
        # First register a user
        user_data = test_data.create_test_user()
        client.post(
            "/api/v1/auth/register",
            json={
                "username": user_data["username"],
                "email": user_data["email"],
                "password": "testpass123"
            }
        )
        
        # Test login
        response = client.post(
            "/api/v1/auth/login",
            data={
                "username": user_data["email"],
                "password": "testpass123"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"

class TestAnalysisEndpoints:
    """Tests for analysis endpoints"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        # Create a test image
        self.test_image = test_data.create_test_image()
        self.test_video = test_data.create_test_video()
        
        # Create a test user and get auth token
        user_data = test_data.create_test_user()
        response = client.post(
            "/api/v1/auth/register",
            json={
                "username": user_data["username"],
                "email": user_data["email"],
                "password": "testpass123"
            }
        )
        self.token = response.json().get("access_token")
        self.headers = {"Authorization": f"Bearer {self.token}"}
        
        yield
        
        # Cleanup
        if self.test_image.exists():
            self.test_image.unlink()
        if self.test_video.exists():
            self.test_video.unlink()
    
    def test_analyze_image(self):
        """Test image analysis endpoint"""
        with open(self.test_image, "rb") as f:
            response = client.post(
                "/api/v1/analyze/image",
                files={"file": ("test.jpg", f, "image/jpeg")},
                headers=self.headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "is_fake" in data
        assert "confidence" in data
        assert "analysis_time" in data
    
    def test_analyze_video(self):
        """Test video analysis endpoint"""
        with open(self.test_video, "rb") as f:
            response = client.post(
                "/api/v1/analyze/video",
                files={"file": ("test.mp4", f, "video/mp4")},
                headers=self.headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "is_fake" in data
        assert "confidence" in data
        assert "analysis_time" in data
        assert "frame_analysis" in data
    
    def test_get_analysis_history(self):
        """Test getting analysis history"""
        response = client.get(
            "/api/v1/analysis/history",
            headers=self.headers
        )
        
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_get_analysis_result(self):
        """Test getting a specific analysis result"""
        # First submit an analysis
        with open(self.test_image, "rb") as f:
            response = client.post(
                "/api/v1/analyze/image",
                files={"file": ("test.jpg", f, "image/jpeg")},
                headers=self.headers
            )
        
        analysis_id = response.json()["id"]
        
        # Now get the result
        response = client.get(
            f"/api/v1/analysis/{analysis_id}",
            headers=self.headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == analysis_id
        assert "is_fake" in data
        assert "confidence" in data

class TestModelEndpoints:
    """Tests for model management endpoints"""
    
    def test_get_models(self):
        """Test getting available models"""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:  # If models are returned
            assert "id" in data[0]
            assert "name" in data[0]
            assert "version" in data[0]
    
    @patch('server.main.ModelManager')
    def test_update_model(self, mock_model_manager):
        """Test updating a model"""
        # Setup mock
        mock_manager = MagicMock()
        mock_manager.update_model.return_value = {
            "status": "success",
            "model_id": "efficientnet_b7",
            "version": "1.1.0"
        }
        mock_model_manager.return_value = mock_manager
        
        response = client.post(
            "/api/v1/models/update",
            json={"model_id": "efficientnet_b7"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "model_id" in data
        assert "version" in data

class TestHealthEndpoints:
    """Tests for health and monitoring endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "version" in data
    
    def test_metrics(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        # Check if the response looks like Prometheus metrics
        assert "# HELP" in response.text
        assert "# TYPE" in response.text
