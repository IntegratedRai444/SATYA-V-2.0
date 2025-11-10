"""
Pytest configuration and fixtures for comprehensive testing
"""
import os
import sys
import pytest
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, List, Optional
from unittest.mock import MagicMock, patch, Mock

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

# Set up test environment variables
os.environ['ENVIRONMENT'] = 'test'
os.environ['PYTHONPATH'] = project_root
os.environ['TESTING'] = 'true'

# Mock imports that cause issues during testing
sys.modules['tensorflow'] = MagicMock()
sys.modules['torch'] = MagicMock()

# Import the FastAPI app after setting up the environment
from fastapi.testclient import TestClient
from server.main import app

# ===== Core Fixtures =====

@pytest.fixture(scope="session")
def session_temp_dir():
    """Create a temporary directory that persists for the test session"""
    temp_dir = Path(tempfile.mkdtemp(prefix="satya_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def temp_dir(tmp_path):
    """Create and return a temporary directory"""
    return tmp_path

# ===== API Testing Fixtures =====

@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app"""
    with TestClient(app) as test_client:
        yield test_client

# ===== Model Testing Fixtures =====

@pytest.fixture
def sample_model_file(temp_dir):
    """Create a sample model file for testing"""
    model_file = temp_dir / "test_model.pth"
    model_file.touch()
    return model_file

@pytest.fixture
def mock_models_dir(session_temp_dir):
    """Create a mock models directory structure"""
    models_dir = session_temp_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Create test model directories
    test_models = ["efficientnet_b7", "xception", "microsoft_video_auth"]
    for model_name in test_models:
        model_dir = models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        (model_dir / "model.pth").touch()
        
        # Add a config file
        config = {
            "name": model_name,
            "version": "1.0.0",
            "input_shape": [224, 224, 3],
            "classes": ["real", "fake"]
        }
        with open(model_dir / "config.json", 'w') as f:
            json.dump(config, f)
    
    return models_dir

# ===== Test Data Fixtures =====

@pytest.fixture
def test_image(session_temp_dir) -> Path:
    """Generate a test image file"""
    from PIL import Image, ImageDraw
    
    img_path = session_temp_dir / "test_image.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Test Image", fill='white')
    img.save(img_path)
    return img_path

@pytest.fixture
def test_video(session_temp_dir) -> Path:
    """Generate a test video file"""
    import cv2
    import numpy as np
    
    video_path = session_temp_dir / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 20.0, (640, 480))
    
    for _ in range(30):  # 30 frames
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    return video_path

# ===== Authentication Fixtures =====

@pytest.fixture
def auth_headers(client) -> Dict[str, str]:
    """Get authentication headers for API requests"""
    # Register test user
    client.post("/api/v1/auth/register", json={
        "email": "test@example.com",
        "password": "testpass123",
        "username": "testuser"
    })
    
    # Login to get token
    response = client.post("/api/v1/auth/login", data={
        "username": "test@example.com",
        "password": "testpass123"
    })
    
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

# ===== Mock Services =====

@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services for testing"""
    with patch('server.services.email.EmailService') as mock_email, \
         patch('server.services.storage.StorageService') as mock_storage, \
         patch('server.services.model.ModelManager') as mock_model:
        
        # Configure email mock
        mock_email.return_value.send_email.return_value = True
        
        # Configure storage mock
        mock_storage.return_value.upload_file.return_value = "test_file_id"
        
        # Configure model mock
        mock_model.return_value.predict.return_value = {
            "is_fake": False,
            "confidence": 0.95,
            "details": {}
        }
        
        yield {"email": mock_email, "storage": mock_storage, "model": mock_model}

# ===== Performance Testing =====

@pytest.fixture
def performance_test_config():
    """Configuration for performance tests"""
    return {
        "users": 10,
        "spawn_rate": 2,
        "duration": "30s",
        "min_rps": 5,
        "max_failure_rate": 0.01,
        "p95_latency_ms": 1000
    }

# ===== Accessibility Testing =====

@pytest.fixture
def axe():
    """Create an Axe accessibility testing instance"""
    from axe_playwright_python.sync_playwright import Axe
    return Axe()

# ===== Global Setup and Teardown =====

def pytest_sessionstart(session):
    """Run before test session starts"""
    # Create test output directories
    test_output = Path("test_output")
    test_output.mkdir(exist_ok=True)
    
    # Clear previous test results
    for item in test_output.glob("*"):
        if item.is_file():
            item.unlink()
        else:
            shutil.rmtree(item)

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Setup code here runs before each test
    start_time = time.time()
    
    yield  # Test runs here
    
    # Teardown code here runs after each test
    test_duration = time.time() - start_time
    if test_duration > 1.0:  # Log slow tests
        print(f"\nTest took {test_duration:.2f}s")

# ===== Custom Markers =====

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers",
        "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers",
        "accessibility: mark test as an accessibility test"
    )
    config.addinivalue_line(
        "markers",
        "e2e: mark test as an end-to-end test"
    )
