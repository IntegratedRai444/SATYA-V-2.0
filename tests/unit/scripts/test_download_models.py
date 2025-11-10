"""
Unit tests for download_models.py
"""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil
import sys
import os

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the module to test
from scripts import download_models

# Test data
TEST_MODEL_URLS = {
    "test_model": {
        "url": "https://example.com/test_model.pth",
        "dest": "test_model/model.pth"
    }
}

class TestDownloadModels:
    """Test cases for download_models.py"""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, tmp_path):
        """Setup and teardown for each test"""
        # Save original values
        self.original_base_dir = download_models.BASE_DIR
        self.original_model_urls = download_models.MODEL_URLS
        
        # Set up test directory
        self.test_dir = tmp_path / "test_models"
        self.test_dir.mkdir()
        download_models.BASE_DIR = self.test_dir
        download_models.MODEL_URLS = TEST_MODEL_URLS
        
        yield  # Test runs here
        
        # Cleanup
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        
        # Restore original values
        download_models.BASE_DIR = self.original_base_dir
        download_models.MODEL_URLS = self.original_model_urls
    
    def test_setup_directories(self):
        """Test that setup_directories creates the correct directory structure"""
        # Act
        download_models.setup_directories()
        
        # Assert
        for model_name in TEST_MODEL_URLS:
            model_dir = download_models.BASE_DIR / model_name
            assert model_dir.exists()
            assert model_dir.is_dir()
    
    @patch('download_models.gdown.download')
    def test_download_model_success(self, mock_download):
        """Test successful model download"""
        # Arrange
        url = "https://example.com/test_model.pth"
        dest = self.test_dir / "test_model.pth"
        mock_download.return_value = str(dest)
        
        # Act
        download_models.download_model(url, dest)
        
        # Assert
        mock_download.assert_called_once_with(url, str(dest), quiet=False)
    
    @patch('download_models.gdown.download')
    def test_download_model_failure(self, mock_download, caplog):
        """Test model download failure"""
        # Arrange
        url = "https://example.com/test_model.pth"
        dest = self.test_dir / "test_model.pth"
        mock_download.side_effect = Exception("Download failed")
        
        # Act
        download_models.download_model(url, dest)
        
        # Assert
        assert "Failed to download" in caplog.text
    
    @patch('download_models.gdown.download')
    def test_main_function(self, mock_download, caplog):
        """Test the main function"""
        # Arrange
        mock_download.return_value = str(self.test_dir / "test_model.pth")
        
        # Act
        with patch('download_models.verify_models') as mock_verify:
            download_models.main()
        
        # Assert
        assert "Setting up model directories" in caplog.text
        assert "Downloading models" in caplog.text
        mock_verify.assert_called_once()
    
    def test_verify_models_missing(self, caplog):
        """Test verify_models with missing models"""
        # Act
        with pytest.raises(SystemExit):
            download_models.verify_models()
        
        # Assert
        assert "Missing model file" in caplog.text
    
    def test_verify_models_present(self, caplog):
        """Test verify_models with all models present"""
        # Arrange
        for model_name in TEST_MODEL_URLS:
            model_path = download_models.BASE_DIR / TEST_MODEL_URLS[model_name]["dest"]
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.touch()
        
        # Act
        download_models.verify_models()
        
        # Assert - No exception should be raised
        assert "All required models are present" in caplog.text
