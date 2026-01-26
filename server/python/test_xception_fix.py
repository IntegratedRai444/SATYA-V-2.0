#!/usr/bin/env python3
"""
Test script to verify Xception model loading fix
"""
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_xception_loading():
    """Test if Xception model loads successfully."""
    try:
        # Import the model loader
        from model_loader import XceptionLoader, ModelManager
        
        # Test Xception model loading
        xception_path = Path("models/xception/model.pth")
        
        if not xception_path.exists():
            logger.error(f"Xception model file not found at {xception_path}")
            return False
            
        logger.info("Testing Xception model loading...")
        
        # Create loader instance
        loader = XceptionLoader(xception_path)
        
        # Try to load the model
        loader.load()
        
        logger.info("✅ Xception model loaded successfully!")
        
        # Test model info
        model_info = loader.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load Xception model: {e}")
        return False

def test_model_manager():
    """Test ModelManager with all models."""
    try:
        from model_loader import ModelManager
        
        logger.info("Testing ModelManager...")
        
        # Create model manager
        manager = ModelManager("models")
        
        # Try to load all models
        manager.load_models()
        
        logger.info(f"✅ ModelManager loaded {len(manager.models)} models")
        logger.info(f"Available models: {list(manager.models.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ModelManager test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔧 Testing Xception model fix...")
    
    # Test individual Xception loading
    xception_ok = test_xception_loading()
    
    # Test ModelManager
    manager_ok = test_model_manager()
    
    if xception_ok and manager_ok:
        logger.info("🎉 All tests passed! Xception model is now functional.")
    else:
        logger.error("💥 Some tests failed. Please check the errors above.")
        sys.exit(1)
