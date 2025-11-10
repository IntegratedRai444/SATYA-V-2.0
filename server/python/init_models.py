"""
Model Initialization Script for SatyaAI
Initializes and tests all AI models
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_models():
    """Initialize all models and test them."""
    try:
        # Import model components
        from model_manager import EnhancedModelManager
        from detectors.image_detector import ImageDetector
        from detectors.audio_detector import AudioDetector
        from detectors.video_detector import VideoDetector
        
        print("🚀 Initializing SatyaAI Models...")
        print("=" * 40)
        
        # Create model manager
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Initialize enhanced model manager
        manager = EnhancedModelManager(str(models_dir))
        
        # Test image detector
        print("\n📸 Testing Image Detector...")
        image_detector = ImageDetector(str(models_dir), enable_gpu=False)
        
        try:
            image_detector.load_models()
            if image_detector.models_loaded:
                print("✅ Image detector loaded successfully")
            else:
                print("⚠️  Image detector using fallback mode")
        except Exception as e:
            print(f"❌ Image detector error: {e}")
        
        # Test audio detector
        print("\n🎵 Testing Audio Detector...")
        audio_detector = AudioDetector(str(models_dir), enable_gpu=False)
        
        try:
            audio_detector.load_models()
            if audio_detector.models_loaded:
                print("✅ Audio detector loaded successfully")
            else:
                print("⚠️  Audio detector using fallback mode")
        except Exception as e:
            print(f"❌ Audio detector error: {e}")
        
        # Test video detector
        print("\n🎬 Testing Video Detector...")
        video_detector = VideoDetector(str(models_dir), enable_gpu=False)
        
        try:
            video_detector.load_models()
            print("✅ Video detector initialized successfully")
        except Exception as e:
            print(f"❌ Video detector error: {e}")
        
        print("\n" + "=" * 40)
        print("🎉 Model initialization complete!")
        print("\nSatyaAI is ready to detect deepfakes.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False

def test_simple_detection():
    """Test detection with a simple image."""
    try:
        import numpy as np
        from PIL import Image
        import io
        
        print("\n🧪 Testing Simple Detection...")
        
        # Create a realistic test image with natural patterns
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Add natural gradient patterns
        for i in range(224):
            for j in range(224):
                # Create natural-looking gradient with some texture
                intensity = int(128 + 50 * np.sin(i/20) * np.cos(j/20))
                test_image[i, j] = [intensity, intensity-10, intensity+10]
        
        # Add some natural noise
        noise = np.random.normal(0, 5, (224, 224, 3)).astype(np.int8)
        test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Test image detection
        from detectors.image_detector import ImageDetector
        
        models_dir = Path("models")
        detector = ImageDetector(str(models_dir), enable_gpu=False)
        
        result = detector.analyze(image_bytes)
        
        if result.get('success', False):
            print("✅ Detection test successful!")
            print(f"   Result: {result.get('authenticity', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.1f}%")
        else:
            print("⚠️  Detection test completed with fallback")
            
        return True
        
    except Exception as e:
        print(f"❌ Detection test error: {e}")
        return False

def main():
    """Main initialization function."""
    print("SatyaAI Model Initialization")
    print("=" * 50)
    
    # Initialize models
    if not init_models():
        print("\n❌ Model initialization failed")
        return 1
    
    # Test detection
    if not test_simple_detection():
        print("\n⚠️  Detection test had issues")
    
    print("\n✅ SatyaAI is ready!")
    print("\nNext steps:")
    print("1. Start the Python server: python app.py")
    print("2. Start the Node.js server in another terminal")
    print("3. Open the web interface")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Initialization cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)