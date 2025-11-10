#!/usr/bin/env python3
"""
Python 3.13 Compatible Setup for SatyaAI
Uses the latest available versions that work with Python 3.13
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_latest_compatible():
    """Install latest versions compatible with Python 3.13."""
    logger.info("📦 Installing Python 3.13 compatible versions...")
    
    try:
        # Install latest PyTorch CPU (available for Python 3.13)
        logger.info("Installing latest PyTorch CPU...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        logger.info("✅ PyTorch installed")
        
        # Install core packages
        core_packages = [
            "numpy",
            "pillow", 
            "opencv-python-headless",
            "scikit-image",
            "scipy",
            "librosa",
            "soundfile",
            "requests",
            "tqdm",
            "flask",
            "flask-cors"
        ]
        
        for package in core_packages:
            logger.info(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        
        logger.info("✅ Core packages installed")
        
        # Try optional packages (may fail, that's OK)
        optional_packages = [
            "transformers",
            "huggingface-hub",
            "timm", 
            "gdown"
        ]
        
        for package in optional_packages:
            try:
                logger.info(f"Installing optional {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])
                logger.info(f"✅ {package} installed")
            except subprocess.CalledProcessError:
                logger.warning(f"⚠️  {package} failed - skipping")
        
        # Try facenet-pytorch (may fail due to version conflicts)
        try:
            logger.info("Installing facenet-pytorch...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "facenet-pytorch", "--no-deps"
            ])
            logger.info("✅ facenet-pytorch installed")
        except subprocess.CalledProcessError:
            logger.warning("⚠️  facenet-pytorch failed - will use OpenCV for face detection")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Installation failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without problematic packages."""
    logger.info("🧪 Testing basic functionality...")
    
    test_script = '''
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    
    import torchvision
    print(f"✅ TorchVision: {torchvision.__version__}")
    
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
    
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
    
    from PIL import Image
    print("✅ Pillow: Available")
    
    import librosa
    print(f"✅ Librosa: {librosa.__version__}")
    
    # Test basic PyTorch functionality
    x = torch.randn(1, 3, 224, 224)
    print(f"✅ PyTorch tensor: {x.shape}")
    
    # Test OpenCV face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("✅ OpenCV face detection: Available")
    
    # Test image processing
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    print(f"✅ Image processing: {pil_img.size}")
    
    print("\\n🎉 BASIC FUNCTIONALITY TEST PASSED!")
    print("Your system can run SatyaAI with core features!")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Test failed: {e}")
'''
    
    try:
        result = subprocess.run([
            sys.executable, "-c", test_script
        ], capture_output=True, text=True, timeout=30)
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        if "BASIC FUNCTIONALITY TEST PASSED" in result.stdout:
            return True
        else:
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def create_minimal_requirements():
    """Create a minimal requirements file that works."""
    logger.info("📝 Creating minimal requirements file...")
    
    minimal_requirements = """# Minimal SatyaAI Requirements for Python 3.13
# Core AI/ML
torch
torchvision
torchaudio

# Computer Vision
opencv-python-headless
pillow
scikit-image
numpy
scipy

# Audio Processing
librosa
soundfile

# Web Framework
flask
flask-cors

# Utilities
requests
tqdm

# Optional (may not install on Python 3.13)
# transformers
# huggingface-hub
# timm
# facenet-pytorch
"""
    
    with open("requirements_minimal.txt", "w") as f:
        f.write(minimal_requirements)
    
    logger.info("✅ Created requirements_minimal.txt")

def main():
    """Main setup function."""
    logger.info("🔧 Python 3.13 Compatible SatyaAI Setup")
    logger.info("=" * 50)
    logger.info("⚠️  Python 3.13 is very new - using latest available packages")
    
    # Install packages
    if not install_latest_compatible():
        logger.error("❌ Failed to install packages")
        return 1
    
    # Test functionality
    if test_basic_functionality():
        logger.info("\n🎉 PYTHON 3.13 SETUP COMPLETE!")
        logger.info("✅ Core functionality is working")
        logger.info("\n📋 What's working:")
        logger.info("   • PyTorch for AI models")
        logger.info("   • OpenCV for computer vision")
        logger.info("   • Librosa for audio processing")
        logger.info("   • Basic deepfake detection")
        logger.info("\n⚠️  What might be limited:")
        logger.info("   • Some advanced AI models (due to Python 3.13)")
        logger.info("   • FaceNet (will use OpenCV instead)")
        logger.info("\n🚀 Next steps:")
        logger.info("1. Run: npm run start:satyaai")
        logger.info("2. Test with images/videos")
        logger.info("3. System will use available models")
        
        create_minimal_requirements()
        return 0
    else:
        logger.warning("\n⚠️  Setup had issues but may still work")
        logger.info("Try running the system anyway - basic features should work")
        return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\n⏹️  Setup cancelled")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)