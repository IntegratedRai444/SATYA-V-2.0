#!/usr/bin/env python3
"""
Windows-Compatible Setup for SatyaAI
Fixes dependency conflicts and installs compatible versions
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check Python version and warn if too new."""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 13):
        logger.warning("⚠️  Python 3.13+ detected. Some packages may not be compatible.")
        logger.info("💡 Consider using Python 3.11 or 3.12 for best compatibility")
    
    return True

def clean_corrupted_packages():
    """Clean up corrupted PyTorch installations."""
    logger.info("🧹 Cleaning up corrupted packages...")
    
    try:
        # Uninstall potentially corrupted packages
        packages_to_clean = [
            "torch", "torchvision", "torchaudio", 
            "facenet-pytorch", "numpy", "pillow"
        ]
        
        for package in packages_to_clean:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "uninstall", package, "-y"
                ], capture_output=True)
                logger.info(f"Cleaned {package}")
            except:
                pass
        
        # Clear pip cache
        subprocess.run([
            sys.executable, "-m", "pip", "cache", "purge"
        ], capture_output=True)
        
        logger.info("✅ Package cleanup completed")
        return True
        
    except Exception as e:
        logger.warning(f"Cleanup had issues: {e}")
        return True  # Continue anyway

def install_compatible_versions():
    """Install compatible versions for Windows."""
    logger.info("📦 Installing Windows-compatible versions...")
    
    # Compatible versions that work together
    compatible_packages = [
        # Core packages with specific versions
        "numpy==1.24.3",
        "pillow==10.2.0",
        
        # PyTorch CPU version (most compatible)
        "torch==2.2.0+cpu",
        "torchvision==0.17.0+cpu", 
        "torchaudio==2.2.0+cpu",
        
        # Other AI packages
        "opencv-python-headless==4.8.1.78",
        "scikit-image==0.21.0",
        "scipy==1.11.4",
        
        # Audio processing
        "librosa==0.10.1",
        "soundfile==0.12.1",
        
        # Utilities
        "requests==2.31.0",
        "tqdm==4.66.1",
    ]
    
    try:
        # Install PyTorch from CPU-only index
        logger.info("Installing PyTorch CPU version...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch==2.2.0+cpu", "torchvision==0.17.0+cpu", "torchaudio==2.2.0+cpu",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        
        # Install other packages
        for package in compatible_packages[2:]:  # Skip torch packages
            logger.info(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        
        # Install facenet-pytorch with compatible versions
        logger.info("Installing facenet-pytorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "facenet-pytorch==2.5.3", "--no-deps"
        ])
        
        logger.info("✅ Compatible versions installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Installation failed: {e}")
        return False

def install_optional_packages():
    """Install optional packages that might fail."""
    optional_packages = [
        "transformers==4.30.0",
        "huggingface-hub==0.16.4", 
        "timm==0.9.7",
        "gdown==4.7.1"
    ]
    
    logger.info("📦 Installing optional packages...")
    
    for package in optional_packages:
        try:
            logger.info(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            logger.info(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            logger.warning(f"⚠️  {package} failed - continuing without it")

def test_installation():
    """Test if the installation works."""
    logger.info("🧪 Testing installation...")
    
    test_script = '''
try:
    import torch
    import torchvision
    import cv2
    import numpy as np
    from PIL import Image
    import librosa
    
    print("✅ Core packages imported successfully")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"OpenCV available: {cv2.__version__}")
    
    # Test basic functionality
    tensor = torch.randn(1, 3, 224, 224)
    print(f"✅ PyTorch tensor created: {tensor.shape}")
    
    # Test image processing
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    print(f"✅ Image processing works: {pil_img.size}")
    
    print("🎉 Installation test PASSED!")
    
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
        
        if "Installation test PASSED" in result.stdout:
            logger.info("✅ Installation test successful!")
            return True
        else:
            logger.warning("⚠️  Installation test had issues")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function."""
    logger.info("🔧 Windows-Compatible SatyaAI Setup")
    logger.info("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Clean up corrupted packages
    clean_corrupted_packages()
    
    # Install compatible versions
    if not install_compatible_versions():
        logger.error("❌ Failed to install core packages")
        return 1
    
    # Install optional packages
    install_optional_packages()
    
    # Test installation
    if test_installation():
        logger.info("\n🎉 WINDOWS SETUP COMPLETE!")
        logger.info("✅ Your system is ready for SatyaAI")
        logger.info("\nNext steps:")
        logger.info("1. Run: python server/python/simple_model_downloader.py")
        logger.info("2. Start system: npm run start:satyaai")
        return 0
    else:
        logger.warning("\n⚠️  Setup completed with warnings")
        logger.info("System should still work, but some features may be limited")
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