"""
Model Setup Script
Downloads and configures required deepfake detection models.
"""
import os
import torch
import gdown
from pathlib import Path
from tqdm import tqdm

# Configuration
MODELS_DIR = Path("../models")
MODEL_URLS = {
    "efficientnet_b4_deepfake.pth": "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_ID",
    "xception_deepfake.pth": "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_ID",
    "resnet50_deepfake.pth": "https://drive.google.com/uc?id=YOUR_GOOGLE_DRIVE_ID"
}

def download_model(url: str, output_path: Path):
    """Download a model file with progress bar."""
    if output_path.exists():
        print(f"Model already exists: {output_path}")
        return True
        
    try:
        print(f"Downloading {output_path.name}...")
        gdown.download(url, str(output_path), quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading {output_path.name}: {e}")
        return False

def setup_models():
    """Download and verify all required models."""
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download each model
    success = True
    for model_name, url in MODEL_URLS.items():
        model_path = MODELS_DIR / model_name
        if not download_model(url, model_path):
            success = False
    
    # Verify PyTorch installation and GPU availability
    print("\nEnvironment Check:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    return success

if __name__ == "__main__":
    print("Setting up deepfake detection models...")
    if setup_models():
        print("\n✅ Model setup completed successfully!")
    else:
        print("\n❌ Some models failed to download. Please check the error messages above.")
