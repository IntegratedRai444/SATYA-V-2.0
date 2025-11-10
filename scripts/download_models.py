"""
Model Downloader for SatyaAI
Downloads and sets up all required model files.
"""
import os
import torch
import gdown
import shutil
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base model directory
BASE_DIR = Path(__file__).parent.parent / "models"
MODEL_URLS = {
    "efficientnet_b7": {
        "url": "https://drive.google.com/uc?id=YOUR_EFFICIENTNET_MODEL_ID",
        "dest": "dfdc_efficientnet_b7/model.pth"
    },
    "xception": {
        "url": "https://drive.google.com/uc?id=YOUR_XCEPTION_MODEL_ID",
        "dest": "xception/model.pth"
    },
    "video_authenticator": {
        "url": "https://drive.google.com/uc?id=YOUR_VIDEO_AUTH_MODEL_ID",
        "dest": "microsoft_video_auth/model.pth"
    },
    "audio_detector": {
        "url": "https://drive.google.com/uc?id=YOUR_AUDIO_DETECTOR_ID",
        "dest": "stanford_audio_detector/model.pth"
    }
}

def setup_directories():
    """Create necessary directories for models."""
    for model_dir in ["dfdc_efficientnet_b7", "xception", "microsoft_video_auth", "stanford_audio_detector"]:
        (BASE_DIR / model_dir).mkdir(parents=True, exist_ok=True)

def download_model(url: str, dest: Path):
    """Download a model file using gdown."""
    if not dest.exists():
        logger.info(f"Downloading {url} to {dest}")
        try:
            gdown.download(url, str(dest), quiet=False)
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    return True

def verify_models():
    """Verify that all required models are present."""
    missing = []
    for model_name, model_info in MODEL_URLS.items():
        dest = BASE_DIR / model_info["dest"]
        if not dest.exists():
            missing.append(model_name)
    
    if missing:
        logger.warning(f"Missing models: {', '.join(missing)}")
        return False
    return True

def main():
    # Setup directories
    setup_directories()
    
    # Download models
    for model_name, model_info in tqdm(MODEL_URLS.items(), desc="Downloading models"):
        dest = BASE_DIR / model_info["dest"]
        if not dest.exists():
            success = download_model(model_info["url"], dest)
            if not success:
                logger.error(f"Failed to download {model_name}")
    
    # Verify all models
    if verify_models():
        logger.info("All models downloaded and verified successfully!")
    else:
        logger.warning("Some models failed to download or verify.")

if __name__ == "__main__":
    main()
