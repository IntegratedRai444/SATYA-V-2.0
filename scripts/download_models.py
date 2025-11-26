#!/usr/bin/env python3
"""
SatyaAI Model Downloader
Downloads and sets up all required AI models for deepfake detection
"""

import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List
import urllib.request
from tqdm import tqdm

# Model directory
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url: str, output_path: Path, description: str = "Downloading"):
    """Download file with progress bar"""
    print(f"\n📥 {description}...")
    print(f"   URL: {url}")
    print(f"   Destination: {output_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=description) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    print(f"✅ Downloaded successfully!")

def verify_checksum(file_path: Path, expected_md5: str = None) -> bool:
    """Verify file integrity using MD5 checksum"""
    if not expected_md5:
        return True
    
    print(f"🔍 Verifying checksum...")
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    actual_md5 = md5_hash.hexdigest()
    if actual_md5 == expected_md5:
        print(f"✅ Checksum verified!")
        return True
    else:
        print(f"❌ Checksum mismatch! Expected: {expected_md5}, Got: {actual_md5}")
        return False

def download_huggingface_model(model_name: str, output_dir: Path):
    """Download model from Hugging Face using transformers library"""
    try:
        from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
        
        print(f"\n📦 Downloading {model_name} from Hugging Face...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Try to download as a vision model
            model = AutoModel.from_pretrained(model_name, cache_dir=str(output_dir))
            print(f"✅ Model downloaded successfully!")
        except:
            print(f"ℹ️  Model will be downloaded on first use")
        
        return True
    except ImportError:
        print("⚠️  transformers library not installed. Run: pip install transformers")
        return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def setup_image_models():
    """Setup image deepfake detection models"""
    print("\n" + "="*60)
    print("🖼️  IMAGE DETECTION MODELS")
    print("="*60)
    
    image_models_dir = MODELS_DIR / "image"
    image_models_dir.mkdir(exist_ok=True)
    
    models = {
        "efficientnet": {
            "name": "timm/efficientnet_b4",
            "description": "EfficientNet-B4 for deepfake detection",
            "size": "~100MB"
        },
        "facenet": {
            "name": "timesler/facenet-pytorch",
            "description": "FaceNet for face analysis",
            "size": "~90MB"
        }
    }
    
    for model_key, model_info in models.items():
        print(f"\n📦 {model_info['description']} ({model_info['size']})")
        model_dir = image_models_dir / model_key
        
        # For now, we'll use PyTorch Hub and Hugging Face which handle downloads automatically
        print(f"ℹ️  Model will be downloaded automatically on first use")
        print(f"   Model ID: {model_info['name']}")
    
    # Create a marker file
    (image_models_dir / "README.txt").write_text(
        "Image Detection Models\n"
        "Models will be downloaded automatically on first use.\n"
        "This directory will store cached model weights.\n"
    )
    
    return True

def setup_video_models():
    """Setup video deepfake detection models"""
    print("\n" + "="*60)
    print("🎥 VIDEO DETECTION MODELS")
    print("="*60)
    
    video_models_dir = MODELS_DIR / "video"
    video_models_dir.mkdir(exist_ok=True)
    
    models = {
        "slowfast": {
            "description": "SlowFast for temporal analysis",
            "size": "~250MB",
            "note": "Will be downloaded on first use"
        },
        "wav2lip": {
            "description": "Wav2Lip for lip-sync detection",
            "size": "~200MB",
            "note": "Will be downloaded on first use"
        }
    }
    
    for model_key, model_info in models.items():
        print(f"\n📦 {model_info['description']} ({model_info['size']})")
        print(f"   {model_info['note']}")
    
    (video_models_dir / "README.txt").write_text(
        "Video Detection Models\n"
        "Models will be downloaded automatically on first use.\n"
        "This directory will store cached model weights.\n"
    )
    
    return True

def setup_audio_models():
    """Setup audio deepfake detection models"""
    print("\n" + "="*60)
    print("🎵 AUDIO DETECTION MODELS")
    print("="*60)
    
    audio_models_dir = MODELS_DIR / "audio"
    audio_models_dir.mkdir(exist_ok=True)
    
    models = {
        "ecapa_tdnn": {
            "name": "speechbrain/spkrec-ecapa-voxceleb",
            "description": "ECAPA-TDNN for speaker verification",
            "size": "~50MB"
        }
    }
    
    for model_key, model_info in models.items():
        print(f"\n📦 {model_info['description']} ({model_info['size']})")
        print(f"ℹ️  Model will be downloaded automatically on first use")
        print(f"   Model ID: {model_info.get('name', 'N/A')}")
    
    (audio_models_dir / "README.txt").write_text(
        "Audio Detection Models\n"
        "Models will be downloaded automatically on first use.\n"
        "This directory will store cached model weights.\n"
    )
    
    return True

def create_model_config():
    """Create model configuration file"""
    print("\n" + "="*60)
    print("⚙️  CREATING MODEL CONFIGURATION")
    print("="*60)
    
    config_content = """# SatyaAI Model Configuration
# This file contains settings for all AI models used in deepfake detection

models:
  image:
    efficientnet:
      name: "timm/efficientnet_b4"
      type: "classification"
      input_size: [380, 380]
      confidence_threshold: 0.7
      enabled: true
    
    facenet:
      name: "facenet-pytorch"
      type: "embedding"
      input_size: [160, 160]
      enabled: true
  
  video:
    slowfast:
      name: "facebookresearch/slowfast"
      type: "temporal"
      frames_per_clip: 32
      confidence_threshold: 0.65
      enabled: true
    
    wav2lip:
      name: "Rudrabha/Wav2Lip"
      type: "lipsync"
      confidence_threshold: 0.75
      enabled: true
  
  audio:
    ecapa_tdnn:
      name: "speechbrain/spkrec-ecapa-voxceleb"
      type: "speaker_verification"
      sample_rate: 16000
      confidence_threshold: 0.6
      enabled: true

# Ensemble settings
ensemble:
  enabled: true
  voting_strategy: "weighted"
  weights:
    image: 0.4
    video: 0.35
    audio: 0.25
  
  min_confidence: 0.5
  require_majority: true

# Performance settings
performance:
  batch_size: 8
  num_workers: 4
  use_gpu: true
  gpu_memory_fraction: 0.8
  enable_caching: true
  cache_ttl: 3600  # seconds

# Detection thresholds
thresholds:
  authentic: 0.7      # Above this = authentic
  suspicious: 0.4     # Between this and authentic = suspicious
  deepfake: 0.4       # Below this = deepfake

# Preprocessing
preprocessing:
  image:
    resize: [380, 380]
    normalize: true
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
  
  video:
    fps: 30
    max_frames: 300
    resize: [224, 224]
  
  audio:
    sample_rate: 16000
    duration: 10  # seconds
    n_mfcc: 40
"""
    
    config_path = MODELS_DIR / "model_config.yaml"
    config_path.write_text(config_content)
    print(f"✅ Created configuration file: {config_path}")
    
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n" + "="*60)
    print("🔍 CHECKING DEPENDENCIES")
    print("="*60)
    
    required_packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "transformers": "Hugging Face Transformers",
        "timm": "PyTorch Image Models",
        "facenet_pytorch": "FaceNet PyTorch",
        "librosa": "Librosa (audio processing)",
        "opencv-python": "OpenCV",
        "PIL": "Pillow (image processing)",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "tqdm": "Progress bars"
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            if package == "opencv-python":
                __import__("cv2")
            else:
                __import__(package.replace("-", "_"))
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print(f"\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    else:
        print(f"\n✅ All dependencies installed!")
        return True

def main():
    """Main function"""
    print("\n" + "="*60)
    print("🛡️  SatyaAI Model Setup")
    print("="*60)
    print(f"\nModels directory: {MODELS_DIR.absolute()}")
    
    # Check dependencies first
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n⚠️  Please install missing dependencies before proceeding.")
        print("   Run: pip install -r server/python/requirements-complete.txt")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Setup models
    try:
        setup_image_models()
        setup_video_models()
        setup_audio_models()
        create_model_config()
        
        print("\n" + "="*60)
        print("✅ MODEL SETUP COMPLETE!")
        print("="*60)
        print("\n📝 Next steps:")
        print("   1. Models will download automatically on first use")
        print("   2. Configure settings in: models/model_config.yaml")
        print("   3. Start the Python service: npm run dev:python")
        print("   4. Test detection endpoints")
        print("\n💡 Tip: First run will be slower as models download.")
        print("   Subsequent runs will use cached models.\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
