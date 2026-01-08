#!/usr/bin/env python3
"""
AI Model Validation Script
Validates that all required AI models and dependencies are available
Run this before starting the server to ensure STRICT MODE will work
"""

import os
import sys
from pathlib import Path


def check_python_packages():
    """Check if required Python packages are installed."""
    print("🔍 Checking Python packages...")
    missing = []

    packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "facenet_pytorch": "FaceNet-PyTorch",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "PIL": "Pillow",
    }

    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT INSTALLED")
            missing.append(name)

    return missing


def check_model_files():
    """Check if required model files exist."""
    print("\n🔍 Checking AI model files...")
    missing = []

    models = {
        "EfficientNet-B4": "models/efficientnet_b4_deepfake.bin",
        "ResNet50": "models/resnet50_deepfake.pth",
        "Haar Cascade": "models/haarcascade_frontalface_default.xml",
    }

    for name, path in models.items():
        full_path = Path(__file__).parent / path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {name} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {name} - NOT FOUND at {full_path}")
            missing.append(name)

    return missing


def test_model_loading():
    """Test if models can actually be loaded."""
    print("\n🔍 Testing model loading...")
    errors = []

    try:
        print("  Testing MTCNN...")
        from facenet_pytorch import MTCNN

        mtcnn = MTCNN(keep_all=True, device="cpu")
        print("  ✅ MTCNN loaded successfully")
    except Exception as e:
        print(f"  ❌ MTCNN failed: {e}")
        errors.append(f"MTCNN: {e}")

    try:
        print("  Testing InceptionResnetV1...")
        from facenet_pytorch import InceptionResnetV1

        model = InceptionResnetV1(pretrained="vggface2", device="cpu").eval()
        print("  ✅ InceptionResnetV1 loaded successfully")
    except Exception as e:
        print(f"  ❌ InceptionResnetV1 failed: {e}")
        errors.append(f"InceptionResnetV1: {e}")

    try:
        print("  Testing PyTorch...")
        import torch

        x = torch.randn(1, 3, 160, 160)
        print(
            f"  ✅ PyTorch working (device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')})"
        )
    except Exception as e:
        print(f"  ❌ PyTorch failed: {e}")
        errors.append(f"PyTorch: {e}")

    return errors


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("  SatyaAI - STRICT MODE Validation")
    print("  Checking Real AI Model Requirements")
    print("=" * 60)
    print()

    # Check packages
    missing_packages = check_python_packages()

    # Check model files
    missing_models = check_model_files()

    # Test loading
    loading_errors = test_model_loading()

    # Summary
    print("\n" + "=" * 60)
    print("  VALIDATION SUMMARY")
    print("=" * 60)

    all_good = not missing_packages and not missing_models and not loading_errors

    if all_good:
        print("\n✅ ALL CHECKS PASSED!")
        print("   System is ready for STRICT MODE operation.")
        print("   Real AI models will be used for all detections.")
        print()
        return 0
    else:
        print("\n❌ VALIDATION FAILED!")
        print()

        if missing_packages:
            print("Missing Python packages:")
            for pkg in missing_packages:
                print(f"  - {pkg}")
            print("\nInstall with:")
            print(
                "  pip install torch torchvision facenet-pytorch opencv-python numpy pillow"
            )
            print()

        if missing_models:
            print("Missing model files:")
            for model in missing_models:
                print(f"  - {model}")
            print("\nDownload models with:")
            print("  python scripts/download_models.py")
            print()

        if loading_errors:
            print("Model loading errors:")
            for error in loading_errors:
                print(f"  - {error}")
            print()

        print("⚠️  STRICT MODE CANNOT OPERATE WITHOUT THESE REQUIREMENTS")
        print("   System will fail if started in strict mode.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
