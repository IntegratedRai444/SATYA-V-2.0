#!/usr/bin/env python3
"""
Backend setup and fix script
Installs dependencies and fixes common issues
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description}: SUCCESS")
            return True
        else:
            print(f"❌ {description}: FAILED")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description}: ERROR - {e}")
        return False

def main():
    print("🚀 SATYA-V-2.0 Backend Setup")
    print("=" * 50)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info >= (3, 13):
        print("⚠️ Python 3.13+ detected. Some dependencies may not work.")
    
    # Install core dependencies
    print("\n📦 Installing Core Dependencies:")
    success = True
    
    # Install basic requirements first
    success &= run_command("pip install fastapi uvicorn python-multipart", "FastAPI and Uvicorn")
    success &= run_command("pip install numpy scikit-learn scipy pandas", "Scientific Computing")
    success &= run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", "PyTorch")
    success &= run_command("pip install opencv-python-headless pillow", "Computer Vision")
    success &= run_command("pip install librosa soundfile pydub", "Audio Processing")
    success &= run_command("pip install moviepy ffmpeg-python", "Video Processing")
    success &= run_command("pip install matplotlib seaborn", "Data Visualization")
    success &= run_command("pip install requests python-dotenv tqdm joblib", "Utilities")
    success &= run_command("pip install piexif exifread fpdf", "File Processing")
    success &= run_command("pip install pytest httpx", "Testing")
    
    # Try optional dependencies
    print("\n🔍 Installing Optional Dependencies:")
    run_command("pip install mediapipe", "Mediapipe (may fail on Python 3.13+)")
    run_command("pip install face-recognition", "Face Recognition (may need dlib)")
    
    # Create necessary directories
    print("\n📁 Creating Directories:")
    os.makedirs("reports", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    print("✅ Created reports/ and uploads/ directories")
    
    # Test imports
    print("\n🧪 Testing Imports:")
    test_imports = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("librosa", "Librosa"),
    ]
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            success = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 SETUP SUMMARY")
    print(f"{'='*50}")
    
    if success:
        print("🎉 Backend setup completed successfully!")
        print("✅ You can now run: python -m uvicorn backend.main:app --reload")
    else:
        print("⚠️ Some dependencies failed to install.")
        print("💡 Try installing them manually or check Python version compatibility")
    
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 