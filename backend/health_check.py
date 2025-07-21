#!/usr/bin/env python3
"""
Backend Health Check and Fix Script
Identifies issues and provides fixes
"""
import os
import sys
import importlib
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: EXISTS")
        return True
    else:
        print(f"❌ {description}: MISSING")
        return False

def check_import(module_name, description):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {description}: OK")
        return True
    except ImportError as e:
        print(f"❌ {description}: FAILED - {e}")
        return False
    except Exception as e:
        print(f"⚠️ {description}: ERROR - {e}")
        return False

def check_directory_structure():
    """Check backend directory structure"""
    print("📁 Checking Directory Structure:")
    
    required_dirs = [
        ("backend", "Backend root"),
        ("backend/routes", "Routes directory"),
        ("backend/models", "Models directory"),
        ("backend/utils", "Utils directory"),
        ("backend/tests", "Tests directory"),
    ]
    
    required_files = [
        ("backend/main.py", "Main application"),
        ("backend/requirements.txt", "Dependencies"),
        ("backend/routes/__init__.py", "Routes init"),
        ("backend/models/__init__.py", "Models init"),
        ("backend/utils/__init__.py", "Utils init"),
        ("backend/routes/image.py", "Image routes"),
        ("backend/routes/video.py", "Video routes"),
        ("backend/routes/audio.py", "Audio routes"),
        ("backend/routes/webcam.py", "Webcam routes"),
        ("backend/routes/auth.py", "Auth routes"),
        ("backend/routes/system.py", "System routes"),
        ("backend/routes/assistant.py", "Assistant routes"),
        ("backend/models/image_model.py", "Image model"),
        ("backend/models/video_model.py", "Video model"),
        ("backend/models/audio_model.py", "Audio model"),
        ("backend/models/webcam_model.py", "Webcam model"),
        ("backend/utils/image_utils.py", "Image utils"),
        ("backend/utils/video_utils.py", "Video utils"),
        ("backend/utils/audio_utils.py", "Audio utils"),
        ("backend/utils/report_utils.py", "Report utils"),
    ]
    
    success = True
    
    for dir_path, description in required_dirs:
        success &= check_file_exists(dir_path, description)
    
    for file_path, description in required_files:
        success &= check_file_exists(file_path, description)
    
    return success

def check_imports():
    """Check all backend imports"""
    print("\n🔧 Checking Backend Imports:")
    
    imports_to_check = [
        ("backend.main", "Main app"),
        ("backend.routes.image", "Image routes"),
        ("backend.routes.video", "Video routes"),
        ("backend.routes.audio", "Audio routes"),
        ("backend.routes.webcam", "Webcam routes"),
        ("backend.routes.auth", "Auth routes"),
        ("backend.routes.system", "System routes"),
        ("backend.routes.assistant", "Assistant routes"),
        ("backend.models.image_model", "Image model"),
        ("backend.models.video_model", "Video model"),
        ("backend.models.audio_model", "Audio model"),
        ("backend.models.webcam_model", "Webcam model"),
        ("backend.utils.image_utils", "Image utils"),
        ("backend.utils.video_utils", "Video utils"),
        ("backend.utils.audio_utils", "Audio utils"),
        ("backend.utils.report_utils", "Report utils"),
    ]
    
    success = True
    for module, description in imports_to_check:
        success &= check_import(module, description)
    
    return success

def check_dependencies():
    """Check external dependencies"""
    print("\n📦 Checking Dependencies:")
    
    dependencies = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("librosa", "Librosa"),
        ("matplotlib", "Matplotlib"),
        ("fpdf", "FPDF"),
    ]
    
    success = True
    for module, name in dependencies:
        success &= check_import(module, name)
    
    # Check optional dependencies
    print("\n🔍 Checking Optional Dependencies:")
    try:
        import mediapipe
        print("✅ Mediapipe: Available")
    except ImportError:
        print("⚠️ Mediapipe: Not available (expected on Python 3.13+)")
    
    try:
        import face_recognition
        print("✅ Face Recognition: Available")
    except ImportError:
        print("⚠️ Face Recognition: Not available (may need dlib)")
    
    return success

def check_code_issues():
    """Check for common code issues"""
    print("\n🔍 Checking Code Issues:")
    
    issues_found = []
    
    # Check for common import issues
    files_to_check = [
        "backend/models/video_model.py",
        "backend/main.py",
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    if "from models." in content and "from backend.models." not in content:
                        issues_found.append(f"❌ {file_path}: Relative import issue")
                    else:
                        print(f"✅ {file_path}: Import paths OK")
            except Exception as e:
                issues_found.append(f"❌ {file_path}: Read error - {e}")
    
    # Check for missing directories
    required_dirs = ["reports", "uploads"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            issues_found.append(f"❌ Missing directory: {dir_name}")
        else:
            print(f"✅ Directory exists: {dir_name}")
    
    if issues_found:
        print("\n⚠️ Issues Found:")
        for issue in issues_found:
            print(issue)
        return False
    else:
        print("✅ No code issues found")
        return True

def main():
    print("🏥 SATYA-V-2.0 Backend Health Check")
    print("=" * 60)
    
    print(f"🐍 Python version: {sys.version}")
    print(f"📂 Working directory: {os.getcwd()}")
    
    # Run all checks
    structure_ok = check_directory_structure()
    imports_ok = check_imports()
    deps_ok = check_dependencies()
    code_ok = check_code_issues()
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 HEALTH CHECK SUMMARY")
    print(f"{'='*60}")
    
    all_ok = structure_ok and imports_ok and deps_ok and code_ok
    
    print(f"Directory Structure: {'✅ OK' if structure_ok else '❌ ISSUES'}")
    print(f"Import Issues: {'✅ OK' if imports_ok else '❌ ISSUES'}")
    print(f"Dependencies: {'✅ OK' if deps_ok else '❌ ISSUES'}")
    print(f"Code Issues: {'✅ OK' if code_ok else '❌ ISSUES'}")
    
    if all_ok:
        print("\n🎉 BACKEND IS HEALTHY!")
        print("✅ Ready to run: python -m uvicorn backend.main:app --reload")
    else:
        print("\n⚠️ BACKEND HAS ISSUES")
        print("💡 Run: python backend/setup_backend.py to fix dependencies")
        print("💡 Check the issues above and fix them manually")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 