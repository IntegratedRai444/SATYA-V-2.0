#!/usr/bin/env python3
"""
Fast Backend Runner - Optimized for quick startup without uvicorn
Uses minimal dependencies and lazy loading
"""
import sys
import os
import subprocess
import time

def check_dependency(module_name):
    """Check if a dependency is available"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def install_minimal_deps():
    """Install only essential dependencies"""
    print("📦 Installing minimal dependencies for fast startup...")
    
    essential_packages = [
        "fastapi",
        "python-multipart",
        "numpy",
        "torch",
        "torchvision",
        "Pillow",
        "requests",
        "fpdf"
    ]
    
    for package in essential_packages:
        print(f"  Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"  ⚠️ Failed to install {package}")

def main():
    print("⚡ SATYA-V-2.0 Fast Backend Runner (No Uvicorn)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check essential dependencies
    essential_deps = [
        ("fastapi", "FastAPI"),
        ("numpy", "NumPy"),
    ]
    
    missing_deps = []
    for module, name in essential_deps:
        if not check_dependency(module):
            missing_deps.append((module, name))
    
    if missing_deps:
        print("⚠️ Missing essential dependencies:")
        for module, name in missing_deps:
            print(f"  - {name} ({module})")
        
        print("\n🔄 Installing minimal dependencies...")
        install_minimal_deps()
        
        # Check again
        for module, name in missing_deps:
            if not check_dependency(module):
                print(f"❌ Failed to install {name}")
                print("💡 Please install manually: pip install fastapi numpy")
                return
    
    # Check if backend directory exists
    if not os.path.exists("backend"):
        print("❌ Backend directory not found!")
        return
    
    # Check if main.py exists
    if not os.path.exists("backend/main.py"):
        print("❌ backend/main.py not found!")
        return
    
    load_time = time.time() - start_time
    print(f"✅ Dependencies ready ({load_time:.2f}s)")
    print("✅ Backend files found")
    print("⚠️ Note: Uvicorn removed - using alternative server options")
    
    # Provide server options
    print("\n🌐 Server Options (no uvicorn):")
    print("1. Minimal mode (basic HTTP server): python run_backend_minimal.py")
    print("2. Install uvicorn manually: pip install uvicorn")
    print("3. Use alternative ASGI server: pip install hypercorn")
    print("-" * 60)
    
    choice = input("Choose option (1-3) or press Enter for minimal mode: ").strip()
    
    if choice == "2":
        print("🔄 Installing uvicorn...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn"])
            print("✅ Uvicorn installed. You can now use the original backend.")
            print("💡 Run: python -m uvicorn backend.main:app --reload")
        except subprocess.CalledProcessError:
            print("❌ Failed to install uvicorn")
    elif choice == "3":
        print("🔄 Installing hypercorn...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "hypercorn"])
            print("✅ Hypercorn installed.")
            print("💡 Run: python -m hypercorn backend.main:app --bind 0.0.0.0:8000")
        except subprocess.CalledProcessError:
            print("❌ Failed to install hypercorn")
    else:
        print("🚀 Starting minimal mode...")
        try:
            subprocess.run([sys.executable, "run_backend_minimal.py"])
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
        except Exception as e:
            print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main() 