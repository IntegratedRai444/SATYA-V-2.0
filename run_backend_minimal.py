#!/usr/bin/env python3
"""
Minimal Backend Runner - Starts instantly without uvicorn or heavy ML dependencies
Perfect for development and testing
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
    """Install only essential dependencies for minimal mode"""
    print("📦 Installing minimal dependencies...")
    
    minimal_packages = [
        "fastapi",
        "python-multipart"
    ]
    
    for package in minimal_packages:
        print(f"  Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"  ⚠️ Failed to install {package}")

def main():
    print("⚡ SATYA-V-2.0 Minimal Backend Runner (No Uvicorn)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check essential dependencies
    essential_deps = [
        ("fastapi", "FastAPI"),
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
                print("💡 Please install manually: pip install fastapi")
                return
    
    # Check if minimal backend exists
    if not os.path.exists("backend/main_minimal.py"):
        print("❌ backend/main_minimal.py not found!")
        return
    
    load_time = time.time() - start_time
    print(f"✅ Dependencies ready ({load_time:.2f}s)")
    print("✅ Minimal backend found")
    print("✅ No uvicorn required - using basic HTTP server")
    
    # Start the minimal server
    print("\n🌐 Starting Minimal HTTP server (no uvicorn)...")
    print("📡 Server will be available at: http://localhost:8000")
    print("🔑 Demo login: username=demo, password=demo")
    print("🔄 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Run minimal backend directly
        subprocess.run([
            sys.executable, "backend/main_minimal.py"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main() 