#!/usr/bin/env python3
"""
Simple backend runner script (No Uvicorn)
Checks dependencies and starts the FastAPI server with alternative options
"""
import sys
import os
import subprocess

def check_dependency(module_name):
    """Check if a dependency is available"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def install_dependency(package_name):
    """Install a dependency"""
    print(f"📦 Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🚀 SATYA-V-2.0 Backend Runner (No Uvicorn)")
    print("=" * 50)
    
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
        print("⚠️ Missing dependencies:")
        for module, name in missing_deps:
            print(f"  - {name} ({module})")
        
        print("\n🔄 Installing missing dependencies...")
        for module, name in missing_deps:
            if install_dependency(module):
                print(f"✅ {name} installed successfully")
            else:
                print(f"❌ Failed to install {name}")
                print("💡 Please install dependencies manually:")
                print("   pip install fastapi numpy")
                return
    
    # Check if backend directory exists
    if not os.path.exists("backend"):
        print("❌ Backend directory not found!")
        return
    
    # Check if main.py exists
    if not os.path.exists("backend/main.py"):
        print("❌ backend/main.py not found!")
        return
    
    print("✅ All dependencies available")
    print("✅ Backend files found")
    print("⚠️ Note: Uvicorn has been removed from dependencies")
    
    # Provide server options
    print("\n🌐 Choose your server option:")
    print("1. Minimal mode (basic HTTP server) - Fastest")
    print("2. Install uvicorn manually")
    print("3. Install hypercorn (alternative ASGI server)")
    print("4. Use gunicorn (if available)")
    print("-" * 50)
    
    choice = input("Choose option (1-4) or press Enter for minimal mode: ").strip()
    
    if choice == "2":
        print("🔄 Installing uvicorn...")
        if install_dependency("uvicorn"):
            print("✅ Uvicorn installed successfully!")
            print("🌐 Starting FastAPI server with uvicorn...")
            print("📡 Server will be available at: http://localhost:8000")
            print("📚 API docs at: http://localhost:8000/docs")
            print("🔄 Press Ctrl+C to stop the server")
            print("-" * 50)
            try:
                subprocess.run([
                    sys.executable, "-m", "uvicorn", 
                    "backend.main:app", 
                    "--host", "0.0.0.0", 
                    "--port", "8000",
                    "--reload"
                ])
            except KeyboardInterrupt:
                print("\n🛑 Server stopped by user")
            except Exception as e:
                print(f"❌ Error starting server: {e}")
        else:
            print("❌ Failed to install uvicorn")
    
    elif choice == "3":
        print("🔄 Installing hypercorn...")
        if install_dependency("hypercorn"):
            print("✅ Hypercorn installed successfully!")
            print("🌐 Starting FastAPI server with hypercorn...")
            print("📡 Server will be available at: http://localhost:8000")
            print("📚 API docs at: http://localhost:8000/docs")
            print("🔄 Press Ctrl+C to stop the server")
            print("-" * 50)
            try:
                subprocess.run([
                    sys.executable, "-m", "hypercorn", 
                    "backend.main:app", 
                    "--bind", "0.0.0.0:8000"
                ])
            except KeyboardInterrupt:
                print("\n🛑 Server stopped by user")
            except Exception as e:
                print(f"❌ Error starting server: {e}")
        else:
            print("❌ Failed to install hypercorn")
    
    elif choice == "4":
        if check_dependency("gunicorn"):
            print("✅ Gunicorn found!")
            print("🌐 Starting FastAPI server with gunicorn...")
            print("📡 Server will be available at: http://localhost:8000")
            print("🔄 Press Ctrl+C to stop the server")
            print("-" * 50)
            try:
                subprocess.run([
                    sys.executable, "-m", "gunicorn", 
                    "backend.main:app", 
                    "-w", "1", 
                    "-b", "0.0.0.0:8000"
                ])
            except KeyboardInterrupt:
                print("\n🛑 Server stopped by user")
            except Exception as e:
                print(f"❌ Error starting server: {e}")
        else:
            print("❌ Gunicorn not available")
            print("💡 Install with: pip install gunicorn")
    
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