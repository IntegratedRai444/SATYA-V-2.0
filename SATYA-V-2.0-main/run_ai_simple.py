#!/usr/bin/env python3
"""
Simple Pure AI Deepfake Detection System
Real neural networks with minimal dependencies
"""

import os
import sys
import time
import subprocess
import requests
from datetime import datetime

def check_basic_dependencies():
    """Check basic dependencies"""
    basic_packages = ['torch', 'numpy']
    missing = []
    
    for package in basic_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - MISSING")
    
    return missing

def install_basic_dependencies():
    """Install basic dependencies"""
    print("\n🔧 Installing basic Pure AI dependencies...")
    
    try:
        # Install PyTorch
        print("Installing PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        
        # Install other basics
        print("Installing other dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "numpy", "flask", "flask-cors", "requests"
        ])
        
        print("✅ Basic dependencies installed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_simple_pure_ai_server(port=5002):
    """Start simple pure AI server"""
    print(f"\n🤖 Starting Simple Pure AI server on port {port}...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(script_dir, "server", "python", "pure_ai_server.py")
    
    if not os.path.exists(server_script):
        print(f"❌ Pure AI server script not found at {server_script}")
        return False
    
    env = os.environ.copy()
    env["PORT"] = str(port)
    
    try:
        if os.name == 'nt':  # Windows
            subprocess.Popen([sys.executable, server_script], 
                            env=env, 
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
        else:  # Unix/Linux/Mac
            subprocess.Popen([sys.executable, server_script],
                            env=env,
                            start_new_session=True)
        
        print("⏳ Waiting for Simple Pure AI server to start...")
        return check_server_status(port)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

def check_server_status(port=5002):
    """Check if server is running"""
    url = f"http://localhost:{port}/health"
    max_attempts = 10
    wait_seconds = 2
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Simple Pure AI server is running on port {port}")
                print(f"🤖 Status: {data.get('status', 'unknown')}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_attempts - 1:
            dots = "." * (attempt % 3 + 1)
            print(f"⏳ Waiting for server{dots}", end='\r')
            time.sleep(wait_seconds)
    
    print(f"❌ Failed to connect to server on port {port}")
    return False

def main():
    """Main entry point"""
    print("=" * 60)
    print("🤖 Simple Pure AI Deepfake Detection System")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check dependencies
    print("\n🔍 Checking basic dependencies...")
    missing_packages = check_basic_dependencies()
    
    if missing_packages:
        print(f"\n⚠️ Missing {len(missing_packages)} basic dependencies")
        install_choice = input("Install basic dependencies? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            if not install_basic_dependencies():
                print("❌ Failed to install dependencies. Cannot start system.")
                return False
        else:
            print("❌ System requires basic dependencies to be installed.")
            return False
    
    # Start server
    port = int(os.environ.get("PYTHON_PORT", 5002))
    success = start_simple_pure_ai_server(port)
    
    if success:
        print("\n🎉 Simple Pure AI server started successfully!")
        print(f"🌐 Frontend: http://localhost:5173")
        print(f"🤖 Backend: http://localhost:{port}")
        print(f"📊 Health Check: http://localhost:{port}/health")
        print("\n🤖 Pure AI Features Available:")
        print("   • Real PyTorch neural networks (if available)")
        print("   • Fallback analysis (always available)")
        print("   • Real-time processing")
        print("   • Graceful dependency handling")
        print("\n🔍 Use /api/analyze/*/pure-ai endpoints for Pure AI analysis")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 