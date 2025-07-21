#!/usr/bin/env python3
"""
Pure AI Deepfake Detection System Launcher
Real neural networks with actual ML model inference
"""
import os
import sys
import time
import subprocess
import requests
from datetime import datetime

def check_pure_ai_dependencies():
    """Check if pure AI dependencies are installed"""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'scikit-learn',
        'opencv-python-headless', 'Pillow', 'face-recognition', 'librosa'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    return missing_packages

def install_pure_ai_dependencies():
    """Install pure AI dependencies"""
    print("\n🔧 Installing Pure AI dependencies...")
    
    try:
        # Install PyTorch first
        print("Installing PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
        
        # Install other dependencies
        print("Installing other dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "numpy>=1.21.0", "scikit-learn>=1.0.0", "scipy>=1.7.0",
            "opencv-python-headless>=4.8.0", "Pillow>=10.0.0",
            "librosa>=0.10.0", "soundfile>=0.12.0", "pydub>=0.25.0",
            "flask>=2.3.0", "flask-cors>=4.0.0", "Werkzeug>=2.3.0",
            "pandas>=1.5.0", "matplotlib>=3.6.0",
            "requests>=2.31.0", "python-dotenv>=1.0.0", "joblib>=1.3.0"
        ])
        
        # Try to install face-recognition
        try:
            print("Installing face-recognition...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "face-recognition>=1.3.0"
            ])
        except subprocess.CalledProcessError:
            print("⚠️ face-recognition installation failed - will use fallback methods")
        
        print("✅ Pure AI dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def start_pure_ai_server(port=5002, debug=False):
    """Start the pure AI server"""
    print(f"\n🤖 Starting Pure AI server on port {port}...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(script_dir, "server", "python", "pure_ai_server.py")
    
    if not os.path.exists(server_script):
        print(f"❌ Pure AI server script not found at {server_script}")
        return False
    
    env = os.environ.copy()
    env["PORT"] = str(port)
    
    try:
        if debug:
            subprocess.run([sys.executable, server_script], env=env)
        else:
            if os.name == 'nt':  # Windows
                subprocess.Popen([sys.executable, server_script], 
                                env=env, 
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:  # Unix/Linux/Mac
                subprocess.Popen([sys.executable, server_script],
                                env=env,
                                start_new_session=True)
            
            print("⏳ Waiting for Pure AI server to start...")
            return check_server_status(port)
    except Exception as e:
        print(f"❌ Failed to start Pure AI server: {e}")
        return False

def check_server_status(port=5002):
    """Check if the pure AI server is running"""
    url = f"http://localhost:{port}/health"
    max_attempts = 15
    wait_seconds = 2
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Pure AI server is running on port {port}")
                print(f"🤖 Pure AI Status: {data.get('pure_ai_status', 'unknown')}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_attempts - 1:
            dots = "." * (attempt % 3 + 1)
            print(f"⏳ Waiting for server{dots}", end='\r')
            time.sleep(wait_seconds)
    
    print(f"❌ Failed to connect to Pure AI server on port {port}")
    return False

def test_pure_ai_endpoints(port=5002):
    """Test pure AI analysis endpoints"""
    print("\n🧪 Testing Pure AI analysis endpoints...")
    
    base_url = f"http://localhost:{port}"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print("❌ Health endpoint failed")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False
    
    # Test pure AI image analysis endpoint
    try:
        test_data = {"imageData": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD"}
        response = requests.post(f"{base_url}/api/analyze/image/pure-ai", 
                                json=test_data, timeout=15)
        if response.status_code == 200:
            data = response.json()
            print("✅ Pure AI image analysis endpoint working")
            print(f"🤖 Neural Network Score: {data.get('neural_network_scores', {}).get('resnet50', 'N/A')}")
        else:
            print("❌ Pure AI image analysis endpoint failed")
    except Exception as e:
        print(f"⚠️ Pure AI image analysis endpoint error: {e}")
    
    print("🎉 Pure AI endpoint testing completed!")
    return True

def main():
    """Main entry point"""
    print("=" * 60)
    print("🤖 Pure AI Deepfake Detection System")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check dependencies
    print("\n🔍 Checking Pure AI dependencies...")
    missing_packages = check_pure_ai_dependencies()
    
    if missing_packages:
        print(f"\n⚠️ Missing {len(missing_packages)} Pure AI dependencies")
        install_choice = input("Install Pure AI dependencies? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            if not install_pure_ai_dependencies():
                print("❌ Failed to install dependencies. Cannot start Pure AI system.")
                return False
        else:
            print("❌ Pure AI system requires all dependencies to be installed.")
            return False
    
    # Start server
    port = int(os.environ.get("PYTHON_PORT", 5002))
    debug_mode = "--debug" in sys.argv
    
    if "--test" in sys.argv:
        # Just test endpoints
        return test_pure_ai_endpoints(port)
    else:
        # Start the server
        success = start_pure_ai_server(port, debug_mode)
        
        if success:
            print("\n🎉 Pure AI server started successfully!")
            print(f"🌐 Frontend: http://localhost:5173")
            print(f"🤖 Backend: http://localhost:{port}")
            print(f"📊 Health Check: http://localhost:{port}/health")
            print("\n🤖 Pure AI Features Available:")
            print("   • Real ResNet50 neural network inference")
            print("   • EfficientNet texture analysis")
            print("   • Face recognition with 128D encodings")
            print("   • Audio CNN spectrogram analysis")
            print("   • Ensemble classification with Random Forest")
            print("   • Real-time neural network processing")
            print("\n🔍 Use /api/analyze/*/pure-ai endpoints for Pure AI analysis")
        
        return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 