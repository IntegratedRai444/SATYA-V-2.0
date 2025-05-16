#!/usr/bin/env python3
"""
SatyaAI - Advanced Deepfake Detection System
Main entry point script
"""
import os
import sys
import time
import subprocess
import requests
from datetime import datetime

def check_dependencies():
    """Check if required Python dependencies are installed"""
    try:
        import flask
        import numpy
        import cv2
        import matplotlib
        import PIL
        print("✅ All required dependencies are installed.")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages with: pip install flask numpy opencv-python-headless matplotlib pillow flask-cors")
        return False

def start_server(port=5000, debug=False):
    """Start the SatyaAI server"""
    if not check_dependencies():
        return False
    
    print(f"🚀 Starting SatyaAI Python server on port {port}...")
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the Python server script
    server_script = os.path.join(script_dir, "server", "python", "main.py")
    
    # Check if the server script exists
    if not os.path.exists(server_script):
        print(f"❌ Server script not found at {server_script}")
        return False
    
    # Set environment variables
    env = os.environ.copy()
    env["PORT"] = str(port)
    
    # Start the server process
    try:
        if debug:
            # Run in foreground with debug output
            subprocess.run([sys.executable, server_script], env=env)
        else:
            # Run in background
            if os.name == 'nt':  # Windows
                subprocess.Popen([sys.executable, server_script], 
                                env=env, 
                                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:  # Unix/Linux/Mac
                subprocess.Popen([sys.executable, server_script],
                                env=env,
                                start_new_session=True)
            
            # Wait for server to start
            print("⏳ Waiting for server to start...")
            return check_server_status(port)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

def check_server_status(port=5000):
    """Check if the server is running"""
    url = f"http://localhost:{port}/health"
    max_attempts = 10
    wait_seconds = 1
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                print(f"✅ SatyaAI server is running on port {port}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_attempts - 1:
            dot_animation = "." * (attempt % 3 + 1)
            print(f"⏳ Waiting for server{dot_animation}", end='\r')
            time.sleep(wait_seconds)
    
    print(f"❌ Failed to connect to server on port {port} after {max_attempts} attempts")
    return False

def main():
    """Main entry point"""
    print("=" * 50)
    print("🔍 SatyaAI - Advanced Deepfake Detection System")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    port = int(os.environ.get("PYTHON_PORT", 5001))
    debug_mode = "--debug" in sys.argv
    
    if "--check" in sys.argv:
        # Just check if the server is running
        return check_server_status(port)
    else:
        # Start the server
        return start_server(port, debug_mode)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)