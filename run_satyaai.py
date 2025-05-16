#!/usr/bin/env python3
"""
SatyaAI - Advanced Deepfake Detection System
Main entry point script
"""

import os
import sys
import argparse
import subprocess
import time
import logging
import platform

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('satyaai.log')
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required Python dependencies are installed"""
    required_packages = [
        'flask', 'numpy', 'pillow', 'flask-cors'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])
            logger.info("Successfully installed missing packages")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {e}")
            return False
    
    return True

def start_server(port=5000, debug=False):
    """Start the SatyaAI server"""
    script_path = os.path.join('server', 'python', 'main.py')
    
    if not os.path.exists(script_path):
        logger.error(f"Server script not found at: {script_path}")
        return False
    
    try:
        cmd = [sys.executable, script_path, '--port', str(port)]
        if debug:
            cmd.append('--debug')
        
        logger.info(f"Starting SatyaAI server on port {port}")
        server_process = subprocess.Popen(cmd)
        
        time.sleep(2)  # Give it a moment to start
        
        if server_process.poll() is not None:
            logger.error("Server failed to start")
            return False
        
        logger.info("Server started successfully")
        return server_process
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        return False

def check_server_status(port=5000):
    """Check if the server is running"""
    import urllib.request
    import json
    
    try:
        response = urllib.request.urlopen(f"http://localhost:{port}/health")
        data = response.read().decode('utf-8')
        
        try:
            status = json.loads(data)
            if status.get('status') == 'ready':
                logger.info("Server is running and ready")
                return True
        except json.JSONDecodeError:
            pass
        
        logger.warning("Server is running but not ready")
        return False
    except Exception:
        logger.warning("Server is not running")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SatyaAI Deepfake Detection System')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    logger.info("Starting SatyaAI - Advanced Deepfake Detection System")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    
    # Check dependencies
    logger.info("Checking dependencies...")
    if not check_dependencies():
        logger.error("Failed to satisfy dependencies. Exiting.")
        return 1
    
    # Check if server is already running
    if check_server_status(args.port):
        logger.info(f"SatyaAI server is already running on port {args.port}")
        return 0
    
    # Start server
    server_process = start_server(args.port, args.debug)
    if not server_process:
        logger.error("Failed to start SatyaAI server. Exiting.")
        return 1
    
    try:
        # Wait for server to be ready
        for _ in range(10):
            if check_server_status(args.port):
                break
            time.sleep(1)
        
        # Keep the server running
        logger.info("SatyaAI system is now running. Press Ctrl+C to stop.")
        server_process.wait()
    except KeyboardInterrupt:
        logger.info("Shutting down SatyaAI...")
        server_process.terminate()
        server_process.wait()
        logger.info("SatyaAI shutdown complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())