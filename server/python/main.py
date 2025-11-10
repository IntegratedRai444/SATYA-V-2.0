#!/usr/bin/env python3
"""
SatyaAI Main Entry Point
Main entry point for the SatyaAI Python server with comprehensive initialization
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_logging():
    """Setup logging configuration"""
    log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('satyaai.log')
        ]
    )

def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'flask',
        'flask_cors',
        'numpy',
        'cv2',
        'PIL',
        'torch',
        'torchvision'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def initialize_models():
    """Initialize AI models"""
    try:
        from model_manager import EnhancedModelManager
        model_manager = EnhancedModelManager()
        # Initialize models (if needed)
        # model_manager.initialize_models()  # Uncomment if you have initialization logic
        print("✅ AI models initialized successfully")
        return True
    except Exception as e:
        print(f"⚠️  Model initialization warning: {e}")
        return True  # Continue even if models fail to load

def main():
    """Main entry point for SatyaAI Python server"""
    print("=" * 60)
    print("SatyaAI - Advanced Deepfake Detection System")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Initialize models
        initialize_models()
        
        # Import and configure Flask application
        from app import app
        
        # Get configuration from environment
        port = int(os.environ.get('PORT', 5001))
        host = os.environ.get('HOST', '0.0.0.0')
        debug = os.environ.get('FLASK_ENV', 'production') == 'development'
        
        logger.info(f"Starting SatyaAI Python server on {host}:{port}")
        logger.info(f"Debug mode: {debug}")
        logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'production')}")
        
        print(f"🚀 Starting SatyaAI Python server on {host}:{port}")
        print(f"🔧 Debug mode: {debug}")
        print(f"🌍 Environment: {os.environ.get('FLASK_ENV', 'production')}")
        print("🎯 Ready to process deepfake detection requests!")
        
        # Start the Flask application
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,
            use_reloader=False  # Disable reloader to prevent double initialization
        )
        
    except ImportError as e:
        logger.error(f"Failed to import Flask application: {e}")
        print(f"❌ Failed to import Flask application: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()