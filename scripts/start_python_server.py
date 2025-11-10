#!/usr/bin/env python3
"""
SatyaAI - Unified Python Server Entry Point

This script serves as the single entry point for starting the SatyaAI Python backend.
It initializes all necessary components and starts the appropriate server based on the environment.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Add server/python to Python path
server_dir = Path(__file__).parent / 'server' / 'python'
sys.path.insert(0, str(server_dir))

def setup_logging(debug: bool = False) -> None:
    """Configure logging for the application."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('satyaai_python.log')
        ]
    )
    
    # Reduce logging from some noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

def start_development_server(host: str, port: int, debug: bool = True) -> None:
    """Start the development server with hot reloading."""
    from app import create_app
    
    app = create_app()
    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=True,
        threaded=True
    )

def start_production_server(host: str, port: int, workers: int = 4) -> None:
    """Start the production server using Gunicorn."""
    try:
        import gunicorn.app.base
    except ImportError:
        logging.error("Gunicorn is required for production mode. Install with: pip install gunicorn")
        sys.exit(1)
    
    from gunicorn.six import iteritems
    
    class StandaloneApplication(gunicorn.app.base.BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()
            
        def load_config(self):
            config = {
                key: value 
                for key, value in iteritems(self.options) 
                if key in self.cfg.settings and value is not None
            }
            for key, value in iteritems(config):
                self.cfg.set(key.lower(), value)
                
        def load(self):
            return self.application
    
    from app import create_app
    
    app = create_app()
    
    # Gunicorn config
    options = {
        'bind': f'{host}:{port}',
        'workers': workers,
        'worker_class': 'gthread',
        'threads': 4,
        'timeout': 120,
        'loglevel': 'info',
        'accesslog': '-',
        'errorlog': '-',
        'worker_tmp_dir': '/dev/shm',  # For better performance on Linux
    }
    
    StandaloneApplication(app, options).run()

def initialize_models() -> None:
    """Initialize all AI models."""
    try:
        from model_manager import EnhancedModelManager
        model_manager = EnhancedModelManager()
        # Add any model initialization code here
        logging.info("AI models initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing models: {e}")
        # Continue without models if initialization fails
        pass

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SatyaAI Python Server')
    
    # Server configuration
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5001,
                       help='Port to run the server on')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with hot reloading')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of worker processes (production only)')
    parser.add_argument('--init-models', action='store_true',
                       help='Initialize AI models before starting the server')
    
    return parser.parse_args()

def main() -> None:
    """Main entry point for the SatyaAI Python server."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    # Print banner
    print("\n" + "=" * 60)
    print("SatyaAI - Deepfake Detection System")
    print(f"Python Server (PID: {os.getpid()})")
    print(f"Mode: {'DEVELOPMENT' if args.debug else 'PRODUCTION'}")
    print("=" * 60 + "\n")
    
    try:
        # Initialize models if requested
        if args.init_models:
            logger.info("Initializing AI models...")
            initialize_models()
        
        # Start the appropriate server
        if args.debug:
            logger.info(f"Starting development server on http://{args.host}:{args.port}")
            start_development_server(args.host, args.port, debug=True)
        else:
            logger.info(f"Starting production server on http://{args.host}:{args.port} with {args.workers} workers")
            start_production_server(args.host, args.port, workers=args.workers)
            
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
