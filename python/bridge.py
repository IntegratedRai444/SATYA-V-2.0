#!/usr/bin/env python3
"""
Python Bridge for SATYA AI
Handles communication between Node.js and Python services
"""

import sys
import json
import traceback
import argparse
from pathlib import Path

class PythonBridge:
    def __init__(self, models_dir=None):
        """Initialize the Python bridge with optional models directory"""
        self.models_dir = Path(models_dir) if models_dir else None
        self.initialized = False
        self.setup_models()
        self.initialized = True

    def setup_models(self):
        """Initialize any required models or services"""
        if self.models_dir and self.models_dir.exists():
            print(f"[PYTHON] Models directory: {self.models_dir}", file=sys.stderr)
        # Add model loading logic here

    def handle_command(self, command, *args, **kwargs):
        """
        Handle incoming commands from Node.js
        
        Args:
            command: The command to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            dict: Response with success status and result/error
        """
        try:
            if not hasattr(self, command):
                return self._error_response(f"Unknown command: {command}")
                
            method = getattr(self, command)
            result = method(*args, **kwargs)
            return self._success_response(result)
            
        except Exception as e:
            return self._error_response(str(e), traceback.format_exc())
    
    def test_connection(self):
        """Test connection to the Python bridge"""
        return {"status": "ok", "message": "Python bridge is running"}
    
    def process_text(self, text, **kwargs):
        """Process text using the AI model"""
        # Add your text processing logic here
        return {"processed_text": text.upper(), **kwargs}
    
    def _success_response(self, result=None):
        """Create a success response"""
        return {
            "success": True,
            "result": result
        }
    
    def _error_response(self, message, trace=None):
        """Create an error response"""
        return {
            "success": False,
            "error": message,
            "trace": trace
        }

def main():
    """Main entry point for the Python bridge"""
    parser = argparse.ArgumentParser(description='Python Bridge for SATYA AI')
    parser.add_argument('--models-dir', type=str, help='Path to models directory')
    args = parser.parse_args()
    
    # Initialize the bridge
    bridge = PythonBridge(models_dir=args.models_dir)
    
    print("[PYTHON] Bridge initialized. Ready to process commands...", file=sys.stderr)
    
    # Process commands from stdin
    for line in sys.stdin:
        try:
            # Parse the command
            data = json.loads(line.strip())
            command = data.get('command')
            args = data.get('args', [])
            kwargs = data.get('kwargs', {})
            
            # Handle the command
            if command == 'exit':
                break
                
            response = bridge.handle_command(command, *args, **kwargs)
            
        except json.JSONDecodeError:
            response = {"success": False, "error": "Invalid JSON input"}
        except Exception as e:
            response = {"success": False, "error": str(e), "trace": traceback.format_exc()}
        
        # Send the response
        print(json.dumps(response), flush=True)

if __name__ == '__main__':
    main()
