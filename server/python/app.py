from flask import Flask, request, jsonify
import os
import numpy as np
import time
import uuid
from datetime import datetime
import base64
import io
from PIL import Image
import logging
import traceback
from models import get_model, face_model, audio_model, video_model, multimodal_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('satyaai')

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for local development
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/api/python/analyze/image', methods=['POST'])
def analyze_image_endpoint():
    try:
        logger.info("Received image analysis request")
        
        if 'image' not in request.files and 'imageData' not in request.json:
            logger.warning("No image data provided")
            return jsonify({"error": "No image provided"}), 400
        
        # Process image data
        image_data = None
        
        # Handle base64 image data
        if 'imageData' in request.json:
            logger.info("Processing base64 image data")
            image_data = request.json['imageData']
            if image_data.startswith('data:image'):
                # Strip the data URL prefix if present
                image_data = image_data.split(',')[1]
            
            # Decode base64 to binary
            binary_data = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(binary_data))
        else:
            # Handle file upload
            logger.info("Processing uploaded image file")
            file = request.files['image']
            image = Image.open(file.stream)
        
        # Use our advanced face analysis model
        logger.info("Running advanced face analysis model")
        result = face_model.analyze(image)
        
        logger.info(f"Analysis complete with result: {result['authenticity']}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in image analysis: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/python/analyze/video', methods=['POST'])
def analyze_video_endpoint():
    try:
        logger.info("Received video analysis request")
        
        if 'video' not in request.files:
            logger.warning("No video file provided")
            return jsonify({"error": "No video provided"}), 400
        
        # Handle video file
        file = request.files['video']
        video_data = file.read()
        
        # Use our advanced video analysis model
        logger.info("Running advanced video analysis model")
        result = video_model.analyze(video_data)
        
        logger.info(f"Analysis complete with result: {result['authenticity']}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in video analysis: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/python/analyze/audio', methods=['POST'])
def analyze_audio_endpoint():
    try:
        logger.info("Received audio analysis request")
        
        if 'audio' not in request.files:
            logger.warning("No audio file provided")
            return jsonify({"error": "No audio provided"}), 400
        
        # Handle audio file
        file = request.files['audio']
        audio_data = file.read()
        
        # Use our advanced audio analysis model
        logger.info("Running advanced audio analysis model")
        result = audio_model.analyze(audio_data)
        
        logger.info(f"Analysis complete with result: {result['authenticity']}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in audio analysis: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/python/analyze/multimodal', methods=['POST'])
def analyze_multimodal_endpoint():
    try:
        logger.info("Received multimodal analysis request")
        
        # Check if any files were provided
        files = request.files
        if not files:
            logger.warning("No files provided for multimodal analysis")
            return jsonify({"error": "No files provided"}), 400
        
        # Get data for each modality
        data_dict = {}
        
        if 'image' in files:
            logger.info("Processing image component")
            image_file = files['image']
            image = Image.open(image_file.stream)
            data_dict['image'] = image
        
        if 'audio' in files:
            logger.info("Processing audio component")
            audio_file = files['audio']
            data_dict['audio'] = audio_file.read()
        
        if 'video' in files:
            logger.info("Processing video component")
            video_file = files['video']
            data_dict['video'] = video_file.read()
        
        if not data_dict:
            logger.warning("No valid files provided")
            return jsonify({"error": "No valid files provided"}), 400
        
        # Use our advanced multimodal fusion model
        logger.info("Running advanced multimodal fusion model")
        result = multimodal_model.analyze(data_dict)
        
        logger.info(f"Analysis complete with result: {result['authenticity']}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in multimodal analysis: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/python/analyze/webcam', methods=['POST'])
def analyze_webcam_endpoint():
    try:
        logger.info("Received webcam analysis request")
        
        if 'imageData' not in request.json:
            logger.warning("No webcam image data provided")
            return jsonify({"error": "No webcam image data provided"}), 400
        
        # Process base64 image data
        image_data = request.json['imageData']
        if image_data.startswith('data:image'):
            # Strip the data URL prefix
            image_data = image_data.split(',')[1]
        
        # Decode base64 to binary
        binary_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(binary_data))
        
        # Use our advanced face analysis model for webcam
        logger.info("Running advanced face analysis model on webcam frame")
        result = face_model.analyze(image)
        
        # Add real-time analysis flag
        result['realtime_analysis'] = True
        
        logger.info(f"Analysis complete with result: {result['authenticity']}")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in webcam analysis: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/python/models/info', methods=['GET'])
def models_info_endpoint():
    """Return information about available models"""
    models_info = {
        'face': face_model.get_model_info(),
        'audio': audio_model.get_model_info(),
        'video': video_model.get_model_info(),
        'multimodal': multimodal_model.get_model_info()
    }
    
    return jsonify({
        'models': models_info,
        'total_models': len(models_info)
    })

@app.route('/api/python/status', methods=['GET'])
def status():
    """Return system status information"""
    return jsonify({
        "status": "operational",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": [
            face_model.name,
            audio_model.name,
            video_model.name,
            multimodal_model.name
        ],
        "system_info": {
            "platform": os.name,
            "python_version": os.sys.version
        }
    })

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PYTHON_API_PORT', 5001))
    
    logger.info(f"Starting SatyaAI Advanced Detection API on port {port}")
    logger.info("Loaded models:")
    logger.info(f"  - Face Analysis: {face_model.name} v{face_model.version}")
    logger.info(f"  - Audio Analysis: {audio_model.name} v{audio_model.version}")
    logger.info(f"  - Video Analysis: {video_model.name} v{video_model.version}")
    logger.info(f"  - Multimodal Analysis: {multimodal_model.name} v{multimodal_model.version}")
    
    app.run(host='0.0.0.0', port=port, debug=True)