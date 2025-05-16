from flask import Flask, request, jsonify, send_file
import os
import sys
import json
import hashlib
import base64
from io import BytesIO
from functools import wraps
from datetime import datetime
import logging

# Import our custom modules
from deepfake_analyzer import (
    analyze_image, analyze_video, analyze_audio, analyze_multimodal,
    analyze_webcam, verify_satyachain, check_darkweb,
    analyze_language_lip_sync, analyze_emotion_conflict
)
from auth_manager import (
    authenticate_user, validate_session, logout, 
    get_active_sessions_count, clean_expired_sessions
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Authentication middleware decorator
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get token from header
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return jsonify({"error": "Authentication required", "code": "auth_required"}), 401
        
        # Extract token
        try:
            session_token = auth_header.split(' ')[1]
        except IndexError:
            return jsonify({"error": "Invalid authorization format", "code": "invalid_auth_format"}), 401
        
        # Validate session
        session_result = validate_session(session_token)
        
        if not session_result["valid"]:
            return jsonify({"error": session_result["message"], "code": "invalid_session"}), 401
        
        # Add user info to request
        request.user = {
            "user_id": session_result["user_id"],
            "username": session_result["username"],
            "is_admin": session_result["is_admin"]
        }
        
        return f(*args, **kwargs)
    
    return decorated_function

# Authentication routes
@app.route('/login', methods=['POST'])
def login_route():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    
    result = authenticate_user(username, password)
    
    if not result["success"]:
        return jsonify({"error": result["message"]}), 401
    
    return jsonify(result), 200

@app.route('/logout', methods=['POST'])
@require_auth
def logout_route():
    auth_header = request.headers.get('Authorization')
    session_token = auth_header.split(' ')[1]
    
    result = logout(session_token)
    
    return jsonify(result), 200 if result["success"] else 400

@app.route('/session', methods=['GET'])
@require_auth
def session_route():
    return jsonify({
        "valid": True,
        "user": request.user
    }), 200

# Analysis routes (all require authentication)
@app.route('/analyze/image', methods=['POST'])
@require_auth
def analyze_image_endpoint():
    # Handle either JSON data (base64) or file upload
    if request.is_json:
        data = request.json
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        try:
            # Clean up base64 data if needed
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Convert base64 to binary
            image_buffer = base64.b64decode(image_data)
            
            # Call the analyzer
            result = analyze_image(image_buffer)
            
            # Add user ID for tracking
            result["user_id"] = request.user["user_id"]
            
            return jsonify(result), 200
        
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return jsonify({"error": f"Error analyzing image: {str(e)}"}), 500
    
    else:
        # Handle file upload
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        try:
            # Read file data
            image_buffer = file.read()
            
            # Call the analyzer
            result = analyze_image(image_buffer)
            
            # Add user ID for tracking
            result["user_id"] = request.user["user_id"]
            
            return jsonify(result), 200
        
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return jsonify({"error": f"Error analyzing image: {str(e)}"}), 500

@app.route('/analyze/video', methods=['POST'])
@require_auth
def analyze_video_endpoint():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"error": "No video selected"}), 400
    
    try:
        # Read file data
        video_buffer = file.read()
        
        # Call the analyzer
        result = analyze_video(video_buffer, file.filename)
        
        # Add user ID for tracking
        result["user_id"] = request.user["user_id"]
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return jsonify({"error": f"Error analyzing video: {str(e)}"}), 500

@app.route('/analyze/audio', methods=['POST'])
@require_auth
def analyze_audio_endpoint():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({"error": "No audio selected"}), 400
    
    try:
        # Read file data
        audio_buffer = file.read()
        
        # Call the analyzer
        result = analyze_audio(audio_buffer, file.filename)
        
        # Add user ID for tracking
        result["user_id"] = request.user["user_id"]
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}")
        return jsonify({"error": f"Error analyzing audio: {str(e)}"}), 500

@app.route('/analyze/multimodal', methods=['POST'])
@require_auth
def analyze_multimodal_endpoint():
    # Check for at least one file
    if len(request.files) == 0:
        return jsonify({"error": "No files provided"}), 400
    
    try:
        # Get files if available
        image_buffer = None
        if 'image' in request.files and request.files['image'].filename:
            image_buffer = request.files['image'].read()
        
        audio_buffer = None
        if 'audio' in request.files and request.files['audio'].filename:
            audio_buffer = request.files['audio'].read()
        
        video_buffer = None
        if 'video' in request.files and request.files['video'].filename:
            video_buffer = request.files['video'].read()
        
        # Check if at least one buffer is present
        if not any([image_buffer, audio_buffer, video_buffer]):
            return jsonify({"error": "No valid files provided"}), 400
        
        # Call the analyzer
        result = analyze_multimodal(image_buffer, audio_buffer, video_buffer)
        
        # Add user ID for tracking
        result["user_id"] = request.user["user_id"]
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error analyzing multimodal data: {str(e)}")
        return jsonify({"error": f"Error analyzing multimodal data: {str(e)}"}), 500

@app.route('/analyze/webcam', methods=['POST'])
@require_auth
def analyze_webcam_endpoint():
    if not request.is_json:
        return jsonify({"error": "Expected JSON data"}), 400
    
    data = request.json
    image_data = data.get('image_data')
    
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400
    
    try:
        # Clean up base64 data if needed
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Call the analyzer
        result = analyze_webcam(image_data)
        
        # Add user ID for tracking
        result["user_id"] = request.user["user_id"]
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error analyzing webcam data: {str(e)}")
        return jsonify({"error": f"Error analyzing webcam data: {str(e)}"}), 500

# Advanced features routes (all require authentication)
@app.route('/verify/blockchain', methods=['POST'])
@require_auth
def verify_blockchain_endpoint():
    if not request.is_json:
        return jsonify({"error": "Expected JSON data"}), 400
    
    data = request.json
    media_hash = data.get('media_hash')
    
    if not media_hash:
        return jsonify({"error": "Media hash is required"}), 400
    
    try:
        result = verify_satyachain(media_hash)
        
        # Add user ID for tracking
        result["user_id"] = request.user["user_id"]
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error verifying on blockchain: {str(e)}")
        return jsonify({"error": f"Error verifying on blockchain: {str(e)}"}), 500

@app.route('/check/darkweb', methods=['POST'])
@require_auth
def check_darkweb_endpoint():
    if not request.is_json:
        return jsonify({"error": "Expected JSON data"}), 400
    
    data = request.json
    media_hash = data.get('media_hash')
    
    if not media_hash:
        return jsonify({"error": "Media hash is required"}), 400
    
    try:
        result = check_darkweb(media_hash)
        
        # Add user ID for tracking
        result["user_id"] = request.user["user_id"]
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error checking darkweb: {str(e)}")
        return jsonify({"error": f"Error checking darkweb: {str(e)}"}), 500

@app.route('/analyze/lip-sync', methods=['POST'])
@require_auth
def analyze_lip_sync_endpoint():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"error": "No video selected"}), 400
    
    # Get language from form data or use default
    language = request.form.get('language', 'english')
    
    try:
        # Read file data
        video_buffer = file.read()
        
        # Call the analyzer
        result = analyze_language_lip_sync(video_buffer, language)
        
        # Add user ID for tracking
        result["user_id"] = request.user["user_id"]
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error analyzing lip sync: {str(e)}")
        return jsonify({"error": f"Error analyzing lip sync: {str(e)}"}), 500

@app.route('/analyze/emotion-conflict', methods=['POST'])
@require_auth
def analyze_emotion_conflict_endpoint():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"error": "No video selected"}), 400
    
    try:
        # Read file data
        video_buffer = file.read()
        
        # Call the analyzer
        result = analyze_emotion_conflict(video_buffer)
        
        # Add user ID for tracking
        result["user_id"] = request.user["user_id"]
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Error analyzing emotion conflict: {str(e)}")
        return jsonify({"error": f"Error analyzing emotion conflict: {str(e)}"}), 500

# Information endpoints
@app.route('/models/info', methods=['GET'])
@require_auth
def models_info_endpoint():
    """Return information about available models"""
    model_info = {
        "image": {
            "name": "SatyaAI Vision Analyzer",
            "version": "2.0",
            "type": "CNN",
            "features": ["Facial forensics", "Metadata analysis", "Artifact detection"]
        },
        "video": {
            "name": "SatyaAI Temporal Analyzer",
            "version": "2.1",
            "type": "LSTM+CNN",
            "features": ["Frame-by-frame analysis", "Temporal consistency", "Motion tracking"]
        },
        "audio": {
            "name": "SatyaAI Audio Analyzer",
            "version": "1.5",
            "type": "Wavenet",
            "features": ["Voice cloning detection", "Frequency analysis", "Acoustic anomalies"]
        },
        "multimodal": {
            "name": "SatyaAI Fusion",
            "version": "3.0",
            "type": "Transformer",
            "features": ["Cross-modal verification", "Integrated analysis", "Conflicting evidence detection"]
        },
        "blockchain": {
            "name": "SatyaChain™",
            "version": "1.0",
            "type": "Distributed Ledger",
            "features": ["Immutable verification", "Proof of authenticity", "Cryptographic signatures"]
        },
        "darkweb": {
            "name": "DarkNet Crawler",
            "version": "1.2",
            "type": "Search Engine",
            "features": ["Dark web scanning", "Media matching", "Origin tracing"]
        },
        "lip_sync": {
            "name": "MultiLingual LipSync",
            "version": "1.4",
            "type": "Acoustic-Visual",
            "features": ["Multi-language support", "Phoneme analysis", "Visual-audio correlation"]
        },
        "emotion": {
            "name": "Emotion Conflict Analyzer",
            "version": "1.0",
            "type": "Multimodal Pattern Matcher",
            "features": ["Face-voice emotion matching", "Temporal pattern analysis", "Incongruence detection"]
        }
    }
    
    return jsonify(model_info), 200

@app.route('/status', methods=['GET'])
def status():
    """Return system status information"""
    # Only return basic info if not authenticated
    active_sessions = get_active_sessions_count()
    basic_status = {
        "status": "running",
        "version": "1.0.0",
        "active_sessions": active_sessions,
        "timestamp": datetime.now().isoformat()
    }
    
    # Return more detailed info if authenticated
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            session_token = auth_header.split(' ')[1]
            session_result = validate_session(session_token)
            
            if session_result["valid"] and session_result.get("is_admin", False):
                # Clean expired sessions
                cleaned = clean_expired_sessions()
                
                # Return detailed status for admin
                return jsonify({
                    **basic_status,
                    "detailed": True,
                    "cleaned_sessions": cleaned,
                    "memory_usage": {
                        "active_sessions_size": sys.getsizeof(json.dumps(active_sessions)),
                        "python_version": sys.version,
                    },
                    "system": {
                        "platform": sys.platform,
                        "cpu_count": os.cpu_count()
                    }
                }), 200
        except:
            pass
    
    return jsonify(basic_status), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ready"}), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)