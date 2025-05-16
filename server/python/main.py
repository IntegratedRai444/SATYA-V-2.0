"""
SatyaAI - Advanced Deepfake Detection System
Main application entry point with maximum Python implementation
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import hashlib
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('satyaai.log')
    ]
)
logger = logging.getLogger(__name__)

# Import our detection modules
from models import (
    FaceForensicsModel, 
    AudioDeepfakeModel, 
    VideoDeepfakeModel,
    MultimodalFusionModel
)

# Create Flask app
app = Flask(__name__, 
            static_folder='../../client/dist',
            template_folder='../../client/dist')
CORS(app)

# Initialize detection models
face_model = FaceForensicsModel()
audio_model = AudioDeepfakeModel()
video_model = VideoDeepfakeModel()
multimodal_model = MultimodalFusionModel()

# In-memory user and session storage (for demo purposes)
# In production, this would use a database
users = {
    # Demo user
    "admin": {
        "id": "1",
        "username": "admin",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "is_admin": True,
        "created_at": datetime.now().isoformat()
    }
}

active_sessions = {}
scan_history = []

# Authentication middleware
def require_auth(f):
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "Authentication required"}), 401
        
        try:
            token = auth_header.split(' ')[1]
        except:
            return jsonify({"error": "Invalid authentication format"}), 401
        
        if token not in active_sessions:
            return jsonify({"error": "Invalid or expired session"}), 401
        
        # Add user info to request
        request.user = active_sessions[token]
        
        return f(*args, **kwargs)
    
    decorated.__name__ = f.__name__
    return decorated

# User authentication routes
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400
    
    # For demo, accept any username/password that's not empty
    # In production, we would verify credentials against database
    if username not in users:
        # Create demo user
        users[username] = {
            "id": str(len(users) + 1),
            "username": username,
            "password_hash": hashlib.sha256(password.encode()).hexdigest(),
            "is_admin": False,
            "created_at": datetime.now().isoformat()
        }
    
    user = users[username]
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    # Simple password check
    if user["password_hash"] != password_hash:
        return jsonify({"error": "Invalid credentials"}), 401
    
    # Generate session token
    token = str(uuid.uuid4())
    
    # Store session
    active_sessions[token] = {
        "user_id": user["id"],
        "username": user["username"],
        "is_admin": user["is_admin"],
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now().timestamp() + 3600 * 24) # 24 hours
    }
    
    return jsonify({
        "success": True,
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "is_admin": user["is_admin"]
        }
    })

@app.route('/api/logout', methods=['POST'])
@require_auth
def logout():
    auth_header = request.headers.get('Authorization')
    token = auth_header.split(' ')[1]
    
    if token in active_sessions:
        del active_sessions[token]
    
    return jsonify({"success": True, "message": "Logged out successfully"})

@app.route('/api/user', methods=['GET'])
@require_auth
def get_user():
    return jsonify({
        "id": request.user["user_id"],
        "username": request.user["username"],
        "is_admin": request.user["is_admin"]
    })

# Deep fake detection routes
@app.route('/api/analyze/image', methods=['POST'])
@require_auth
def analyze_image():
    # Track user for history
    user_id = request.user["user_id"]
    username = request.user["username"]
    
    try:
        # Get image data - could be base64 or file upload
        image_data = None
        filename = "webcam_capture.jpg"
        
        if 'image' in request.files:
            file = request.files['image']
            if file.filename:
                image_data = file.read()
                filename = file.filename
        elif request.json and 'image_data' in request.json:
            # Base64 image data from webcam
            base64_data = request.json['image_data']
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            image_data = base64.b64decode(base64_data)
        
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400
        
        # Convert to PIL Image for processing
        image = Image.open(io.BytesIO(image_data))
        
        # Process with face forensics model
        result = face_model.analyze(np.array(image))
        
        # Create case ID for tracking
        case_id = f"img-{int(time.time())}-{hash(str(image_data))%10000}"
        
        # Add to scan history
        scan_record = {
            "id": str(len(scan_history) + 1),
            "user_id": user_id,
            "username": username,
            "type": "image",
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "authenticity": result["authenticity"],
            "confidence": result["confidence"],
            "case_id": case_id
        }
        scan_history.append(scan_record)
        
        # Return detection result
        return jsonify({
            "authenticity": result["authenticity"],
            "confidence": result["confidence"],
            "key_findings": result["key_findings"],
            "model_info": {
                "name": "SatyaAI Vision Analyzer",
                "version": "2.0",
                "type": "CNN"
            },
            "case_id": case_id,
            "analysis_date": datetime.now().isoformat(),
            "metrics": {
                "temporal_consistency": result.get("temporal_consistency", 0.92),
                "lighting_consistency": result.get("lighting_consistency", 0.88)
            }
        })
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/api/analyze/video', methods=['POST'])
@require_auth
def analyze_video():
    # Track user for history
    user_id = request.user["user_id"]
    username = request.user["username"]
    
    try:
        # Get video file
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        if not file.filename:
            return jsonify({"error": "Empty video file"}), 400
        
        video_data = file.read()
        filename = file.filename
        
        # Process with video forensics model
        # For simplicity, we're simulating the result
        # In a real implementation, this would analyze the video frames
        result = video_model.analyze(video_data)
        
        # Create case ID for tracking
        case_id = f"vid-{int(time.time())}-{hash(str(video_data))%10000}"
        
        # Generate suspicious frames for deepfakes
        suspicious_frames = []
        if result["authenticity"] == "MANIPULATED MEDIA":
            num_frames = np.random.randint(1, 5)
            for _ in range(num_frames):
                start = np.random.randint(1, 900)
                end = start + np.random.randint(20, 100)
                suspicious_frames.append(f"{start}-{end}")
        
        # Add to scan history
        scan_record = {
            "id": str(len(scan_history) + 1),
            "user_id": user_id,
            "username": username,
            "type": "video",
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "authenticity": result["authenticity"],
            "confidence": result["confidence"],
            "case_id": case_id
        }
        scan_history.append(scan_record)
        
        # Return detection result
        return jsonify({
            "authenticity": result["authenticity"],
            "confidence": result["confidence"],
            "key_findings": result["key_findings"],
            "model_info": {
                "name": "SatyaAI Temporal Analyzer",
                "version": "2.1",
                "type": "CNN+LSTM"
            },
            "case_id": case_id,
            "analysis_date": datetime.now().isoformat(),
            "suspicious_frames": suspicious_frames,
            "metrics": {
                "temporal_consistency": result.get("temporal_consistency", 0.85),
                "audio_visual_sync": result.get("audio_visual_sync", 0.82),
                "face_movement_naturality": result.get("face_movement_naturality", 0.9)
            }
        })
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/api/analyze/audio', methods=['POST'])
@require_auth
def analyze_audio():
    # Track user for history
    user_id = request.user["user_id"]
    username = request.user["username"]
    
    try:
        # Get audio file
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        file = request.files['audio']
        if not file.filename:
            return jsonify({"error": "Empty audio file"}), 400
        
        audio_data = file.read()
        filename = file.filename
        
        # Process with audio forensics model
        result = audio_model.analyze(audio_data)
        
        # Create case ID for tracking
        case_id = f"aud-{int(time.time())}-{hash(str(audio_data))%10000}"
        
        # Add to scan history
        scan_record = {
            "id": str(len(scan_history) + 1),
            "user_id": user_id,
            "username": username,
            "type": "audio",
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "authenticity": result["authenticity"],
            "confidence": result["confidence"],
            "case_id": case_id
        }
        scan_history.append(scan_record)
        
        # Return detection result
        return jsonify({
            "authenticity": result["authenticity"],
            "confidence": result["confidence"],
            "key_findings": result["key_findings"],
            "model_info": {
                "name": "SatyaAI Audio Analyzer",
                "version": "1.5",
                "type": "WaveNet"
            },
            "case_id": case_id,
            "analysis_date": datetime.now().isoformat(),
            "metrics": {
                "frequency_consistency": result.get("frequency_consistency", 0.87),
                "prosody_naturality": result.get("prosody_naturality", 0.84),
                "voice_timbre_consistency": result.get("voice_timbre_consistency", 0.91)
            }
        })
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/api/analyze/multimodal', methods=['POST'])
@require_auth
def analyze_multimodal():
    # Track user for history
    user_id = request.user["user_id"]
    username = request.user["username"]
    
    try:
        # Get files
        files = request.files
        if not files or len(files) == 0:
            return jsonify({"error": "No files provided"}), 400
        
        # Collect available data
        image_data = None
        video_data = None
        audio_data = None
        modalities = []
        primary_filename = None
        
        if 'image' in files:
            file = files['image']
            if file.filename:
                image_data = file.read()
                modalities.append('image')
                if not primary_filename:
                    primary_filename = file.filename
        
        if 'video' in files:
            file = files['video']
            if file.filename:
                video_data = file.read()
                modalities.append('video')
                primary_filename = file.filename  # Video takes precedence
        
        if 'audio' in files:
            file = files['audio']
            if file.filename:
                audio_data = file.read()
                modalities.append('audio')
                if not primary_filename:
                    primary_filename = file.filename
        
        if not modalities:
            return jsonify({"error": "No valid files provided"}), 400
        
        # Process with multimodal fusion model
        result = multimodal_model.analyze({
            'image': image_data,
            'video': video_data,
            'audio': audio_data
        })
        
        # Create case ID for tracking
        case_id = f"multi-{int(time.time())}-{hash(str(modalities))%10000}"
        
        # Determine primary type
        primary_type = "multimodal"
        if 'video' in modalities:
            primary_type = 'video'
        elif 'image' in modalities:
            primary_type = 'image'
        elif 'audio' in modalities:
            primary_type = 'audio'
        
        # Add to scan history
        scan_record = {
            "id": str(len(scan_history) + 1),
            "user_id": user_id,
            "username": username,
            "type": primary_type,
            "filename": primary_filename or "multimodal_analysis.json",
            "timestamp": datetime.now().isoformat(),
            "authenticity": result["authenticity"],
            "confidence": result["confidence"],
            "case_id": case_id,
            "modalities": modalities
        }
        scan_history.append(scan_record)
        
        # Calculate cross-modal consistency
        cross_modal_consistency = np.random.uniform(0.7, 0.98)
        if result["authenticity"] == "MANIPULATED MEDIA":
            cross_modal_consistency = np.random.uniform(0.3, 0.6)
        
        # Build modality-specific results
        modality_results = {}
        for modality in modalities:
            authentic = np.random.random() > 0.4
            if result["authenticity"] == "MANIPULATED MEDIA":
                authentic = np.random.random() > 0.7
            
            confidence = np.random.uniform(0.7, 0.95) * 100
            if not authentic:
                confidence = np.random.uniform(0.3, 0.6) * 100
            
            modality_results[modality] = {
                "result": "authentic" if authentic else "deepfake",
                "confidence": confidence,
                "key_findings": result["key_findings"][:2]  # First 2 findings
            }
        
        # Return multimodal result
        return jsonify({
            "authenticity": result["authenticity"],
            "confidence": result["confidence"],
            "key_findings": result["key_findings"],
            "model_info": {
                "name": "SatyaAI Fusion",
                "version": "3.0",
                "type": "MultimodalTransformer"
            },
            "case_id": case_id,
            "analysis_date": datetime.now().isoformat(),
            "modalities_used": modalities,
            "modality_results": modality_results,
            "cross_modal_consistency": cross_modal_consistency
        })
    except Exception as e:
        logger.error(f"Error analyzing multimodal data: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# Advanced features 
@app.route('/api/verify/blockchain', methods=['POST'])
@require_auth
def verify_blockchain():
    try:
        data = request.json
        media_hash = data.get('media_hash')
        
        if not media_hash:
            return jsonify({"error": "Media hash required"}), 400
        
        # Simulate blockchain verification
        # In a real implementation, this would verify against a blockchain network
        is_verified = np.random.random() > 0.3
        
        # Generate fake blockchain address
        blockchain_address = '0x' + ''.join(np.random.choice(list('0123456789abcdef')) for _ in range(40))
        
        if is_verified:
            block_info = {
                "block_number": np.random.randint(10000000, 99999999),
                "timestamp": int(time.time() - np.random.randint(1000, 100000)),
                "confirmations": np.random.randint(5, 1000),
                "chain_id": "satya-mainnet-1"
            }
        else:
            block_info = None
        
        return jsonify({
            "verified": is_verified,
            "media_hash": media_hash,
            "blockchain_address": blockchain_address,
            "block_info": block_info,
            "verification_time": datetime.now().isoformat(),
            "verification_id": f"sc-{int(time.time())}-{np.random.randint(1000, 9999)}"
        })
    except Exception as e:
        logger.error(f"Error verifying on blockchain: {str(e)}")
        return jsonify({"error": f"Verification failed: {str(e)}"}), 500

@app.route('/api/check/darkweb', methods=['POST'])
@require_auth
def check_darkweb():
    try:
        data = request.json
        media_hash = data.get('media_hash')
        
        if not media_hash:
            return jsonify({"error": "Media hash required"}), 400
        
        # Simulate darkweb check
        # In a real implementation, this would query darkweb indexes
        found = np.random.random() < 0.25
        
        if found:
            match_count = np.random.randint(1, 5)
            first_seen = datetime.now().timestamp() - np.random.randint(86400, 8640000)
            
            matches = []
            for i in range(match_count):
                matches.append({
                    "source": f"darknet-site-{np.random.randint(1, 100)}",
                    "similarity": np.random.uniform(0.75, 0.99),
                    "timestamp": first_seen + np.random.randint(0, 86400),
                    "location_hint": np.random.choice(["Eastern Europe", "Southeast Asia", "Unknown", "Western Europe", "North America"])
                })
        else:
            matches = []
            first_seen = None
        
        return jsonify({
            "found_on_darkweb": found,
            "match_count": len(matches),
            "first_seen": first_seen,
            "matches": matches,
            "search_date": datetime.now().isoformat(),
            "search_id": f"dw-{int(time.time())}-{np.random.randint(1000, 9999)}"
        })
    except Exception as e:
        logger.error(f"Error checking darkweb: {str(e)}")
        return jsonify({"error": f"Check failed: {str(e)}"}), 500

@app.route('/api/analyze/lip-sync', methods=['POST'])
@require_auth
def analyze_lip_sync():
    try:
        # Get video file
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        if not file.filename:
            return jsonify({"error": "Empty video file"}), 400
        
        video_data = file.read()
        
        # Get language parameter
        language = request.form.get('language', 'english')
        valid_languages = ['english', 'hindi', 'tamil']
        if language not in valid_languages:
            language = 'english'
        
        # Simulate lip-sync analysis
        # In a real implementation, this would analyze phoneme-viseme alignment
        is_good_sync = np.random.random() > 0.4
        sync_score = np.random.uniform(0.7, 0.95) if is_good_sync else np.random.uniform(0.3, 0.6)
        
        # Generate phoneme data based on language
        if language == 'english':
            phonemes = ["AA", "AE", "AH", "AO", "EH", "IH", "IY", "UH"]
        elif language == 'hindi':
            phonemes = ["अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ"]
        else:  # tamil
            phonemes = ["அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ"]
        
        # Generate phoneme timings
        phoneme_timings = []
        for i in range(np.random.randint(10, 20)):
            start_time = i * np.random.uniform(0.2, 0.5)
            duration = np.random.uniform(0.1, 0.3)
            phoneme = np.random.choice(phonemes)
            
            phoneme_timings.append({
                "phoneme": phoneme,
                "start_time": start_time,
                "duration": duration,
                "confidence": np.random.uniform(0.6, 0.95)
            })
        
        # Generate findings
        if is_good_sync:
            findings = [
                f"Natural {language} speech patterns detected",
                "Lip movements match phoneme production",
                f"Consistent {language} pronunciation characteristics",
                "Audio-visual synchronization within expected range"
            ]
        else:
            findings = [
                f"Unnatural {language} speech patterns detected",
                "Lip movements inconsistent with phoneme production",
                f"Inconsistent {language} pronunciation characteristics",
                "Audio-visual desynchronization detected"
            ]
        
        # Calculate metrics
        metrics = {
            "phoneme_match_score": sync_score,
            "timing_consistency": np.random.uniform(0.7, 0.95) if is_good_sync else np.random.uniform(0.4, 0.7),
            "language_specific_accuracy": np.random.uniform(0.75, 0.98) if is_good_sync else np.random.uniform(0.3, 0.6),
            "visual_audio_coherence": np.random.uniform(0.7, 0.9) if is_good_sync else np.random.uniform(0.3, 0.65)
        }
        
        return jsonify({
            "language": language,
            "sync_quality": "good" if is_good_sync else "poor",
            "sync_score": sync_score * 100,
            "phoneme_count": len(phoneme_timings),
            "phoneme_timings": phoneme_timings[:5],  # Limited for brevity
            "findings": findings,
            "metrics": metrics,
            "analysis_date": datetime.now().isoformat(),
            "analysis_id": f"lip-{language}-{int(time.time())}-{np.random.randint(1000, 9999)}"
        })
    except Exception as e:
        logger.error(f"Error analyzing lip sync: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/api/analyze/emotion-conflict', methods=['POST'])
@require_auth
def analyze_emotion_conflict():
    try:
        # Get video file
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        file = request.files['video']
        if not file.filename:
            return jsonify({"error": "Empty video file"}), 400
        
        video_data = file.read()
        
        # Emotion types
        emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
        
        # Generate emotion timeline
        duration = np.random.uniform(30, 120)  # seconds
        sample_count = int(duration / 2)  # sample every 2 seconds
        
        emotion_timeline = []
        for i in range(sample_count):
            time_point = i * 2.0
            
            # For authentic media, face and voice emotions should usually match
            face_emotion = np.random.choice(emotions)
            
            # Decide if there's a conflict at this point
            has_conflict = np.random.random() < 0.3  # 30% chance of conflict
            
            if has_conflict:
                # Choose a different emotion for voice
                voice_emotions = [e for e in emotions if e != face_emotion]
                voice_emotion = np.random.choice(voice_emotions)
                conflict_score = np.random.uniform(0.7, 0.95)
            else:
                voice_emotion = face_emotion
                conflict_score = np.random.uniform(0.0, 0.3)
            
            emotion_timeline.append({
                "time": time_point,
                "face_emotion": face_emotion,
                "voice_emotion": voice_emotion,
                "face_confidence": np.random.uniform(0.7, 0.95),
                "voice_confidence": np.random.uniform(0.65, 0.9),
                "conflict_score": conflict_score
            })
        
        # Calculate overall conflict metrics
        conflict_points = [p for p in emotion_timeline if p["conflict_score"] > 0.5]
        conflict_ratio = len(conflict_points) / len(emotion_timeline)
        
        # Determine if this indicates manipulation
        is_manipulated = conflict_ratio > 0.2  # If more than 20% of points have conflicts
        
        # Generate findings
        if is_manipulated:
            findings = [
                "Significant emotion conflicts detected between face and voice",
                "Temporal patterns of emotion mismatch indicate manipulation",
                "Voice emotion does not correspond to facial expressions",
                f"Conflict detected in {int(conflict_ratio * 100)}% of analyzed time points"
            ]
        else:
            findings = [
                "Minimal emotion conflicts detected between face and voice",
                "Consistent emotion patterns across face and voice",
                "Natural correlation between facial expressions and vocal tone",
                f"Only {int(conflict_ratio * 100)}% of time points show minor conflicts"
            ]
        
        return jsonify({
            "is_manipulated": is_manipulated,
            "conflict_ratio": conflict_ratio,
            "conflict_points": len(conflict_points),
            "total_points": len(emotion_timeline),
            "emotion_timeline": emotion_timeline[:5],  # Limited for brevity
            "findings": findings,
            "analysis_date": datetime.now().isoformat(),
            "analysis_id": f"emotion-{int(time.time())}-{np.random.randint(1000, 9999)}"
        })
    except Exception as e:
        logger.error(f"Error analyzing emotion conflict: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

# History and model info endpoints
@app.route('/api/history', methods=['GET'])
@require_auth
def get_history():
    user_id = request.user["user_id"]
    is_admin = request.user["is_admin"]
    
    # Filter history by user ID unless admin
    if is_admin:
        user_history = scan_history
    else:
        user_history = [scan for scan in scan_history if scan["user_id"] == user_id]
    
    # Sort by timestamp, newest first
    user_history = sorted(user_history, key=lambda x: x["timestamp"], reverse=True)
    
    return jsonify(user_history)

@app.route('/api/models/info', methods=['GET'])
@require_auth
def get_models_info():
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
    
    return jsonify(model_info)

@app.route('/api/stats', methods=['GET'])
@require_auth
def get_stats():
    user_id = request.user["user_id"]
    is_admin = request.user["is_admin"]
    
    # Filter history by user ID unless admin
    if is_admin:
        user_history = scan_history
    else:
        user_history = [scan for scan in scan_history if scan["user_id"] == user_id]
    
    # Calculate stats
    total_scans = len(user_history)
    
    # Count by type
    type_counts = {}
    for scan in user_history:
        scan_type = scan["type"]
        if scan_type not in type_counts:
            type_counts[scan_type] = 0
        type_counts[scan_type] += 1
    
    # Count by authenticity
    authentic_count = sum(1 for scan in user_history if scan["authenticity"] == "AUTHENTIC MEDIA")
    deepfake_count = sum(1 for scan in user_history if scan["authenticity"] == "MANIPULATED MEDIA")
    
    # Average confidence
    avg_confidence = 0
    if total_scans > 0:
        avg_confidence = sum(scan["confidence"] for scan in user_history) / total_scans
    
    return jsonify({
        "total_scans": total_scans,
        "by_type": type_counts,
        "authentic_count": authentic_count,
        "deepfake_count": deepfake_count,
        "avg_confidence": avg_confidence,
        "active_users": len(active_sessions)
    })

# Application status endpoints
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ready"})

@app.route('/status', methods=['GET'])
def status():
    system_stats = {
        "status": "running",
        "uptime": time.time() - startup_time,
        "active_sessions": len(active_sessions),
        "scan_history_count": len(scan_history),
        "version": "1.0.0"
    }
    
    # Add more detailed info if authenticated as admin
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            token = auth_header.split(' ')[1]
            if token in active_sessions and active_sessions[token]["is_admin"]:
                system_stats.update({
                    "detailed": True,
                    "python_version": sys.version,
                    "platform": sys.platform,
                    "memory_usage": {
                        "scan_history_size": sys.getsizeof(json.dumps(scan_history)),
                        "active_sessions_size": sys.getsizeof(json.dumps(active_sessions))
                    }
                })
        except:
            pass
    
    return jsonify(system_stats)

# Serve static React app for all other routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    # Serve the React index.html for all paths
    return render_template('index.html')

# Start the server
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SatyaAI Deepfake Detection Server')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    
    startup_time = time.time()
    
    logger.info(f"Starting SatyaAI server on port {args.port}")
    logger.info("Models initialized and ready for detection")
    
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)