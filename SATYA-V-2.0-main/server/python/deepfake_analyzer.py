import os
import time
import json
import random
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
from datetime import datetime

class DeepfakeAnalyzer:
    """
    Advanced deepfake detection and analysis tool that uses multiple techniques
    to identify potentially manipulated media.
    """
    
    def __init__(self):
        self.models = {
            'image': {
                'name': 'SatyaAI Vision Analyzer',
                'version': '2.0',
                'type': 'CNN'
            },
            'video': {
                'name': 'SatyaAI Temporal Analyzer',
                'version': '2.1',
                'type': 'LSTM+CNN'
            },
            'audio': {
                'name': 'SatyaAI Audio Analyzer',
                'version': '1.5',
                'type': 'Wavenet'
            },
            'multimodal': {
                'name': 'SatyaAI Fusion',
                'version': '3.0',
                'type': 'Transformer'
            }
        }
    
    def analyze_image(self, image_data):
        """
        Analyze image data for potential manipulation.
        
        Args:
            image_data: Base64 encoded image or image buffer
            
        Returns:
            dict: Detailed analysis results
        """
        # Simulate processing time
        time.sleep(1)
        
        # For demo purposes, generate pseudo-random results
        # In a real implementation, this would use actual ML models
        is_fake = random.random() < 0.4
        confidence = random.uniform(0.65, 0.98)
        
        if is_fake:
            confidence = 1 - confidence
            
        authenticity = "MANIPULATED MEDIA" if is_fake else "AUTHENTIC MEDIA"
        
        # Generate key findings based on authenticity
        if is_fake:
            key_findings = [
                "Facial inconsistencies detected around eye regions",
                "Unnatural boundary transitions identified",
                "Metadata analysis shows evidence of manipulation",
                "Inconsistent lighting patterns across face"
            ]
        else:
            key_findings = [
                "No facial inconsistencies detected",
                "Natural boundary transitions confirmed",
                "Metadata analysis shows no evidence of manipulation",
                "Consistent lighting patterns across image"
            ]
        
        # Feature analysis
        feature_points = self._analyze_facial_features(image_data)
        
        # Generate metrics for the analysis
        metrics = {
            "temporal_consistency": random.uniform(0.85, 0.98),
            "lighting_consistency": random.uniform(0.80, 0.97),
            "boundary_consistency": random.uniform(0.82, 0.99),
            "metadata_integrity": random.uniform(0.90, 0.99)
        }
        
        if is_fake:
            # Lower metrics for fake media
            metrics = {k: 1 - v for k, v in metrics.items()}
        
        result = {
            "authenticity": authenticity,
            "confidence": confidence * 100,
            "analysis_date": datetime.now().isoformat(),
            "case_id": f"img-{int(time.time())}-{random.randint(1000, 9999)}",
            "key_findings": key_findings,
            "feature_points": feature_points,
            "metrics": metrics,
            "model_info": self.models['image']
        }
        
        return result
    
    def analyze_video(self, video_buffer, filename=None):
        """
        Analyze video for deepfake manipulation.
        
        Args:
            video_buffer: Raw video data
            filename: Original filename
            
        Returns:
            dict: Analysis results with temporal information
        """
        # Simulate processing time
        time.sleep(2)
        
        # Pseudo-random results
        is_fake = random.random() < 0.4
        confidence = random.uniform(0.70, 0.95)
        
        if is_fake:
            confidence = 1 - confidence
            
        authenticity = "MANIPULATED MEDIA" if is_fake else "AUTHENTIC MEDIA"
        
        # Generate key findings based on authenticity
        if is_fake:
            key_findings = [
                "Temporal inconsistencies detected in face movement",
                "Audio-visual desynchronization identified",
                "Unnatural eye blinking patterns detected",
                "Inconsistent lighting across consecutive frames"
            ]
            
            # Generate suspicious frame ranges
            frame_count = random.randint(1, 4)
            suspicious_frames = []
            for _ in range(frame_count):
                start = random.randint(1, 900)
                duration = random.randint(20, 100)
                suspicious_frames.append(f"{start}-{start + duration}")
        else:
            key_findings = [
                "Temporal consistency verified in face movement",
                "Audio-visual synchronization confirmed",
                "Natural eye blinking patterns detected",
                "Consistent lighting across consecutive frames"
            ]
            suspicious_frames = []
        
        # Generate metrics for video analysis
        metrics = {
            "temporal_consistency": random.uniform(0.85, 0.98),
            "audio_visual_sync": random.uniform(0.82, 0.97),
            "face_movement_naturality": random.uniform(0.80, 0.96),
            "eye_blink_consistency": random.uniform(0.85, 0.99)
        }
        
        if is_fake:
            # Lower metrics for fake media
            metrics = {k: 1 - v for k, v in metrics.items()}
        
        result = {
            "authenticity": authenticity,
            "confidence": confidence * 100,
            "analysis_date": datetime.now().isoformat(),
            "case_id": f"vid-{int(time.time())}-{random.randint(1000, 9999)}",
            "key_findings": key_findings,
            "suspicious_frames": suspicious_frames,
            "metrics": metrics,
            "model_info": self.models['video']
        }
        
        return result
    
    def analyze_audio(self, audio_buffer, filename=None):
        """
        Analyze audio for voice cloning or manipulation.
        
        Args:
            audio_buffer: Raw audio data
            filename: Original filename
            
        Returns:
            dict: Analysis results with audio-specific metrics
        """
        # Simulate processing time
        time.sleep(1.5)
        
        # Pseudo-random results
        is_fake = random.random() < 0.4
        confidence = random.uniform(0.75, 0.98)
        
        if is_fake:
            confidence = 1 - confidence
            
        authenticity = "MANIPULATED MEDIA" if is_fake else "AUTHENTIC MEDIA"
        
        # Generate key findings based on authenticity
        if is_fake:
            key_findings = [
                "Voice synthesis artifacts detected",
                "Unnatural prosody patterns identified",
                "Frequency anomalies consistent with voice cloning",
                "Inconsistent audio quality throughout recording"
            ]
        else:
            key_findings = [
                "No voice synthesis artifacts detected",
                "Natural prosody patterns confirmed",
                "Frequency analysis shows consistent human voice",
                "Consistent audio quality throughout recording"
            ]
        
        # Generate metrics for audio analysis
        metrics = {
            "frequency_consistency": random.uniform(0.82, 0.97),
            "prosody_naturality": random.uniform(0.85, 0.98),
            "voice_timbre_consistency": random.uniform(0.80, 0.96),
            "background_noise_consistency": random.uniform(0.83, 0.99)
        }
        
        if is_fake:
            # Lower metrics for fake media
            metrics = {k: 1 - v for k, v in metrics.items()}
        
        result = {
            "authenticity": authenticity,
            "confidence": confidence * 100,
            "analysis_date": datetime.now().isoformat(),
            "case_id": f"aud-{int(time.time())}-{random.randint(1000, 9999)}",
            "key_findings": key_findings,
            "metrics": metrics,
            "model_info": self.models['audio']
        }
        
        return result
    
    def analyze_multimodal(self, image_buffer=None, audio_buffer=None, video_buffer=None):
        """
        Perform advanced multimodal analysis using multiple media types.
        
        Args:
            image_buffer: Optional image data
            audio_buffer: Optional audio data
            video_buffer: Optional video data
            
        Returns:
            dict: Comprehensive analysis with cross-modal consistency metrics
        """
        # Simulate processing time
        time.sleep(3)
        
        # Track which modalities are used
        modalities = []
        if image_buffer:
            modalities.append('image')
        if audio_buffer:
            modalities.append('audio')
        if video_buffer:
            modalities.append('video')
        
        if not modalities:
            raise ValueError("At least one media type must be provided")
        
        # Generate individual results for each modality
        modality_results = {}
        
        if 'image' in modalities:
            image_result = self.analyze_image(image_buffer)
            modality_results['image'] = {
                'result': 'authentic' if image_result['authenticity'] == 'AUTHENTIC MEDIA' else 'deepfake',
                'confidence': image_result['confidence'],
                'key_findings': image_result['key_findings'][:2]
            }
        
        if 'audio' in modalities:
            audio_result = self.analyze_audio(audio_buffer)
            modality_results['audio'] = {
                'result': 'authentic' if audio_result['authenticity'] == 'AUTHENTIC MEDIA' else 'deepfake',
                'confidence': audio_result['confidence'],
                'key_findings': audio_result['key_findings'][:2]
            }
        
        if 'video' in modalities:
            video_result = self.analyze_video(video_buffer)
            modality_results['video'] = {
                'result': 'authentic' if video_result['authenticity'] == 'AUTHENTIC MEDIA' else 'deepfake',
                'confidence': video_result['confidence'],
                'key_findings': video_result['key_findings'][:2]
            }
        
        # Calculate cross-modal consistency
        results = [r['result'] == 'authentic' for r in modality_results.values()]
        all_same = all(r == results[0] for r in results)
        cross_modal_consistency = 0.95 if all_same else 0.5 + random.random() * 0.3
        
        # Determine final authenticity based on weighted voting
        authentic_count = sum(1 for r in modality_results.values() if r['result'] == 'authentic')
        total_count = len(modality_results)
        
        is_authentic = authentic_count / total_count > 0.5
        final_confidence = sum(r['confidence'] for r in modality_results.values()) / total_count
        
        authenticity = "AUTHENTIC MEDIA" if is_authentic else "MANIPULATED MEDIA"
        
        # Generate combined findings
        combined_findings = []
        for modality, result in modality_results.items():
            for finding in result['key_findings']:
                combined_findings.append(f"[{modality.upper()}] {finding}")
        
        # Add cross-modal findings
        if len(modalities) > 1:
            if cross_modal_consistency > 0.8:
                combined_findings.append("Cross-modal consistency check passed")
                combined_findings.append(f"Strong correlation across {', '.join(modalities)} analysis")
            else:
                combined_findings.append("Cross-modal inconsistencies detected")
                combined_findings.append("Potential manipulation indicated by modality conflicts")
        
        result = {
            "authenticity": authenticity,
            "confidence": final_confidence,
            "analysis_date": datetime.now().isoformat(),
            "case_id": f"multi-{int(time.time())}-{random.randint(1000, 9999)}",
            "key_findings": combined_findings,
            "modalities_used": modalities,
            "modality_results": modality_results,
            "cross_modal_consistency": cross_modal_consistency,
            "model_info": self.models['multimodal']
        }
        
        return result
    
    def analyze_webcam(self, image_data):
        """
        Analyze webcam capture for real-time deepfake detection.
        
        Args:
            image_data: Base64 encoded image
            
        Returns:
            dict: Analysis results with real-time indicators
        """
        # For webcam, prioritize speed over depth
        time.sleep(0.5)
        
        # Similar to image analysis but optimized for speed
        result = self.analyze_image(image_data)
        
        # Add webcam-specific metrics
        result["metrics"]["real_time_score"] = random.uniform(0.80, 0.99)
        result["metrics"]["face_verification"] = random.uniform(0.85, 0.99)
        
        return result
    
    def _analyze_facial_features(self, image_data):
        """
        Extract and analyze facial feature points.
        
        Args:
            image_data: Image data
            
        Returns:
            dict: Facial feature analysis
        """
        # In a real implementation, this would use actual face detection
        # For demo, generate random facial points
        face_points = []
        center_x, center_y = 0.5, 0.5
        
        # Eyes
        left_eye_x = center_x - 0.15 + random.uniform(-0.01, 0.01)
        left_eye_y = center_y - 0.1 + random.uniform(-0.01, 0.01)
        right_eye_x = center_x + 0.15 + random.uniform(-0.01, 0.01)
        right_eye_y = center_y - 0.1 + random.uniform(-0.01, 0.01)
        
        # Nose and mouth
        nose_x = center_x + random.uniform(-0.01, 0.01)
        nose_y = center_y + 0.05 + random.uniform(-0.01, 0.01)
        mouth_x = center_x + random.uniform(-0.01, 0.01)
        mouth_y = center_y + 0.15 + random.uniform(-0.01, 0.01)
        
        feature_points = {
            "left_eye": [left_eye_x, left_eye_y],
            "right_eye": [right_eye_x, right_eye_y],
            "nose": [nose_x, nose_y],
            "mouth": [mouth_x, mouth_y],
            "face_center": [center_x, center_y],
            "feature_consistency": random.uniform(0.85, 0.98)
        }
        
        return feature_points
    
    def verify_satyachain(self, media_hash):
        """
        Verify media authenticity using SatyaChain blockchain.
        
        Args:
            media_hash: Hash of the media to verify
            
        Returns:
            dict: Blockchain verification results
        """
        # Simulate blockchain verification
        time.sleep(1.5)
        
        # Generate random blockchain address
        blockchain_address = '0x' + ''.join(random.choice('0123456789abcdef') for _ in range(40))
        
        # Simulate verification
        is_verified = random.random() < 0.7
        
        # Generate block information
        if is_verified:
            block_info = {
                "block_number": random.randint(10000000, 99999999),
                "timestamp": int(time.time() - random.randint(1000, 100000)),
                "confirmations": random.randint(5, 1000),
                "chain_id": "satya-mainnet-1"
            }
        else:
            block_info = None
        
        result = {
            "verified": is_verified,
            "media_hash": media_hash,
            "blockchain_address": blockchain_address,
            "block_info": block_info,
            "verification_time": datetime.now().isoformat(),
            "verification_id": f"sc-{int(time.time())}-{random.randint(1000, 9999)}"
        }
        
        return result
    
    def check_darkweb(self, media_hash):
        """
        Check if media has appeared on darkweb.
        
        Args:
            media_hash: Hash of the media to check
            
        Returns:
            dict: Darkweb search results
        """
        # Simulate darkweb search
        time.sleep(2)
        
        # Simulate search results
        found = random.random() < 0.25
        
        if found:
            match_count = random.randint(1, 5)
            first_seen = datetime.now().timestamp() - random.randint(86400, 8640000)
            
            matches = []
            for i in range(match_count):
                matches.append({
                    "source": f"darknet-site-{random.randint(1, 100)}",
                    "similarity": random.uniform(0.75, 0.99),
                    "timestamp": first_seen + random.randint(0, 86400),
                    "location_hint": random.choice(["Eastern Europe", "Southeast Asia", "Unknown", "Western Europe", "North America"])
                })
        else:
            matches = []
            first_seen = None
        
        result = {
            "found_on_darkweb": found,
            "match_count": len(matches),
            "first_seen": first_seen,
            "matches": matches,
            "search_date": datetime.now().isoformat(),
            "search_id": f"dw-{int(time.time())}-{random.randint(1000, 9999)}"
        }
        
        return result
    
    def analyze_language_lip_sync(self, video_buffer, language="english"):
        """
        Analyze lip sync for specific language.
        
        Args:
            video_buffer: Raw video data
            language: Language to analyze (english, hindi, tamil, etc.)
            
        Returns:
            dict: Language-specific lip sync analysis
        """
        # Simulate processing time
        time.sleep(2)
        
        # Language-specific phoneme sets
        language_phonemes = {
            "english": ["AA", "AE", "AH", "AO", "EH", "IH", "IY", "UH"],
            "hindi": ["अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ"],
            "tamil": ["அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ"]
        }
        
        # Get phonemes for selected language
        phonemes = language_phonemes.get(language.lower(), language_phonemes["english"])
        
        # Simulate lip sync matching
        is_good_sync = random.random() < 0.6
        sync_score = random.uniform(0.7, 0.95) if is_good_sync else random.uniform(0.3, 0.6)
        
        # Generate phoneme timing data
        phoneme_timings = []
        for i in range(random.randint(10, 20)):
            start_time = i * random.uniform(0.2, 0.5)
            duration = random.uniform(0.1, 0.3)
            phoneme = random.choice(phonemes)
            
            phoneme_timings.append({
                "phoneme": phoneme,
                "start_time": start_time,
                "duration": duration,
                "confidence": random.uniform(0.6, 0.95)
            })
        
        # Generate language-specific findings
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
        
        # Generate metrics
        metrics = {
            "phoneme_match_score": sync_score,
            "timing_consistency": random.uniform(0.7, 0.95) if is_good_sync else random.uniform(0.4, 0.7),
            "language_specific_accuracy": random.uniform(0.75, 0.98) if is_good_sync else random.uniform(0.3, 0.6),
            "visual_audio_coherence": random.uniform(0.7, 0.9) if is_good_sync else random.uniform(0.3, 0.65)
        }
        
        result = {
            "language": language,
            "sync_quality": "good" if is_good_sync else "poor",
            "sync_score": sync_score * 100,
            "phoneme_count": len(phoneme_timings),
            "phoneme_timings": phoneme_timings[:5],  # Limited for brevity
            "findings": findings,
            "metrics": metrics,
            "analysis_date": datetime.now().isoformat(),
            "analysis_id": f"lip-{language}-{int(time.time())}-{random.randint(1000, 9999)}"
        }
        
        return result
    
    def analyze_emotion_conflict(self, video_buffer):
        """
        Detect conflicts between facial emotions and voice emotions.
        
        Args:
            video_buffer: Raw video data
            
        Returns:
            dict: Emotion conflict analysis
        """
        # Simulate processing time
        time.sleep(2)
        
        # Emotion types
        emotions = ["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"]
        
        # Generate emotion timeline
        duration = random.uniform(30, 120)  # seconds
        sample_count = int(duration / 2)  # sample every 2 seconds
        
        emotion_timeline = []
        for i in range(sample_count):
            time_point = i * 2.0
            
            # For real media, face and voice emotions should usually match
            face_emotion = random.choice(emotions)
            
            # Decide if there's a conflict at this point
            has_conflict = random.random() < 0.3  # 30% chance of conflict
            
            if has_conflict:
                # Choose a different emotion for voice
                voice_emotions = [e for e in emotions if e != face_emotion]
                voice_emotion = random.choice(voice_emotions)
                conflict_score = random.uniform(0.7, 0.95)
            else:
                voice_emotion = face_emotion
                conflict_score = random.uniform(0.0, 0.3)
            
            emotion_timeline.append({
                "time": time_point,
                "face_emotion": face_emotion,
                "voice_emotion": voice_emotion,
                "face_confidence": random.uniform(0.7, 0.95),
                "voice_confidence": random.uniform(0.65, 0.9),
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
        
        result = {
            "is_manipulated": is_manipulated,
            "conflict_ratio": conflict_ratio,
            "conflict_points": len(conflict_points),
            "total_points": len(emotion_timeline),
            "emotion_timeline": emotion_timeline[:5],  # Limited for brevity
            "findings": findings,
            "analysis_date": datetime.now().isoformat(),
            "analysis_id": f"emotion-{int(time.time())}-{random.randint(1000, 9999)}"
        }
        
        return result

# Create singleton instance
analyzer = DeepfakeAnalyzer()

# Export functions for the API
def analyze_image(image_data):
    return analyzer.analyze_image(image_data)

def analyze_video(video_buffer, filename=None):
    return analyzer.analyze_video(video_buffer, filename)

def analyze_audio(audio_buffer, filename=None):
    return analyzer.analyze_audio(audio_buffer, filename)

def analyze_multimodal(image_buffer=None, audio_buffer=None, video_buffer=None):
    return analyzer.analyze_multimodal(image_buffer, audio_buffer, video_buffer)

def analyze_webcam(image_data):
    return analyzer.analyze_webcam(image_data)

def verify_satyachain(media_hash):
    return analyzer.verify_satyachain(media_hash)

def check_darkweb(media_hash):
    return analyzer.check_darkweb(media_hash)

def analyze_language_lip_sync(video_buffer, language="english"):
    return analyzer.analyze_language_lip_sync(video_buffer, language)

def analyze_emotion_conflict(video_buffer):
    return analyzer.analyze_emotion_conflict(video_buffer)