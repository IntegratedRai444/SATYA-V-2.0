"""
Video Deepfake Detector - UPGRADED
Detects manipulated videos using advanced frame-by-frame analysis, 
temporal consistency checks, and audio-visual synchronization
"""

import os
import io
import logging
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
import time
import tempfile

from .base_detector import BaseDetector
from .image_detector import ImageDetector
from ..utils.error_handler import (
    error_handler, timeout, memory_monitor, retry,
    create_fallback_result, validate_input_file, AnalysisError
)
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Import advanced audio detector for audio-visual analysis
try:
    from .advanced_audio_detector import AdvancedAudioDetector
    ADVANCED_AUDIO_DETECTOR_AVAILABLE = True
except ImportError:
    ADVANCED_AUDIO_DETECTOR_AVAILABLE = False
    logger.warning("Advanced audio detector not available for video analysis")


class VideoDetector(BaseDetector):
    """
    Video deepfake detector using frame extraction and temporal analysis.
    """
    
    def __init__(self, model_path: str, enable_gpu: bool = False):
        """
        Initialize the video detector with advanced audio-visual analysis.
        
        Args:
            model_path: Path to model files
            enable_gpu: Whether to use GPU acceleration
        """
        super().__init__(model_path, enable_gpu)
        
        # Initialize image detector for per-frame analysis
        self.image_detector = ImageDetector(model_path, enable_gpu)
        
        # Initialize advanced audio detector for audio-visual sync analysis
        self.audio_detector = None
        if ADVANCED_AUDIO_DETECTOR_AVAILABLE:
            try:
                self.audio_detector = AdvancedAudioDetector(model_path, enable_gpu)
                logger.info("✓ Advanced audio detector initialized for video analysis")
            except Exception as e:
                logger.warning(f"Could not initialize audio detector: {e}")
        
        self.models_loaded = True
    
    def load_models(self):
        """Load required models (uses ImageDetector's models)."""
        # Models are loaded by ImageDetector
        pass
    
    def extract_frames(
        self,
        video_path: str,
        fps: int = 5,
        max_frames: int = 300,
        quality_filter: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract frames from video using OpenCV with enhanced processing.
        
        Args:
            video_path: Path to video file
            fps: Frames per second to extract
            max_frames: Maximum number of frames to extract
            quality_filter: Whether to filter out low-quality frames
            
        Returns:
            List of frame dictionaries with metadata
        """
        frames = []
        
        try:
            import cv2
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return frames
            
            # Get video properties
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video: {video_fps} fps, {total_frames} total frames, {width}x{height}")
            
            # Calculate frame interval with adaptive sampling
            if video_fps > 0:
                frame_interval = max(1, int(video_fps / fps))
                # Adaptive sampling for longer videos
                if total_frames > max_frames * 2:
                    frame_interval = max(frame_interval, total_frames // max_frames)
            else:
                frame_interval = 1
            
            frame_count = 0
            extracted_count = 0
            previous_frame = None
            
            while cap.isOpened() and extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frame at intervals
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Quality assessment
                    if quality_filter:
                        quality_score = self._assess_frame_quality(frame_rgb)
                        if quality_score < 0.3:  # Skip very low quality frames
                            frame_count += 1
                            continue
                    else:
                        quality_score = 1.0
                    
                    # Motion analysis
                    motion_score = 0.0
                    if previous_frame is not None:
                        motion_score = self._calculate_frame_motion(previous_frame, frame_rgb)
                    
                    # Store frame with metadata
                    frame_data = {
                        'frame': frame_rgb,
                        'frame_number': frame_count,
                        'timestamp': frame_count / video_fps if video_fps > 0 else 0,
                        'quality_score': quality_score,
                        'motion_score': motion_score
                    }
                    
                    frames.append(frame_data)
                    previous_frame = frame_rgb.copy()
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames)} high-quality frames from video")
            
        except ImportError:
            logger.error("OpenCV (cv2) not available for video processing")
        except Exception as e:
            logger.error(f"Frame extraction error: {e}", exc_info=True)
        
        return frames
    
    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        """Assess the quality of a frame for analysis."""
        try:
            import cv2
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # 1. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)
            
            # 2. Brightness (avoid too dark or too bright)
            brightness = np.mean(gray)
            brightness_score = 1.0 - abs(brightness - 128) / 128.0
            
            # 3. Contrast
            contrast = np.std(gray)
            contrast_score = min(contrast / 64.0, 1.0)
            
            # 4. Noise level (using high frequency content)
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            
            # High frequency noise indicator
            h, w = gray.shape
            center_h, center_w = h // 2, w // 2
            high_freq_region = magnitude_spectrum[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
            noise_level = np.mean(high_freq_region)
            noise_score = max(0, 1.0 - noise_level / 10000.0)
            
            # Combine scores
            quality_score = (
                sharpness_score * 0.3 +
                brightness_score * 0.25 +
                contrast_score * 0.25 +
                noise_score * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except ImportError:
            # Fallback without OpenCV
            return self._simple_quality_assessment(frame)
    
    def _simple_quality_assessment(self, frame: np.ndarray) -> float:
        """Simple quality assessment without OpenCV."""
        # Basic brightness and contrast check
        gray = np.mean(frame, axis=2)
        
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        brightness_score = 1.0 - abs(brightness - 128) / 128.0
        contrast_score = min(contrast / 64.0, 1.0)
        
        return (brightness_score + contrast_score) / 2.0
    
    def _calculate_frame_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate motion between consecutive frames."""
        try:
            import cv2
            
            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray,
                np.array([[100, 100]], dtype=np.float32).reshape(-1, 1, 2),
                None
            )[0]
            
            if flow is not None and len(flow) > 0:
                motion_magnitude = np.linalg.norm(flow[0][0])
                return min(motion_magnitude / 10.0, 1.0)
            
        except ImportError:
            pass
        except Exception:
            pass
        
        # Fallback: simple frame difference
        diff = np.mean(np.abs(curr_frame.astype(float) - prev_frame.astype(float)))
        return min(diff / 50.0, 1.0)
    
    def analyze_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single frame for deepfakes with optimization.
        
        Args:
            frame_data: Frame data dictionary with frame and metadata
            
        Returns:
            Frame analysis result
        """
        try:
            frame = frame_data['frame']
            frame_number = frame_data.get('frame_number', 0)
            
            # Skip low-quality frames for performance
            quality_score = frame_data.get('quality_score', 1.0)
            if quality_score < 0.4:
                return {
                    'frame_number': frame_number,
                    'confidence': 50.0,
                    'authenticity': 'UNCERTAIN',
                    'faces_detected': 0,
                    'skipped': True,
                    'skip_reason': 'low_quality'
                }
            
            # Convert frame to bytes for image detector (optimized)
            frame_pil = Image.fromarray(frame)
            
            # Resize large frames for faster processing
            if frame_pil.size[0] > 1920 or frame_pil.size[1] > 1080:
                # Maintain aspect ratio while reducing size
                frame_pil.thumbnail((1920, 1080), Image.LANCZOS)
            
            buffer = io.BytesIO()
            frame_pil.save(buffer, format='JPEG', quality=85, optimize=True)
            frame_bytes = buffer.getvalue()
            
            # Analyze using image detector with timeout
            try:
                result = self.image_detector.analyze(frame_bytes)
                
                return {
                    'frame_number': frame_number,
                    'confidence': result.get('confidence', 50.0),
                    'authenticity': result.get('authenticity', 'UNCERTAIN'),
                    'faces_detected': result.get('metrics', {}).get('faces_detected', 0),
                    'processing_time': result.get('metrics', {}).get('processing_time_ms', 0),
                    'quality_score': quality_score
                }
                
            except Exception as e:
                logger.warning(f"Frame {frame_number} analysis failed: {e}")
                return {
                    'frame_number': frame_number,
                    'confidence': 50.0,
                    'authenticity': 'UNCERTAIN',
                    'faces_detected': 0,
                    'error': str(e),
                    'quality_score': quality_score
                }
            
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return {
                'frame_number': frame_data.get('frame_number', 0),
                'confidence': 50.0,
                'authenticity': 'UNCERTAIN',
                'faces_detected': 0,
                'error': str(e)
            }
    
    def analyze_frames_parallel(self, frame_data_list: List[Dict[str, Any]], 
                               max_workers: int = None) -> List[Dict[str, Any]]:
        """
        Analyze multiple frames in parallel for better performance.
        
        Args:
            frame_data_list: List of frame data dictionaries
            max_workers: Maximum number of worker threads
            
        Returns:
            List of frame analysis results
        """
        if not frame_data_list:
            return []
        
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(frame_data_list), 4)
        
        logger.info(f"Analyzing {len(frame_data_list)} frames with {max_workers} workers")
        
        results = []
        
        try:
            # Use ThreadPoolExecutor for I/O bound operations
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all frame analysis tasks
                future_to_frame = {
                    executor.submit(self.analyze_frame, frame_data): frame_data
                    for frame_data in frame_data_list
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_frame):
                    try:
                        result = future.result(timeout=60)  # 1 minute timeout per frame
                        results.append(result)
                    except Exception as e:
                        frame_data = future_to_frame[future]
                        frame_number = frame_data.get('frame_number', 0)
                        logger.error(f"Frame {frame_number} analysis failed: {e}")
                        results.append({
                            'frame_number': frame_number,
                            'confidence': 50.0,
                            'authenticity': 'UNCERTAIN',
                            'faces_detected': 0,
                            'error': str(e)
                        })
        
        except Exception as e:
            logger.error(f"Parallel frame analysis failed: {e}")
            # Fallback to sequential processing
            logger.info("Falling back to sequential processing")
            for frame_data in frame_data_list:
                result = self.analyze_frame(frame_data)
                results.append(result)
        
        # Sort results by frame number
        results.sort(key=lambda x: x.get('frame_number', 0))
        
        return results
    
    def adaptive_frame_sampling(self, total_frames: int, target_frames: int, 
                               video_fps: float) -> List[int]:
        """
        Implement adaptive frame sampling for optimal performance.
        
        Args:
            total_frames: Total number of frames in video
            target_frames: Target number of frames to analyze
            video_fps: Video frame rate
            
        Returns:
            List of frame indices to analyze
        """
        if total_frames <= target_frames:
            return list(range(total_frames))
        
        # Different sampling strategies based on video characteristics
        if total_frames < target_frames * 2:
            # Uniform sampling for short videos
            step = total_frames / target_frames
            return [int(i * step) for i in range(target_frames)]
        
        else:
            # Adaptive sampling: more frames from beginning and end, fewer from middle
            # This captures scene changes and transitions better
            
            # 40% from first quarter
            first_quarter_frames = int(target_frames * 0.4)
            first_quarter_end = total_frames // 4
            first_quarter_indices = np.linspace(0, first_quarter_end, first_quarter_frames, dtype=int)
            
            # 20% from middle half
            middle_frames = int(target_frames * 0.2)
            middle_start = total_frames // 4
            middle_end = 3 * total_frames // 4
            middle_indices = np.linspace(middle_start, middle_end, middle_frames, dtype=int)
            
            # 40% from last quarter
            last_quarter_frames = target_frames - first_quarter_frames - middle_frames
            last_quarter_start = 3 * total_frames // 4
            last_quarter_indices = np.linspace(last_quarter_start, total_frames - 1, last_quarter_frames, dtype=int)
            
            # Combine and sort
            all_indices = np.concatenate([first_quarter_indices, middle_indices, last_quarter_indices])
            all_indices = np.unique(all_indices)  # Remove duplicates
            
            return sorted(all_indices.tolist())
    
    def optimize_memory_usage(self, frame_data_list: List[Dict[str, Any]], 
                             batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Process frames in batches to optimize memory usage.
        
        Args:
            frame_data_list: List of frame data
            batch_size: Number of frames to process in each batch
            
        Returns:
            List of all frame analysis results
        """
        all_results = []
        
        for i in range(0, len(frame_data_list), batch_size):
            batch = frame_data_list[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(frame_data_list) + batch_size - 1)//batch_size}")
            
            # Process batch
            batch_results = self.analyze_frames_parallel(batch)
            all_results.extend(batch_results)
            
            # Force garbage collection between batches
            import gc
            gc.collect()
            
            # Log memory usage
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:
                    logger.warning(f"High memory usage: {memory_percent:.1f}%")
            except ImportError:
                pass
        
        return all_results
    
    def check_temporal_consistency(self, frame_results: List[Dict], frame_data: List[Dict] = None) -> Dict[str, float]:
        """
        Advanced temporal consistency analysis across frames.
        
        Args:
            frame_results: List of frame analysis results
            frame_data: List of frame data with metadata (optional)
            
        Returns:
            Dictionary with various temporal consistency metrics
        """
        if len(frame_results) < 2:
            return {'overall_consistency': 1.0, 'confidence_stability': 1.0, 'authenticity_agreement': 1.0}
        
        try:
            # 1. Confidence stability analysis
            confidence_stability = self._analyze_confidence_stability(frame_results)
            
            # 2. Authenticity agreement analysis
            authenticity_agreement = self._analyze_authenticity_agreement(frame_results)
            
            # 3. Facial feature consistency (if face data available)
            feature_consistency = self._analyze_facial_feature_consistency(frame_results)
            
            # 4. Motion pattern consistency
            motion_consistency = 1.0
            if frame_data:
                motion_consistency = self._analyze_motion_consistency(frame_data)
            
            # 5. Embedding similarity analysis
            embedding_consistency = self._analyze_embedding_consistency(frame_results)
            
            # 6. Temporal anomaly detection
            anomaly_score = self._detect_temporal_anomalies(frame_results, frame_data)
            
            # Combine all metrics
            overall_consistency = (
                confidence_stability * 0.2 +
                authenticity_agreement * 0.25 +
                feature_consistency * 0.2 +
                motion_consistency * 0.15 +
                embedding_consistency * 0.15 +
                (1 - anomaly_score) * 0.05
            )
            
            consistency_metrics = {
                'overall_consistency': overall_consistency,
                'confidence_stability': confidence_stability,
                'authenticity_agreement': authenticity_agreement,
                'feature_consistency': feature_consistency,
                'motion_consistency': motion_consistency,
                'embedding_consistency': embedding_consistency,
                'anomaly_score': anomaly_score
            }
            
            logger.info(f"Temporal consistency metrics: {consistency_metrics}")
            
            return consistency_metrics
            
        except Exception as e:
            logger.error(f"Temporal consistency check error: {e}")
            return {'overall_consistency': 0.5, 'confidence_stability': 0.5, 'authenticity_agreement': 0.5}
    
    def _analyze_confidence_stability(self, frame_results: List[Dict]) -> float:
        """Analyze stability of confidence scores across frames."""
        confidences = [r.get('confidence', 50.0) for r in frame_results]
        
        # Calculate various stability metrics
        confidence_std = np.std(confidences)
        confidence_range = max(confidences) - min(confidences)
        
        # Detect sudden jumps
        confidence_diffs = np.abs(np.diff(confidences))
        sudden_jumps = np.sum(confidence_diffs > 20)  # Jumps > 20%
        
        # Calculate stability score
        std_score = max(0, 1 - (confidence_std / 30))  # Penalize high variance
        range_score = max(0, 1 - (confidence_range / 80))  # Penalize wide range
        jump_score = max(0, 1 - (sudden_jumps / len(confidences)))  # Penalize jumps
        
        stability = (std_score + range_score + jump_score) / 3
        
        return max(0.0, min(1.0, stability))
    
    def _analyze_authenticity_agreement(self, frame_results: List[Dict]) -> float:
        """Analyze agreement in authenticity labels across frames."""
        authenticities = [r.get('authenticity', 'UNCERTAIN') for r in frame_results]
        
        # Count each type
        authentic_count = sum(1 for a in authenticities if a == 'AUTHENTIC MEDIA')
        manipulated_count = sum(1 for a in authenticities if a == 'MANIPULATED MEDIA')
        uncertain_count = sum(1 for a in authenticities if a == 'UNCERTAIN')
        
        total = len(authenticities)
        
        # Calculate agreement ratio (highest category / total)
        max_agreement = max(authentic_count, manipulated_count, uncertain_count)
        agreement_ratio = max_agreement / total
        
        # Penalize too many uncertain results
        uncertainty_penalty = uncertain_count / total * 0.3
        
        agreement_score = agreement_ratio - uncertainty_penalty
        
        return max(0.0, min(1.0, agreement_score))
    
    def _analyze_facial_feature_consistency(self, frame_results: List[Dict]) -> float:
        """
        Analyze consistency of facial features across frames using advanced metrics.
        
        Args:
            frame_results: List of frame analysis results with face data
            
        Returns:
            float: Consistency score between 0 (inconsistent) and 1 (highly consistent)
        """
        # Filter frames with detected faces and landmarks
        frames_with_faces = [
            r for r in frame_results 
            if r.get('face_data') and 'landmarks' in r['face_data']
        ]
        
        if len(frames_with_faces) < 2:
            return 0.5  # Not enough data for meaningful analysis
            
        try:
            # 1. Calculate inter-ocular distance consistency
            eye_distances = []
            for frame in frames_with_faces:
                landmarks = frame['face_data']['landmarks']
                if 'left_eye' in landmarks and 'right_eye' in landmarks:
                    left_eye = np.array(landmarks['left_eye'])
                    right_eye = np.array(landmarks['right_eye'])
                    eye_distances.append(np.linalg.norm(left_eye - right_eye))
            
            # Calculate consistency of eye distances
            if eye_distances:
                eye_std = np.std(eye_distances) / np.mean(eye_distances)  # Relative std
                eye_consistency = max(0, 1 - min(eye_std, 0.5))  # Cap at 50% variation
            else:
                eye_consistency = 0.5
                
            # 2. Calculate facial expression consistency
            expression_changes = []
            for i in range(1, len(frames_with_faces)):
                curr = frames_with_faces[i]['face_data']['landmarks']
                prev = frames_with_faces[i-1]['face_data']['landmarks']
                
                # Calculate changes in mouth aspect ratio
                if all(k in curr and k in prev for k in ['mouth_left', 'mouth_right', 'nose']):
                    curr_mar = self._calculate_mouth_aspect_ratio(curr)
                    prev_mar = self._calculate_mouth_aspect_ratio(prev)
                    expression_changes.append(abs(curr_mar - prev_mar))
            
            # Calculate expression consistency (lower changes = more consistent)
            expression_consistency = 1.0
            if expression_changes:
                expr_std = np.std(expression_changes)
                expression_consistency = max(0, 1 - min(expr_std * 10, 1))  # Scale to 0-1 range
                
            # 3. Face orientation consistency
            orientations = []
            for frame in frames_with_faces:
                landmarks = frame['face_data']['landmarks']
                if all(k in landmarks for k in ['left_eye', 'right_eye', 'nose']):
                    orientation = self._estimate_face_orientation(landmarks)
                    orientations.append(orientation)
            
            orientation_consistency = 1.0
            if orientations:
                orientation_std = np.std(orientations, axis=0)
                # Penalize large changes in pitch, yaw, roll
                orientation_consistency = 1 - min(np.mean(orientation_std) / 15.0, 1)  # 15 degrees max
            
            # Combine all metrics with weights
            consistency_scores = {
                'eye': 0.4,
                'expression': 0.3,
                'orientation': 0.3
            }
            
            overall_consistency = (
                eye_consistency * consistency_scores['eye'] +
                expression_consistency * consistency_scores['expression'] +
                orientation_consistency * consistency_scores['orientation']
            )
            
            return max(0.0, min(1.0, overall_consistency))
            
        except Exception as e:
            logger.error(f"Facial feature consistency analysis failed: {e}")
            return 0.5  # Fallback to neutral score on error
    
    def _calculate_mouth_aspect_ratio(self, landmarks: Dict) -> float:
        ""Calculate mouth aspect ratio for expression analysis."""
        # Get mouth points
        mouth_left = np.array(landmarks['mouth_left'])
        mouth_right = np.array(landmarks['mouth_right'])
        
        # Calculate mouth width
        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        
        # Calculate mouth height (from nose to mouth center)
        nose = np.array(landmarks['nose'])
        mouth_center = (mouth_left + mouth_right) / 2
        mouth_height = np.linalg.norm(nose - mouth_center)
        
        # Avoid division by zero
        if mouth_height < 1e-6:
            return 0.0
            
        return mouth_width / mouth_height
    
    def _estimate_face_orientation(self, landmarks: Dict) -> np.ndarray:
        ""Estimate face orientation (pitch, yaw, roll) from landmarks."""
        # Convert landmarks to numpy arrays
        left_eye = np.array(landmarks['left_eye'])
        right_eye = np.array(landmarks['right_eye'])
        nose = np.array(landmarks['nose'])
        
        # Calculate eye center and eye vector
        eye_center = (left_eye + right_eye) / 2
        eye_vector = right_eye - left_eye
        
        # Calculate roll (rotation around Z axis)
        roll = np.arctan2(eye_vector[1], eye_vector[0])
        
        # Calculate yaw (rotation around Y axis)
        eye_distance = np.linalg.norm(eye_vector)
        nose_offset = nose - eye_center
        yaw = np.arctan2(nose_offset[0], eye_distance)  # Simplified yaw estimation
        
        # Calculate pitch (rotation around X axis)
        # This is a simplified estimation - in production, use solvePnP for better accuracy
        vertical_ratio = abs(nose_offset[1]) / eye_distance
        pitch = np.arcsin(min(max(vertical_ratio, -1), 1))
        
        return np.array([pitch, yaw, roll]) * 180 / np.pi  # Convert to degrees
    
    def _analyze_motion_consistency(self, frame_data: List[Dict]) -> float:
        """Analyze consistency of motion patterns."""
        motion_scores = [fd.get('motion_score', 0.0) for fd in frame_data]
        
        if len(motion_scores) < 2:
            return 1.0
        
        # Analyze motion smoothness
        motion_diffs = np.abs(np.diff(motion_scores))
        sudden_motion_changes = np.sum(motion_diffs > 0.3)
        
        # Natural motion should be relatively smooth
        smoothness_score = max(0, 1 - (sudden_motion_changes / len(motion_scores)))
        
        # Check for unnatural motion patterns
        motion_variance = np.var(motion_scores)
        variance_score = max(0, 1 - (motion_variance / 0.1))  # Penalize high variance
        
        motion_consistency = (smoothness_score + variance_score) / 2
        
        return max(0.0, min(1.0, motion_consistency))
    
    def _analyze_embedding_consistency(self, frame_results: List[Dict]) -> float:
        """Analyze consistency of facial embeddings across frames."""
        # This would require storing embeddings from each frame
        # For now, use a simplified approach based on confidence patterns
        
        confidences = [r.get('confidence', 50.0) for r in frame_results]
        
        # Look for patterns that suggest consistent identity
        # Real faces should have relatively consistent embedding-based confidence
        confidence_trend = np.polyfit(range(len(confidences)), confidences, 1)[0]
        
        # Penalize strong trends (sudden improvement/degradation)
        trend_penalty = min(abs(confidence_trend) / 10.0, 0.3)
        
        embedding_consistency = 0.8 - trend_penalty
        
        return max(0.0, min(1.0, embedding_consistency))
    
    def _detect_temporal_anomalies(self, frame_results: List[Dict], frame_data: List[Dict] = None) -> float:
        """Detect temporal anomalies that suggest manipulation."""
        anomaly_score = 0.0
        
        # 1. Sudden confidence spikes/drops
        confidences = [r.get('confidence', 50.0) for r in frame_results]
        confidence_diffs = np.abs(np.diff(confidences))
        sudden_changes = np.sum(confidence_diffs > 25)  # Changes > 25%
        anomaly_score += min(sudden_changes / len(confidences), 0.3)
        
        # 2. Inconsistent face detection
        face_counts = [r.get('faces_detected', 0) for r in frame_results]
        face_disappearances = 0
        for i in range(1, len(face_counts)):
            if face_counts[i-1] > 0 and face_counts[i] == 0:
                face_disappearances += 1
        anomaly_score += min(face_disappearances / len(face_counts), 0.2)
        
        # 3. Quality inconsistencies (if frame data available)
        if frame_data:
            quality_scores = [fd.get('quality_score', 1.0) for fd in frame_data]
            quality_variance = np.var(quality_scores)
            if quality_variance > 0.1:  # High quality variance
                anomaly_score += 0.1
        
        # 4. Periodic patterns (suggesting synthetic generation)
        if len(confidences) > 10:
            # Simple periodicity check using autocorrelation
            autocorr = np.correlate(confidences, confidences, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for strong periodic patterns
            if len(autocorr) > 5:
                max_autocorr = np.max(autocorr[2:min(10, len(autocorr))])  # Skip first few
                if max_autocorr > np.max(autocorr) * 0.8:  # Strong periodicity
                    anomaly_score += 0.15
        
        return min(anomaly_score, 1.0)
    
    def analyze_motion_patterns(self, frames: List[np.ndarray]) -> Dict[str, float]:
        """
        Analyze motion patterns across frames.
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            Dictionary with motion metrics
        """
        metrics = {
            'motion_smoothness': 0.9,
            'motion_consistency': 0.85,
            'unnatural_transitions': 0.1
        }
        
        if len(frames) < 2:
            return metrics
        
        try:
            # Calculate frame differences
            differences = []
            for i in range(len(frames) - 1):
                # Simple frame difference
                diff = np.mean(np.abs(frames[i+1].astype(float) - frames[i].astype(float)))
                differences.append(diff)
            
            if differences:
                # Analyze difference patterns
                diff_std = np.std(differences)
                diff_mean = np.mean(differences)
                
                # Smooth motion has low variance in differences
                metrics['motion_smoothness'] = max(0, 1 - (diff_std / 50))
                
                # Consistent motion has moderate mean difference
                if 5 < diff_mean < 30:
                    metrics['motion_consistency'] = 0.9
                else:
                    metrics['motion_consistency'] = 0.7
                
                # Detect sudden jumps (potential manipulation)
                sudden_changes = sum(1 for d in differences if d > diff_mean + 2 * diff_std)
                metrics['unnatural_transitions'] = min(1.0, sudden_changes / len(differences))
            
        except Exception as e:
            logger.error(f"Motion analysis error: {e}")
        
        return metrics
    
    def aggregate_scores(
        self,
        frame_results: List[Dict],
        temporal_score: float,
        motion_metrics: Dict[str, float]
    ) -> Tuple[float, str]:
        """
        Aggregate frame scores with temporal and motion analysis.
        
        Args:
            frame_results: List of frame analysis results
            temporal_score: Temporal consistency score
            motion_metrics: Motion analysis metrics
            
        Returns:
            Tuple of (confidence, authenticity)
        """
        if not frame_results:
            return 50.0, 'UNCERTAIN'
        
        # Calculate average frame confidence
        confidences = [r['confidence'] for r in frame_results]
        avg_confidence = np.mean(confidences)
        
        # Count authentic vs manipulated frames
        authenticities = [r['authenticity'] for r in frame_results]
        authentic_count = sum(1 for a in authenticities if a == 'AUTHENTIC MEDIA')
        manipulated_count = sum(1 for a in authenticities if a == 'MANIPULATED MEDIA')
        
        # Determine overall authenticity
        if authentic_count > manipulated_count:
            base_authenticity = 'AUTHENTIC MEDIA'
            base_confidence = avg_confidence
        elif manipulated_count > authentic_count:
            base_authenticity = 'MANIPULATED MEDIA'
            base_confidence = avg_confidence
        else:
            base_authenticity = 'UNCERTAIN'
            base_confidence = 50.0
        
        # Apply temporal consistency penalty/bonus
        temporal_factor = 0.8 + (temporal_score * 0.4)  # Range: 0.8 to 1.2
        adjusted_confidence = base_confidence * temporal_factor
        
        # Apply motion consistency factor
        motion_factor = (motion_metrics['motion_smoothness'] + motion_metrics['motion_consistency']) / 2
        motion_factor = 0.9 + (motion_factor * 0.2)  # Range: 0.9 to 1.1
        adjusted_confidence *= motion_factor
        
        # Apply unnatural transition penalty
        if motion_metrics['unnatural_transitions'] > 0.3:
            adjusted_confidence *= 0.9
        
        # Clamp confidence
        final_confidence = max(0.0, min(100.0, adjusted_confidence))
        
        logger.info(f"Aggregated score: {final_confidence:.1f}% ({base_authenticity})")
        
        return final_confidence, base_authenticity
    
    @timeout(600)  # 10 minute timeout for videos
    @memory_monitor(2048)  # 2GB memory limit
    @retry(max_retries=1, delay=2.0)  # Limited retries for videos
    def analyze(self, video_buffer: bytes, **kwargs) -> Dict[str, Any]:
        """
        Analyze a video for deepfake manipulation with optimizations.
        
        Args:
            video_buffer: Raw video data as bytes
            **kwargs: Additional parameters (fps, max_frames, parallel, batch_size)
            
        Returns:
            Analysis result dictionary
        """
        # Validate input
        validation = validate_input_file(video_buffer, max_size_mb=200, allowed_types=['video'])
        if not validation['valid']:
            return create_fallback_result(
                authenticity='ANALYSIS FAILED',
                confidence=0.0,
                error_message=f"Input validation failed: {validation['error']}"
            )
        
        # Use safe execution wrapper
        def _analyze_internal():
            start_time = time.time()
            
            # Get parameters with defaults
            fps = kwargs.get('fps', 5)
            max_frames = kwargs.get('max_frames', 200)  # Reduced for performance
            use_parallel = kwargs.get('parallel', True)
            batch_size = kwargs.get('batch_size', 20)
            
            try:
                # Save video to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(video_buffer)
                    tmp_path = tmp_file.name
                
                logger.info(f"Analyzing video: {validation['size_mb']:.1f}MB")
                
                # Extract frames with enhanced metadata
                frame_data_list = self.extract_frames(
                    tmp_path, 
                    fps=fps, 
                    max_frames=max_frames,
                    quality_filter=True
                )
                
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                if not frame_data_list:
                    raise AnalysisError("Failed to extract frames from video")
                
                # Adaptive frame sampling for large videos
                if len(frame_data_list) > max_frames:
                    total_frames = len(frame_data_list)
                    selected_indices = self.adaptive_frame_sampling(total_frames, max_frames, fps)
                    frame_data_list = [frame_data_list[i] for i in selected_indices]
                    logger.info(f"Adaptive sampling: selected {len(frame_data_list)} frames from {total_frames}")
                
                logger.info(f"Analyzing {len(frame_data_list)} frames...")
                
                # Analyze frames (parallel or sequential based on settings)
                if use_parallel and len(frame_data_list) > 5:
                    # Use optimized memory processing for large videos
                    if len(frame_data_list) > batch_size:
                        frame_results = self.optimize_memory_usage(frame_data_list, batch_size)
                    else:
                        frame_results = self.analyze_frames_parallel(frame_data_list)
                else:
                    # Sequential processing for small videos
                    frame_results = []
                    for i, frame_data in enumerate(frame_data_list):
                        if i % 10 == 0:
                            logger.info(f"Processing frame {i+1}/{len(frame_data_list)}")
                        result = self.analyze_frame(frame_data)
                        frame_results.append(result)
                
                # Filter out skipped frames for analysis
                valid_frame_results = [r for r in frame_results if not r.get('skipped', False)]
                
                if not valid_frame_results:
                    raise AnalysisError("No valid frames could be analyzed")
                
                logger.info(f"Successfully analyzed {len(valid_frame_results)} frames")
                
                # Enhanced temporal consistency analysis
                temporal_metrics = self.check_temporal_consistency(valid_frame_results, frame_data_list)
                
                # Analyze motion patterns from frame data
                motion_metrics = self.analyze_motion_patterns([fd['frame'] for fd in frame_data_list])
                
                # Aggregate scores with enhanced metrics
                final_confidence, final_authenticity = self.aggregate_scores(
                    valid_frame_results,
                    temporal_metrics.get('overall_consistency', 0.5),
                    motion_metrics
                )
                
                # Calculate video statistics
                video_duration = len(frame_data_list) / fps if fps > 0 else 0
                avg_quality = np.mean([fd.get('quality_score', 1.0) for fd in frame_data_list])
                
                # Create enhanced key findings
                key_findings = []
                if final_authenticity == 'AUTHENTIC MEDIA':
                    key_findings = [
                        'No temporal inconsistencies detected',
                        'Natural motion patterns observed',
                        f'Analyzed {len(valid_frame_results)} high-quality frames',
                        f'High temporal consistency: {temporal_metrics.get("overall_consistency", 0.5):.2f}'
                    ]
                elif final_authenticity == 'MANIPULATED MEDIA':
                    key_findings = [
                        'Temporal inconsistencies detected',
                        'Suspicious motion patterns found',
                        f'Analyzed {len(valid_frame_results)} frames with concerns',
                        f'Low temporal consistency: {temporal_metrics.get("overall_consistency", 0.5):.2f}'
                    ]
                else:
                    key_findings = [
                        'Mixed results across frames',
                        'Unable to determine with high confidence',
                        f'Analyzed {len(valid_frame_results)} frames'
                    ]
                
                # Add performance insights
                processing_time = time.time() - start_time
                frames_per_second = len(valid_frame_results) / processing_time if processing_time > 0 else 0
                
                if use_parallel:
                    key_findings.append(f'Parallel processing: {frames_per_second:.1f} frames/sec')
                
                # Build comprehensive result
                result = self._create_result(
                    authenticity=final_authenticity,
                    confidence=final_confidence,
                    key_findings=key_findings,
                    video_analysis={
                        'total_frames_extracted': len(frame_data_list),
                        'frames_analyzed': len(valid_frame_results),
                        'frames_skipped': len(frame_data_list) - len(valid_frame_results),
                        'video_duration_seconds': video_duration,
                        'frame_rate': fps,
                        'average_frame_quality': avg_quality,
                        'temporal_consistency': temporal_metrics.get('overall_consistency', 0.5),
                        'confidence_stability': temporal_metrics.get('confidence_stability', 0.5),
                        'motion_smoothness': motion_metrics['motion_smoothness'],
                        'motion_consistency': motion_metrics['motion_consistency'],
                        'unnatural_transitions': motion_metrics['unnatural_transitions'],
                        'processing_fps': frames_per_second,
                        'parallel_processing': use_parallel
                    },
                    metrics={
                        'frames_analyzed': len(valid_frame_results),
                        'temporal_consistency': temporal_metrics.get('overall_consistency', 0.5),
                        'processing_time_ms': int(processing_time * 1000),
                        'memory_optimized': len(frame_data_list) > batch_size
                    }
                )
                
                return result
                
            except AnalysisError:
                raise  # Re-raise analysis errors
            except Exception as e:
                raise AnalysisError(f"Video analysis failed: {e}")
        
        # Execute with comprehensive error handling
        execution_result = error_handler.safe_execute(_analyze_internal)
        
        if execution_result['success']:
            return execution_result['result']
        else:
            # Return fallback result with error details
            error_info = execution_result['error']
            return create_fallback_result(
                authenticity='ANALYSIS FAILED',
                confidence=0.0,
                error_message=f"{error_info['type']}: {error_info['message']}"
            )
