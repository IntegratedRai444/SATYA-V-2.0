# video_utils.py
# Add video frame extraction, aggregation, lip sync, and timeline helpers here. 

import cv2
import numpy as np
import io
import os
import logging
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_video(video_path: str) -> Dict[str, Any]:
    """Extract video metadata and basic information.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video metadata
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get basic video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps else 0
        
        # Get video codec information
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        return {
            "frames": frame_count,
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "codec": codec,
            "aspect_ratio": f"{width}:{height}",
            "resolution": f"{width}x{height}",
            "is_hd": height >= 720,
            "is_4k": width >= 3840 and height >= 2160
        }
        
    except Exception as e:
        logger.error(f"Error preprocessing video: {e}")
        raise

def extract_frames(
    video_path: str, 
    target_fps: float = 2,
    max_frames: int = None,
    target_size: Tuple[int, int] = None,
    grayscale: bool = False
) -> List[np.ndarray]:
    """Extract frames from video with optional downsampling and resizing.
    
    Args:
        video_path: Path to the video file
        target_fps: Target frames per second (default: 2)
        max_frames: Maximum number of frames to extract (default: None for all)
        target_size: Optional target (width, height) for resizing frames
        grayscale: Whether to convert frames to grayscale
        
    Returns:
        List of extracted frames as numpy arrays
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(round(video_fps / target_fps)))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames to achieve target FPS
            if frame_count % frame_interval == 0:
                # Convert color space
                if grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize if target size is specified
                if target_size:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                
                frames.append(frame)
                
                # Stop if we've reached max_frames
                if max_frames and len(frames) >= max_frames:
                    break
            
            frame_count += 1
            
    finally:
        cap.release()
    
    return frames

def analyze_frames(
    frames: List[np.ndarray],
    model_name: str = "efficientnet_b7",
    batch_size: int = 8,
    use_gpu: bool = True
) -> List[Dict]:
    """Analyze video frames using the specified deepfake detection model.
    
    Args:
        frames: List of video frames as numpy arrays
        model_name: Name of the model to use (e.g., 'efficientnet_b7', 'xception')
        batch_size: Batch size for inference
        use_gpu: Whether to use GPU if available
        
    Returns:
        List of analysis results per frame
    """
    try:
        # Import here to avoid circular imports
        from backend.models.video_model_enhanced import DeepFakeDetector
        
        # Initialize detector
        detector = DeepFakeDetector(model_name=model_name, use_gpu=use_gpu)
        
        # Process frames in batches
        results = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_results = detector.predict_batch(batch)
            results.extend(batch_results)
        
        return results
    except Exception as e:
        logger.error(f"Error analyzing frames: {e}")
        raise

def analyze_video_advanced(video_path: str, output_dir: str, generate_overlay: bool = True) -> Dict:
    """
    Perform advanced video analysis including temporal consistency checks and deepfake detection.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save output files
        generate_overlay: Whether to generate an annotated video overlay
        
    Returns:
        Dictionary containing analysis results
    """
    from .face_analysis import analyze_facial_landmarks, detect_face_swaps
    from .temporal_analysis import check_temporal_consistency
    from .audio_analysis import extract_audio_features, check_av_sync
    
    try:
        # 1. Extract and preprocess frames
        frames = extract_frames(video_path, target_fps=5)  # Lower FPS for faster processing
        
        # 2. Perform deepfake detection on frames
        frame_results = analyze_frames(frames)
        
        # 3. Analyze facial landmarks and expressions
        face_analysis = analyze_facial_landmarks(frames)
        
        # 4. Check for face swaps or manipulations
        face_swaps = detect_face_swaps(face_analysis)
        
        # 5. Perform temporal consistency analysis
        temporal_results = check_temporal_consistency(frames, frame_results)
        
        # 6. Extract and analyze audio if available
        audio_features = extract_audio_features(video_path)
        av_sync = check_av_sync(video_path, frame_results)
        
        # 7. Generate timeline and metrics
        timeline = generate_timeline(frame_results)
        
        # 8. Generate overlay video if requested
        overlay_path = None
        if generate_overlay:
            overlay_path = generate_video_overlay(
                video_path=video_path,
                frame_results=frame_results,
                output_path=os.path.join(output_dir, 'overlay.mp4')
            )
        
        # 9. Compile final results
        result = {
            'metadata': preprocess_video(video_path),
            'frame_results': frame_results,
            'face_analysis': face_analysis,
            'face_swaps': face_swaps,
            'temporal_analysis': temporal_results,
            'audio_analysis': {
                'features': audio_features,
                'av_sync': av_sync
            },
            'timeline': timeline,
            'overlay_video': overlay_path,
            'summary': {
                'is_manipulated': any(fr.get('is_manipulated', False) for fr in frame_results),
                'manipulation_score': sum(fr.get('manipulation_score', 0) for fr in frame_results) / len(frame_results) if frame_results else 0,
                'confidence': sum(fr.get('confidence', 0) for fr in frame_results) / len(frame_results) if frame_results else 0,
                'anomalies_detected': sum(1 for fr in frame_results if fr.get('has_anomalies', False)),
                'analysis_complete': True
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Advanced video analysis failed: {e}")
        raise

def generate_timeline(frame_results: List[Dict]) -> Dict:
    """
    Generate a detailed timeline of video analysis results.
    
    Args:
        frame_results: List of frame analysis results
        
    Returns:
        Dict containing timeline data for visualization
    """
    if not frame_results:
        return {
            "anomalies": [],
            "confidence_trend": [],
            "segment_analysis": [],
            "key_events": [],
            "summary_metrics": {
                "total_frames": 0,
                "authentic_frames": 0,
                "manipulated_frames": 0,
                "avg_confidence": 0.0,
                "manipulation_score": 0.0
            }
        }
    
    # Calculate basic metrics
    total_frames = len(frame_results)
    authentic_frames = sum(1 for fr in frame_results if fr.get("label") == "AUTHENTIC")
    manipulated_frames = sum(1 for fr in frame_results if fr.get("label") == "MANIPULATED")
    uncertain_frames = total_frames - authentic_frames - manipulated_frames
    
    # Calculate confidence trend (smoothed)
    confidences = [fr.get("confidence", 0.0) for fr in frame_results]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Smooth confidence values (simple moving average)
    window_size = max(1, int(total_frames * 0.05))  # 5% of total frames
    smoothed_confidences = []
    for i in range(total_frames):
        start = max(0, i - window_size // 2)
        end = min(total_frames, i + window_size // 2 + 1)
        window = confidences[start:end]
        smoothed_confidences.append(sum(window) / len(window))
    
    # Detect key events (sudden drops in confidence)
    key_events = []
    confidence_drops = []
    threshold = 0.3  # 30% drop in confidence
    
    for i in range(1, len(smoothed_confidences)):
        if smoothed_confidences[i] < smoothed_confidences[i-1] * (1 - threshold):
            drop_amount = (smoothed_confidences[i-1] - smoothed_confidences[i]) / smoothed_confidences[i-1]
            confidence_drops.append((i, drop_amount))
    
    # Get top 3 most significant drops
    significant_drops = sorted(confidence_drops, key=lambda x: -x[1])[:3]
    for frame_idx, drop_amount in significant_drops:
        key_events.append({
            "frame": frame_idx,
            "type": "confidence_drop",
            "severity": min(1.0, drop_amount * 2),  # Scale to 0-1
            "confidence_before": smoothed_confidences[frame_idx-1],
            "confidence_after": smoothed_confidences[frame_idx],
            "description": f"Significant confidence drop ({drop_amount*100:.1f}%)"
        })
    
    # Segment analysis (split video into segments and analyze each)
    segment_size = max(1, total_frames // 10)  # 10 segments
    segments = []
    
    for i in range(0, total_frames, segment_size):
        segment = frame_results[i:i+segment_size]
        if not segment:
            continue
            
        seg_authentic = sum(1 for fr in segment if fr.get("label") == "AUTHENTIC")
        seg_manipulated = sum(1 for fr in segment if fr.get("label") == "MANIPULATED")
        seg_confidence = sum(fr.get("confidence", 0.0) for fr in segment) / len(segment)
        
        segments.append({
            "start_frame": i,
            "end_frame": min(i + segment_size - 1, total_frames - 1),
            "authentic_frames": seg_authentic,
            "manipulated_frames": seg_manipulated,
            "avg_confidence": seg_confidence,
            "manipulation_score": seg_manipulated / len(segment)
        })
    
    # Calculate overall manipulation score
    manipulation_score = manipulated_frames / total_frames if total_frames > 0 else 0.0
    
    return {
        "anomalies": [i for i, fr in enumerate(frame_results) 
                      if fr.get("label") == "MANIPULATED"],
        "confidence_trend": [
            {"frame": i, "confidence": conf, "smoothed": smoothed_confidences[i]}
            for i, conf in enumerate(confidences)
        ],
        "segment_analysis": segments,
        "key_events": key_events,
        "summary_metrics": {
            "total_frames": total_frames,
            "authentic_frames": authentic_frames,
            "manipulated_frames": manipulated_frames,
            "uncertain_frames": uncertain_frames,
            "avg_confidence": avg_confidence,
            "manipulation_score": manipulation_score
        }
    }

def generate_video_overlay(video_bytes: bytes, frame_results: List[Dict], output_path: str = None) -> str:
    """
    Generate an annotated video with overlays showing detection results.
    
    Args:
        video_bytes: Original video bytes
        frame_results: List of frame analysis results
        output_path: Optional custom output path
        
    Returns:
        Path to the generated overlay video
    """
    import tempfile
    import os
    from pathlib import Path
    import cv2
    import numpy as np
    from typing import Tuple, Optional
    
    # Create output directory if it doesn't exist
    output_dir = Path("reports/overlays")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set output path
    if not output_path:
        output_path = str(output_dir / "overlay_output.mp4")
    
    # Create a temporary file for the input video
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(video_bytes)
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_idx = 0
        success = True
        
        while success and frame_idx < total_frames and frame_idx < len(frame_results):
            success, frame = cap.read()
            if not success:
                break
                
            # Get frame result
            result = frame_results[frame_idx]
            
            # Draw face bounding boxes if available
            if 'face_data' in result:
                face_data = result['face_data']
                if 'box' in face_data:
                    x, y, w, h = face_data['box']
                    color = (0, 255, 0) if result.get('label') == 'AUTHENTIC' else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw landmarks if available
                    if 'landmarks' in face_data:
                        landmarks = face_data['landmarks']
                        for point in landmarks.values():
                            if isinstance(point, (list, tuple)) and len(point) >= 2:
                                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 255), -1)
            
            # Add confidence score
            confidence = result.get('confidence', 0.0)
            confidence_text = f"Confidence: {confidence:.1%}"
            cv2.putText(frame, confidence_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add authenticity label
            label = result.get('label', 'UNKNOWN')
            label_color = (0, 255, 0) if label == 'AUTHENTIC' else (0, 0, 255)
            cv2.putText(frame, label, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)
            
            # Add frame number
            cv2.putText(frame, f"Frame: {frame_idx}", (width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add manipulation indicators if available
            if 'anomalies' in result:
                for anomaly in result['anomalies']:
                    cv2.putText(frame, "ANOMALY DETECTED", (width // 2 - 100, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Write the frame
            out.write(frame)
            frame_idx += 1
        
        # Release resources
        cap.release()
        out.release()
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error generating video overlay: {e}")
        # Clean up temp file in case of error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise