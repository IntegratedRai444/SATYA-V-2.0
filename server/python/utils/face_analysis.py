"""Facial analysis utilities for deepfake detection."""
import cv2
import numpy as np
import dlib
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Initialize face detector and predictor
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except Exception as e:
    logger.warning(f"Could not initialize face detector: {e}")
    detector = None
    predictor = None

def analyze_facial_landmarks(frames: List[np.ndarray]) -> List[Dict]:
    """
    Analyze facial landmarks across video frames.
    
    Args:
        frames: List of video frames as numpy arrays
        
    Returns:
        List of face analysis results per frame
    """
    if detector is None or predictor is None:
        logger.warning("Face detector not available. Skipping facial landmark analysis.")
        return [{"error": "Face detector not available"} for _ in frames]
    
    results = []
    
    for i, frame in enumerate(frames):
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = detector(gray)
            
            if not faces:
                results.append({"faces_detected": 0})
                continue
                
            # For simplicity, process only the first face
            face = faces[0]
            
            # Get facial landmarks
            landmarks = predictor(gray, face)
            
            # Convert landmarks to list of (x, y) coordinates
            landmarks_list = [(p.x, p.y) for p in landmarks.parts()]
            
            # Calculate facial features
            left_eye = _get_eye_aspect_ratio(landmarks_list[36:42])
            right_eye = _get_eye_aspect_ratio(landmarks_list[42:48])
            mouth_aspect_ratio = _get_mouth_aspect_ratio([
                landmarks_list[48], landmarks_list[49], 
                landmarks_list[50], landmarks_list[51],
                landmarks_list[52], landmarks_list[53],
                landmarks_list[54], landmarks_list[55],
                landmarks_list[56], landmarks_list[57],
                landmarks_list[58], landmarks_list[59]
            ])
            
            results.append({
                "frame": i,
                "faces_detected": len(faces),
                "landmarks": landmarks_list,
                "eye_aspect_ratio": (left_eye + right_eye) / 2,
                "mouth_aspect_ratio": mouth_aspect_ratio,
                "face_box": {
                    "x": face.left(),
                    "y": face.top(),
                    "width": face.width(),
                    "height": face.height()
                }
            })
            
        except Exception as e:
            logger.error(f"Error analyzing frame {i}: {e}")
            results.append({"frame": i, "error": str(e)})
    
    return results

def detect_face_swaps(face_analysis: List[Dict]) -> List[Dict]:
    """
    Detect potential face swaps or manipulations based on facial landmark analysis.
    
    Args:
        face_analysis: Results from analyze_facial_landmarks
        
    Returns:
        List of detected face swap events
    """
    if not face_analysis:
        return []
    
    events = []
    
    # Simple heuristic: look for sudden changes in face position or size
    for i in range(1, len(face_analysis)):
        prev = face_analysis[i-1]
        curr = face_analysis[i]
        
        if "face_box" not in prev or "face_box" not in curr:
            continue
            
        prev_box = prev["face_box"]
        curr_box = curr["face_box"]
        
        # Calculate position and size differences
        dx = abs(curr_box["x"] - prev_box["x"]) / max(1, prev_box["width"])
        dy = abs(curr_box["y"] - prev_box["y"]) / max(1, prev_box["height"])
        dw = abs(curr_box["width"] - prev_box["width"]) / max(1, prev_box["width"])
        
        # If change is too abrupt, it might be a face swap
        if dx > 0.5 or dy > 0.5 or dw > 0.3:
            events.append({
                "frame_start": i-1,
                "frame_end": i,
                "type": "potential_face_swap",
                "confidence": min(0.99, max(dx, dy, dw)),
                "metrics": {
                    "position_change_x": dx,
                    "position_change_y": dy,
                    "size_change": dw
                }
            })
    
    return events

def _get_eye_aspect_ratio(eye_points: List[Tuple[float, float]]) -> float:
    """Calculate the eye aspect ratio (EAR) for blink detection."""
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

def _get_mouth_aspect_ratio(mouth_points: List[Tuple[float, float]]) -> float:
    """Calculate the mouth aspect ratio (MAR) for mouth opening detection."""
    # Compute the euclidean distances between the three sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = np.linalg.norm(np.array(mouth_points[2]) - np.array(mouth_points[10]))  # 51, 59
    B = np.linalg.norm(np.array(mouth_points[4]) - np.array(mouth_points[8]))   # 53, 57
    
    # Compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[6]))  # 49, 55
    
    # Compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    
    return mar
