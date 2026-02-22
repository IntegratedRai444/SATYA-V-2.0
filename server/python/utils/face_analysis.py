"""Facial analysis utilities for deepfake detection."""
import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Try to import MediaPipe with fallback
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None
    logger = logging.getLogger(__name__)
    logger.warning("MediaPipe not available - face analysis will be limited")

logger = logging.getLogger(__name__)

# Initialize MediaPipe Face Mesh
if MEDIAPIPE_AVAILABLE:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
else:
    face_mesh = None
    mp_face_mesh = None

detector = face_mesh  # For backward compatibility
predictor = face_mesh  # For backward compatibility


def analyze_facial_landmarks(frames: List[np.ndarray]) -> List[Dict]:
    """
    Analyze facial landmarks across video frames using MediaPipe.

    Args:
        frames: List of video frames as numpy arrays in RGB format

    Returns:
        List of face analysis results per frame
    """
    if not MEDIAPIPE_AVAILABLE or face_mesh is None:
        logger.warning("Face mesh not available. Using fallback face detection.")
        return [{"faces_detected": 0, "fallback_used": True} for _ in frames]

    results = []

    for i, frame in enumerate(frames):
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and get face landmarks
            results_mp = face_mesh.process(rgb_frame)

            if not results_mp.multi_face_landmarks:
                results.append({"faces_detected": 0})
                continue

            # For simplicity, process only the first face
            face_landmarks = results_mp.multi_face_landmarks[0]

            # Get image dimensions
            h, w, _ = frame.shape

            # Convert landmarks to list of (x, y) coordinates
            landmarks_list = [
                (int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark
            ]

            # Get face bounding box
            x_coords = [lm.x * w for lm in face_landmarks.landmark]
            y_coords = [lm.y * h for lm in face_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # MediaPipe Face Mesh has different landmark indices than dlib's 68-point model
            # We'll use the following indices for facial features (approximate mapping):
            # Left eye: [33, 160, 158, 133, 153, 144]
            # Right eye: [362, 385, 387, 263, 373, 380]
            # Mouth: [61, 291, 39, 181, 0, 17, 269, 405]

            # Calculate eye aspect ratio (using different indices for MediaPipe)
            left_eye = _get_eye_aspect_ratio(
                [
                    (
                        int(face_landmarks.landmark[33].x * w),
                        int(face_landmarks.landmark[33].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[160].x * w),
                        int(face_landmarks.landmark[160].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[158].x * w),
                        int(face_landmarks.landmark[158].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[133].x * w),
                        int(face_landmarks.landmark[133].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[153].x * w),
                        int(face_landmarks.landmark[153].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[144].x * w),
                        int(face_landmarks.landmark[144].y * h),
                    ),
                ]
            )

            right_eye = _get_eye_aspect_ratio(
                [
                    (
                        int(face_landmarks.landmark[362].x * w),
                        int(face_landmarks.landmark[362].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[385].x * w),
                        int(face_landmarks.landmark[385].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[387].x * w),
                        int(face_landmarks.landmark[387].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[263].x * w),
                        int(face_landmarks.landmark[263].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[373].x * w),
                        int(face_landmarks.landmark[373].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[380].x * w),
                        int(face_landmarks.landmark[380].y * h),
                    ),
                ]
            )

            # Calculate mouth aspect ratio
            mouth_aspect_ratio = _get_mouth_aspect_ratio(
                [
                    (
                        int(face_landmarks.landmark[61].x * w),
                        int(face_landmarks.landmark[61].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[291].x * w),
                        int(face_landmarks.landmark[291].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[39].x * w),
                        int(face_landmarks.landmark[39].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[181].x * w),
                        int(face_landmarks.landmark[181].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[0].x * w),
                        int(face_landmarks.landmark[0].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[17].x * w),
                        int(face_landmarks.landmark[17].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[269].x * w),
                        int(face_landmarks.landmark[269].y * h),
                    ),
                    (
                        int(face_landmarks.landmark[405].x * w),
                        int(face_landmarks.landmark[405].y * h),
                    ),
                ]
            )

            results.append(
                {
                    "frame": i,
                    "faces_detected": len(results_mp.multi_face_landmarks),
                    "landmarks": landmarks_list,
                    "eye_aspect_ratio": (left_eye + right_eye) / 2,
                    "mouth_aspect_ratio": mouth_aspect_ratio,
                    "face_box": {
                        "x": x_min,
                        "y": y_min,
                        "width": x_max - x_min,
                        "height": y_max - y_min,
                    },
                }
            )

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
        prev = face_analysis[i - 1]
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
            events.append(
                {
                    "frame_start": i - 1,
                    "frame_end": i,
                    "type": "potential_face_swap",
                    "confidence": min(0.99, max(dx, dy, dw)),
                    "metrics": {
                        "position_change_x": dx,
                        "position_change_y": dy,
                        "size_change": dw,
                    },
                }
            )

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
    B = np.linalg.norm(np.array(mouth_points[4]) - np.array(mouth_points[8]))  # 53, 57

    # Compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[6]))  # 49, 55

    # Compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    return mar
