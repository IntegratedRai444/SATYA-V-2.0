"""
Advanced Face Detector
Combines MediaPipe, DeepFace, and dlib for comprehensive face analysis
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class AdvancedFaceDetector:
    """
    Advanced face detection using multiple state-of-the-art models.
    Combines MediaPipe, DeepFace, and dlib for robust face analysis.
    """
    
    def __init__(self, enable_gpu: bool = False):
        """
        Initialize the advanced face detector.
        
        Args:
            enable_gpu: Whether to use GPU acceleration
        """
        self.enable_gpu = enable_gpu
        self.mediapipe_detector = None
        self.deepface_available = False
        self.dlib_available = False
        
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        """Initialize all face detection models."""
        # Initialize MediaPipe
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mediapipe_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1 for full range, 0 for short range
                min_detection_confidence=0.5
            )
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe face detection initialized")
        except ImportError:
            logger.warning("MediaPipe not available")
            
        # Initialize DeepFace
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            self.deepface_available = True
            logger.info("DeepFace initialized")
        except ImportError:
            logger.warning("DeepFace not available")
            
        # Initialize dlib
        try:
            import dlib
            self.dlib = dlib
            self.dlib_detector = dlib.get_frontal_face_detector()
            # Try to load shape predictor for landmarks
            try:
                self.dlib_predictor = dlib.shape_predictor(
                    "models/shape_predictor_68_face_landmarks.dat"
                )
                self.dlib_available = True
                logger.info("dlib face detection initialized with landmarks")
            except:
                self.dlib_available = False
                logger.warning("dlib shape predictor not found")
        except ImportError:
            logger.warning("dlib not available")
    
    def detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect faces using all available detectors.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            Dictionary containing detection results from all models
        """
        results = {
            'faces_detected': 0,
            'detections': [],
            'mediapipe_results': None,
            'deepface_results': None,
            'dlib_results': None
        }
        
        # MediaPipe detection
        if self.mediapipe_detector:
            mp_results = self._detect_with_mediapipe(image)
            results['mediapipe_results'] = mp_results
            if mp_results['faces']:
                results['faces_detected'] = max(
                    results['faces_detected'], 
                    len(mp_results['faces'])
                )
        
        # DeepFace detection
        if self.deepface_available:
            df_results = self._detect_with_deepface(image)
            results['deepface_results'] = df_results
            if df_results['faces']:
                results['faces_detected'] = max(
                    results['faces_detected'],
                    len(df_results['faces'])
                )
        
        # dlib detection
        if self.dlib_available:
            dlib_results = self._detect_with_dlib(image)
            results['dlib_results'] = dlib_results
            if dlib_results['faces']:
                results['faces_detected'] = max(
                    results['faces_detected'],
                    len(dlib_results['faces'])
                )
        
        # Combine detections
        results['detections'] = self._combine_detections(results)
        
        return results
    
    def _detect_with_mediapipe(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect faces using MediaPipe."""
        try:
            results = self.mediapipe_detector.process(image)
            faces = []
            
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w = image.shape[:2]
                    
                    face_info = {
                        'bbox': {
                            'x': int(bbox.xmin * w),
                            'y': int(bbox.ymin * h),
                            'width': int(bbox.width * w),
                            'height': int(bbox.height * h)
                        },
                        'confidence': detection.score[0],
                        'keypoints': self._extract_mediapipe_keypoints(detection)
                    }
                    faces.append(face_info)
            
            # Get face mesh for detailed landmarks
            mesh_results = self.face_mesh.process(image)
            landmarks = []
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    landmarks.append(self._extract_face_mesh(face_landmarks, image.shape))
            
            return {
                'success': True,
                'faces': faces,
                'landmarks': landmarks,
                'detector': 'mediapipe'
            }
            
        except Exception as e:
            logger.error(f"MediaPipe detection failed: {e}")
            return {'success': False, 'faces': [], 'error': str(e)}
    
    def _detect_with_deepface(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect faces using DeepFace."""
        try:
            # DeepFace expects BGR format
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Detect faces and extract attributes
            faces = self.deepface.extract_faces(
                img_path=image_bgr,
                detector_backend='retinaface',  # Best accuracy
                enforce_detection=False
            )
            
            face_list = []
            for face in faces:
                face_info = {
                    'bbox': face['facial_area'],
                    'confidence': face['confidence'],
                    'face_image': face['face']
                }
                
                # Analyze facial attributes
                try:
                    analysis = self.deepface.analyze(
                        img_path=face['face'],
                        actions=['age', 'gender', 'race', 'emotion'],
                        enforce_detection=False
                    )
                    face_info['attributes'] = analysis[0] if isinstance(analysis, list) else analysis
                except:
                    pass
                
                face_list.append(face_info)
            
            return {
                'success': True,
                'faces': face_list,
                'detector': 'deepface'
            }
            
        except Exception as e:
            logger.error(f"DeepFace detection failed: {e}")
            return {'success': False, 'faces': [], 'error': str(e)}
    
    def _detect_with_dlib(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect faces using dlib."""
        try:
            # Convert to grayscale for dlib
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            dets = self.dlib_detector(gray, 1)
            
            faces = []
            for det in dets:
                face_info = {
                    'bbox': {
                        'x': det.left(),
                        'y': det.top(),
                        'width': det.right() - det.left(),
                        'height': det.bottom() - det.top()
                    },
                    'confidence': 1.0  # dlib doesn't provide confidence
                }
                
                # Get 68 facial landmarks if predictor available
                if self.dlib_available:
                    shape = self.dlib_predictor(gray, det)
                    landmarks = []
                    for i in range(68):
                        landmarks.append({
                            'x': shape.part(i).x,
                            'y': shape.part(i).y
                        })
                    face_info['landmarks'] = landmarks
                
                faces.append(face_info)
            
            return {
                'success': True,
                'faces': faces,
                'detector': 'dlib'
            }
            
        except Exception as e:
            logger.error(f"dlib detection failed: {e}")
            return {'success': False, 'faces': [], 'error': str(e)}
    
    def _extract_mediapipe_keypoints(self, detection) -> List[Dict]:
        """Extract keypoints from MediaPipe detection."""
        keypoints = []
        if hasattr(detection.location_data, 'relative_keypoints'):
            for kp in detection.location_data.relative_keypoints:
                keypoints.append({
                    'x': kp.x,
                    'y': kp.y
                })
        return keypoints
    
    def _extract_face_mesh(self, face_landmarks, image_shape) -> List[Dict]:
        """Extract face mesh landmarks from MediaPipe."""
        h, w = image_shape[:2]
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append({
                'x': int(landmark.x * w),
                'y': int(landmark.y * h),
                'z': landmark.z
            })
        return landmarks
    
    def _combine_detections(self, results: Dict) -> List[Dict]:
        """
        Combine detections from all models using consensus.
        Uses IoU (Intersection over Union) to match detections.
        """
        all_detections = []
        
        # Collect all detections
        if results['mediapipe_results'] and results['mediapipe_results']['faces']:
            for face in results['mediapipe_results']['faces']:
                all_detections.append({
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'source': 'mediapipe',
                    'data': face
                })
        
        if results['deepface_results'] and results['deepface_results']['faces']:
            for face in results['deepface_results']['faces']:
                all_detections.append({
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'source': 'deepface',
                    'data': face
                })
        
        if results['dlib_results'] and results['dlib_results']['faces']:
            for face in results['dlib_results']['faces']:
                all_detections.append({
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'source': 'dlib',
                    'data': face
                })
        
        # Group overlapping detections
        combined = self._group_overlapping_detections(all_detections)
        
        return combined
    
    def _group_overlapping_detections(
        self, 
        detections: List[Dict], 
        iou_threshold: float = 0.5
    ) -> List[Dict]:
        """Group overlapping detections using IoU."""
        if not detections:
            return []
        
        grouped = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], start=i+1):
                if j in used:
                    continue
                    
                iou = self._calculate_iou(det1['bbox'], det2['bbox'])
                if iou > iou_threshold:
                    group.append(det2)
                    used.add(j)
            
            # Merge group into single detection
            merged = self._merge_detections(group)
            grouped.append(merged)
        
        return grouped
    
    def _calculate_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1 = max(bbox1['x'], bbox2['x'])
        y1 = max(bbox1['y'], bbox2['y'])
        x2 = min(bbox1['x'] + bbox1['width'], bbox2['x'] + bbox2['width'])
        y2 = min(bbox1['y'] + bbox1['height'], bbox2['y'] + bbox2['height'])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = bbox1['width'] * bbox1['height']
        area2 = bbox2['width'] * bbox2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_detections(self, group: List[Dict]) -> Dict:
        """Merge multiple detections into one using weighted average."""
        if len(group) == 1:
            return group[0]
        
        # Calculate weighted average of bounding boxes
        total_conf = sum(d['confidence'] for d in group)
        
        merged_bbox = {
            'x': int(sum(d['bbox']['x'] * d['confidence'] for d in group) / total_conf),
            'y': int(sum(d['bbox']['y'] * d['confidence'] for d in group) / total_conf),
            'width': int(sum(d['bbox']['width'] * d['confidence'] for d in group) / total_conf),
            'height': int(sum(d['bbox']['height'] * d['confidence'] for d in group) / total_conf)
        }
        
        return {
            'bbox': merged_bbox,
            'confidence': total_conf / len(group),
            'sources': [d['source'] for d in group],
            'consensus': len(group),
            'data': [d['data'] for d in group]
        }
    
    def analyze_face_quality(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze face image quality for deepfake detection.
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Quality metrics dictionary
        """
        quality = {
            'sharpness': self._calculate_sharpness(face_image),
            'brightness': self._calculate_brightness(face_image),
            'contrast': self._calculate_contrast(face_image),
            'symmetry': self._calculate_symmetry(face_image),
            'resolution': face_image.shape[:2]
        }
        
        # Overall quality score
        quality['overall_score'] = (
            quality['sharpness'] * 0.3 +
            quality['brightness'] * 0.2 +
            quality['contrast'] * 0.2 +
            quality['symmetry'] * 0.3
        )
        
        return quality
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        # Normalize to 0-1 range
        return min(variance / 1000.0, 1.0)
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.mean(gray) / 255.0
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return np.std(gray) / 128.0
    
    def _calculate_symmetry(self, image: np.ndarray) -> float:
        """Calculate face symmetry score."""
        h, w = image.shape[:2]
        left_half = image[:, :w//2]
        right_half = cv2.flip(image[:, w//2:], 1)
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calculate similarity
        diff = np.abs(left_half.astype(float) - right_half.astype(float))
        similarity = 1.0 - (np.mean(diff) / 255.0)
        
        return similarity
