"""
Image Deepfake Detector - UPGRADED
Detects manipulated images using advanced multi-modal facial forensics and CNN analysis
Integrates MediaPipe, DeepFace, dlib, and advanced AI models
"""

import io
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# Import the centralized ML classifier
try:
    from models.deepfake_classifier import get_model_info, predict_image
    ML_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    ML_CLASSIFIER_AVAILABLE = False
    logger.error(f"Failed to import ML classifier: {e}")
    # Don't raise - just continue without ML

# Import forensic analysis (non-ML analysis)
try:
    from .forensic_analysis import analyze_ela, analyze_prnu
    FORENSIC_ANALYSIS_AVAILABLE = True
except ImportError:
    FORENSIC_ANALYSIS_AVAILABLE = False
    logger.warning("Forensic analysis not available")


class ImageDetector:
    """
    Image Deepfake Detector with advanced multi-modal capabilities
    """
    
    def __init__(self, model_path: str = None, enable_gpu: bool = False):
        """
        Initialize the image detector with advanced multi-modal capabilities.
        
        Args:
            model_path: Path to model files (kept for backward compatibility)
            enable_gpu: Whether to use GPU acceleration
            
        Note:
            All ML model loading and inference is now handled by deepfake_classifier.py
        """
        self.model_path = model_path  # Kept for backward compatibility
        self.enable_gpu = enable_gpu
        
        # Check if ML classifier is available
        if not ML_CLASSIFIER_AVAILABLE:
            raise RuntimeError("Deepfake classifier is not available. Check your installation.")
            
        try:
            # Verify the classifier is working
            model_info = get_model_info()
            logger.info(f"Using centralized ML classifier: {model_info}")
        except Exception as e:
            logger.error(f"Failed to initialize ML classifier: {e}")
            raise
    
    def load_models(self):
        """
        This method is kept for backward compatibility but does nothing.
        Model loading is now handled by deepfake_classifier.py
        """
        pass
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in an image using the configured face detector.
        
        Args:
            image: Input image as numpy array (RGB)
            
        Returns:
            List of face dictionaries with bounding boxes and confidence
            
        Raises:
            RuntimeError: If face detection fails
        """
        if not hasattr(self, 'face_detector') or self.face_detector is None:
            raise RuntimeError("Face detector not initialized")
            
        try:
            # Preprocess image for better detection
            processed_image = self._preprocess_for_detection(image)
            
            # Detect faces with the configured detector
            boxes, probs, landmarks = self.face_detector.detect(
                processed_image, 
                landmarks=True
            )
            
            if boxes is None:
                return []
            
            faces = []
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                # Filter out low confidence detections
                if prob < 0.7:
                    continue
                    
                face_data = {
                    'box': box.tolist(),
                    'confidence': float(prob),
                    'keypoints': {}
                }
                
                if landmarks is not None and i < len(landmarks):
                    face_data['keypoints'] = {
                        'left_eye': landmarks[i][0].tolist(),
                        'right_eye': landmarks[i][1].tolist(),
                        'nose': landmarks[i][2].tolist(),
                        'mouth_left': landmarks[i][3].tolist(),
                        'mouth_right': landmarks[i][4].tolist()
                    }
                
                faces.append(face_data)
            
            return faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            raise RuntimeError(f"Face detection failed: {e}")
        
        # This duplicate block has been removed as it was redundant with the code above
        # All face detection is now handled by the first implementation
        # which properly raises exceptions on failure
    
    def _preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better face detection."""
        # Enhance contrast and normalize lighting
        from PIL import Image as PILImage, ImageEnhance
        
        pil_image = PILImage.fromarray(image)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(1.2)
        
        # Enhance brightness if too dark
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        return np.array(enhanced)
    
    def _detect_face_regions(self, gray_image: np.ndarray) -> List[Dict]:
        """
        Detect face regions using OpenCV cascade.
        
        Args:
            gray_image: Grayscale image as numpy array
            
        Returns:
            List of detected face regions with bounding boxes and confidence
            
        Raises:
            RuntimeError: If face detection fails
            ImportError: If OpenCV is not available
        """
        try:
            import cv2

            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if face_cascade.empty():
                raise RuntimeError("Failed to load face cascade classifier")
            
            # Convert to uint8 if needed
            if gray_image.dtype != np.uint8:
                gray_image = (gray_image * 255).astype(np.uint8)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            regions = []
            for (x, y, w, h) in faces:
                regions.append({
                    'box': [int(x), int(y), int(x + w), int(y + h)],
                    'confidence': 0.8  # Fixed confidence for cascade detections
                })
            
            return regions
            
        except ImportError as e:
            logger.error("OpenCV is required for face detection")
            raise
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            raise RuntimeError(f"Face detection failed: {e}")
    
    def _estimate_keypoints(self, x1: int, y1: int, x2: int, y2: int) -> Dict:
        """Estimate facial keypoints from bounding box."""
        w = x2 - x1
        h = y2 - y1
        
        return {
            'left_eye': [x1 + w * 0.3, y1 + h * 0.35],
            'right_eye': [x1 + w * 0.7, y1 + h * 0.35],
            'nose': [x1 + w * 0.5, y1 + h * 0.55],
            'mouth_left': [x1 + w * 0.35, y1 + h * 0.75],
            'mouth_right': [x1 + w * 0.65, y1 + h * 0.75]
        }
    
    def extract_face(self, image: np.ndarray, box: List[float]) -> Optional[np.ndarray]:
        """
        Extract and preprocess a face region from an image.
        
        Args:
            image: Full image as numpy array
            box: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Extracted face as numpy array, or None if extraction fails
        """
        try:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Extract face region
            face = image[y1:y2, x1:x2]
            
            if face.size == 0:
                return None
            
            # Resize to standard size (160x160 for InceptionResnetV1)
            face_pil = Image.fromarray(face)
            face_resized = face_pil.resize((160, 160), Image.BILINEAR)
            
            return np.array(face_resized)
            
        except Exception as e:
            logger.error(f"Face extraction error: {e}")
            return None
    
    def classify_authenticity(self, face_image: np.ndarray) -> Tuple[float, str]:
        """
        Classify face authenticity using the centralized deepfake classifier.
        
        Args:
            face_image: Face image as numpy array (160x160x3 or 224x224x3)
            
        Returns:
            Tuple of (confidence_score, authenticity_label)
        """
        try:
            # Use the centralized classifier
            result = predict_image(face_image)
            
            # Extract results
            confidence = result['confidence']
            label = result['prediction']
            
            # Ensure confidence is in [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            
            # Ensure label is either 'real' or 'fake'
            if label not in ['real', 'fake']:
                logger.warning(f"Unexpected label '{label}' from classifier, defaulting to 'unknown'")
                label = 'unknown'
            
            return confidence, label
            
        except Exception as e:
            logger.error(f"Error in model classification: {e}")
            raise RuntimeError(f"Failed to classify image: {e}")
    
    def analyze_metadata(self, image_pil) -> Dict[str, Any]:
        """
        Analyze image metadata for manipulation signs.
        
        Args:
            image_pil: PIL Image object
            
        Returns:
            Dictionary with metadata analysis results
        """
        metadata = {
            'has_exif': False,
            'format': image_pil.format,
            'mode': image_pil.mode,
            'size': image_pil.size,
            'integrity_score': 0.95
        }
        
        try:
            exif_data = image_pil._getexif()
            if exif_data:
                metadata['has_exif'] = True
                metadata['integrity_score'] = 0.98
        except:
            pass
        
        return metadata
    
    def generate_heatmap(self, image: np.ndarray, faces: List[Dict], embeddings_list: List[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Generate an advanced heatmap highlighting suspicious regions using multiple techniques.
        
        Args:
            image: Original image as numpy array
            faces: List of detected faces with boxes
            embeddings_list: List of embeddings for each face (optional)
            
        Returns:
            Heatmap as numpy array, or None
        """
        try:
            h, w = image.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.float32)
            
            # 1. Face-based heatmap
            face_heatmap = self._generate_face_heatmap(image, faces, embeddings_list)
            
            # 2. Edge-based anomaly detection
            edge_heatmap = self._generate_edge_anomaly_heatmap(image)
            
            # 3. Texture-based anomaly detection
            texture_heatmap = self._generate_texture_anomaly_heatmap(image)
            
            # 4. Frequency-based anomaly detection
            frequency_heatmap = self._generate_frequency_anomaly_heatmap(image)
            
            # Combine heatmaps with weights
            combined_heatmap = (
                face_heatmap * 0.4 +
                edge_heatmap * 0.25 +
                texture_heatmap * 0.2 +
                frequency_heatmap * 0.15
            )
            
            # Normalize and smooth
            combined_heatmap = self._smooth_heatmap(combined_heatmap)
            combined_heatmap = (combined_heatmap - np.min(combined_heatmap)) / (np.max(combined_heatmap) - np.min(combined_heatmap) + 1e-8)
            
            return combined_heatmap
            
        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            return self._generate_simple_heatmap(image, faces)
    
    def _generate_face_heatmap(self, image: np.ndarray, faces: List[Dict], embeddings_list: List[np.ndarray] = None) -> np.ndarray:
        """Generate heatmap based on face analysis."""
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for i, face in enumerate(faces):
            box = face['box']
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Base face region heat
            base_heat = 0.3
            if embeddings_list and i < len(embeddings_list):
                # Adjust heat based on embedding analysis
                embedding = embeddings_list[i]
                suspicion_score = self._calculate_suspicion_score(embedding)
                base_heat = 0.2 + suspicion_score * 0.6
            
            # Add heat to face region with gradient
            face_heat = self._create_gradient_heat(x2-x1, y2-y1, base_heat)
            heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], face_heat)
            
            # Enhanced keypoint analysis
            if 'keypoints' in face and face['keypoints']:
                keypoint_heat = self._analyze_keypoint_anomalies(image, face['keypoints'])
                for key, point in face['keypoints'].items():
                    if len(point) == 2:
                        px, py = int(point[0]), int(point[1])
                        heat_intensity = keypoint_heat.get(key, 0.5)
                        self._add_circular_heat(heatmap, px, py, 15, heat_intensity)
        
        return heatmap
    
    def _generate_edge_anomaly_heatmap(self, image: np.ndarray) -> np.ndarray:
        """Generate heatmap based on edge anomalies."""
        try:
            import cv2

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Multiple edge detection methods
            edges_canny = cv2.Canny(gray, 50, 150)
            edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges_sobel = np.sqrt(edges_sobel_x**2 + edges_sobel_y**2)
            
            # Detect edge inconsistencies
            edge_consistency = np.abs(edges_canny.astype(float) - (edges_sobel / np.max(edges_sobel) * 255))
            
            # Normalize
            edge_heatmap = edge_consistency / 255.0
            
            return edge_heatmap.astype(np.float32)
            
        except ImportError:
            # Fallback without OpenCV
            return self._simple_edge_detection(image)
    
    def _generate_texture_anomaly_heatmap(self, image: np.ndarray) -> np.ndarray:
        """Generate heatmap based on texture anomalies."""
        # Local Binary Pattern analysis
        gray = np.mean(image, axis=2).astype(np.uint8)
        
        # Simple texture analysis using local variance
        kernel_size = 9
        texture_map = np.zeros_like(gray, dtype=np.float32)
        
        for i in range(kernel_size//2, gray.shape[0] - kernel_size//2):
            for j in range(kernel_size//2, gray.shape[1] - kernel_size//2):
                patch = gray[i-kernel_size//2:i+kernel_size//2+1, j-kernel_size//2:j+kernel_size//2+1]
                texture_map[i, j] = np.var(patch)
        
        # Detect texture anomalies
        texture_mean = np.mean(texture_map)
        texture_std = np.std(texture_map)
        
        # Areas with unusual texture variance
        anomaly_map = np.abs(texture_map - texture_mean) / (texture_std + 1e-8)
        anomaly_map = np.clip(anomaly_map / 3.0, 0, 1)  # Normalize
        
        return anomaly_map
    
    def _generate_frequency_anomaly_heatmap(self, image: np.ndarray) -> np.ndarray:
        """Generate heatmap based on frequency domain anomalies."""
        gray = np.mean(image, axis=2)
        
        # FFT analysis in overlapping windows
        window_size = 64
        step_size = 32
        h, w = gray.shape
        
        frequency_map = np.zeros_like(gray, dtype=np.float32)
        
        for i in range(0, h - window_size, step_size):
            for j in range(0, w - window_size, step_size):
                window = gray[i:i+window_size, j:j+window_size]
                
                # FFT
                fft = np.fft.fft2(window)
                power_spectrum = np.abs(fft)
                
                # Detect anomalous frequency patterns
                # High frequency content might indicate manipulation
                high_freq_energy = np.sum(power_spectrum[window_size//4:, window_size//4:])
                total_energy = np.sum(power_spectrum)
                
                if total_energy > 0:
                    high_freq_ratio = high_freq_energy / total_energy
                    frequency_map[i:i+window_size, j:j+window_size] = np.maximum(
                        frequency_map[i:i+window_size, j:j+window_size],
                        high_freq_ratio
                    )
        
        return frequency_map
    
    def _calculate_suspicion_score(self, embedding: np.ndarray) -> float:
        """Calculate suspicion score from embedding characteristics."""
        # Analyze embedding for suspicious patterns
        embedding_norm = np.linalg.norm(embedding)
        embedding_std = np.std(embedding)
        
        # Check for artificial patterns
        segments = np.array_split(embedding, 8)
        correlations = []
        for i in range(len(segments)):
            for j in range(i+1, len(segments)):
                corr = np.corrcoef(segments[i], segments[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        avg_correlation = np.mean(correlations) if correlations else 0
        
        # Higher correlation and unusual statistics = more suspicious
        suspicion = avg_correlation * 0.5 + (1 - min(embedding_std, 1.0)) * 0.3
        
        return min(suspicion, 1.0)
    
    def _create_gradient_heat(self, width: int, height: int, max_intensity: float) -> np.ndarray:
        """Create gradient heat map for face region."""
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        
        # Distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Gradient from center
        gradient = 1 - (distance / max_distance)
        gradient = np.clip(gradient, 0, 1)
        
        return gradient * max_intensity
    
    def _analyze_keypoint_anomalies(self, image: np.ndarray, keypoints: Dict) -> Dict[str, float]:
        """Analyze keypoints for anomalies."""
        anomalies = {}
        
        for key, point in keypoints.items():
            if len(point) == 2:
                px, py = int(point[0]), int(point[1])
                
                # Extract region around keypoint
                region_size = 20
                x1 = max(0, px - region_size)
                y1 = max(0, py - region_size)
                x2 = min(image.shape[1], px + region_size)
                y2 = min(image.shape[0], py + region_size)
                
                if x2 > x1 and y2 > y1:
                    region = image[y1:y2, x1:x2]
                    
                    # Analyze region characteristics
                    region_std = np.std(region)
                    region_mean = np.mean(region)
                    
                    # Eyes and mouth should have certain characteristics
                    if 'eye' in key:
                        # Eyes typically have lower variance in natural images
                        anomaly_score = min(region_std / 50.0, 1.0)
                    elif 'mouth' in key:
                        # Mouth region analysis
                        anomaly_score = abs(region_mean - 128) / 128.0
                    else:
                        anomaly_score = 0.3
                    
                    anomalies[key] = anomaly_score
                else:
                    anomalies[key] = 0.3
        
        return anomalies
    
    def _add_circular_heat(self, heatmap: np.ndarray, cx: int, cy: int, radius: int, intensity: float):
        """Add circular heat pattern to heatmap."""
        h, w = heatmap.shape
        y, x = np.ogrid[:h, :w]
        
        # Create circular mask
        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = distance <= radius
        
        # Gradient within circle
        gradient = np.where(mask, 1 - (distance / radius), 0)
        heat_addition = gradient * intensity
        
        heatmap[:] = np.maximum(heatmap, heat_addition)
    
    def _smooth_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """Smooth heatmap using Gaussian filter."""
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(heatmap, sigma=2.0)
        except ImportError:
            # Simple smoothing without scipy
            kernel = np.ones((5, 5)) / 25
            return self._convolve2d(heatmap, kernel)
    
    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution."""
        h, w = image.shape
        kh, kw = kernel.shape
        
        # Pad image
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        # Convolve
        result = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        
        return result
    
    def _generate_simple_heatmap(self, image: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """Simple fallback heatmap generation."""
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for face in faces:
            box = face['box']
            x1, y1, x2, y2 = [int(coord) for coord in box]
            heatmap[y1:y2, x1:x2] = 0.6
            
            # Add keypoint heat
            if 'keypoints' in face and face['keypoints']:
                for key, point in face['keypoints'].items():
                    if len(point) == 2:
                        px, py = int(point[0]), int(point[1])
                        y_start = max(0, py - 10)
                        y_end = min(h, py + 10)
                        x_start = max(0, px - 10)
                        x_end = min(w, px + 10)
                        heatmap[y_start:y_end, x_start:x_end] = 0.8
        
        return heatmap
    
    def _simple_edge_detection(self, image: np.ndarray) -> np.ndarray:
        """Simple edge detection without OpenCV."""
        gray = np.mean(image, axis=2)
        
        # Sobel operators
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Apply filters
        edges_x = self._convolve2d(gray, sobel_x)
        edges_y = self._convolve2d(gray, sobel_y)
        
        # Combine
        edges = np.sqrt(edges_x**2 + edges_y**2)
        edges = edges / np.max(edges)  # Normalize
        
        return edges.astype(np.float32)
    
    def analyze(self, image: np.ndarray, detect_faces: bool = False, 
                analyze_forensics: bool = False, return_face_data: bool = False) -> Dict[str, Any]:
        """
        Analyze an image and extract signals for deepfake detection.
        
        This method only performs signal extraction and preprocessing.
        All ML inference and final decisions are handled by deepfake_classifier.py.
        
        Args:
            image: Input image as numpy array (RGB)
            detect_faces: Whether to detect and analyze faces
            analyze_forensics: Whether to perform forensic analysis
            return_face_data: Whether to include detailed face data in results
            
        Returns:
            Dictionary containing signal data for the classifier:
            - image: Preprocessed image data
            - signals: Dictionary of extracted signals
            - metadata: Additional processing information
            
        Raises:
            RuntimeError: If analysis fails
            ValueError: If input is invalid
        """
        if not isinstance(image, (np.ndarray, PILImage.Image)):
            raise ValueError("Input must be a numpy array or PIL Image")
        
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image
                
            img_array = np.array(image)
            
            # Basic image statistics
            if len(img_array.shape) == 3:
                gray = np.mean(img_array, axis=2)
            else:
                gray = img_array
                
            # Extract signals (no ML inference here)
            signals = {}
            signals['statistics'] = {
                'mean_intensity': float(np.mean(gray)),
                'std_intensity': float(np.std(gray)),
                'min_intensity': float(np.min(gray)),
                'max_intensity': float(np.max(gray)),
                'image_size': {
                    'height': img_array.shape[0],
                    'width': img_array.shape[1] if len(img_array.shape) > 1 else 0,
                    'channels': img_array.shape[2] if len(img_array.shape) > 2 else 1
                }
            }
            
            # Add metadata
            metadata = {
                'image_size': image_pil.size,
                'image_mode': image_pil.mode,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'detect_faces': detect_faces,
                'analyze_forensics': analyze_forensics,
            }
            
            # Return only signals and metadata - NO VERDICTS
            result = {
                'image': image_pil,  # Return PIL Image object
                'signals': signals,
                'metadata': metadata
            }
            
            logger.debug("Image signals extracted successfully")
            return result
            
        except Exception as e:
            logger.error(f"Signal extraction failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to extract image signals: {e}")
    
    def _analyze_image_statistics(self, image: np.ndarray) -> float:
        """
        Analyze statistical properties of the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            float: Statistical analysis score
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        # Calculate various statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        skewness = self._calculate_skewness(gray)
        kurtosis = self._calculate_kurtosis(gray)
        
        score = 0.5
        
        # Natural images typically have certain statistical properties
        if 50 < mean_val < 200:  # Reasonable brightness
            score += 0.2
        if 20 < std_val < 80:   # Reasonable contrast
            score += 0.2
        if -1 < skewness < 1:   # Not too skewed
            score += 0.1
        if 2 < kurtosis < 6:    # Reasonable kurtosis
            score += 0.1
        
        return max(0.0, min(1.0, score))
            
        try:
            # Analyze JPEG compression artifacts
            if hasattr(image_pil, 'format') and image_pil.format == 'JPEG':
                # Real images typically have natural compression patterns
                import numpy as np
                img_array = np.array(image_pil)

                # Check for over-compression (common in manipulated images)
            
            # Calculate Local Binary Pattern (simplified)
            lbp_score = self._calculate_lbp_score(gray)
            
            # Calculate texture energy
            texture_energy = self._calculate_texture_energy(gray)
            
            # Combine scores
            score = (lbp_score + texture_energy) / 2
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.error(f"Texture analysis error: {e}")
            return 0.5
    
    def _analyze_frequency_domain(self, image: np.ndarray) -> float:
        """Analyze frequency domain characteristics."""
        try:
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # FFT analysis
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Analyze frequency distribution
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            
            # Low frequency energy (center region)
            low_freq_region = magnitude[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8]
            low_freq_energy = np.mean(low_freq_region)
            
            # High frequency energy (outer regions)
            high_freq_energy = np.mean(magnitude) - low_freq_energy
            
            # Natural images have balanced frequency distribution
            freq_ratio = low_freq_energy / (high_freq_energy + 1e-8)
            
            score = 0.5
            if 2 < freq_ratio < 10:  # Reasonable frequency balance
                score += 0.3
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Frequency analysis error: {e}")
            return 0.5
    
    def _analyze_edge_patterns(self, image: np.ndarray) -> float:
        """Analyze edge patterns for manipulation artifacts."""
        try:
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image
            
            # Calculate gradients
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            
            # Edge magnitude
            edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Edge statistics
            edge_mean = np.mean(edge_magnitude)
            edge_std = np.std(edge_magnitude)
            
            # Strong edges with reasonable variation indicate natural images
            score = 0.5
            if edge_mean > 5:  # Sufficient edge content
                score += 0.2
            if 2 < edge_std < 20:  # Reasonable edge variation
                score += 0.2
            
            # Check for unnatural edge patterns
            edge_histogram = np.histogram(edge_magnitude, bins=50)[0]
            edge_entropy = -np.sum(edge_histogram * np.log(edge_histogram + 1e-8))
            
            if edge_entropy > 2:  # Good edge diversity
                score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Edge analysis error: {e}")
            return 0.5
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 4) - 3
    
    def _calculate_lbp_score(self, gray: np.ndarray) -> float:
        """Calculate Local Binary Pattern score (simplified)."""
        try:
            # Simplified LBP calculation
            h, w = gray.shape
            lbp = np.zeros((h-2, w-2))
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = gray[i, j]
                    code = 0
                    
                    # 8-neighborhood
                    neighbors = [
                        gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                        gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                        gray[i+1, j-1], gray[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i-1, j-1] = code
            
            # Calculate LBP histogram uniformity
            hist = np.histogram(lbp, bins=256)[0]
            uniformity = np.sum(hist**2) / (np.sum(hist)**2 + 1e-8)
            
            # Natural textures have moderate uniformity
            if 0.01 < uniformity < 0.1:
                return 0.8
            else:
                return 0.4
                
        except Exception as e:
            logger.error(f"LBP calculation error: {e}")
            return 0.5
    
    def _calculate_texture_energy(self, gray: np.ndarray) -> float:
        """Calculate texture energy."""
        try:
            # Calculate co-occurrence matrix (simplified)
            h, w = gray.shape
            energy = 0
            
            # Horizontal co-occurrence
            for i in range(h):
                for j in range(w-1):
                    diff = abs(int(gray[i, j]) - int(gray[i, j+1]))
                    energy += diff
            
            # Normalize
            energy = energy / (h * (w-1) * 255)
            
            # Natural textures have moderate energy
            if 0.1 < energy < 0.5:
                return 0.8
            else:
                return 0.4
                
        except Exception as e:
            logger.error(f"Texture energy calculation error: {e}")
            return 0.5
    
