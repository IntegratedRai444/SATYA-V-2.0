
import io
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Configure logging
import logging
logger = logging.getLogger(__name__)

# Initialize with safe defaults
ML_CLASSIFIER_AVAILABLE = False
ADVANCED_IMAGE_MODEL_AVAILABLE = False

# Try to import the centralized ML classifier
try:
    from models.deepfake_classifier import get_model_info, predict_image, is_model_available
    ML_CLASSIFIER_AVAILABLE = is_model_available()
    if not ML_CLASSIFIER_AVAILABLE:
        logger.warning("ML classifier is not available. Running in limited mode.")
    else:
        logger.info("ML classifier is available")
except ImportError as e:
    logger.warning(f"Failed to import ML classifier: {e}")
    # Continue without ML

try:
    from models.image_model import SwinTransformer as AdvancedImageDetector
    # Just check if we can import it, don't try to use it yet
    ADVANCED_IMAGE_MODEL_AVAILABLE = True
    logger.info("Advanced image model (Swin Transformer) available")
except ImportError as e:
    ADVANCED_IMAGE_MODEL_AVAILABLE = False
    logger.warning(f"Advanced image model not available: {e}")
except Exception as e:
    ADVANCED_IMAGE_MODEL_AVAILABLE = False
    logger.warning(f"Error initializing advanced image model: {e}")

# Import forensic analysis (non-ML analysis)
try:
    from .forensic_analysis import analyze_ela, analyze_prnu
    FORENSIC_ANALYSIS_AVAILABLE = True
except ImportError:
    FORENSIC_ANALYSIS_AVAILABLE = False
    logger.warning("Forensic analysis not available")

# Import enhanced forensic methods from detector.py
class ForensicAnalyzer:
    """Advanced forensic analysis capabilities integrated from detector.py"""
    
    @staticmethod
    def analyze_ela(image: np.ndarray) -> Dict[str, Any]:
        """Error Level Analysis for detecting compression artifacts"""
        try:
            from PIL import Image, ImageEnhance
            
            # Convert to PIL if needed
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Save at quality 95, then resave at quality 75 and compare
            original_temp = io.BytesIO()
            pil_image.save(original_temp, 'JPEG', quality=95)
            original_temp.seek(0)
            
            # Resave at lower quality
            resaved_temp = io.BytesIO()
            temp_img = Image.open(original_temp)
            temp_img.save(resaved_temp, 'JPEG', quality=75)
            resaved_temp.seek(0)
            
            # Calculate difference
            from PIL import ImageChops
            ela_img = ImageChops.difference(Image.open(original_temp), Image.open(resaved_temp))
            
            # Enhance the difference
            ela_img = ImageEnhance.Contrast(ela_img).enhance(10.0)
            
            # Calculate ELA score
            ela_array = np.array(ela_img.convert('L'))
            ela_score = np.mean(ela_array) / 255.0
            
            return {
                'ela_score': ela_score,
                'ela_mean': float(np.mean(ela_array)),
                'ela_std': float(np.std(ela_array)),
                'analysis': 'High ELA scores may indicate manipulation'
            }
            
        except Exception as e:
            logger.error(f"ELA analysis failed: {e}")
            return {'ela_score': 0.5, 'error': str(e)}
    
    @staticmethod
    def analyze_prnu(image: np.ndarray) -> Dict[str, Any]:
        """Photo Response Non-Uniformity analysis"""
        try:
            # Simple PRNU approximation using noise patterns
            gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
            
            # Apply wavelet-like denoising
            from scipy import ndimage
            
            # Gaussian blur as reference
            blurred = ndimage.gaussian_filter(gray, sigma=2)
            
            # Calculate noise residual
            noise_residual = gray.astype(float) - blurred.astype(float)
            
            # Calculate PRNU characteristics
            prnu_mean = np.mean(noise_residual)
            prnu_std = np.std(noise_residual)
            prnu_skewness = float((noise_residual - prnu_mean)**3).mean() / (prnu_std**3 + 1e-8)
            
            # PRNU consistency score (lower is more consistent)
            prnu_consistency = prnu_std / (np.abs(prnu_mean) + 1e-8)
            
            return {
                'prnu_mean': float(prnu_mean),
                'prnu_std': float(prnu_std),
                'prnu_skewness': prnu_skewness,
                'prnu_consistency': float(prnu_consistency),
                'analysis': 'PRNU inconsistencies may indicate manipulation'
            }
            
        except Exception as e:
            logger.error(f"PRNU analysis failed: {e}")
            return {'prnu_consistency': 1.0, 'error': str(e)}
    
    @staticmethod
    def analyze_frequency_domain(image: np.ndarray) -> Dict[str, Any]:
        """Frequency domain analysis for detecting manipulation artifacts"""
        try:
            gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
            
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
            
            # Frequency ratio
            freq_ratio = low_freq_energy / (high_freq_energy + 1e-8)
            
            # Detect unusual frequency patterns
            freq_anomaly_score = 0.0
            if freq_ratio < 2:  # Too much high frequency
                freq_anomaly_score += 0.3
            if freq_ratio > 10:  # Too much low frequency
                freq_anomaly_score += 0.3
            
            return {
                'low_freq_energy': float(low_freq_energy),
                'high_freq_energy': float(high_freq_energy),
                'frequency_ratio': float(freq_ratio),
                'freq_anomaly_score': freq_anomaly_score,
                'analysis': 'Unusual frequency patterns may indicate manipulation'
            }
            
        except Exception as e:
            logger.error(f"Frequency analysis failed: {e}")
            return {'freq_anomaly_score': 0.0, 'error': str(e)}
    
    @staticmethod
    def analyze_texture_patterns(image: np.ndarray) -> Dict[str, Any]:
        """Texture analysis for detecting inconsistencies"""
        try:
            gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
            
            # Local Binary Pattern approximation
            kernel_size = 9
            texture_map = np.zeros_like(gray, dtype=np.float32)
            
            for i in range(kernel_size//2, gray.shape[0] - kernel_size//2):
                for j in range(kernel_size//2, gray.shape[1] - kernel_size//2):
                    patch = gray[i-kernel_size//2:i+kernel_size//2+1, j-kernel_size//2:j+kernel_size//2+1]
                    texture_map[i, j] = np.var(patch)
            
            # Analyze texture patterns
            texture_mean = np.mean(texture_map)
            texture_std = np.std(texture_map)
            
            # Detect texture anomalies
            anomaly_map = np.abs(texture_map - texture_mean) / (texture_std + 1e-8)
            anomaly_score = np.mean(anomaly_map > 2.0)  # Percentage of anomalous regions
            
            return {
                'texture_mean': float(texture_mean),
                'texture_std': float(texture_std),
                'anomaly_score': float(anomaly_score),
                'analysis': 'Texture inconsistencies may indicate manipulation'
            }
            
        except Exception as e:
            logger.error(f"Texture analysis failed: {e}")
            return {'anomaly_score': 0.0, 'error': str(e)}


class ImageDetector:
    """
    Image Deepfake Detector with advanced multi-modal capabilities
    """
    
    def __init__(self, model_path: str = None, enable_gpu: bool = False, use_advanced_model: bool = False):
        """
        Initialize image detector with advanced multi-modal capabilities.
        
        Args:
            model_path: Path to model files (kept for backward compatibility)
            enable_gpu: Whether to use GPU acceleration
            use_advanced_model: Whether to use advanced Swin Transformer model
            
        Note:
            All ML model loading and inference is now handled by deepfake_classifier.py
            Advanced model provides Swin Transformer with multi-scale attention
        """
        self.model_path = model_path  # Kept for backward compatibility
        self.enable_gpu = enable_gpu
        self.use_advanced_model = use_advanced_model
        self.models_available = False
        self.advanced_model = None
        
        # Initialize advanced model if requested and available
        if use_advanced_model and ADVANCED_IMAGE_MODEL_AVAILABLE:
            try:
                self.advanced_model = AdvancedImageDetector(
                    model_path=model_path,
                    device='cuda' if enable_gpu else 'cpu'
                )
                logger.info("Advanced Swin Transformer model initialized")
                self.models_available = True
            except Exception as e:
                logger.error(f"Failed to initialize advanced model: {e}")
                self.advanced_model = None
        
        # Check if ML classifier is available
        if ML_CLASSIFIER_AVAILABLE:
            try:
                # Verify classifier is working
                model_info = get_model_info()
                logger.info(f"Using centralized ML classifier: {model_info}")
                self.models_available = True
            except Exception as e:
                logger.error(f"Failed to initialize ML classifier: {e}")
        
        if not self.models_available:
            logger.warning("No ML models available. Running in limited mode with basic analysis only.")
            
        # Set up basic image processing
        self._setup_basic_analysis()
    
    def _setup_basic_analysis(self):
        """
        Initialize basic image analysis components that don't require ML models.
        This method sets up the basic analysis pipeline that will work even without ML models.
        """
        try:
            # Initialize basic image processing components
            self.forensic_analyzer = ForensicAnalyzer()
            logger.info("Basic image analysis components initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize basic analysis components: {e}")
            # Set a flag to indicate basic analysis is not available
            self.basic_analysis_available = False
        else:
            self.basic_analysis_available = True
    
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
        Classify face authenticity using both standard and advanced ML models.
        
        Args:
            face_image: Face image as numpy array (160x160x3 or 224x224x3)
            
        Returns:
            Tuple of (confidence_score, authenticity_label)
        """
        try:
            # Use advanced model if available and enabled
            if self.advanced_model and self.use_advanced_model:
                logger.info("Using advanced Swin Transformer model for analysis")
                result = self.advanced_model.predict_deepfake(face_image)
                confidence = result.get('confidence', 0.5)
                label = result.get('prediction', 'unknown')
            else:
                # Use centralized classifier
                result = predict_image(face_image)
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
                analyze_forensics: bool = True, return_face_data: bool = False) -> Dict[str, Any]:
        """
        Enhanced image analysis with comprehensive forensic analysis and ML integration.
        
        Args:
            image: Input image as numpy array (RGB)
            detect_faces: Whether to detect and analyze faces
            analyze_forensics: Whether to perform forensic analysis
            return_face_data: Whether to include detailed face data in results
            
        Returns:
            Dictionary containing comprehensive analysis results:
            - ml_analysis: ML model predictions
            - forensic_analysis: Detailed forensic analysis
            - technical_details: Processing information
            - authenticity: Final authenticity assessment
            - confidence: Overall confidence score
        """
        if not isinstance(image, (np.ndarray, Image.Image)):
            raise ValueError("Input must be a numpy array or PIL Image")
        
        start_time = time.time()
        
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image
                image = np.array(image_pil)
                
            # Initialize result structure
            result = {
                'success': True,
                'authenticity': 'UNCERTAIN',
                'confidence': 0.5,
                'analysis_date': datetime.utcnow().isoformat(),
                'ml_analysis': {},
                'forensic_analysis': {},
                'technical_details': {
                    'image_size': image_pil.size,
                    'image_mode': image_pil.mode,
                    'processing_timestamp': datetime.utcnow().isoformat(),
                    'detect_faces': detect_faces,
                    'analyze_forensics': analyze_forensics,
                },
                'key_findings': [],
                'detailed_analysis': {}
            }
            
            # 1. ML Analysis (if available)
            if ML_CLASSIFIER_AVAILABLE:
                try:
                    ml_result = predict_image(image)
                    result['ml_analysis'] = {
                        'prediction': ml_result['prediction'],
                        'confidence': ml_result['confidence'],
                        'class_probs': ml_result.get('class_probs', {}),
                        'inference_time': ml_result.get('inference_time', 0.0),
                        'performance_metrics': ml_result.get('performance_metrics', {})
                    }
                    result['key_findings'].append(f"ML model predicts: {ml_result['prediction']} with {ml_result['confidence']:.2f} confidence")
                except Exception as e:
                    logger.error(f"ML analysis failed: {e}")
                    result['ml_analysis'] = {'error': str(e)}
                    result['key_findings'].append("ML analysis failed - using forensic analysis only")
            else:
                result['key_findings'].append("ML models not available - using forensic analysis only")
            
            # 2. Forensic Analysis (if requested)
            if analyze_forensics:
                try:
                    forensic_results = self._comprehensive_forensic_analysis(image)
                    result['forensic_analysis'] = forensic_results
                    
                    # Extract key forensic findings
                    if 'ela_analysis' in forensic_results:
                        ela_score = forensic_results['ela_analysis'].get('ela_score', 0.5)
                        if ela_score > 0.7:
                            result['key_findings'].append(f"High ELA score ({ela_score:.2f}) suggests possible manipulation")
                    
                    if 'frequency_analysis' in forensic_results:
                        freq_score = forensic_results['frequency_analysis'].get('freq_anomaly_score', 0.0)
                        if freq_score > 0.3:
                            result['key_findings'].append(f"Frequency anomalies detected ({freq_score:.2f})")
                    
                    if 'texture_analysis' in forensic_results:
                        texture_score = forensic_results['texture_analysis'].get('anomaly_score', 0.0)
                        if texture_score > 0.2:
                            result['key_findings'].append(f"Texture inconsistencies detected ({texture_score:.2f})")
                            
                except Exception as e:
                    logger.error(f"Forensic analysis failed: {e}")
                    result['forensic_analysis'] = {'error': str(e)}
                    result['key_findings'].append("Forensic analysis failed")
            
            # 3. Face Analysis (if requested)
            if detect_faces:
                try:
                    face_results = self._analyze_faces(image)
                    result['face_analysis'] = face_results
                    result['key_findings'].append(f"Detected {len(face_results.get('faces', []))} face(s)")
                except Exception as e:
                    logger.error(f"Face analysis failed: {e}")
                    result['face_analysis'] = {'error': str(e)}
            
            # 4. Comprehensive scoring and final assessment
            authenticity, confidence = self._calculate_comprehensive_score(result)
            result['authenticity'] = authenticity
            result['confidence'] = confidence
            
            # 5. Processing time
            processing_time = time.time() - start_time
            result['technical_details']['processing_time_seconds'] = processing_time
            
            if os.environ.get('PYTHON_ENV') == 'development':
                logger.debug(f"Comprehensive image analysis completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}", exc_info=True)
            return {
                'success': False,
                'authenticity': 'UNCERTAIN',
                'confidence': 0.0,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _comprehensive_forensic_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive forensic analysis using multiple techniques.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing detailed forensic analysis results
        """
        forensic_results = {}
        
        try:
            # 1. Error Level Analysis
            ela_result = ForensicAnalyzer.analyze_ela(image)
            forensic_results['ela_analysis'] = ela_result
            
            # 2. PRNU Analysis
            prnu_result = ForensicAnalyzer.analyze_prnu(image)
            forensic_results['prnu_analysis'] = prnu_result
            
            # 3. Frequency Domain Analysis
            freq_result = ForensicAnalyzer.analyze_frequency_domain(image)
            forensic_results['frequency_analysis'] = freq_result
            
            # 4. Texture Analysis
            texture_result = ForensicAnalyzer.analyze_texture_patterns(image)
            forensic_results['texture_analysis'] = texture_result
            
            # 5. Statistical Analysis
            stat_result = self._analyze_image_statistics(image)
            forensic_results['statistical_analysis'] = {
                'stat_score': stat_result,
                'analysis': 'Statistical properties analysis'
            }
            
            # 6. Metadata Analysis
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image
            
            metadata_result = self.analyze_metadata(image_pil)
            forensic_results['metadata_analysis'] = metadata_result
            
            # 7. Overall forensic score
            forensic_score = self._calculate_forensic_score(forensic_results)
            forensic_results['overall_forensic_score'] = forensic_score
            
            logger.info("Comprehensive forensic analysis completed")
            return forensic_results
            
        except Exception as e:
            logger.error(f"Forensic analysis failed: {e}")
            return {'error': str(e), 'overall_forensic_score': 0.5}
    
    def _calculate_forensic_score(self, forensic_results: Dict[str, Any]) -> float:
        """
        Calculate overall forensic score based on all analysis results.
        
        Args:
            forensic_results: Dictionary containing all forensic analysis results
            
        Returns:
            float: Overall forensic score (0.0 = definitely fake, 1.0 = definitely real)
        """
        scores = []
        weights = []
        
        # ELA Analysis (weight: 0.25)
        if 'ela_analysis' in forensic_results:
            ela_score = forensic_results['ela_analysis'].get('ela_score', 0.5)
            # High ELA score suggests manipulation, so invert
            forensic_ela_score = 1.0 - min(ela_score, 1.0)
            scores.append(forensic_ela_score)
            weights.append(0.25)
        
        # PRNU Analysis (weight: 0.20)
        if 'prnu_analysis' in forensic_results:
            prnu_consistency = forensic_results['prnu_analysis'].get('prnu_consistency', 1.0)
            # Lower consistency suggests manipulation
            forensic_prnu_score = 1.0 / (1.0 + prnu_consistency)
            scores.append(forensic_prnu_score)
            weights.append(0.20)
        
        # Frequency Analysis (weight: 0.20)
        if 'frequency_analysis' in forensic_results:
            freq_anomaly = forensic_results['frequency_analysis'].get('freq_anomaly_score', 0.0)
            # Higher anomaly suggests manipulation
            forensic_freq_score = 1.0 - min(freq_anomaly, 1.0)
            scores.append(forensic_freq_score)
            weights.append(0.20)
        
        # Texture Analysis (weight: 0.15)
        if 'texture_analysis' in forensic_results:
            texture_anomaly = forensic_results['texture_analysis'].get('anomaly_score', 0.0)
            # Higher anomaly suggests manipulation
            forensic_texture_score = 1.0 - min(texture_anomaly, 1.0)
            scores.append(forensic_texture_score)
            weights.append(0.15)
        
        # Statistical Analysis (weight: 0.10)
        if 'statistical_analysis' in forensic_results:
            stat_score = forensic_results['statistical_analysis'].get('stat_score', 0.5)
            scores.append(stat_score)
            weights.append(0.10)
        
        # Metadata Analysis (weight: 0.10)
        if 'metadata_analysis' in forensic_results:
            integrity_score = forensic_results['metadata_analysis'].get('integrity_score', 0.95)
            scores.append(integrity_score)
            weights.append(0.10)
        
        # Calculate weighted average
        if scores and weights:
            forensic_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        else:
            forensic_score = 0.5  # Neutral if no analysis available
        
        return max(0.0, min(1.0, forensic_score))
    
    def _calculate_comprehensive_score(self, result: Dict[str, Any]) -> Tuple[str, float]:
        """
        Calculate comprehensive authenticity score combining ML and forensic analysis.
        
        Args:
            result: Dictionary containing all analysis results
            
        Returns:
            Tuple of (authenticity_label, confidence_score)
        """
        ml_score = 0.5
        forensic_score = 0.5
        
        # Extract ML score
        if 'ml_analysis' in result and result['ml_analysis']:
            ml_analysis = result['ml_analysis']
            if 'prediction' in ml_analysis and 'confidence' in ml_analysis:
                if ml_analysis['prediction'] == 'real':
                    ml_score = ml_analysis['confidence']
                else:  # 'fake'
                    ml_score = 1.0 - ml_analysis['confidence']
        
        # Extract forensic score
        if 'forensic_analysis' in result and result['forensic_analysis']:
            forensic_score = result['forensic_analysis'].get('overall_forensic_score', 0.5)
        
        # Combine scores with weights (ML: 0.6, Forensic: 0.4)
        combined_score = ml_score * 0.6 + forensic_score * 0.4
        
        # Determine authenticity label
        if combined_score >= 0.7:
            authenticity = 'REAL'
        elif combined_score <= 0.3:
            authenticity = 'FAKE'
        else:
            authenticity = 'UNCERTAIN'
        
        # Calculate confidence based on agreement between methods
        ml_confidence = result.get('ml_analysis', {}).get('confidence', 0.5)
        forensic_confidence = abs(forensic_score - 0.5) * 2  # Convert to 0-1 scale
        
        # Higher confidence when methods agree
        score_diff = abs(ml_score - forensic_score)
        agreement_bonus = max(0, 1.0 - score_diff) * 0.2
        
        confidence = (ml_confidence + forensic_confidence) / 2 + agreement_bonus
        confidence = max(0.0, min(1.0, confidence))
        
        return authenticity, confidence
    
    def _analyze_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze faces in the image for manipulation detection.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing face analysis results
        """
        try:
            # Try to use advanced face detection if available
            if hasattr(self, 'detect_faces'):
                faces = self.detect_faces(image)
            else:
                # Fallback to basic face detection
                faces = self._detect_face_regions(np.mean(image, axis=2) if len(image.shape) == 3 else image)
            
            face_results = {
                'faces': faces,
                'face_count': len(faces),
                'analysis': 'Face detection and analysis completed'
            }
            
            # Analyze each face for manipulation
            for i, face in enumerate(faces):
                if 'box' in face:
                    box = face['box']
                    face_image = self.extract_face(image, box)
                    if face_image is not None:
                        # Classify face authenticity
                        face_confidence, face_label = self.classify_authenticity(face_image)
                        face['authenticity'] = face_label
                        face['confidence'] = face_confidence
            
            return face_results
            
        except Exception as e:
            logger.error(f"Face analysis failed: {e}")
            return {'error': str(e), 'faces': [], 'face_count': 0}
    
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
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return float(np.mean(((data - mean) / std) ** 4)) - 3.0
    
    def _calculate_lbp_score(self, gray: np.ndarray) -> float:
        """Calculate Local Binary Pattern score."""
        try:
            # Simple LBP approximation
            h, w = gray.shape
            lbp = np.zeros_like(gray)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = gray[i, j]
                    code = 0
                    
                    # 8 neighbors
                    neighbors = [
                        gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                        gray[i, j+1], gray[i+1, j+1], gray[i+1, j], gray[i+1, j-1], gray[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i, j] = code
            
            # Calculate LBP histogram uniformity
            hist, _ = np.histogram(lbp, bins=256, range=(0, 255))
            hist = hist.astype(float)
            hist = hist / (np.sum(hist) + 1e-8)
            
            # Uniformity score (lower is more uniform)
            uniformity = np.sum(hist ** 2)
            return 1.0 - min(uniformity, 1.0)
            
        except Exception as e:
            logger.error(f"LBP calculation error: {e}")
            return 0.5
    
    def _calculate_texture_energy(self, gray: np.ndarray) -> float:
        """Calculate texture energy."""
        try:
            # Simple texture energy calculation
            kernel_size = 5
            energy_map = np.zeros_like(gray, dtype=np.float32)
            
            for i in range(kernel_size//2, gray.shape[0] - kernel_size//2):
                for j in range(kernel_size//2, gray.shape[1] - kernel_size//2):
                    patch = gray[i-kernel_size//2:i+kernel_size//2+1, j-kernel_size//2:j+kernel_size//2+1]
                    energy_map[i, j] = np.sum(patch ** 2)
            
            # Normalize and return average energy
            avg_energy = np.mean(energy_map) / (255 ** 2)
            return min(avg_energy, 1.0)
            
        except Exception as e:
            logger.error(f"Texture energy calculation error: {e}")
            return 0.5
