"""
Detector Singleton Service
Ensures ML detectors are initialized only once and shared across the application
Hybrid loading: Image = Eager, Others = Lazy
"""

import logging
import os
from typing import Optional, Dict, Any
from threading import Lock

logger = logging.getLogger(__name__)

class DetectorSingleton:
    """
    Thread-safe singleton for ML detectors
    Hybrid loading strategy: Image models preload eagerly, others load lazily
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # 🔥 PRODUCTION: Hybrid loading configuration
        self._detectors = {}
        self._config = {}
        self._eager_loaded = set()  # Track eagerly loaded models
        self._initialized = True
        
        # Check if image model should be eagerly loaded
        self._eager_load_image = os.getenv('EAGER_LOAD_IMAGE', 'true').lower() == 'true'
        
        logger.info(f"🔒 DetectorSingleton initialized (hybrid loading: image={'eager' if self._eager_load_image else 'lazy'})")
    
    async def eager_load_image_detector(self):
        """
        🔥 PRODUCTION: Eagerly load image detector during warmup
        This ensures image_detector=true after warmup, no lazy loading dependency
        """
        if not self._eager_load_image:
            logger.info("⚠️ Eager image loading disabled via EAGER_LOAD_IMAGE=false")
            return False
            
        if 'image' in self._eager_loaded:
            logger.info("✅ Image detector already eagerly loaded")
            return True
            
        try:
            with self._lock:
                if 'image' in self._detectors:
                    logger.info("✅ Image detector already available (cached)")
                    self._eager_loaded.add('image')
                    return True
                
                logger.info("🔥 Eager loading image detector (warmup initialization)")
                
                # 🔥 PRODUCTION: Safe memory guards for eager loading
                import torch
                torch.set_num_threads(2)  # Limit threads for memory efficiency
                
                # Enable GPU by default if available
                enable_gpu = self._is_cuda_available()
                
                # Initialize image detector with memory safety
                with torch.no_grad():  # Memory efficient
                    from detectors.image_detector import ImageDetector
                    detector = ImageDetector(enable_gpu=enable_gpu)
                
                if detector is not None:
                    self._detectors['image'] = detector
                    self._eager_loaded.add('image')
                    logger.info("✅ Image detector eagerly loaded and cached")
                    return True
                else:
                    logger.error("❌ Image detector initialization returned None")
                    return False
                    
        except MemoryError as e:
            logger.error(f"❌ Image detector eager loading failed - MemoryError: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Image detector eager loading failed: {type(e).__name__}: {e}")
            return False
    
    def get_detector(self, detector_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Get a detector instance, initializing it if necessary
        Hybrid loading: Image = eager (if preloaded), Others = lazy
        """
        with self._lock:
            if detector_type in self._detectors:
                return self._detectors[detector_type]
            
            # For image detector, check if it was eagerly loaded
            if detector_type == 'image' and detector_type in self._eager_loaded:
                logger.info("✅ Using eagerly loaded image detector")
                return self._detectors.get(detector_type)
            
            # Lazy loading for other detectors or fallback
            logger.info(f"🔧 Lazy loading {detector_type} detector")
            detector = self._initialize_detector(detector_type, config or {})
            if detector:
                self._detectors[detector_type] = detector
                logger.info(f"✅ {detector_type} detector cached in singleton")
            
            return detector
    
    def get_image_detector(self):
        """
        🔥 PRODUCTION: Explicit method to get image detector
        Forces initialization if not already loaded
        """
        with self._lock:
            if 'image' in self._detectors:
                return self._detectors['image']
            
            # Force initialization if not available
            logger.info("🔥 Forcing image detector initialization")
            detector = self._initialize_detector('image', {})
            if detector:
                self._detectors['image'] = detector
                logger.info("✅ Image detector force-initialized and cached")
            
            return detector
    
    def _initialize_detector(self, detector_type: str, config: Dict[str, Any]):
        """
        Initialize a single detector of the given type
        """
        try:
            # Enable GPU by default if available
            enable_gpu = config.get('enable_gpu', self._is_cuda_available())
            
            if detector_type == 'image':
                from detectors.image_detector import ImageDetector
                return ImageDetector(
                    enable_gpu=enable_gpu
                )
            elif detector_type == 'video':
                from detectors.video_detector import VideoDetector
                device = 'cuda' if enable_gpu and self._is_cuda_available() else 'cpu'
                return VideoDetector(config={'device': device} if device == 'cuda' else {})
            elif detector_type == 'audio':
                from detectors.audio_detector import AudioDetector
                device = 'cuda' if enable_gpu and self._is_cuda_available() else 'cpu'
                return AudioDetector(device=device)
            elif detector_type == 'text':
                from detectors.text_nlp_detector import TextNLPDetector
                return TextNLPDetector()
            else:
                logger.warning(f"Unknown detector type: {detector_type}")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to initialize {detector_type} detector: {e}")
            return None
    
    def _is_cuda_available(self):
        """Check if CUDA is available with robust error handling"""
        try:
            import torch
            # First check if CUDA is available
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA not available - will use CPU")
                return False
            
            # Test CUDA device access to catch DLL issues
            try:
                # Try to access CUDA device 0
                device = torch.device('cuda:0')
                # Test with a simple tensor operation
                test_tensor = torch.randn(1, 3, 224, 224).to(device)
                _ = test_tensor.sum()  # Simple operation
                logger.info("✅ CUDA device test passed")
                return True
            except Exception as cuda_error:
                logger.error(f"❌ CUDA device test failed: {cuda_error}")
                logger.warning("⚠️ Falling back to CPU due to CUDA DLL/device issues")
                return False
                
        except ImportError:
            logger.warning("⚠️ PyTorch not available")
            return False
        except Exception as e:
            logger.error(f"❌ CUDA availability check failed: {e}")
            return False
    
    def get_all_detectors(self) -> Dict[str, Any]:
        """Get all initialized detectors"""
        with self._lock:
            return self._detectors.copy()
    
    def is_detector_available(self, detector_type: str) -> bool:
        """Check if a detector is available"""
        with self._lock:
            return detector_type in self._detectors and self._detectors[detector_type] is not None
    
    def get_detector_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all detectors including eager loading info"""
        with self._lock:
            status = {}
            for detector_type, detector in self._detectors.items():
                status[detector_type] = {
                    'available': detector is not None,
                    'initialized': detector is not None,
                    'type': detector_type,
                    'eager_loaded': detector_type in self._eager_loaded
                }
            return status
    
    def is_image_detector_eager_loaded(self) -> bool:
        """Check if image detector was eagerly loaded"""
        with self._lock:
            return 'image' in self._eager_loaded and 'image' in self._detectors
    
    def clear_detector(self, detector_type: str):
        """Clear a specific detector from cache"""
        with self._lock:
            if detector_type in self._detectors:
                del self._detectors[detector_type]
                logger.info(f"🗑️ Cleared {detector_type} detector from singleton")
    
    def clear_all_detectors(self):
        """Clear all detectors from cache"""
        with self._lock:
            self._detectors.clear()
            logger.info("🗑️ Cleared all detectors from singleton")

# Global singleton instance
_detector_singleton = None

def get_detector_singleton():
    """Get the global detector singleton instance"""
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = DetectorSingleton()
    return _detector_singleton
