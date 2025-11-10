"""
SatyaAI Core Module
Main interface for the deepfake detection system
"""

import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime

# Import detectors
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.image_detector import ImageDetector
from detectors.video_detector import VideoDetector
from detectors.audio_detector import AudioDetector
from detectors.fusion_engine import FusionEngine

logger = logging.getLogger(__name__)


class SatyaAICore:
    """
    Main SatyaAI detection engine that coordinates all modalities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SatyaAI with configuration.
        
        Args:
            config: Configuration dictionary with model paths and settings
        """
        self.config = config
        self.model_path = config.get('MODEL_PATH', './models')
        self.enable_gpu = config.get('ENABLE_GPU', False)
        
        # Initialize detectors
        self.image_detector = None
        self.video_detector = None
        self.audio_detector = None
        self.fusion_engine = None
        
        # Initialize components
        self._initialize_components()
        
        logger.info("SatyaAI Core initialized successfully")
    
    def _initialize_components(self):
        """Initialize all detection components."""
        try:
            # Initialize detectors
            self.image_detector = ImageDetector(self.model_path, self.enable_gpu)
            self.video_detector = VideoDetector(self.model_path, self.enable_gpu)
            self.audio_detector = AudioDetector(self.model_path, self.enable_gpu)
            
            # Initialize fusion engine
            self.fusion_engine = FusionEngine()
            
            logger.info("All detection components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            # Continue with limited functionality
    
    def analyze_image(self, image_buffer: bytes) -> Dict[str, Any]:
        """
        Analyze an image for deepfake detection.
        
        Args:
            image_buffer: Image data as bytes
            
        Returns:
            Analysis result dictionary
        """
        try:
            if not self.image_detector:
                return self._create_fallback_result("Image detector not available")
            
            logger.info("Starting image analysis")
            result = self.image_detector.analyze(image_buffer)
            
            # Add metadata
            result.update({
                'analysis_date': datetime.now().isoformat(),
                'case_id': f"img-{int(datetime.now().timestamp())}",
                'server_version': '2.0.0'
            })
            
            logger.info(f"Image analysis completed: {result.get('authenticity', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return self._create_error_result(f"Image analysis failed: {str(e)}")
    
    def analyze_video(self, video_buffer: bytes) -> Dict[str, Any]:
        """
        Analyze a video for deepfake detection.
        
        Args:
            video_buffer: Video data as bytes
            
        Returns:
            Analysis result dictionary
        """
        try:
            if not self.video_detector:
                return self._create_fallback_result("Video detector not available")
            
            logger.info("Starting video analysis")
            result = self.video_detector.analyze(video_buffer)
            
            # Add metadata
            result.update({
                'analysis_date': datetime.now().isoformat(),
                'case_id': f"vid-{int(datetime.now().timestamp())}",
                'server_version': '2.0.0'
            })
            
            logger.info(f"Video analysis completed: {result.get('authenticity', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return self._create_error_result(f"Video analysis failed: {str(e)}")
    
    def analyze_audio(self, audio_buffer: bytes) -> Dict[str, Any]:
        """
        Analyze audio for deepfake detection.
        
        Args:
            audio_buffer: Audio data as bytes
            
        Returns:
            Analysis result dictionary
        """
        try:
            if not self.audio_detector:
                return self._create_fallback_result("Audio detector not available")
            
            logger.info("Starting audio analysis")
            result = self.audio_detector.analyze(audio_buffer)
            
            # Add metadata
            result.update({
                'analysis_date': datetime.now().isoformat(),
                'case_id': f"aud-{int(datetime.now().timestamp())}",
                'server_version': '2.0.0'
            })
            
            logger.info(f"Audio analysis completed: {result.get('authenticity', 'Unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return self._create_error_result(f"Audio analysis failed: {str(e)}")
    
    def analyze_multimodal(
        self, 
        image_buffer: Optional[bytes] = None,
        audio_buffer: Optional[bytes] = None,
        video_buffer: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Perform multimodal analysis combining multiple media types.
        
        Args:
            image_buffer: Optional image data
            audio_buffer: Optional audio data
            video_buffer: Optional video data
            
        Returns:
            Unified analysis result
        """
        try:
            if not self.fusion_engine:
                return self._create_fallback_result("Fusion engine not available")
            
            logger.info("Starting multimodal analysis")
            
            # Analyze each modality
            results = {}
            
            if image_buffer:
                results['image'] = self.analyze_image(image_buffer)
            
            if video_buffer:
                results['video'] = self.analyze_video(video_buffer)
            
            if audio_buffer:
                results['audio'] = self.analyze_audio(audio_buffer)
            
            if not results:
                return self._create_error_result("No valid media provided for analysis")
            
            # Fuse results
            fused_result = self.fusion_engine.fuse(results)
            
            logger.info(f"Multimodal analysis completed: {fused_result.get('authenticity', 'Unknown')}")
            return fused_result
            
        except Exception as e:
            logger.error(f"Multimodal analysis failed: {e}")
            return self._create_error_result(f"Multimodal analysis failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models and their status.
        
        Returns:
            Model information dictionary
        """
        return {
            'image_detector': {
                'available': self.image_detector is not None,
                'models_loaded': self.image_detector.models_loaded if self.image_detector else False
            },
            'video_detector': {
                'available': self.video_detector is not None,
                'models_loaded': self.video_detector.models_loaded if self.video_detector else False
            },
            'audio_detector': {
                'available': self.audio_detector is not None,
                'models_loaded': self.audio_detector.models_loaded if self.audio_detector else False
            },
            'fusion_engine': {
                'available': self.fusion_engine is not None
            }
        }
    
    def _create_fallback_result(self, message: str) -> Dict[str, Any]:
        """Create a fallback result when detectors are not available."""
        return {
            'success': False,
            'authenticity': 'ANALYSIS UNAVAILABLE',
            'confidence': 0.0,
            'analysis_date': datetime.now().isoformat(),
            'key_findings': [message],
            'error': message
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create an error result."""
        return {
            'success': False,
            'authenticity': 'ANALYSIS FAILED',
            'confidence': 0.0,
            'analysis_date': datetime.now().isoformat(),
            'key_findings': [f'Error: {error_message}'],
            'error': error_message
        }


# Global instance
_satyaai_instance = None


def get_satyaai_instance(config: Optional[Dict[str, Any]] = None) -> SatyaAICore:
    """
    Get or create the global SatyaAI instance.
    
    Args:
        config: Configuration dictionary (only used on first call)
        
    Returns:
        SatyaAI instance
    """
    global _satyaai_instance
    
    if _satyaai_instance is None:
        if config is None:
            config = {
                'MODEL_PATH': os.environ.get('MODEL_PATH', './models'),
                'ENABLE_GPU': os.environ.get('ENABLE_GPU', 'False').lower() == 'true'
            }
        
        _satyaai_instance = SatyaAICore(config)
    
    return _satyaai_instance


def reset_satyaai_instance():
    """Reset the global instance (useful for testing)."""
    global _satyaai_instance
    _satyaai_instance = None