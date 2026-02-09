"""
Detectors Models Module
Provides access to all detector classes and model information.
"""

from .audio_detector import AudioDetector
from .image_detector import ImageDetector
from .video_detector import VideoDetector
from .text_nlp_detector import TextNLPDetector
from .multimodal_fusion_detector import MultimodalFusionDetector

def get_model_info():
    """
    Get information about available models.
    
    Returns:
        dict: Dictionary containing model classes and their availability
    """
    return {
        'image': ImageDetector,
        'audio': AudioDetector,
        'video': VideoDetector,
        'text': TextNLPDetector,
        'multimodal': MultimodalFusionDetector
    }

__all__ = [
    'AudioDetector',
    'ImageDetector', 
    'VideoDetector',
    'TextNLPDetector',
    'MultimodalFusionDetector',
    'get_model_info'
]
