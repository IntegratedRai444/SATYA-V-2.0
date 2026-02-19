"""
SatyaAI Detectors Package

This package contains various detectors for deepfake analysis.
"""

# Temporarily disable audio detector due to scipy compatibility issues
# from .audio_detector import AudioDetector
from .fusion_engine import FusionEngine
# Core detectors
from .image_detector import ImageDetector
from .multimodal_fusion_detector import MultimodalFusionDetector
from .text_nlp_detector import TextNLPDetector
from .video_detector import VideoDetector

__all__ = [
    # Core detectors
    "ImageDetector",
    "VideoDetector",
    # "AudioDetector",  # Temporarily disabled
    "TextNLPDetector",
    "FusionEngine",
    "MultimodalFusionDetector",
]
