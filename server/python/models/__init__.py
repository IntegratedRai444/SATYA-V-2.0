# SatyaAI Models Package

# Import model classes for external access
try:
    from .image_model import ImageModel
    from .audio_enhanced import AudioModel
    from .video_model import VideoModel
    
    # Legacy aliases for compatibility
    AudioDetector = AudioModel
    VideoDetector = VideoModel
    ImageDetector = ImageModel
    
    __all__ = [
        'ImageModel', 'AudioModel', 'VideoModel',
        'AudioDetector', 'VideoDetector', 'ImageDetector'
    ]
except ImportError as e:
    # Fallback if specific models not available
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Some models not available: {e}")
    
    # Create dummy classes for compatibility
    class AudioDetector:
        def __init__(self, *args, **kwargs):
            pass
    
    class VideoDetector:
        def __init__(self, *args, **kwargs):
            pass
    
    class ImageDetector:
        def __init__(self, *args, **kwargs):
            pass
    
    __all__ = ['AudioDetector', 'VideoDetector', 'ImageDetector']
