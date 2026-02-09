"""
SatyaAI Models Package
Enhanced with comprehensive HuggingFace Transformers integration
"""

# Core models
try:
    from .deepfake_classifier import DeepfakeClassifier
    DEEPFAKE_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DeepfakeClassifier not available: {e}")
    DEEPFAKE_CLASSIFIER_AVAILABLE = False

# Enhanced modality models
try:
    from .image_model import AdvancedImageDetector as ImageModel
    IMAGE_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ImageModel not available: {e}")
    IMAGE_MODEL_AVAILABLE = False

try:
    from .audio_enhanced import AudioDeepfakeDetector as AudioModel
    AUDIO_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AudioModel not available: {e}")
    AUDIO_MODEL_AVAILABLE = False

try:
    from .video_model import VideoDeepfakeDetector as VideoModel
    VIDEO_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VideoModel not available: {e}")
    VIDEO_MODEL_AVAILABLE = False

# New transformer-enhanced models
try:
    from .text_model import TextDeepfakeDetector, analyze_text_deepfake
    TEXT_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TextModel not available: {e}")
    TEXT_MODEL_AVAILABLE = False

try:
    from .multimodal_fusion import MultimodalDeepfakeDetector, analyze_multimodal_deepfake
    MULTIMODAL_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MultimodalFusion not available: {e}")
    MULTIMODAL_MODEL_AVAILABLE = False

# Ensemble and advanced models
try:
    from .ensemble_detector import EnsembleDeepfakeDetector as EnsembleDetector
    ENSEMBLE_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: EnsembleDetector not available: {e}")
    ENSEMBLE_DETECTOR_AVAILABLE = False

# Model registry for easy access
MODEL_REGISTRY = {
    'deepfake_classifier': DeepfakeClassifier if DEEPFAKE_CLASSIFIER_AVAILABLE else None,
    'image': ImageModel if IMAGE_MODEL_AVAILABLE else None,
    'audio': AudioModel if AUDIO_MODEL_AVAILABLE else None,
    'video': VideoModel if VIDEO_MODEL_AVAILABLE else None,
    'text': TextDeepfakeDetector if TEXT_MODEL_AVAILABLE else None,
    'multimodal': MultimodalDeepfakeDetector if MULTIMODAL_MODEL_AVAILABLE else None,
    'ensemble': EnsembleDetector if ENSEMBLE_DETECTOR_AVAILABLE else None,
}

# Transformer availability check
try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
    TRANSFORMERS_VERSION = transformers.__version__
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_VERSION = None

# Utility functions
def get_available_models():
    """Get list of available models."""
    available = []
    for name, model_class in MODEL_REGISTRY.items():
        if model_class is not None:
            available.append(name)
    return available

def create_model(model_type: str, **kwargs):
    """Create a model instance by type."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = MODEL_REGISTRY[model_type]
    if model_class is None:
        raise ValueError(f"Model {model_type} is not available")
    
    return model_class(**kwargs)

def quick_analyze(modality: str, data, **kwargs):
    """Quick analysis function for different modalities."""
    if modality == 'text' and TEXT_MODEL_AVAILABLE:
        return analyze_text_deepfake(data, **kwargs)
    elif modality == 'multimodal' and MULTIMODAL_MODEL_AVAILABLE:
        return analyze_multimodal_deepfake(**data, **kwargs)
    else:
        raise ValueError(f"Quick analysis not available for modality: {modality}")

# Model information
def get_model_info():
    """Get information about available models and transformers."""
    return {
        'available_models': get_available_models(),
        'transformers_available': TRANSFORMERS_AVAILABLE,
        'transformers_version': TRANSFORMERS_VERSION,
        'model_registry': {k: v is not None for k, v in MODEL_REGISTRY.items()}
    }

def get_detector_by_type(model_type: str, **kwargs):
    """Get a detector by type - alias for create_model for compatibility."""
    return create_model(model_type, **kwargs)

# Export main classes
__all__ = [
    'DeepfakeClassifier',
    'ImageModel', 
    'AudioModel',
    'VideoModel',
    'TextDeepfakeDetector',
    'MultimodalDeepfakeDetector',
    'EnsembleDetector',
    'analyze_text_deepfake',
    'analyze_multimodal_deepfake',
    'get_available_models',
    'create_model',
    'get_detector_by_type',
    'quick_analyze',
    'get_model_info'
]

# Version info
__version__ = '2.0.0'
__author__ = 'SatyaAI Team'
__description__ = 'Enhanced deepfake detection models with HuggingFace Transformers'
