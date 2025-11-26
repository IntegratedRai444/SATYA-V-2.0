"""
Strict Mode Configuration for SatyaAI
Enforces REAL AI models ONLY - No fallbacks, no heuristics
"""

# STRICT MODE SETTINGS
STRICT_MODE_ENABLED = True  # Set to False to allow fallbacks

# Strict mode behavior
STRICT_MODE_CONFIG = {
    'require_mtcnn': True,              # Require MTCNN face detector
    'require_inception': True,          # Require InceptionResnetV1 embeddings
    'require_classifier': True,         # Require EfficientNet/ResNet classifier
    'allow_opencv_fallback': False,     # NO OpenCV Haar Cascades fallback
    'allow_heuristic_fallback': False,  # NO heuristic analysis fallback
    'allow_mock_embeddings': False,     # NO mock embeddings
    'fail_on_model_error': True,        # Fail immediately if models can't load
}

# Error messages for strict mode
STRICT_MODE_ERRORS = {
    'mtcnn_missing': 'STRICT MODE: MTCNN face detector not available. Install: pip install facenet-pytorch',
    'inception_missing': 'STRICT MODE: InceptionResnetV1 embedding model not available. Install: pip install facenet-pytorch',
    'classifier_missing': 'STRICT MODE: Deepfake classifier not loaded. Check model files in server/python/models/',
    'torch_missing': 'STRICT MODE: PyTorch not available. Install: pip install torch torchvision',
    'general_error': 'STRICT MODE: Real AI models are REQUIRED. No fallbacks allowed.',
}

# Model requirements
REQUIRED_MODELS = {
    'efficientnet_b4': 'server/python/models/efficientnet_b4_deepfake.bin',
    'resnet50': 'server/python/models/resnet50_deepfake.pth',
    'haarcascade': 'server/python/models/haarcascade_frontalface_default.xml',
}

# Required Python packages for strict mode
REQUIRED_PACKAGES = [
    'torch>=2.0.1',
    'torchvision>=0.15.2',
    'facenet-pytorch>=2.5.3',
    'opencv-python>=4.8.1',
    'numpy>=1.24.3',
    'pillow>=10.0.1',
]

def validate_strict_mode_requirements():
    """
    Validate that all requirements for strict mode are met.
    Returns (is_valid, missing_items)
    """
    missing = []
    
    # Check Python packages
    try:
        import torch
    except ImportError:
        missing.append('torch (PyTorch)')
    
    try:
        from facenet_pytorch import MTCNN, InceptionResnetV1
    except ImportError:
        missing.append('facenet-pytorch')
    
    try:
        import cv2
    except ImportError:
        missing.append('opencv-python')
    
    # Check model files
    import os
    for model_name, model_path in REQUIRED_MODELS.items():
        if not os.path.exists(model_path):
            missing.append(f'{model_name} model file: {model_path}')
    
    is_valid = len(missing) == 0
    return is_valid, missing


def get_strict_mode_status():
    """Get current strict mode status and requirements."""
    is_valid, missing = validate_strict_mode_requirements()
    
    return {
        'strict_mode_enabled': STRICT_MODE_ENABLED,
        'requirements_met': is_valid,
        'missing_requirements': missing,
        'config': STRICT_MODE_CONFIG,
    }


if __name__ == '__main__':
    # Test strict mode requirements
    print("=== STRICT MODE VALIDATION ===")
    print()
    
    status = get_strict_mode_status()
    
    print(f"Strict Mode Enabled: {status['strict_mode_enabled']}")
    print(f"Requirements Met: {status['requirements_met']}")
    print()
    
    if not status['requirements_met']:
        print("❌ MISSING REQUIREMENTS:")
        for item in status['missing_requirements']:
            print(f"  - {item}")
        print()
        print("Install missing packages:")
        print("  pip install torch torchvision facenet-pytorch opencv-python")
    else:
        print("✅ All requirements met! System ready for STRICT MODE operation.")
