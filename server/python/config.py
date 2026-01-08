"""
SatyaAI Configuration
Production-ready configuration for deepfake detection system
"""

import os
from pathlib import Path
from typing import Any, Dict

# Base paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models" / "pretrained"
CACHE_DIR = BASE_DIR / "cache"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    """Base configuration class."""

    # Server Configuration
    HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5001))
    DEBUG = False
    TESTING = False
    ENV = "development"  # Default environment

    @staticmethod
    def init_app(app):
        """Initialize application with this config."""
        pass

    # Security - Will be overridden by DevelopmentConfig or ProductionConfig
    # Security - Will be overridden by DevelopmentConfig or ProductionConfig
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-prod")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
    REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))

    # CORS Configuration
    CORS_ORIGINS = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(",")

    # File Upload Configuration
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    ALLOWED_EXTENSIONS = {
        "image": {"jpg", "jpeg", "png", "gif", "bmp", "webp"},
        "video": {"mp4", "avi", "mov", "mkv", "webm", "flv"},
        "audio": {"mp3", "wav", "flac", "m4a", "aac", "ogg"},
    }

    # Model Configuration
    MODEL_PATH = MODEL_DIR
    ENABLE_GPU = False  # Will be set after torch import
    DEVICE = "cpu"  # Will be set after torch import

    # Detection Thresholds
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
    FACE_DETECTION_THRESHOLD = float(os.getenv("FACE_DETECTION_THRESHOLD", 0.8))

    # Performance Configuration
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", 1000))

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOG_DIR / "satyaai.log"
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Model URLs for downloading
    MODEL_URLS = {
        "deepfake_classifier": {
            "url": "https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/v1.0/final_999_DeepFakeClassifier_tf_efficientnet_b4_ns_0_36",
            "filename": "efficientnet_b4_deepfake.bin",
            "type": "efficientnet_b4",
        },
        "face_xray": {
            "url": "https://github.com/neverUseThisName/FaceX-Zoo/releases/download/v1.0/face_xray_model.pth",
            "filename": "face_xray.pth",
            "type": "xray",
        },
        "capsule_forensics": {
            "url": "https://github.com/nii-yamagishilab/Capsule-Forensics-v2/releases/download/v1.0/capsule_net.pth",
            "filename": "capsule_forensics.pth",
            "type": "capsule",
        },
    }

    # Feature Flags
    ENABLE_FACE_DETECTION = True
    ENABLE_EMBEDDING_ANALYSIS = True
    ENABLE_FREQUENCY_ANALYSIS = True
    ENABLE_METADATA_ANALYSIS = True
    ENABLE_ENSEMBLE_PREDICTION = True
    ENABLE_CACHING = True
    ENABLE_PREPROCESSING = True


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True
    LOG_LEVEL = "DEBUG"
    CACHE_SIZE = 100
    ENV = "development"

    # Development-only defaults (never use in production)
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-not-for-production")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret-not-for-production")


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False
    LOG_LEVEL = "INFO"
    ENV = "production"
    # Use environment variables for sensitive data
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")


# Validate production config only when actually using production
def validate_production_config():
    """Validate production configuration when needed."""
    if not ProductionConfig.SECRET_KEY or not ProductionConfig.JWT_SECRET_KEY:
        raise ValueError("SECRET_KEY and JWT_SECRET_KEY must be set in production")


class TestingConfig(Config):
    """Testing configuration."""

    TESTING = True
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    CACHE_SIZE = 10


# Configuration mapping
config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}

# Import torch after config definition to avoid circular imports
try:
    import torch

    Config.ENABLE_GPU = torch.cuda.is_available()
    Config.DEVICE = "cuda" if Config.ENABLE_GPU else "cpu"
except ImportError:
    Config.ENABLE_GPU = False
    Config.DEVICE = "cpu"


# Validate environment variables on startup
def validate_environment():
    """Validate required environment variables based on environment."""
    env = os.getenv("ENV", "development")

    if env == "production":
        required_vars = {
            "FLASK_SECRET_KEY": "Flask secret key for session management",
            "JWT_SECRET_KEY": "JWT secret key for token signing",
        }

        missing = []
        for var, description in required_vars.items():
            if not os.getenv(var):
                missing.append(f"{var} ({description})")

        if missing:
            error_msg = (
                "Missing required environment variables in production:\n"
                + "\n".join(f"  - {var}" for var in missing)
            )
            raise ValueError(error_msg)

    # Log configuration status
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Environment: {env}")
    logger.info(f"GPU Available: {Config.ENABLE_GPU}")
    logger.info(f"Device: {Config.DEVICE}")


# Auto-validate on import in production
if os.getenv("ENV") == "production":
    try:
        validate_environment()
    except ValueError as e:
        import logging

        logging.error(f"Configuration validation failed: {e}")
        raise
