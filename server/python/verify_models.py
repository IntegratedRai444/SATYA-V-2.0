import logging
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detectors.audio_detector import AudioDetector
from detectors.image_detector import ImageDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelVerifier")

print(f"DEBUG: ImageDetector module: {ImageDetector.__module__}")
import inspect

print(f"DEBUG: ImageDetector file: {inspect.getfile(ImageDetector)}")
print(
    f"DEBUG: ImageDetector init signature: {inspect.signature(ImageDetector.__init__)}"
)


def verify_models():
    logger.info("--- Verifying Image Detector ---")
    # try:
    image_detector = ImageDetector(enable_gpu=True)
    if image_detector.models_loaded:
        logger.info("✅ Image Detector models loaded successfully.")
        if hasattr(image_detector, "vit_model"):
            logger.info("   - ViT Model: Loaded")
        if hasattr(image_detector, "face_detector"):
            logger.info("   - FaceNet: Loaded")
    else:
        logger.error("❌ Image Detector models failed to load.")
    # except Exception as e:
    #     logger.error(f"❌ Image Detector initialization failed: {e}")

    logger.info("\n--- Verifying Audio Detector ---")
    try:
        audio_detector = AudioDetector(enable_gpu=True)
        if audio_detector.models_loaded:
            logger.info("✅ Audio Detector models loaded successfully.")
            if hasattr(audio_detector, "model"):
                logger.info("   - Wav2Vec2 Model: Loaded")
        else:
            logger.error("❌ Audio Detector models failed to load.")
    except Exception as e:
        logger.error(f"❌ Audio Detector initialization failed: {e}")


if __name__ == "__main__":
    verify_models()
