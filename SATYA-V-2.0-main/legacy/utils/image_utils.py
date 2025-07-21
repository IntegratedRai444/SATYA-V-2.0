# image_utils.py
# Add image preprocessing, EXIF extraction, ELA, GAN artifact detection helpers here. 
from PIL import Image
import numpy as np
import exifread
import io

def preprocess_image(image_bytes, size=(224, 224)):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(size)
    arr = np.asarray(image) / 255.0  # Normalize to 0-1
    return arr

def extract_exif(image_bytes):
    tags = {}
    try:
        with io.BytesIO(image_bytes) as f:
            exif_tags = exifread.process_file(f, details=False)
            for tag in exif_tags:
                tags[tag] = str(exif_tags[tag])
    except Exception as e:
        tags['error'] = str(e)
    return tags 

def detect_faces(arr):
    # TODO: Integrate MTCNN or similar face detector
    # For now, return a placeholder
    return [{"bbox": [50, 50, 150, 150], "confidence": 0.99}]

def scan_gan_artifacts(arr):
    # TODO: Implement GAN artifact scanning
    # For now, return a placeholder
    return {"checkerboard": True, "patch_inconsistencies": False}

def generate_heatmap(arr):
    # TODO: Generate a heatmap using model output or artifact scan
    # For now, return a placeholder (could be a base64 string or dummy array)
    return "heatmap_placeholder" 