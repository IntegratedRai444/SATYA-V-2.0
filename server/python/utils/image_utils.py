# image_utils.py
# Add image preprocessing, EXIF extraction, ELA, GAN artifact detection helpers here. 
import numpy as np
import io

# Lazy imports for faster startup
_PIL_Image = None
_exifread = None

def get_pil():
    """Lazy load PIL"""
    global _PIL_Image
    if _PIL_Image is None:
        from PIL import Image
        _PIL_Image = Image
    return _PIL_Image

def get_exifread():
    """Lazy load exifread"""
    global _exifread
    if _exifread is None:
        import exifread
        _exifread = exifread
    return _exifread

def preprocess_image(image_bytes, size=(224, 224)):
    Image = get_pil()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(size)
    arr = np.asarray(image) / 255.0  # Normalize to 0-1
    return arr

def extract_exif(image_bytes):
    exifread = get_exifread()
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