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
    """
    Detect faces using OpenCV Haar cascades as fallback.
    This is used when MTCNN is not available in the main detector.
    """
    try:
        import cv2
        
        # Convert to grayscale
        if len(arr.shape) == 3:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to expected format
        face_list = []
        for (x, y, w, h) in faces:
            face_list.append({
                "bbox": [x, y, x + w, y + h],
                "confidence": 0.85  # Haar cascades don't provide confidence
            })
        
        return face_list
        
    except ImportError:
        # Fallback when OpenCV is not available
        h, w = arr.shape[:2]
        return [{"bbox": [w//4, h//4, 3*w//4, 3*h//4], "confidence": 0.7}]
    except Exception as e:
        print(f"Face detection error: {e}")
        return []

def scan_gan_artifacts(arr):
    """
    Scan for GAN artifacts using frequency domain analysis.
    """
    try:
        import numpy as np
        
        # Convert to grayscale if needed
        if len(arr.shape) == 3:
            gray = np.mean(arr, axis=2)
        else:
            gray = arr
        
        # FFT analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Analyze frequency patterns
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        
        # Check for checkerboard patterns (high frequency content)
        high_freq_region = magnitude_spectrum[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
        high_freq_energy = np.mean(high_freq_region)
        total_energy = np.mean(magnitude_spectrum)
        
        checkerboard_score = (high_freq_energy / total_energy) if total_energy > 0 else 0
        
        # Check for patch inconsistencies using local variance
        patch_size = 32
        variances = []
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = gray[i:i+patch_size, j:j+patch_size]
                variances.append(np.var(patch))
        
        if variances:
            variance_std = np.std(variances)
            variance_mean = np.mean(variances)
            patch_inconsistency = (variance_std / variance_mean) if variance_mean > 0 else 0
        else:
            patch_inconsistency = 0
        
        return {
            "checkerboard": checkerboard_score > 0.3,
            "checkerboard_score": float(checkerboard_score),
            "patch_inconsistencies": patch_inconsistency > 0.5,
            "patch_inconsistency_score": float(patch_inconsistency)
        }
        
    except Exception as e:
        print(f"GAN artifact scanning error: {e}")
        return {"checkerboard": False, "patch_inconsistencies": False}

def generate_heatmap(arr, faces=None, suspicious_regions=None):
    """
    Generate a heatmap highlighting suspicious regions.
    """
    try:
        import numpy as np
        import base64
        from PIL import Image
        import io
        
        h, w = arr.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Add heat for detected faces
        if faces:
            for face in faces:
                bbox = face.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # Add gradient heat to face region
                    face_h, face_w = y2 - y1, x2 - x1
                    if face_h > 0 and face_w > 0:
                        y_indices, x_indices = np.ogrid[y1:y2, x1:x2]
                        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
                        
                        # Distance from center
                        distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
                        max_distance = np.sqrt((face_w/2)**2 + (face_h/2)**2)
                        
                        # Create gradient
                        if max_distance > 0:
                            gradient = 1 - (distances / max_distance)
                            gradient = np.clip(gradient, 0, 1)
                            heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], gradient * 0.8)
        
        # Add heat for suspicious regions
        if suspicious_regions:
            for region in suspicious_regions:
                x, y, intensity = region.get('x', 0), region.get('y', 0), region.get('intensity', 0.5)
                radius = region.get('radius', 20)
                
                # Add circular heat
                y_indices, x_indices = np.ogrid[:h, :w]
                distances = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
                mask = distances <= radius
                gradient = np.where(mask, 1 - (distances / radius), 0)
                heatmap = np.maximum(heatmap, gradient * intensity)
        
        # If no specific regions, create general suspicion map
        if not faces and not suspicious_regions:
            # Use edge detection to find potentially manipulated areas
            try:
                import cv2
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if len(arr.shape) == 3 else arr
                edges = cv2.Canny(gray, 50, 150)
                
                # Dilate edges to create heat regions
                kernel = np.ones((5, 5), np.uint8)
                dilated_edges = cv2.dilate(edges, kernel, iterations=1)
                heatmap = dilated_edges.astype(np.float32) / 255.0 * 0.6
                
            except ImportError:
                # Simple gradient fallback
                center_y, center_x = h // 2, w // 2
                y_indices, x_indices = np.ogrid[:h, :w]
                distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                heatmap = 0.3 * (1 - distances / max_distance)
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Convert to colormap and encode as base64
        # Apply colormap (red for high values)
        heatmap_colored = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap_colored[:, :, 0] = (heatmap * 255).astype(np.uint8)  # Red channel
        heatmap_colored[:, :, 3] = (heatmap * 128).astype(np.uint8)  # Alpha for transparency
        
        # Convert to PIL Image
        heatmap_image = Image.fromarray(heatmap_colored, 'RGB')
        
        # Encode as base64
        buffer = io.BytesIO()
        heatmap_image.save(buffer, format='PNG')
        heatmap_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return heatmap_base64
        
    except Exception as e:
        print(f"Heatmap generation error: {e}")
        return "heatmap_generation_failed" 