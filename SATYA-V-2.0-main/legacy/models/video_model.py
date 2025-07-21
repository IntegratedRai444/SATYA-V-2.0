import numpy as np
from models.image_model import predict_deepfake

def predict_video_deepfake(frames):
    """
    Takes a list of frame arrays and returns per-frame results using image model inference.
    """
    results = []
    for i, frame in enumerate(frames):
        # Use image model's predict_deepfake for each frame
        label, confidence, explanation = predict_deepfake(frame)
        explanation = [f'Frame {i}: ' + exp for exp in explanation]
        results.append({
            'frame': i,
            'label': label,
            'confidence': confidence,
            'explanation': explanation
        })
    return results 