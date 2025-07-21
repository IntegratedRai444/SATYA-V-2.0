import librosa
import numpy as np

def predict_audio_deepfake(audio_bytes):
    """
    Takes audio bytes and returns (label, confidence, explanation).
    Uses librosa to extract MFCC features and applies a simple heuristic.
    """
    try:
        # Load audio from bytes
        import io
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc)
        mfcc_var = np.var(mfcc)
        # Simple heuristic: if variance is very low, might be synthetic
        if mfcc_var < 50:
            label = 'FAKE'
            confidence = 85.0
            explanation = [f'Low MFCC variance ({mfcc_var:.2f}) suggests synthetic audio.']
        else:
            label = 'REAL'
            confidence = 90.0
            explanation = [f'Normal MFCC variance ({mfcc_var:.2f}) suggests real audio.']
    except Exception as e:
        label = 'FAKE'
        confidence = 50.0
        explanation = [f'Audio analysis error: {e}']
    return label, confidence, explanation 