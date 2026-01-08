# audio_utils.py
# Add audio preprocessing, spectrogram, fingerprint, and clone check helpers here.

import io

import librosa
import numpy as np


def preprocess_audio(audio_bytes):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    y = librosa.util.normalize(y)
    duration = librosa.get_duration(y=y, sr=sr)
    channels = 1  # librosa loads as mono
    return {"duration": duration, "channels": channels, "sr": sr}


def segment_audio(audio_bytes, segment_length=1.0):
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    total_duration = librosa.get_duration(y=y, sr=sr)
    segments = []
    for start in np.arange(0, total_duration, segment_length):
        end = min(start + segment_length, total_duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segments.append(y[start_sample:end_sample])
    return segments


def generate_spectrogram(audio_bytes, output_path=None):
    """
    Generate spectrogram visualization from audio bytes.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        import os

        import matplotlib.pyplot as plt
        import numpy as np

        # Load audio
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Create plot
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            mel_spec_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel-frequency Spectrogram")
        plt.tight_layout()

        # Save to file or return path
        if output_path is None:
            output_path = f"reports/spectrogram_{int(time.time())}.png"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return output_path

    except ImportError as e:
        print(f"Missing dependencies for spectrogram generation: {e}")
        return "reports/spectrogram_unavailable.png"
    except Exception as e:
        print(f"Spectrogram generation error: {e}")
        return "reports/spectrogram_error.png"


def detect_voice_clone(audio_bytes):
    """
    Detect voice cloning using spectral analysis and pattern matching.
    """
    try:
        # Load audio
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

        # Extract features for clone detection
        # 1. MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # 2. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        # 3. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)

        # Simple heuristic-based clone detection
        # Real implementation would use trained models

        # Check for unnatural spectral patterns
        spectral_variance = np.var(spectral_centroids)
        rolloff_variance = np.var(spectral_rolloff)
        zcr_mean = np.mean(zcr)

        # Calculate clone probability based on features
        clone_indicators = 0

        # Unnatural spectral variance (too consistent)
        if spectral_variance < 1000:
            clone_indicators += 0.3

        # Unusual rolloff patterns
        if rolloff_variance < 100000:
            clone_indicators += 0.2

        # Unnatural zero crossing rate
        if zcr_mean < 0.01 or zcr_mean > 0.3:
            clone_indicators += 0.2

        # MFCC consistency check
        mfcc_consistency = np.mean(mfcc_std)
        if mfcc_consistency < 1.0:  # Too consistent
            clone_indicators += 0.3

        clone_probability = min(1.0, clone_indicators)

        return {
            "clone_probability": float(clone_probability),
            "spectral_variance": float(spectral_variance),
            "rolloff_variance": float(rolloff_variance),
            "zcr_mean": float(zcr_mean),
            "mfcc_consistency": float(mfcc_consistency),
        }

    except Exception as e:
        print(f"Voice clone detection error: {e}")
        return {"clone_probability": 0.5, "error": str(e)}


def check_pitch_jitter(audio_bytes):
    """
    Check pitch, jitter, and shimmer for voice quality analysis.
    """
    try:
        # Load audio
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

        # Extract pitch using librosa
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)

        # Get the most prominent pitch at each time frame
        pitch_contour = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_contour.append(pitch)

        if len(pitch_contour) < 10:
            return {
                "pitch": 0,
                "jitter": 0,
                "shimmer": 0,
                "error": "Insufficient pitch data",
            }

        pitch_contour = np.array(pitch_contour)

        # Calculate pitch statistics
        mean_pitch = np.mean(pitch_contour)

        # Calculate jitter (pitch period variation)
        pitch_periods = 1.0 / (pitch_contour + 1e-8)
        period_diffs = np.abs(np.diff(pitch_periods))
        jitter = (
            np.mean(period_diffs) / np.mean(pitch_periods)
            if np.mean(pitch_periods) > 0
            else 0
        )

        # Calculate shimmer (amplitude variation) - simplified
        # Frame-based energy analysis
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)  # 10ms hop

        frame_energies = []
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i : i + frame_length]
            energy = np.sum(frame**2)
            frame_energies.append(energy)

        if len(frame_energies) > 1:
            energy_diffs = np.abs(np.diff(frame_energies))
            shimmer = (
                np.mean(energy_diffs) / np.mean(frame_energies)
                if np.mean(frame_energies) > 0
                else 0
            )
        else:
            shimmer = 0

        return {
            "pitch": float(mean_pitch),
            "jitter": float(min(jitter, 1.0)),  # Cap at 1.0
            "shimmer": float(min(shimmer, 1.0)),  # Cap at 1.0
            "pitch_range": float(np.max(pitch_contour) - np.min(pitch_contour)),
            "pitch_std": float(np.std(pitch_contour)),
        }

    except Exception as e:
        print(f"Pitch/jitter analysis error: {e}")
        return {"pitch": 0, "jitter": 0, "shimmer": 0, "error": str(e)}


def match_voiceprint(audio_bytes, reference_bytes=None):
    """
    Match voiceprint against reference (or analyze for consistency).
    """
    try:
        # Load primary audio
        y1, sr1 = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

        # Extract voiceprint features
        features1 = extract_voiceprint_features(y1, sr1)

        if reference_bytes is not None:
            # Load reference audio
            y2, sr2 = librosa.load(io.BytesIO(reference_bytes), sr=None, mono=True)
            features2 = extract_voiceprint_features(y2, sr2)

            # Calculate similarity
            match_score = calculate_feature_similarity(features1, features2)
        else:
            # Analyze consistency within the same audio
            # Split audio into segments and check consistency
            segment_length = len(y1) // 4
            if segment_length > sr1:  # At least 1 second per segment
                segments = [
                    y1[i : i + segment_length]
                    for i in range(0, len(y1), segment_length)
                ]

                segment_features = []
                for segment in segments[:4]:  # Use first 4 segments
                    if len(segment) > sr1 // 2:  # At least 0.5 seconds
                        features = extract_voiceprint_features(segment, sr1)
                        segment_features.append(features)

                if len(segment_features) >= 2:
                    # Calculate average similarity between segments
                    similarities = []
                    for i in range(len(segment_features)):
                        for j in range(i + 1, len(segment_features)):
                            sim = calculate_feature_similarity(
                                segment_features[i], segment_features[j]
                            )
                            similarities.append(sim)

                    match_score = np.mean(similarities) if similarities else 0.5
                else:
                    match_score = 0.5
            else:
                match_score = 0.5

        return {
            "match_score": float(match_score),
            "features_extracted": len(features1),
            "analysis_type": "reference_match"
            if reference_bytes
            else "consistency_check",
        }

    except Exception as e:
        print(f"Voiceprint matching error: {e}")
        return {"match_score": 0.5, "error": str(e)}


def extract_voiceprint_features(y, sr):
    """Extract features for voiceprint analysis."""
    features = {}

    try:
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features["mfcc_mean"] = np.mean(mfccs, axis=1)
        features["mfcc_std"] = np.std(mfccs, axis=1)

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["spectral_centroid"] = np.mean(spectral_centroids)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features["spectral_rolloff"] = np.mean(spectral_rolloff)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features["zcr"] = np.mean(zcr)

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features["chroma_mean"] = np.mean(chroma, axis=1)

    except Exception as e:
        print(f"Feature extraction error: {e}")

    return features


def calculate_feature_similarity(features1, features2):
    """Calculate similarity between two feature sets."""
    try:
        similarities = []

        # Compare MFCC features
        if "mfcc_mean" in features1 and "mfcc_mean" in features2:
            mfcc_sim = np.corrcoef(features1["mfcc_mean"], features2["mfcc_mean"])[0, 1]
            if not np.isnan(mfcc_sim):
                similarities.append(abs(mfcc_sim))

        # Compare spectral features
        spectral_features = ["spectral_centroid", "spectral_rolloff", "zcr"]
        for feature in spectral_features:
            if feature in features1 and feature in features2:
                # Normalize and compare
                val1 = features1[feature]
                val2 = features2[feature]
                if val1 != 0 and val2 != 0:
                    similarity = 1 - abs(val1 - val2) / max(abs(val1), abs(val2))
                    similarities.append(max(0, similarity))

        # Compare chroma features
        if "chroma_mean" in features1 and "chroma_mean" in features2:
            chroma_sim = np.corrcoef(
                features1["chroma_mean"], features2["chroma_mean"]
            )[0, 1]
            if not np.isnan(chroma_sim):
                similarities.append(abs(chroma_sim))

        return np.mean(similarities) if similarities else 0.5

    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return 0.5
