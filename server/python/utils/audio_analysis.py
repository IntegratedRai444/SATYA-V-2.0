"""Audio analysis utilities for deepfake detection."""
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Check for ffmpeg availability
FFMPEG_AVAILABLE = False
try:
    subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    FFMPEG_AVAILABLE = True
except (subprocess.CalledProcessError, FileNotFoundError):
    logger.warning("FFmpeg not available - audio extraction from videos will be limited")


def extract_audio_features(video_path: str) -> Dict[str, Any]:
    """
    Extract audio features from a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing audio features
    """
    if not FFMPEG_AVAILABLE:
        return {"error": "FFmpeg not available for audio extraction"}
    
    try:
        # Extract audio from video using ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio_path = temp_audio.name

        # Use ffmpeg to extract audio
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-vn",  # Disable video
                    "-acodec",
                    "pcm_s16le",  # Audio codec
                    "-ar",
                    "16000",  # Sample rate
                    "-ac",
                    "1",  # Mono audio
                    "-y",  # Overwrite output file if it exists
                    temp_audio_path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"Could not extract audio: {e}")
            return {"error": "Could not extract audio from video"}

        # Check if audio file exists and has content
        if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
            return {"error": "No audio track found in video"}

        # Load audio file
        y, sr = librosa.load(temp_audio_path, sr=16000)

        # Clean up temporary file
        try:
            os.unlink(temp_audio_path)
        except Exception as e:
            logger.warning(f"Could not delete temporary audio file: {e}")

        # Extract features
        features = {
            "duration": librosa.get_duration(y=y, sr=sr),
            "rms_energy": float(np.mean(librosa.feature.rms(y=y)[0])),
            "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y))),
            "spectral_centroid": float(
                np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            ),
            "spectral_bandwidth": float(
                np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            ),
            "spectral_rolloff": float(
                np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            ),
            "mfcc": [
                float(x)
                for x in np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
            ],
            "chroma_stft": [
                float(x)
                for x in np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            ],
        }

        # Detect voice activity
        voice_activity = _detect_voice_activity(y, sr)
        features["voice_activity"] = voice_activity

        return features

    except Exception as e:
        logger.error(f"Audio feature extraction failed: {e}")
        return {"error": f"Audio analysis failed: {str(e)}"}


def check_av_sync(video_path: str, frame_results: List[Dict]) -> Dict[str, Any]:
    """
    Check audio-video synchronization.

    Args:
        video_path: Path to the video file
        frame_results: List of frame analysis results

    Returns:
        Dictionary containing A/V sync analysis results
    """
    try:
        # This is a simplified version - in practice, you'd need more sophisticated
        # analysis to detect A/V sync issues

        # Check if we have frame timestamps in the results
        if not frame_results or "timestamp" not in frame_results[0]:
            return {"status": "unknown", "reason": "No timestamp data available"}

        # Calculate frame intervals
        timestamps = [fr.get("timestamp", 0) for fr in frame_results]
        frame_intervals = np.diff(timestamps)

        # Check for inconsistent frame intervals
        mean_interval = np.mean(frame_intervals) if len(frame_intervals) > 0 else 0
        std_interval = np.std(frame_intervals) if len(frame_intervals) > 1 else 0

        # Simple heuristic: if frame intervals vary too much, there might be A/V sync issues
        av_sync_issue = std_interval > mean_interval * 0.1  # More than 10% variation

        return {
            "status": "in_sync" if not av_sync_issue else "potential_issue",
            "mean_frame_interval": float(mean_interval) if mean_interval > 0 else 0,
            "frame_interval_std": float(std_interval) if std_interval > 0 else 0,
            "frame_count": len(frame_results),
            "total_duration": timestamps[-1] - timestamps[0] if timestamps else 0,
        }

    except Exception as e:
        logger.error(f"A/V sync check failed: {e}")
        return {"error": f"A/V sync check failed: {str(e)}"}


def _detect_voice_activity(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    threshold: float = 0.03,
) -> Dict[str, Any]:
    """
    Detect voice activity in an audio signal.

    Args:
        y: Audio time series
        sr: Sample rate
        frame_length: Length of the frame in samples
        hop_length: Number of samples between frames
        threshold: Energy threshold for voice activity

    Returns:
        Dictionary containing voice activity information
    """
    # Calculate short-time energy
    energy = np.array(
        [sum(abs(y[i : i + frame_length] ** 2)) for i in range(0, len(y), hop_length)]
    )

    # Normalize energy
    if len(energy) > 0:
        energy = energy / np.max(energy)

    # Detect voice activity
    is_voice = energy > threshold
    voice_frames = np.where(is_voice)[0]

    # Calculate voice activity ratio
    voice_ratio = np.sum(is_voice) / len(is_voice) if len(is_voice) > 0 else 0

    # Find voice segments
    segments = []
    if len(voice_frames) > 0:
        start = voice_frames[0]
        for i in range(1, len(voice_frames)):
            if voice_frames[i] - voice_frames[i - 1] > 1:  # Non-consecutive frames
                segments.append(
                    {
                        "start": start * hop_length / sr,
                        "end": voice_frames[i - 1] * hop_length / sr,
                        "duration": (voice_frames[i - 1] - start) * hop_length / sr,
                    }
                )
                start = voice_frames[i]

        # Add the last segment
        segments.append(
            {
                "start": start * hop_length / sr,
                "end": voice_frames[-1] * hop_length / sr,
                "duration": (voice_frames[-1] - start) * hop_length / sr,
            }
        )

    return {
        "voice_activity_ratio": float(voice_ratio),
        "voice_segments": segments,
        "total_voice_duration": sum(s["duration"] for s in segments),
        "segment_count": len(segments),
    }


def detect_audio_artifacts(audio_features: Dict[str, Any]) -> List[Dict]:
    """
    Detect potential audio manipulation artifacts.

    Args:
        audio_features: Dictionary of audio features from extract_audio_features

    Returns:
        List of detected artifacts
    """
    artifacts = []

    # Check for signs of audio splicing
    if "zero_crossing_rate" in audio_features and "rms_energy" in audio_features:
        # Sudden changes in zero-crossing rate can indicate edits
        if (
            audio_features["zero_crossing_rate"] > 0.1
            and audio_features["rms_energy"] < 0.01
        ):
            artifacts.append(
                {
                    "type": "potential_audio_edit",
                    "confidence": 0.7,
                    "metrics": {
                        "zero_crossing_rate": audio_features["zero_crossing_rate"],
                        "rms_energy": audio_features["rms_energy"],
                    },
                    "description": "High zero-crossing rate with low energy may indicate audio edits",
                }
            )

    # Check for inconsistent spectral features
    if "spectral_centroid" in audio_features and "spectral_bandwidth" in audio_features:
        if (
            audio_features["spectral_centroid"] < 500
            and audio_features["spectral_bandwidth"] > 3000
        ):
            artifacts.append(
                {
                    "type": "potential_audio_manipulation",
                    "confidence": 0.6,
                    "metrics": {
                        "spectral_centroid": audio_features["spectral_centroid"],
                        "spectral_bandwidth": audio_features["spectral_bandwidth"],
                    },
                    "description": "Spectral characteristics may indicate audio manipulation",
                }
            )

    return artifacts
