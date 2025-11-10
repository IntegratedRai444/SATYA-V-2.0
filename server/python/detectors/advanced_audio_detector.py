"""
Advanced Audio Detector
Uses Wav2Vec2, Librosa, and spectrogram analysis for voice deepfake detection
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import io

logger = logging.getLogger(__name__)


class AdvancedAudioDetector:
    """
    Advanced audio deepfake detector using Wav2Vec2 and spectrogram analysis.
    """
    
    def __init__(self, model_path: str = "models", enable_gpu: bool = False):
        """
        Initialize the advanced audio detector.
        
        Args:
            model_path: Path to model directory
            enable_gpu: Whether to use GPU acceleration
        """
        self.model_path = model_path
        self.enable_gpu = enable_gpu
        self.device = "cuda" if enable_gpu else "cpu"
        
        self.wav2vec2_model = None
        self.wav2vec2_processor = None
        self.sample_rate = 16000
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize Wav2Vec2 and other audio models."""
        try:
            from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
            import torch
            
            # Load Wav2Vec2 for audio classification
            model_name = "facebook/wav2vec2-base"
            
            try:
                self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2  # Real vs Fake
                )
                
                if self.enable_gpu and torch.cuda.is_available():
                    self.wav2vec2_model = self.wav2vec2_model.to(self.device)
                
                self.wav2vec2_model.eval()
                logger.info("Wav2Vec2 model initialized successfully")
                
            except Exception as e:
                logger.warning(f"Could not load Wav2Vec2 model: {e}")
                
        except ImportError:
            logger.warning("Transformers library not available for Wav2Vec2")
    
    def analyze_audio(self, audio_buffer: bytes) -> Dict[str, Any]:
        """
        Analyze audio for deepfake detection.
        
        Args:
            audio_buffer: Audio data as bytes
            
        Returns:
            Comprehensive analysis results
        """
        try:
            import librosa
            import soundfile as sf
            from datetime import datetime
            
            # Load audio
            audio_data, sr = self._load_audio(audio_buffer)
            
            # Resample if needed
            if sr != self.sample_rate:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sr, 
                    target_sr=self.sample_rate
                )
                sr = self.sample_rate
            
            results = {
                'success': True,
                'authenticity': 'UNCERTAIN',
                'confidence': 0.0,
                'analysis_date': datetime.now().isoformat(),
                'audio_info': {
                    'duration': len(audio_data) / sr,
                    'sample_rate': sr,
                    'channels': 1 if audio_data.ndim == 1 else audio_data.shape[0]
                },
                'analyses': {}
            }
            
            # Run multiple analyses
            results['analyses']['wav2vec2'] = self._analyze_with_wav2vec2(audio_data, sr)
            results['analyses']['spectrogram'] = self._analyze_spectrogram(audio_data, sr)
            results['analyses']['prosody'] = self._analyze_prosody(audio_data, sr)
            results['analyses']['artifacts'] = self._detect_artifacts(audio_data, sr)
            results['analyses']['voice_features'] = self._extract_voice_features(audio_data, sr)
            
            # Combine results
            final_result = self._combine_audio_results(results['analyses'])
            results['authenticity'] = final_result['authenticity']
            results['confidence'] = final_result['confidence']
            results['key_findings'] = final_result['findings']
            
            return results
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return {
                'success': False,
                'authenticity': 'UNCERTAIN',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _load_audio(self, audio_buffer: bytes) -> Tuple[np.ndarray, int]:
        """Load audio from bytes buffer."""
        import soundfile as sf
        
        audio_file = io.BytesIO(audio_buffer)
        audio_data, sample_rate = sf.read(audio_file)
        
        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        return audio_data, sample_rate
    
    def _analyze_with_wav2vec2(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze audio using Wav2Vec2 model."""
        if not self.wav2vec2_model or not self.wav2vec2_processor:
            return {
                'available': False,
                'message': 'Wav2Vec2 model not available'
            }
        
        try:
            import torch
            
            # Process audio
            inputs = self.wav2vec2_processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )
            
            if self.enable_gpu:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.wav2vec2_model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
            
            fake_prob = probs[0][1].item()
            
            return {
                'available': True,
                'is_fake': fake_prob > 0.5,
                'fake_probability': float(fake_prob),
                'real_probability': float(1 - fake_prob),
                'confidence': float(max(fake_prob, 1 - fake_prob))
            }
            
        except Exception as e:
            logger.error(f"Wav2Vec2 analysis failed: {e}")
            return {
                'available': False,
                'error': str(e)
            }
    
    def _analyze_spectrogram(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze audio spectrogram for manipulation artifacts."""
        try:
            import librosa
            import librosa.display
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=128,
                fmax=8000
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Compute MFCC
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            
            # Analyze spectral characteristics
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            
            # Detect anomalies in spectrogram
            anomaly_score = self._detect_spectral_anomalies(mel_spec_db)
            
            return {
                'available': True,
                'anomaly_score': float(anomaly_score),
                'spectral_features': {
                    'centroid_mean': float(np.mean(spectral_centroid)),
                    'centroid_std': float(np.std(spectral_centroid)),
                    'rolloff_mean': float(np.mean(spectral_rolloff)),
                    'contrast_mean': float(np.mean(spectral_contrast))
                },
                'mfcc_stats': {
                    'mean': float(np.mean(mfcc)),
                    'std': float(np.std(mfcc))
                },
                'is_suspicious': anomaly_score > 0.6
            }
            
        except Exception as e:
            logger.error(f"Spectrogram analysis failed: {e}")
            return {'available': False, 'error': str(e)}
    
    def _analyze_prosody(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze prosodic features (pitch, rhythm, energy)."""
        try:
            import librosa
            
            # Extract pitch (F0)
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            # Extract energy
            rms = librosa.feature.rms(y=audio)[0]
            
            # Extract zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Analyze naturalness
            naturalness_score = self._calculate_prosody_naturalness(
                pitch_values, rms, zcr
            )
            
            return {
                'available': True,
                'pitch': {
                    'mean': float(np.mean(pitch_values)) if pitch_values else 0,
                    'std': float(np.std(pitch_values)) if pitch_values else 0,
                    'range': float(np.ptp(pitch_values)) if pitch_values else 0
                },
                'energy': {
                    'mean': float(np.mean(rms)),
                    'std': float(np.std(rms))
                },
                'zero_crossing_rate': {
                    'mean': float(np.mean(zcr)),
                    'std': float(np.std(zcr))
                },
                'naturalness_score': float(naturalness_score),
                'is_natural': naturalness_score > 0.6
            }
            
        except Exception as e:
            logger.error(f"Prosody analysis failed: {e}")
            return {'available': False, 'error': str(e)}
    
    def _detect_artifacts(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect audio manipulation artifacts."""
        try:
            import librosa
            
            # Detect clipping
            clipping_ratio = np.sum(np.abs(audio) > 0.99) / len(audio)
            
            # Detect silence/noise patterns
            silence_threshold = 0.01
            silence_ratio = np.sum(np.abs(audio) < silence_threshold) / len(audio)
            
            # Detect abrupt changes (potential splicing)
            diff = np.diff(audio)
            abrupt_changes = np.sum(np.abs(diff) > 0.1) / len(diff)
            
            # Detect frequency domain artifacts
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Look for unnatural frequency patterns
            freq_variance = np.var(magnitude, axis=1)
            freq_anomaly = np.sum(freq_variance > np.percentile(freq_variance, 95)) / len(freq_variance)
            
            # Calculate overall artifact score
            artifact_score = (
                clipping_ratio * 0.3 +
                (1 - silence_ratio) * 0.2 +
                abrupt_changes * 0.3 +
                freq_anomaly * 0.2
            )
            
            return {
                'available': True,
                'artifact_score': float(artifact_score),
                'clipping_ratio': float(clipping_ratio),
                'silence_ratio': float(silence_ratio),
                'abrupt_changes': float(abrupt_changes),
                'frequency_anomaly': float(freq_anomaly),
                'has_artifacts': artifact_score > 0.5
            }
            
        except Exception as e:
            logger.error(f"Artifact detection failed: {e}")
            return {'available': False, 'error': str(e)}
    
    def _extract_voice_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract comprehensive voice features."""
        try:
            import librosa
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            
            # Spectral features
            spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            spec_flatness = librosa.feature.spectral_flatness(y=audio)
            
            # Temporal features
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            
            return {
                'available': True,
                'chroma': {
                    'mean': float(np.mean(chroma)),
                    'std': float(np.std(chroma))
                },
                'spectral_bandwidth': {
                    'mean': float(np.mean(spec_bandwidth)),
                    'std': float(np.std(spec_bandwidth))
                },
                'spectral_flatness': {
                    'mean': float(np.mean(spec_flatness)),
                    'std': float(np.std(spec_flatness))
                },
                'tempo': float(tempo),
                'beat_count': len(beats)
            }
            
        except Exception as e:
            logger.error(f"Voice feature extraction failed: {e}")
            return {'available': False, 'error': str(e)}
    
    def _detect_spectral_anomalies(self, mel_spec_db: np.ndarray) -> float:
        """Detect anomalies in mel spectrogram."""
        # Calculate statistics
        mean_spec = np.mean(mel_spec_db, axis=1)
        std_spec = np.std(mel_spec_db, axis=1)
        
        # Look for unusual patterns
        # Real voices have smooth spectral envelopes
        # Synthetic voices often have artifacts
        
        # Calculate smoothness
        diff = np.diff(mean_spec)
        roughness = np.std(diff)
        
        # Normalize to 0-1 range
        anomaly_score = min(roughness / 10.0, 1.0)
        
        return anomaly_score
    
    def _calculate_prosody_naturalness(
        self,
        pitch_values: List[float],
        rms: np.ndarray,
        zcr: np.ndarray
    ) -> float:
        """Calculate how natural the prosody sounds."""
        if not pitch_values:
            return 0.5
        
        # Natural speech has:
        # 1. Moderate pitch variation
        # 2. Smooth energy transitions
        # 3. Consistent zero-crossing rate
        
        pitch_variation = np.std(pitch_values) / (np.mean(pitch_values) + 1e-6)
        energy_smoothness = 1.0 - (np.std(np.diff(rms)) / (np.mean(rms) + 1e-6))
        zcr_consistency = 1.0 - np.std(zcr)
        
        # Combine scores
        naturalness = (
            (0.3 if 0.1 < pitch_variation < 0.5 else 0.0) +
            energy_smoothness * 0.4 +
            zcr_consistency * 0.3
        )
        
        return min(max(naturalness, 0.0), 1.0)
    
    def _combine_audio_results(self, analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from all audio analyses."""
        scores = []
        findings = []
        
        # Wav2Vec2 results
        if analyses['wav2vec2'].get('available'):
            wav2vec2 = analyses['wav2vec2']
            scores.append(wav2vec2['fake_probability'])
            if wav2vec2['is_fake']:
                findings.append(f"Wav2Vec2 detected synthetic voice ({wav2vec2['confidence']:.1%} confidence)")
        
        # Spectrogram results
        if analyses['spectrogram'].get('available'):
            spec = analyses['spectrogram']
            if spec['is_suspicious']:
                scores.append(spec['anomaly_score'])
                findings.append(f"Spectral anomalies detected (score: {spec['anomaly_score']:.2f})")
        
        # Prosody results
        if analyses['prosody'].get('available'):
            prosody = analyses['prosody']
            if not prosody['is_natural']:
                scores.append(1.0 - prosody['naturalness_score'])
                findings.append(f"Unnatural prosody patterns detected")
        
        # Artifact results
        if analyses['artifacts'].get('available'):
            artifacts = analyses['artifacts']
            if artifacts['has_artifacts']:
                scores.append(artifacts['artifact_score'])
                findings.append(f"Audio manipulation artifacts found")
        
        # Calculate final score
        if scores:
            avg_score = np.mean(scores)
            confidence = min(max(avg_score, 0.0), 1.0)
            
            if confidence > 0.7:
                authenticity = 'FAKE'
            elif confidence < 0.3:
                authenticity = 'REAL'
            else:
                authenticity = 'UNCERTAIN'
        else:
            confidence = 0.5
            authenticity = 'UNCERTAIN'
            findings.append("Insufficient data for confident prediction")
        
        return {
            'authenticity': authenticity,
            'confidence': float(confidence),
            'findings': findings
        }
