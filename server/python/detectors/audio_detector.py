"""
Enhanced Audio Deepfake Detector
Comprehensive audio analysis using Wav2Vec2, spectral analysis, and prosody detection
Combines ML models with signal processing for robust voice cloning detection
"""

import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
import tempfile
import os
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Import audio forensics
try:
    from .audio_forensics import analyze_voiceprint_consistency
    AUDIO_FORENSICS_AVAILABLE = True
except ImportError:
    AUDIO_FORENSICS_AVAILABLE = False
    logger.warning("Audio forensics not available")

# Try to import required libraries
try:
    from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Run: pip install transformers torch")

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not installed. Run: pip install librosa soundfile")

try:
    from scipy import signal as scipy_signal
    from scipy.fft import fft, fftfreq
    from scipy.stats import entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not installed for advanced analysis")


class AudioDetector:
    """
    Comprehensive audio deepfake detector using multiple approaches:
    1. Wav2Vec2 transformer model for ML-based detection
    2. Spectral analysis (FFT, MFCC, Mel-spectrogram)
    3. Prosody analysis (pitch, rhythm, energy)
    4. Voice quality analysis
    5. Artifact detection (synthesis artifacts, noise patterns)
    6. Temporal consistency analysis
    7. Frequency domain analysis
    """
    
    def __init__(self, device='cpu', config: Optional[Dict] = None):
        """Initialize comprehensive audio detector"""
        self.device = device
        self.config = config or self._default_config()
        self.sample_rate = 16000
        
        logger.info(f"🎵 Initializing Enhanced Audio Detector on {device}")
        
        # ML models
        self.model = None
        self.processor = None
        self.models_loaded = False
        
        # Analysis weights
        self.analysis_weights = {
            'ml_classification': 0.40,
            'spectral_analysis': 0.20,
            'prosody_analysis': 0.15,
            'voice_quality': 0.10,
            'artifact_detection': 0.10,
            'temporal_consistency': 0.05
        }
        
        # Load models
        self._load_pretrained_model()
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'models': {
                'wav2vec2': {
                    'enabled': True,
                    'model_options': [
                        "Andyrasika/wav2vec2-large-xlsr-53-deepfake-detection",
                        "facebook/wav2vec2-base"
                    ]
                }
            },
            'analysis': {
                'enable_spectral': True,
                'enable_prosody': True,
                'enable_voice_quality': True,
                'enable_artifacts': True,
                'enable_temporal': True
            },
            'thresholds': {
                'authentic': 0.7,
                'suspicious': 0.4,
                'deepfake': 0.4
            },
            'audio_processing': {
                'max_duration': 30,  # seconds
                'n_mfcc': 13,
                'n_mels': 128,
                'hop_length': 512
            }
        }
    
    def _load_pretrained_model(self):
        """Load pre-trained Wav2Vec2 model"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ Transformers not available, ML detection disabled")
            return False
        
        try:
            logger.info("📦 Loading Wav2Vec2 model...")
            
            # Prioritize specific deepfake detection models
            model_options = self.config['models']['wav2vec2']['model_options']
            
            for model_name in model_options:
                try:
                    logger.info(f"Attempting to load {model_name}...")
                    self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                    self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
                    self.model = self.model.to(self.device)
                    self.model.eval()
                    self.models_loaded = True
                    logger.info(f"✅ Loaded audio model: {model_name}")
                    return True
                except Exception as e:
                    logger.warning(f"Could not load {model_name}: {e}")
                    continue
            
            logger.error("❌ Could not load any Wav2Vec2 model. ML detection will be disabled.")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            return False
    
    def detect(self, audio_path: str) -> Dict:
        """
        Comprehensive audio analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Comprehensive detection results
        """
        try:
            logger.info(f"🔍 Analyzing audio: {audio_path}")
            
            # Load audio
            if LIBROSA_AVAILABLE:
                audio_array, sr = librosa.load(audio_path, sr=self.sample_rate)
            else:
                # Fallback: basic loading
                import wave
                with wave.open(audio_path, 'rb') as wav_file:
                    sr = wav_file.getframerate()
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            duration = len(audio_array) / sr
            logger.info(f"📊 Audio: {duration:.2f}s, {sr}Hz")
            
            # Initialize results
            results = {
                'success': True,
                'authenticity_score': 0.0,
                'confidence': 0.0,
                'label': 'unknown',
                'explanation': '',
                'details': {
                    'audio_info': {
                        'duration': duration,
                        'sample_rate': sr,
                        'samples': len(audio_array)
                    }
                },
                'warnings': [],
                'artifacts_detected': []
            }
            
            # Resample if needed
            if sr != self.sample_rate:
                if LIBROSA_AVAILABLE:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.sample_rate)
                    sr = self.sample_rate
            
            # 1. ML-based classification
            if self.models_loaded:
                ml_result = self._classify_with_wav2vec2(audio_array)
                results['details']['ml_classification'] = ml_result
                logger.info(f"ML Classification: {ml_result['score']:.3f}")
            
            # 2. Spectral analysis
            if self.config['analysis']['enable_spectral'] and LIBROSA_AVAILABLE:
                spectral_result = self._analyze_spectral_features(audio_array, sr)
                results['details']['spectral_analysis'] = spectral_result
                logger.info(f"Spectral Analysis: {spectral_result['score']:.3f}")
            
            # 3. Prosody analysis
            if self.config['analysis']['enable_prosody'] and LIBROSA_AVAILABLE:
                prosody_result = self._analyze_prosody(audio_array, sr)
                results['details']['prosody_analysis'] = prosody_result
                logger.info(f"Prosody Analysis: {prosody_result['score']:.3f}")
            
            # 4. Voice quality analysis
            if self.config['analysis']['enable_voice_quality']:
                voice_quality_result = self._analyze_voice_quality(audio_array, sr)
                results['details']['voice_quality'] = voice_quality_result
                logger.info(f"Voice Quality: {voice_quality_result['score']:.3f}")
            
            # 5. Artifact detection
            if self.config['analysis']['enable_artifacts']:
                artifacts = self._detect_audio_artifacts(audio_array, sr)
                results['artifacts_detected'] = artifacts
                results['details']['artifact_detection'] = {
                    'artifacts_found': len(artifacts),
                    'score': 1.0 - min(1.0, len(artifacts) / 5)
                }
            
            # 6. Temporal consistency
            if self.config['analysis']['enable_temporal']:
                temporal_result = self._analyze_temporal_consistency(audio_array, sr)
                results['details']['temporal_consistency'] = temporal_result
            
            # 7. Frequency domain analysis
            freq_result = self._analyze_frequency_domain(audio_array, sr)
            results['details']['frequency_analysis'] = freq_result
            
            # Combine all scores
            final_score, confidence, label = self._combine_all_audio_scores(results['details'])
            
            results['authenticity_score'] = final_score
            results['confidence'] = confidence
            results['label'] = label
            
            # Generate explanation
            results['explanation'] = self._generate_audio_explanation(results)
            
            # Add recommendations
            results['recommendations'] = self._generate_audio_recommendations(results)
            
            logger.info(f"✅ Audio analysis complete: {label} ({final_score:.3f}, confidence: {confidence:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Audio detection failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'label': 'error',
                'confidence': 0.0,
                'authenticity_score': 0.5
            }
    
    def _classify_with_wav2vec2(self, audio_array: np.ndarray) -> Dict:
        """Classify using Wav2Vec2 model"""
        try:
            import torch
            
            # Process audio
            inputs = self.processor(
                audio_array,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                
                # Assuming class 0 = real, class 1 = fake
                real_prob = probabilities[0][0].item()
                fake_prob = probabilities[0][1].item()
            
            return {
                'real_probability': float(real_prob),
                'fake_probability': float(fake_prob),
                'score': float(real_prob),
                'confidence': float(abs(real_prob - fake_prob)),
                'method': 'wav2vec2_transformer'
            }
            
        except Exception as e:
            logger.error(f"Wav2Vec2 classification failed: {e}")
            return {'score': 0.5, 'error': str(e), 'method': 'wav2vec2_transformer'}
    
    def _analyze_spectral_features(self, audio_array: np.ndarray, sr: int) -> Dict:
        """Comprehensive spectral analysis"""
        try:
            scores = {}
            
            # 1. MFCC analysis
            mfcc_score = self._analyze_mfcc(audio_array, sr)
            scores['mfcc'] = mfcc_score
            
            # 2. Mel-spectrogram analysis
            mel_score = self._analyze_mel_spectrogram(audio_array, sr)
            scores['mel_spectrogram'] = mel_score
            
            # 3. Spectral centroid
            spectral_centroid_score = self._analyze_spectral_centroid(audio_array, sr)
            scores['spectral_centroid'] = spectral_centroid_score
            
            # 4. Spectral rolloff
            spectral_rolloff_score = self._analyze_spectral_rolloff(audio_array, sr)
            scores['spectral_rolloff'] = spectral_rolloff_score
            
            # 5. Zero crossing rate
            zcr_score = self._analyze_zero_crossing_rate(audio_array)
            scores['zero_crossing_rate'] = zcr_score
            
            # Overall spectral score
            overall_score = np.mean(list(scores.values()))
            
            return {
                'score': float(overall_score),
                'component_scores': {k: float(v) for k, v in scores.items()},
                'method': 'comprehensive_spectral_analysis'
            }
            
        except Exception as e:
            logger.error(f"Spectral analysis failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _analyze_mfcc(self, audio_array: np.ndarray, sr: int) -> float:
        """Analyze MFCC features"""
        try:
            n_mfcc = self.config['audio_processing']['n_mfcc']
            mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
            
            # Calculate statistics
            mfcc_mean = np.mean(mfccs, axis=1)
            mfcc_std = np.std(mfccs, axis=1)
            
            # Natural speech has characteristic MFCC patterns
            # Check for unnatural uniformity
            uniformity = np.std(mfcc_std)
            
            # Score based on uniformity (too uniform = suspicious)
            if uniformity > 5:
                return 0.9
            elif uniformity > 2:
                return 0.7
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"MFCC analysis failed: {e}")
            return 0.5
    
    def _analyze_mel_spectrogram(self, audio_array: np.ndarray, sr: int) -> float:
        """Analyze mel-spectrogram"""
        try:
            n_mels = self.config['audio_processing']['n_mels']
            mel_spec = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Analyze energy distribution
            energy_variance = np.var(mel_spec_db)
            
            # Natural speech has varied energy distribution
            if 100 <= energy_variance <= 500:
                return 0.9
            elif 50 <= energy_variance <= 700:
                return 0.7
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Mel-spectrogram analysis failed: {e}")
            return 0.5
    
    def _analyze_spectral_centroid(self, audio_array: np.ndarray, sr: int) -> float:
        """Analyze spectral centroid"""
        try:
            centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sr)
            centroid_mean = np.mean(centroid)
            centroid_std = np.std(centroid)
            
            # Natural speech has moderate centroid values
            if 1000 <= centroid_mean <= 3000 and centroid_std > 200:
                return 0.9
            else:
                return 0.6
                
        except Exception as e:
            logger.error(f"Spectral centroid analysis failed: {e}")
            return 0.5
    
    def _analyze_spectral_rolloff(self, audio_array: np.ndarray, sr: int) -> float:
        """Analyze spectral rolloff"""
        try:
            rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=sr)
            rolloff_mean = np.mean(rolloff)
            
            # Natural speech has characteristic rolloff
            if 2000 <= rolloff_mean <= 6000:
                return 0.9
            else:
                return 0.6
                
        except Exception as e:
            logger.error(f"Spectral rolloff analysis failed: {e}")
            return 0.5
    
    def _analyze_zero_crossing_rate(self, audio_array: np.ndarray) -> float:
        """Analyze zero crossing rate"""
        try:
            zcr = librosa.feature.zero_crossing_rate(audio_array)
            zcr_mean = np.mean(zcr)
            
            # Natural speech has moderate ZCR
            if 0.05 <= zcr_mean <= 0.15:
                return 0.9
            else:
                return 0.6
                
        except Exception as e:
            logger.error(f"ZCR analysis failed: {e}")
            return 0.5
    
    def _analyze_prosody(self, audio_array: np.ndarray, sr: int) -> Dict:
        """Comprehensive prosody analysis"""
        try:
            scores = {}
            
            # 1. Pitch analysis
            pitch_score = self._analyze_pitch(audio_array, sr)
            scores['pitch'] = pitch_score
            
            # 2. Rhythm analysis
            rhythm_score = self._analyze_rhythm(audio_array, sr)
            scores['rhythm'] = rhythm_score
            
            # 3. Energy analysis
            energy_score = self._analyze_energy(audio_array)
            scores['energy'] = energy_score
            
            # 4. Speaking rate
            rate_score = self._analyze_speaking_rate(audio_array, sr)
            scores['speaking_rate'] = rate_score
            
            overall_score = np.mean(list(scores.values()))
            
            return {
                'score': float(overall_score),
                'component_scores': {k: float(v) for k, v in scores.items()},
                'method': 'comprehensive_prosody_analysis'
            }
            
        except Exception as e:
            logger.error(f"Prosody analysis failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _analyze_pitch(self, audio_array: np.ndarray, sr: int) -> float:
        """Analyze pitch patterns"""
        try:
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(y=audio_array, sr=sr)
            
            # Get pitch values
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if not pitch_values:
                return 0.5
            
            # Analyze pitch variation
            pitch_std = np.std(pitch_values)
            pitch_range = np.max(pitch_values) - np.min(pitch_values)
            
            # Natural speech has varied pitch
            if pitch_std > 20 and pitch_range > 50:
                return 0.9
            else:
                return 0.6
                
        except Exception as e:
            logger.error(f"Pitch analysis failed: {e}")
            return 0.5
    
    def _analyze_rhythm(self, audio_array: np.ndarray, sr: int) -> float:
        """Analyze rhythm patterns"""
        try:
            # Onset detection
            onset_env = librosa.onset.onset_strength(y=audio_array, sr=sr)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            
            # Analyze beat consistency
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                beat_variance = np.var(beat_intervals)
                
                # Natural speech has some rhythm variation
                if beat_variance > 10:
                    return 0.9
                else:
                    return 0.6
            
            return 0.7
                
        except Exception as e:
            logger.error(f"Rhythm analysis failed: {e}")
            return 0.5
    
    def _analyze_energy(self, audio_array: np.ndarray) -> float:
        """Analyze energy patterns"""
        try:
            # Calculate RMS energy
            rms = librosa.feature.rms(y=audio_array)
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            
            # Natural speech has varied energy
            if rms_std > 0.01:
                return 0.9
            else:
                return 0.6
                
        except Exception as e:
            logger.error(f"Energy analysis failed: {e}")
            return 0.5
    
    def _analyze_speaking_rate(self, audio_array: np.ndarray, sr: int) -> float:
        """Analyze speaking rate"""
        try:
            # Detect syllables using onset detection
            onset_env = librosa.onset.onset_strength(y=audio_array, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            
            duration = len(audio_array) / sr
            syllables_per_second = len(onsets) / duration
            
            # Natural speech: 2-6 syllables per second
            if 2 <= syllables_per_second <= 6:
                return 0.9
            else:
                return 0.6
                
        except Exception as e:
            logger.error(f"Speaking rate analysis failed: {e}")
            return 0.5
    
    def _analyze_voice_quality(self, audio_array: np.ndarray, sr: int) -> Dict:
        """Analyze voice quality characteristics"""
        try:
            scores = {}
            
            # 1. Harmonic-to-noise ratio
            hnr_score = self._calculate_hnr(audio_array, sr)
            scores['hnr'] = hnr_score
            
            # 2. Jitter (pitch variation)
            jitter_score = self._calculate_jitter(audio_array, sr)
            scores['jitter'] = jitter_score
            
            # 3. Shimmer (amplitude variation)
            shimmer_score = self._calculate_shimmer(audio_array)
            scores['shimmer'] = shimmer_score
            
            overall_score = np.mean(list(scores.values()))
            
            return {
                'score': float(overall_score),
                'component_scores': {k: float(v) for k, v in scores.items()},
                'method': 'voice_quality_analysis'
            }
            
        except Exception as e:
            logger.error(f"Voice quality analysis failed: {e}")
            return {'score': 0.5, 'error': str(e)}
    
    def _calculate_hnr(self, audio_array: np.ndarray, sr: int) -> float:
        """Calculate harmonic-to-noise ratio"""
        try:
            # Simplified HNR calculation
            # Natural voices have higher HNR
            harmonic = np.abs(fft(audio_array))
            noise_floor = np.percentile(harmonic, 10)
            signal_peak = np.percentile(harmonic, 90)
            
            hnr = signal_peak / (noise_floor + 1e-10)
            
            # Higher HNR = more natural
            if hnr > 10:
                return 0.9
            elif hnr > 5:
                return 0.7
            else:
                return 0.5
                
        except:
            return 0.5
    
    def _calculate_jitter(self, audio_array: np.ndarray, sr: int) -> float:
        """Calculate jitter (pitch perturbation)"""
        try:
            # Simplified jitter calculation
            # Natural voices have some jitter
            if LIBROSA_AVAILABLE:
                pitches, _ = librosa.piptrack(y=audio_array, sr=sr)
                pitch_values = [pitches[:, t].max() for t in range(pitches.shape[1])]
                pitch_values = [p for p in pitch_values if p > 0]
                
                if len(pitch_values) > 1:
                    jitter = np.std(pitch_values) / (np.mean(pitch_values) + 1e-10)
                    
                    # Moderate jitter is natural
                    if 0.01 <= jitter <= 0.1:
                        return 0.9
                    else:
                        return 0.6
            
            return 0.7
                
        except:
            return 0.5
    
    def _calculate_shimmer(self, audio_array: np.ndarray) -> float:
        """Calculate shimmer (amplitude perturbation)"""
        try:
            # Calculate amplitude envelope
            amplitude = np.abs(audio_array)
            
            # Calculate local variations
            if len(amplitude) > 1:
                shimmer = np.std(amplitude) / (np.mean(amplitude) + 1e-10)
                
                # Moderate shimmer is natural
                if 0.05 <= shimmer <= 0.3:
                    return 0.9
                else:
                    return 0.6
            
            return 0.7
                
        except:
            return 0.5
    
    def _detect_audio_artifacts(self, audio_array: np.ndarray, sr: int) -> List[str]:
        """Detect specific audio artifacts"""
        artifacts = []
        
        try:
            # 1. Clipping detection
            if self._has_clipping(audio_array):
                artifacts.append("Audio clipping detected")
            
            # 2. Unnatural silence
            if self._has_unnatural_silence(audio_array):
                artifacts.append("Unnatural silence patterns")
            
            # 3. Synthesis artifacts
            if self._has_synthesis_artifacts(audio_array, sr):
                artifacts.append("Synthesis artifacts detected")
            
            # 4. Spectral discontinuities
            if self._has_spectral_discontinuities(audio_array, sr):
                artifacts.append("Spectral discontinuities detected")
            
            # 5. Unnatural noise floor
            if self._has_unnatural_noise_floor(audio_array):
                artifacts.append("Unnatural noise floor")
                
        except Exception as e:
            logger.error(f"Artifact detection failed: {e}")
        
        return artifacts
    
    def _has_clipping(self, audio_array: np.ndarray) -> bool:
        """Detect audio clipping"""
        try:
            # Check for values at maximum
            clipped_samples = np.sum(np.abs(audio_array) > 0.99)
            return clipped_samples > len(audio_array) * 0.01
        except:
            return False
    
    def _has_unnatural_silence(self, audio_array: np.ndarray) -> bool:
        """Detect unnatural silence patterns"""
        try:
            # Find silent regions
            threshold = 0.01
            silent_samples = np.sum(np.abs(audio_array) < threshold)
            silence_ratio = silent_samples / len(audio_array)
            
            # Too much or too little silence is suspicious
            return silence_ratio > 0.5 or silence_ratio < 0.05
        except:
            return False
    
    def _has_synthesis_artifacts(self, audio_array: np.ndarray, sr: int) -> bool:
        """Detect synthesis artifacts"""
        try:
            # Check for unnatural periodicity
            autocorr = np.correlate(audio_array, audio_array, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Strong periodic patterns can indicate synthesis
            peaks = np.where(autocorr > np.max(autocorr) * 0.7)[0]
            return len(peaks) > 10
        except:
            return False
    
    def _has_spectral_discontinuities(self, audio_array: np.ndarray, sr: int) -> bool:
        """Detect spectral discontinuities"""
        try:
            if LIBROSA_AVAILABLE:
                # Calculate spectrogram
                spec = np.abs(librosa.stft(audio_array))
                
                # Check for sudden spectral changes
                spec_diff = np.diff(spec, axis=1)
                large_changes = np.sum(np.abs(spec_diff) > np.percentile(np.abs(spec_diff), 95))
                
                return large_changes > spec.shape[1] * 0.1
            return False
        except:
            return False
    
    def _has_unnatural_noise_floor(self, audio_array: np.ndarray) -> bool:
        """Detect unnatural noise floor"""
        try:
            # Calculate noise floor
            sorted_samples = np.sort(np.abs(audio_array))
            noise_floor = np.mean(sorted_samples[:len(sorted_samples)//10])
            
            # Too clean (low noise) can indicate synthesis
            return noise_floor < 0.001
        except:
            return False
    
    def _analyze_temporal_consistency(self, audio_array: np.ndarray, sr: int) -> Dict:
        """Analyze temporal consistency"""
        try:
            # Divide audio into segments
            segment_length = sr  # 1 second segments
            num_segments = len(audio_array) // segment_length
            
            if num_segments < 2:
                return {'score': 0.7, 'consistent': True}
            
            segment_features = []
            for i in range(num_segments):
                segment = audio_array[i*segment_length:(i+1)*segment_length]
                # Calculate segment features
                energy = np.mean(np.abs(segment))
                zcr = np.mean(librosa.feature.zero_crossing_rate(segment)) if LIBROSA_AVAILABLE else 0
                segment_features.append([energy, zcr])
            
            # Check consistency
            feature_variance = np.var(segment_features, axis=0)
            consistency_score = 1.0 - min(1.0, np.mean(feature_variance) * 10)
            
            return {
                'score': float(consistency_score),
                'consistent': consistency_score > 0.6,
                'num_segments': num_segments
            }
            
        except Exception as e:
            logger.error(f"Temporal consistency analysis failed: {e}")
            return {'score': 0.7, 'consistent': True}
    
    def _analyze_frequency_domain(self, audio_array: np.ndarray, sr: int) -> Dict:
        """Analyze frequency domain characteristics"""
        try:
            # FFT analysis
            fft_values = np.abs(fft(audio_array))
            freqs = fftfreq(len(audio_array), 1/sr)
            
            # Analyze positive frequencies only
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = fft_values[:len(fft_values)//2]
            
            # Check for unnatural frequency patterns
            # 1. Spectral flatness
            spectral_flatness = np.exp(np.mean(np.log(positive_fft + 1e-10))) / (np.mean(positive_fft) + 1e-10)
            
            # 2. Frequency distribution
            low_freq_energy = np.sum(positive_fft[positive_freqs < 1000])
            mid_freq_energy = np.sum(positive_fft[(positive_freqs >= 1000) & (positive_freqs < 4000)])
            high_freq_energy = np.sum(positive_fft[positive_freqs >= 4000])
            
            total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
            
            # Natural speech has balanced frequency distribution
            if total_energy > 0:
                low_ratio = low_freq_energy / total_energy
                mid_ratio = mid_freq_energy / total_energy
                high_ratio = high_freq_energy / total_energy
                
                # Check if distribution is natural
                is_natural = 0.3 <= low_ratio <= 0.6 and 0.2 <= mid_ratio <= 0.5
                score = 0.9 if is_natural else 0.6
            else:
                score = 0.5
            
            return {
                'score': float(score),
                'spectral_flatness': float(spectral_flatness),
                'frequency_distribution': {
                    'low': float(low_ratio) if total_energy > 0 else 0,
                    'mid': float(mid_ratio) if total_energy > 0 else 0,
                    'high': float(high_ratio) if total_energy > 0 else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Frequency domain analysis failed: {e}")
            return {'score': 0.5}
    
    def _combine_all_audio_scores(self, details: Dict) -> Tuple[float, float, str]:
        """Combine all audio analysis scores"""
        scores = []
        weights = []
        
        # ML classification
        if 'ml_classification' in details:
            scores.append(details['ml_classification']['score'])
            weights.append(self.analysis_weights['ml_classification'])
        
        # Spectral analysis
        if 'spectral_analysis' in details:
            scores.append(details['spectral_analysis']['score'])
            weights.append(self.analysis_weights['spectral_analysis'])
        
        # Prosody analysis
        if 'prosody_analysis' in details:
            scores.append(details['prosody_analysis']['score'])
            weights.append(self.analysis_weights['prosody_analysis'])
        
        # Voice quality
        if 'voice_quality' in details:
            scores.append(details['voice_quality']['score'])
            weights.append(self.analysis_weights['voice_quality'])
        
        # Artifact detection
        if 'artifact_detection' in details:
            scores.append(details['artifact_detection']['score'])
            weights.append(self.analysis_weights['artifact_detection'])
        
        # Temporal consistency
        if 'temporal_consistency' in details:
            scores.append(details['temporal_consistency']['score'])
            weights.append(self.analysis_weights['temporal_consistency'])
        
        if not scores:
            return 0.5, 0.0, 'unknown'
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted average
        final_score = sum(s * w for s, w in zip(scores, normalized_weights))
        
        # Confidence based on agreement
        confidence = 1.0 - np.std(scores) if len(scores) > 1 else 0.7
        
        # Determine label
        thresholds = self.config.get('thresholds', {})
        authentic_threshold = thresholds.get('authentic', 0.7)
        deepfake_threshold = thresholds.get('deepfake', 0.4)
        
        if final_score >= authentic_threshold:
            label = 'authentic'
        elif final_score <= deepfake_threshold:
            label = 'deepfake'
        else:
            label = 'suspicious'
        
        return float(final_score), float(confidence), label
    
    def _generate_audio_explanation(self, results: Dict) -> str:
        """Generate comprehensive explanation"""
        label = results.get('label', 'unknown')
        score = results.get('authenticity_score', 0)
        confidence = results.get('confidence', 0)
        
        explanations = []
        
        # Main verdict
        if label == 'authentic':
            explanations.append(f"Audio appears to be authentic (score: {score:.2f}, confidence: {confidence:.2f}).")
        elif label == 'deepfake':
            explanations.append(f"Audio shows strong signs of voice cloning/synthesis (score: {score:.2f}, confidence: {confidence:.2f}).")
        else:
            explanations.append(f"Audio shows suspicious characteristics (score: {score:.2f}, confidence: {confidence:.2f}).")
        
        # Artifacts
        if results.get('artifacts_detected'):
            explanations.append(f"Detected artifacts: {', '.join(results['artifacts_detected'][:3])}.")
        
        # ML classification
        if 'ml_classification' in results.get('details', {}):
            ml = results['details']['ml_classification']
            if 'fake_probability' in ml and ml['fake_probability'] > 0.6:
                explanations.append(f"ML model detected {ml['fake_probability']*100:.1f}% probability of synthesis.")
        
        return " ".join(explanations)
    
    def _generate_audio_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations"""
        recommendations = []
        
        label = results.get('label', 'unknown')
        
        if label == 'deepfake':
            recommendations.append("Do not trust this audio for voice verification")
            recommendations.append("Likely voice cloning or synthesis detected")
        elif label == 'suspicious':
            recommendations.append("Exercise caution with this audio")
            recommendations.append("Consider additional verification methods")
        else:
            recommendations.append("Audio appears authentic, but verify source")
        
        if results.get('artifacts_detected'):
            recommendations.append("Multiple synthesis artifacts detected")
        
        return recommendations


# Singleton instance
_audio_detector_instance = None

def get_audio_detector() -> AudioDetector:
    """Get or create audio detector instance"""
    global _audio_detector_instance
    if _audio_detector_instance is None:
        _audio_detector_instance = AudioDetector()
    return _audio_detector_instance
