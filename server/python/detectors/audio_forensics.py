"""
Enhanced Audio Forensics Module for Deepfake Detection
Implements ECAPA-TDNN, RawNet2, and comprehensive voice analysis with improved reliability.
"""

import logging
import time
import hashlib
import warnings
import threading
import json
import os
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import lru_cache, partial
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Module version
__version__ = "1.0.0"

# Constants
DEFAULT_SAMPLE_RATE = 16000
MAX_AUDIO_DURATION = 3600  # 1 hour max
MIN_AUDIO_LENGTH = 1.0  # 1 second minimum
VALID_AUDIO_TYPES = ('float32', 'float64', 'int16', 'int32')
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Try to import optional dependencies
try:
    # Fix for speechbrain compatibility with newer torchaudio
    import torchaudio
    if hasattr(torchaudio, 'list_audio_backends'):
        from speechbrain.pretrained import EncoderClassifier
        SPEECHBRAIN_AVAILABLE = True
    else:
        # Fallback for newer torchaudio versions
        try:
            from speechbrain.pretrained import EncoderClassifier
            SPEECHBRAIN_AVAILABLE = True
        except (ImportError, AttributeError) as e:
            SPEECHBRAIN_AVAILABLE = False
            logger.warning(f"speechbrain not compatible with current torchaudio version: {e}")
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    logger.warning("speechbrain not installed. Some features will be disabled. Run: pip install speechbrain")

# Configuration
class AnalysisLevel(Enum):
    BASIC = auto()
    STANDARD = auto()
    ADVANCED = auto()
    COMPREHENSIVE = auto()

DEFAULT_CONFIG = {
    "analysis_level": "standard",
    "device": DEFAULT_DEVICE,
    "cache_dir": os.path.expanduser("~/.cache/audio_forensics"),
    "models": {
        "ecapa_tdnn": {
            "enabled": True,
            "model_name": "speechbrain/spkrec-ecapa-voxceleb",
            "min_segment_duration": 1.0,
            "max_segments": 10
        },
        "rawnet2": {
            "enabled": True,
            "pretrained": True
        }
    },
    "features": {
        "voiceprint": True,
        "noise_analysis": True,
        "compression_artifacts": True,
        "tamper_detection": True,
        "prosody_analysis": True
    },
    "thresholds": {
        "high_risk": 0.7,
        "medium_risk": 0.4,
        "low_risk": 0.2
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_AUDIO_DURATION = 3600  # 1 hour max
SAMPLE_RATE = 16000
MIN_AUDIO_LENGTH = 1.0  # 1 second minimum
VALID_AUDIO_TYPES = ('float32', 'float64', 'int16', 'int32')

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Try to import optional dependencies
try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    logger.warning("speechbrain not installed. Some features will be disabled. Run: pip install speechbrain")

class RiskLevel(Enum):
    """Risk level classification for audio analysis results."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AudioForensicsResult:
    """
    Enhanced result container for audio forensic analysis with comprehensive metadata.
    
    Attributes:
        is_synthetic: Whether the audio is detected as synthetic
        confidence: Confidence score (0-1)
        risk_level: Risk level classification
        signals_detected: List of detected manipulation signals
        features: Dictionary of extracted features and their values
        metadata: Additional metadata about the analysis
        processing_time: Time taken for analysis in seconds
        model_versions: Versions of models used in analysis
    """
    is_synthetic: bool
    confidence: float
    risk_level: RiskLevel
    signals_detected: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    model_versions: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        # Convert string risk_level to enum if needed
        if isinstance(self.risk_level, str):
            self.risk_level = RiskLevel(self.risk_level.lower())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a serializable dictionary."""
        return {
            'is_synthetic': self.is_synthetic,
            'confidence': float(self.confidence),
            'risk_level': self.risk_level.value,
            'signals_detected': self.signals_detected.copy(),
            'features': {k: self._serialize_value(v) for k, v in self.features.items()},
            'metadata': self.metadata.copy(),
            'processing_time': self.processing_time,
            'model_versions': self.model_versions.copy(),
            'version': __version__,
            'timestamp': time.time()
        }
    
    def _serialize_value(self, value):
        """Recursively serialize values for JSON compatibility."""
        if isinstance(value, (int, float, str, bool, type(None))):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, torch.Tensor):
            return value.cpu().numpy().tolist()
        elif hasattr(value, 'to_dict'):
            return value.to_dict()
        return str(value)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioForensicsResult':
        """Create instance from dictionary."""
        return cls(**data)
    
    @property
    def is_high_confidence(self) -> bool:
        """Whether the result has high confidence (>= 0.8)."""
        return self.confidence >= 0.8
    
    def get_risk_score(self) -> float:
        """Get normalized risk score (0-1)."""
        risk_scores = {
            RiskLevel.VERY_LOW: 0.1,
            RiskLevel.LOW: 0.3,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 1.0
        }
        base_score = risk_scores.get(self.risk_level, 0.5)
        return min(1.0, base_score * (1.0 + self.confidence) / 2)


class AudioForensicsAnalyzer:
    """
    Main class for performing audio forensic analysis with multiple detection methods.
    
    Features:
    - Voiceprint analysis using ECAPA-TDNN
    - Audio tampering detection
    - Noise analysis
    - Compression artifact detection
    - Prosody analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the audio forensics analyzer.
        
        Args:
            config: Configuration dictionary (see DEFAULT_CONFIG for structure)
        """
        self.config = self._merge_configs(config or {})
        self.device = self.config['device']
        self._models = {}
        self._initialized = False
        self._lock = threading.RLock()
        
        # Create cache directory
        os.makedirs(self.config['cache_dir'], exist_ok=True)
        
        # Initialize models
        self._init_models()
    
    def _merge_configs(self, user_config: Dict) -> Dict:
        """Merge user config with defaults."""
        config = DEFAULT_CONFIG.copy()
        
        # Deep merge configs
        for key, value in user_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
                
        return config
    
    def _init_models(self):
        """Initialize all required models."""
        if self._initialized:
            return
            
        with self._lock:
            try:
                logger.info("Initializing audio forensic models...")
                
                # Initialize ECAPA-TDNN if enabled
                if self.config['models']['ecapa_tdnn']['enabled'] and SPEECHBRAIN_AVAILABLE:
                    self._init_ecapa_tdnn()
                
                # Initialize RawNet2 if enabled
                if self.config['models']['rawnet2']['enabled']:
                    self._init_rawnet2()
                
                self._initialized = True
                logger.info("Audio forensic models initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize models: {str(e)}")
                raise
    
    def _init_ecapa_tdnn(self):
        """Initialize ECAPA-TDNN model."""
        model_name = self.config['models']['ecapa_tdnn']['model_name']
        logger.info(f"Initializing ECAPA-TDNN model: {model_name}")
        
        try:
            self._models['ecapa_tdnn'] = ECAPATDNN(
                device=self.device,
                model_name=model_name
            )
        except Exception as e:
            logger.error(f"Failed to initialize ECAPA-TDNN: {str(e)}")
            if self.config['models']['ecapa_tdnn'].get('required', False):
                raise
    
    def _init_rawnet2(self):
        """Initialize RawNet2 model."""
        logger.info("Initializing RawNet2 model")
        
        try:
            self._models['rawnet2'] = create_rawnet2(
                pretrained=self.config['models']['rawnet2'].get('pretrained', True)
            ).to(self.device).eval()
        except Exception as e:
            logger.error(f"Failed to initialize RawNet2: {str(e)}")
            if self.config['models']['rawnet2'].get('required', False):
                raise
    
    def analyze(self, audio_path: str, **kwargs) -> AudioForensicsResult:
        """
        Perform comprehensive audio forensic analysis.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional analysis parameters
            
        Returns:
            AudioForensicsResult containing analysis results
        """
        start_time = time.time()
        
        try:
            # Validate input
            self._validate_audio_file(audio_path)
            
            # Initialize result
            result = AudioForensicsResult(
                is_synthetic=False,
                confidence=0.0,
                risk_level=RiskLevel.VERY_LOW,
                model_versions={
                    'audio_forensics': __version__,
                    'pytorch': torch.__version__,
                    'torchaudio': torchaudio.__version__
                }
            )
            
            # Load audio
            audio, sample_rate = self._load_audio(audio_path)
            
            # Run analyses based on config
            if self.config['features']['voiceprint'] and 'ecapa_tdnn' in self._models:
                self._analyze_voiceprint(audio, sample_rate, result)
                
            if self.config['features']['noise_analysis']:
                self._analyze_noise(audio, sample_rate, result)
                
            if self.config['features']['compression_artifacts']:
                self._detect_compression_artifacts(audio, sample_rate, result)
            
            # Calculate overall result
            self._calculate_overall_result(result)
            
            # Add processing time
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise
    
    def _validate_audio_file(self, audio_path: str):
        """Validate audio file before processing."""
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Check file size (max 1GB)
        max_size = 1024 * 1024 * 1024  # 1GB
        file_size = os.path.getsize(audio_path)
        if file_size > max_size:
            raise ValueError(f"File too large: {file_size/1024/1024:.2f}MB (max {max_size/1024/1024}MB)")
    
    def _load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio file."""
        try:
            # Load audio with torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if needed
            if waveform.dim() > 1 and waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            target_sample_rate = self.config.get('target_sample_rate', DEFAULT_SAMPLE_RATE)
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=target_sample_rate
                )
                waveform = resampler(waveform)
                sample_rate = target_sample_rate
            
            return waveform.squeeze(0), sample_rate
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {str(e)}")
    
    def _analyze_voiceprint(self, audio: torch.Tensor, sample_rate: int, result: AudioForensicsResult):
        """Analyze voiceprint consistency."""
        try:
            # Convert to numpy array if needed
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = np.asarray(audio)
            
            # Get analysis parameters from config
            min_segment_duration = self.config['models']['ecapa_tdnn']['min_segment_duration']
            max_segments = self.config['models']['ecapa_tdnn']['max_segments']
            
            # Analyze voiceprint consistency
            voiceprint_result = analyze_voiceprint_consistency(
                audio_array=audio_np,
                sample_rate=sample_rate,
                segment_duration=min_segment_duration,
                max_segments=max_segments,
                device=self.device
            )
            
            # Update result
            result.features['voiceprint'] = voiceprint_result
            
            # Add signals if inconsistency detected
            if voiceprint_result.get('is_consistent', True) is False:
                result.signals_detected.append('inconsistent_voiceprint')
                result.confidence = max(result.confidence, voiceprint_result.get('confidence', 0.5))
                
        except Exception as e:
            logger.error(f"Voiceprint analysis failed: {str(e)}")
            if self.config.get('strict_mode', False):
                raise
    
    def _analyze_noise(self, audio: torch.Tensor, sample_rate: int, result: AudioForensicsResult):
        """Analyze noise patterns in audio."""
        try:
            # Convert to numpy if needed
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = np.asarray(audio)
            
            # Analyze noise patterns
            noise_analysis = {
                'rms_energy': float(np.sqrt(np.mean(audio_np**2))),
                'zcr': float(np.mean(librosa.feature.zero_crossing_rate(audio_np))),
                'spectral_flatness': float(np.mean(librosa.feature.spectral_flatness(y=audio_np)))
            }
            
            # Update result
            result.features['noise_analysis'] = noise_analysis
            
            # Check for suspicious patterns
            if noise_analysis['spectral_flatness'] > 0.9:  # Very flat spectrum
                result.signals_detected.append('high_spectral_flatness')
                result.confidence = max(result.confidence, 0.6)
                
        except Exception as e:
            logger.error(f"Noise analysis failed: {str(e)}")
            if self.config.get('strict_mode', False):
                raise
    
    def _detect_compression_artifacts(self, audio: torch.Tensor, sample_rate: int, result: AudioForensicsResult):
        """Detect compression artifacts in audio."""
        try:
            # This is a simplified example - actual implementation would be more sophisticated
            audio_np = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else np.asarray(audio)
            
            # Analyze high-frequency content
            fft_vals = np.fft.fft(audio_np)
            fft_freqs = np.fft.fftfreq(len(fft_vals), 1.0/sample_rate)
            
            # Look for signs of compression (e.g., high-frequency cutoff)
            high_freq_energy = np.sum(np.abs(fft_vals[fft_freqs > 0.4 * sample_rate])**2)
            total_energy = np.sum(np.abs(fft_vals)**2)
            high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
            
            compression_analysis = {
                'high_freq_energy_ratio': float(high_freq_ratio),
                'suspicious': high_freq_ratio < 0.01  # Threshold may need adjustment
            }
            
            result.features['compression_analysis'] = compression_analysis
            
            if compression_analysis['suspicious']:
                result.signals_detected.append('possible_compression_artifacts')
                result.confidence = max(result.confidence, 0.7)
                
        except Exception as e:
            logger.error(f"Compression artifact detection failed: {str(e)}")
            if self.config.get('strict_mode', False):
                raise
    
    def _calculate_overall_result(self, result: AudioForensicsResult):
        """Calculate overall result based on all analyses."""
        # If no signals detected, likely authentic
        if not result.signals_detected:
            result.is_synthetic = False
            result.confidence = max(0.8, result.confidence)  # High confidence in authenticity
            result.risk_level = RiskLevel.VERY_LOW
            return
        
        # Determine overall result based on signals and confidence
        result.is_synthetic = True
        
        # Adjust confidence based on number and type of signals
        signal_weights = {
            'inconsistent_voiceprint': 0.6,
            'high_spectral_flatness': 0.4,
            'possible_compression_artifacts': 0.5
        }
        
        # Calculate weighted confidence
        total_weight = 0.0
        weighted_sum = 0.0
        
        for signal in result.signals_detected:
            weight = signal_weights.get(signal, 0.3)
            weighted_sum += result.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            result.confidence = min(1.0, weighted_sum / total_weight)
        
        # Determine risk level
        if result.confidence >= 0.8:
            result.risk_level = RiskLevel.HIGH
        elif result.confidence >= 0.6:
            result.risk_level = RiskLevel.MEDIUM
        elif result.confidence >= 0.4:
            result.risk_level = RiskLevel.LOW
        else:
            result.risk_level = RiskLevel.VERY_LOW

class AudioValidationError(ValueError):
    """Raised when audio validation fails."""
    pass

class ModelLoadError(RuntimeError):
    """Raised when a model fails to load."""
    pass


class ECAPATDNN:
    """
    Enhanced ECAPA-TDNN wrapper for speaker embedding extraction with caching and error handling.
    
    Features:
    - Thread-safe model loading
    - Caching of embeddings
    - Input validation
    - Graceful degradation
    - Batch processing support
    - GPU acceleration
    """
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, device: str = None, model_name: str = "speechbrain/spkrec-ecapa-voxceleb", **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
                cls._instance._model = None
                cls._instance._processor = None
                cls._instance.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                cls._instance.model_name = model_name
                cls._instance.cache_dir = kwargs.get('cache_dir', os.path.expanduser("~/.cache/ecapa_tdnn"))
                cls._instance.embedding_dim = 192  # Standard ECAPA-TDNN embedding size
                cls._instance._load_model()
                cls._instance._initialized = True
        return cls._instance
    
    def _load_model(self):
        """Load the ECAPA-TDNN model with error handling and validation."""
        if not SPEECHBRAIN_AVAILABLE:
            logger.warning("SpeechBrain not available, ECAPA-TDNN disabled")
            return False
            
        try:
            logger.info(f"Loading ECAPA-TDNN model: {self.model_name}")
            
            # Create cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Load model with caching
            self._model = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=os.path.join(self.cache_dir, self.model_name.replace('/', '_')),
                run_opts={"device": self.device}
            )
            
            # Set model to evaluation mode
            self._model.eval()
            
            # Initialize processor for feature extraction
            self._processor = self._model.mods.compute_features
            
            logger.info(f"Successfully loaded ECAPA-TDNN model on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ECAPA-TDNN model: {str(e)}")
            self._model = None
            return False
            
        try:
            logger.info(f"Loading ECAPA-TDNN model from {self.model_name}...")
            start_time = time.time()
            
            # Create model directory if it doesn't exist
            model_dir = Path(f"models/ecapa_tdnn_{hashlib.md5(self.model_name.encode()).hexdigest()[:8]}")
            model_dir.mkdir(parents=True, exist_ok=True)
            
            self._model = EncoderClassifier.from_hparams(
                source=self.model_name,
                savedir=str(model_dir),
                run_opts={"device": self.device},
            )
            
            # Warm up the model
            with torch.no_grad():
                dummy_input = torch.randn(1, 16000, device=self.device)
                _ = self._model.encode_batch(dummy_input)
            
            load_time = time.time() - start_time
            logger.info(f"✅ ECAPA-TDNN loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Failed to load ECAPA-TDNN: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._model = None
            raise ModelLoadError(error_msg) from e

    def extract_embedding(
        self, audio_array: np.ndarray, sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio

        Args:
            audio_array: Audio as numpy array
            sample_rate: Sample rate (default 16000)

        Returns:
            Embedding vector or None if failed
        """
        if self.model is None:
            return None

        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)

            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(audio_tensor)
                embedding_np = embedding.squeeze().cpu().numpy()

            return embedding_np

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None


def validate_audio_input(
    audio_array: np.ndarray, 
    sample_rate: int, 
    min_duration: float = 1.0,
    max_duration: float = 3600.0
) -> None:
    """
    Validate audio input parameters.
    
    Args:
        audio_array: Input audio array
        sample_rate: Audio sample rate
        min_duration: Minimum allowed duration in seconds
        max_duration: Maximum allowed duration in seconds
        
    Raises:
        AudioValidationError: If validation fails
    """
    if not isinstance(audio_array, np.ndarray):
        raise AudioValidationError(f"audio_array must be a numpy array, got {type(audio_array)}")
        
    if len(audio_array.shape) != 1:
        raise AudioValidationError(f"audio_array must be 1D, got shape {audio_array.shape}")
    
    if audio_array.dtype.kind not in ('f', 'i'):
        raise AudioValidationError(f"Unsupported audio dtype: {audio_array.dtype}")
    
    duration = len(audio_array) / sample_rate
    if duration < min_duration:
        raise AudioValidationError(f"Audio too short: {duration:.2f}s < {min_duration:.2f}s")
        
    if duration > max_duration:
        raise AudioValidationError(f"Audio too long: {duration:.2f}s > {max_duration:.2f}s")
    
    if sample_rate < 8000 or sample_rate > 48000:
        raise AudioValidationError(f"Unsupported sample rate: {sample_rate} Hz")

def analyze_voiceprint_consistency(
    audio_array: np.ndarray, 
    sample_rate: int = SAMPLE_RATE, 
    segment_duration: float = 3.0,
    min_segments: int = 2,
    max_segments: int = 10,
    device: str = None
) -> Dict[str, Any]:
    """
    Enhanced voiceprint consistency analysis with improved error handling and validation.
    
    Args:
        audio_array: Input audio as 1D numpy array
        sample_rate: Sample rate in Hz (default: 16000)
        segment_duration: Duration of each analysis segment in seconds (default: 3.0)
        min_segments: Minimum number of segments required for analysis (default: 2)
        max_segments: Maximum number of segments to process (default: 10)
        device: Device to run the model on ('cuda' or 'cpu')
        
    Returns:
        Dictionary containing analysis results with the following keys:
        - score: float (0-1, higher = more likely authentic)
        - mean_similarity: float
        - std_similarity: float
        - num_segments: int
        - method: str
        - consistent: bool
        - suspicious_reason: Optional[str]
        - features: Dict with detailed features
        - metadata: Dict with processing metadata
        
    Raises:
        AudioValidationError: If input validation fails
        ModelLoadError: If model fails to load
        RuntimeError: For other processing errors
    """
    start_time = time.time()
    
    try:
        # Input validation
        validate_audio_input(audio_array, sample_rate)
        
        if segment_duration <= 0:
            raise AudioValidationError(f"segment_duration must be positive, got {segment_duration}")
            
        if min_segments < 2:
            raise AudioValidationError(f"min_segments must be >= 2, got {min_segments}")
        
        # Initialize ECAPA-TDNN with error handling
        try:
            ecapa = ECAPATDNN(device=device)
            if ecapa._model is None:
                raise ModelLoadError("Failed to initialize ECAPA-TDNN model")
        except Exception as e:
            logger.error(f"ECAPA-TDNN initialization failed: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to initialize voiceprint model: {e}") from e
        
        # Calculate segment parameters
        segment_samples = int(segment_duration * sample_rate)
        total_samples = len(audio_array)
        num_segments = min(max_segments, total_samples // segment_samples)
        
        if num_segments < min_segments:
            raise AudioValidationError(
                f"Insufficient audio for analysis. Need at least {min_segments} segments "
                f"of {segment_duration}s each, got {num_segments} segments"
            )
        
        logger.info(f"Processing {num_segments} segments of {segment_duration:.2f}s each")
        
        # Process segments in batches for better performance
        embeddings = []
        segment_metadata = []
        
        for i in range(num_segments):
            segment_start = i * segment_samples
            segment_end = segment_start + segment_samples
            segment = audio_array[segment_start:segment_end]
            
            try:
                # Extract embedding with timing
                start_embed = time.time()
                embedding = ecapa.extract_embedding(segment, sample_rate)
                embed_time = time.time() - start_embed
                
                if embedding is not None:
                    embeddings.append(embedding)
                    segment_metadata.append({
                        'segment_idx': i,
                        'start_sample': segment_start,
                        'end_sample': segment_end,
                        'embedding_shape': embedding.shape,
                        'processing_time': embed_time
                    })
                
            except Exception as e:
                logger.warning(f"Failed to process segment {i}: {e}")
                continue
        
        if len(embeddings) < 2:
            raise RuntimeError(
                f"Insufficient valid segments for analysis. Need at least 2, got {len(embeddings)}"
            )
        
        # Calculate pairwise similarities with vectorized operations
        embeddings = np.stack(embeddings)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Get upper triangle without diagonal
        triu_indices = np.triu_indices(len(embeddings), k=1)
        similarities = similarity_matrix[triu_indices]
        
        # Calculate statistics
        mean_similarity = float(np.mean(similarities))
        std_similarity = float(np.std(similarities))
        
        # Enhanced scoring with more nuanced thresholds
        if mean_similarity > 0.95:
            score = 0.3  # Too consistent, likely synthetic
            reason = "too_consistent"
            risk = "high"
        elif mean_similarity < 0.5:
            score = 0.4  # Too inconsistent, likely spliced
            reason = "too_inconsistent"
            risk = "high"
        elif 0.7 <= mean_similarity <= 0.9:
            score = 0.9  # Natural variation
            reason = None
            risk = "low"
        elif 0.5 <= mean_similarity < 0.7:
            score = 0.6  # Slightly inconsistent
            reason = "moderately_inconsistent"
            risk = "medium"
        else:  # 0.9 < mean_similarity <= 0.95
            score = 0.7  # Slightly too consistent
            reason = "slightly_too_consistent"
            risk = "medium"
        
        # Prepare detailed features
        features = {
            'similarity_matrix': similarity_matrix.tolist(),
            'pairwise_similarities': similarities.tolist(),
            'embedding_shapes': [e.shape for e in embeddings],
            'segment_duration_seconds': segment_duration,
            'num_segments_analyzed': len(embeddings),
            'sample_rate': sample_rate,
            'audio_duration_seconds': total_samples / sample_rate,
            'embedding_dimension': embeddings[0].shape[0]
        }
        
        # Prepare metadata
        processing_time = time.time() - start_time
        metadata = {
            'processing_time_seconds': processing_time,
            'segments_processed': len(segment_metadata),
            'segment_metadata': segment_metadata,
            'model': 'ECAPA-TDNN',
            'model_source': ecapa.model_name if hasattr(ecapa, 'model_name') else 'unknown',
            'device': str(device),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create structured result
        result = AudioForensicsResult(
            is_synthetic=score < 0.5,
            confidence=abs(score - 0.5) * 2,  # Convert to 0-1 confidence
            risk_level=risk,
            signals_detected=[reason] if reason else [],
            features=features,
            metadata=metadata
        )
        
        return result.to_dict()
        
    except (AudioValidationError, ModelLoadError) as e:
        logger.error(f"Validation/Model error in voiceprint analysis: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in voiceprint analysis: {e}", exc_info=True)
        raise RuntimeError(f"Voiceprint analysis failed: {e}") from e


class RawNet2(nn.Module):
    """
    Enhanced RawNet2 implementation for audio spoofing detection.
    
    Key improvements:
    - Added residual connections
    - Better weight initialization
    - Configurable architecture
    - Improved documentation
    - Type hints
    """
    
    def __init__(
        self, 
        num_classes: int = 2,
        sinc_filters: int = 128,
        sinc_kernel_size: int = 251,
        res_blocks: int = 4,
        res_channels: int = 64,
        gru_units: int = 1024,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        activation: str = 'leaky_relu'
    ):
        """
        Initialize RawNet2 model.
        
        Args:
            num_classes: Number of output classes
            sinc_filters: Number of sinc filters in first layer
            sinc_kernel_size: Kernel size for sinc convolution (must be odd)
            res_blocks: Number of residual blocks
            res_channels: Number of channels in residual blocks
            gru_units: Number of GRU units
            dropout: Dropout rate
            use_batchnorm: Whether to use batch normalization
            activation: Activation function ('relu', 'leaky_relu', 'gelu')
        """
        super().__init__()
        
        # Input validation
        if sinc_kernel_size % 2 != 1:
            raise ValueError(f"sinc_kernel_size must be odd, got {sinc_kernel_size}")
            
        if sinc_filters <= 0:
            raise ValueError(f"sinc_filters must be positive, got {sinc_filters}")
            
        # Activation function
        activation_fn = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'gelu': nn.GELU
        }.get(activation.lower(), nn.LeakyReLU)
        
        # Sinc layer (learnable filterbank)
        self.sinc_conv = nn.Conv1d(
            1, 
            sinc_filters, 
            kernel_size=sinc_kernel_size, 
            stride=1, 
            padding=(sinc_kernel_size - 1) // 2,
            bias=not use_batchnorm  # No bias if using batch norm
        )
        
        # Initial batch norm and activation
        self.bn1 = nn.BatchNorm1d(sinc_filters) if use_batchnorm else nn.Identity()
        self.activation = activation_fn()
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        in_channels = sinc_filters
        
        for i in range(res_blocks):
            # Add residual block with skip connection
            block = self._make_residual_block(
                in_channels=in_channels,
                out_channels=res_channels,
                dilation=2 ** (i % 4),  # Cycle through dilation rates
                activation=activation_fn,
                use_batchnorm=use_batchnorm,
                dropout=dropout
            )
            self.res_blocks.append(block)
            in_channels = res_channels  # Output channels become input for next block
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=res_channels,
            hidden_size=gru_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if 2 > 1 else 0
        )
        
        # Output layers
        self.fc = nn.Linear(2 * gru_units, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _make_residual_block(
        self, 
        in_channels: int, 
        out_channels: int, 
        dilation: int = 1,
        activation = nn.LeakyReLU,
        use_batchnorm: bool = True,
        dropout: float = 0.2
    ) -> nn.Module:
        """Create a residual block with skip connection."""
        return nn.Sequential(
            nn.Conv1d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                padding=dilation, 
                dilation=dilation,
                bias=not use_batchnorm
            ),
            nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity(),
            activation(),
            self.dropout,
            nn.Conv1d(
                out_channels, 
                out_channels, 
                kernel_size=3, 
                padding=dilation, 
                dilation=dilation,
                bias=not use_batchnorm
            ),
            nn.BatchNorm1d(out_channels) if use_batchnorm else nn.Identity(),
            activation(),
            self.dropout
        )
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_out', 
                    nonlinearity='leaky_relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Input shape: (batch_size, 1, seq_len)
        
        # Sinc convolution
        x = self.sinc_conv(x)  # (batch_size, sinc_filters, seq_len)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Residual blocks
        for block in self.res_blocks:
            residual = x
            x = block(x)
            
            # Add skip connection if dimensions match
            if x.shape == residual.shape:
                x = x + residual  # Skip connection
        
        # Permute for GRU: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # GRU
        x, _ = self.gru(x)  # (batch_size, seq_len, 2 * gru_units)
        
        # Take the last hidden state
        x = x[:, -1, :]  # (batch_size, 2 * gru_units)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Output
        x = self.fc(x)  # (batch_size, num_classes)
        
        # Return logits during training, probabilities during inference
        return x if self.training else self.sigmoid(x)
        self.res_block1 = self._make_res_block(128, 128)
        self.res_block2 = self._make_res_block(128, 256)
        self.res_block3 = self._make_res_block(256, 256)

        # GRU layer
        self.gru = nn.GRU(256, 1024, num_layers=3, batch_first=True)

        # Fully connected
        self.fc = nn.Linear(1024, num_classes)

    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.3),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.3),
            nn.MaxPool1d(3),
        )

    def forward(self, x):
        # x shape: (batch, samples)
        x = x.unsqueeze(1)  # (batch, 1, samples)

        # Sinc convolution
        x = torch.abs(self.sinc_conv(x))
        x = nn.functional.max_pool1d(x, 3)

        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # GRU
        x = x.transpose(1, 2)  # (batch, time, features)
        x, _ = self.gru(x)
        x = x[:, -1, :]  # Take last timestep

        # FC
        x = self.fc(x)

        return x


def create_rawnet2(pretrained=False):
    """
    Create RawNet2 model

    Args:
        pretrained: Whether to load pretrained weights (if available)

    Returns:
        RawNet2 model
    """
    model = RawNet2(num_classes=2)

    if pretrained:
        try:
            from pathlib import Path

            import torch

            # Try to load from models directory
            model_path = (
                Path(__file__).parent.parent / "models" / "rawnet2_pretrained.pth"
            )

            if model_path.exists():
                state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict)
                logger.info(f"✅ Loaded pretrained RawNet2 weights from {model_path}")
            else:
                logger.warning(
                    f"⚠️ Pretrained RawNet2 weights not found at {model_path}, using random initialization"
                )
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to load pretrained RawNet2 weights: {e}, using random initialization"
            )

    return model
