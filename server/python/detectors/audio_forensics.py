"""
Audio Forensics Module for Deepfake Detection
Implements ECAPA-TDNN, RawNet2, and voiceprint analysis
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import speechbrain
try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    logger.warning("speechbrain not installed. Run: pip install speechbrain")


class ECAPATDNN:
    """
    ECAPA-TDNN wrapper for speaker embedding extraction
    Uses SpeechBrain pretrained models
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        
        if SPEECHBRAIN_AVAILABLE:
            try:
                logger.info("Loading ECAPA-TDNN model...")
                # Load pretrained ECAPA-TDNN from SpeechBrain
                self.model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="models/ecapa_tdnn",
                    run_opts={"device": device}
                )
                logger.info("✅ ECAPA-TDNN loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ECAPA-TDNN: {e}")
                self.model = None
        else:
            logger.warning("SpeechBrain not available, ECAPA-TDNN disabled")
    
    def extract_embedding(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
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


def analyze_voiceprint_consistency(audio_array: np.ndarray, sample_rate: int = 16000, 
                                   segment_duration: float = 3.0) -> Dict:
    """
    Analyze voiceprint consistency across audio segments
    Detects splicing and voice cloning artifacts
    
    Args:
        audio_array: Audio as numpy array
        sample_rate: Sample rate
        segment_duration: Duration of each segment in seconds
        
    Returns:
        Dictionary with consistency analysis
    """
    try:
        # Initialize ECAPA-TDNN
        ecapa = ECAPATDNN()
        
        if ecapa.model is None:
            return {
                'score': 0.5,
                'message': 'ECAPA-TDNN not available',
                'method': 'voiceprint_consistency'
            }
        
        # Split audio into segments
        segment_samples = int(segment_duration * sample_rate)
        num_segments = len(audio_array) // segment_samples
        
        if num_segments < 2:
            return {
                'score': 0.7,
                'message': 'Audio too short for consistency analysis',
                'method': 'voiceprint_consistency'
            }
        
        # Extract embeddings for each segment
        embeddings = []
        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = audio_array[start:end]
            
            embedding = ecapa.extract_embedding(segment, sample_rate)
            if embedding is not None:
                embeddings.append(embedding)
        
        if len(embeddings) < 2:
            return {
                'score': 0.5,
                'message': 'Failed to extract embeddings',
                'method': 'voiceprint_consistency'
            }
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                emb1 = embeddings[i]
                emb2 = embeddings[j]
                
                # Cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities.append(similarity)
        
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        # Scoring logic:
        # - Very high similarity (> 0.95) = possibly synthetic (too consistent)
        # - Moderate similarity (0.7-0.9) = authentic (same speaker, natural variation)
        # - Low similarity (< 0.6) = possibly spliced (different speakers)
        if 0.7 <= mean_similarity <= 0.9:
            score = 0.9
        elif mean_similarity > 0.95:
            score = 0.4  # Too consistent, possibly synthetic
        elif mean_similarity < 0.6:
            score = 0.3  # Too inconsistent, possibly spliced
        else:
            score = 0.6
        
        return {
            'score': float(score),
            'mean_similarity': float(mean_similarity),
            'std_similarity': float(std_similarity),
            'num_segments': num_segments,
            'method': 'voiceprint_consistency',
            'consistent': 0.7 <= mean_similarity <= 0.9,
            'suspicious_reason': 'too_consistent' if mean_similarity > 0.95 else 'too_inconsistent' if mean_similarity < 0.6 else None
        }
        
    except Exception as e:
        logger.error(f"Voiceprint consistency analysis failed: {e}")
        return {'score': 0.5, 'error': str(e)}


class RawNet2(nn.Module):
    """
    RawNet2: End-to-end neural network for audio spoofing detection
    Simplified implementation based on the paper
    """
    
    def __init__(self, num_classes=2):
        super(RawNet2, self).__init__()
        
        # Sinc layer (learnable filterbank)
        self.sinc_conv = nn.Conv1d(1, 128, kernel_size=251, stride=1, padding=125, bias=False)
        
        # Residual blocks
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
            nn.MaxPool1d(3)
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
            model_path = Path(__file__).parent.parent / 'models' / 'rawnet2_pretrained.pth'
            
            if model_path.exists():
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info(f"✅ Loaded pretrained RawNet2 weights from {model_path}")
            else:
                logger.warning(f"⚠️ Pretrained RawNet2 weights not found at {model_path}, using random initialization")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load pretrained RawNet2 weights: {e}, using random initialization")
    
    return model
