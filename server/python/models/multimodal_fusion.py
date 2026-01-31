"""
Multimodal Fusion Model for Deepfake Detection
Combines features from image, audio, video, and text transformers
for comprehensive deepfake detection.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# HuggingFace Transformers
try:
    from transformers import (
        AutoModel, AutoConfig,
        AutoProcessor, AutoImageProcessor,
        AutoTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for fusing different modalities."""
    
    def __init__(self, embed_dims: List[int], num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims[0] // num_heads
        
        # Linear projections for each modality
        self.projections = nn.ModuleList([
            nn.Linear(dim, embed_dims[0]) for dim in embed_dims
        ])
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dims[0], num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dims[0])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of tensors from different modalities
        Returns:
            Fused features
        """
        # Project all features to same dimension
        projected = []
        for i, feat in enumerate(features):
            if feat.shape[-1] != self.embed_dims[i]:
                feat = self.projections[i](feat)
            projected.append(feat)
        
        # Stack for multi-head attention
        stacked = torch.stack(projected, dim=1)  # [batch, modalities, seq_len, embed_dim]
        
        # Apply multi-head attention
        attn_output, _ = self.attention(stacked, stacked, stacked)
        attn_output = self.norm(attn_output)
        
        # Global average pooling over modalities
        fused = attn_output.mean(dim=1)  # [batch, seq_len, embed_dim]
        
        return self.dropout(fused)

class AdaptiveFusion(nn.Module):
    """Adaptive fusion layer that learns optimal combination strategies."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        total_input_dim = sum(input_dims)
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(input_dims)) / len(input_dims))
        
        # Projection layers
        self.input_projs = nn.ModuleList([
            nn.Linear(dim, output_dim // len(input_dims)) for dim in input_dims
        ])
        
        self.output_proj = nn.Linear(output_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of tensors from different modalities
        Returns:
            Fused features
        """
        # Normalize weights
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # Project each modality
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.input_projs)):
            if feat.shape[-1] != proj.out_features:
                feat = proj(feat)
            projected.append(feat * weights[i])
        
        # Sum and project
        fused = sum(projected)
        fused = self.output_proj(fused)
        fused = self.norm(fused)
        
        return self.dropout(fused)

class MultimodalDeepfakeDetector(nn.Module):
    """Comprehensive multimodal deepfake detection using transformers."""
    
    def __init__(
        self,
        image_model_name: str = "google/vit-base-patch16-224",
        audio_model_name: str = "facebook/wav2vec2-base-960h",
        text_model_name: str = "roberta-base-openai-detector",
        video_model_name: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        fusion_strategy: str = "adaptive",  # 'concat', 'attention', 'adaptive'
        num_classes: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes
        
        # Initialize modality encoders
        self.modality_encoders = {}
        self.modality_dims = {}
        
        if TRANSFORMERS_AVAILABLE:
            self._init_transformers(image_model_name, audio_model_name, text_model_name, video_model_name)
        
        # Fusion layer
        if self.modality_encoders:
            embed_dims = list(self.modality_dims.values())
            
            if fusion_strategy == "attention":
                self.fusion = CrossModalAttention(embed_dims)
                fused_dim = embed_dims[0]
            elif fusion_strategy == "adaptive":
                self.fusion = AdaptiveFusion(embed_dims, 512)
                fused_dim = 512
            else:  # concat
                fused_dim = sum(embed_dims)
            
            # Classification head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(fused_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
    
    def _init_transformers(self, image_model_name: str, audio_model_name: str, text_model_name: str, video_model_name: str):
        """Initialize transformer models for each modality."""
        
        # Image encoder
        try:
            self.modality_encoders['image'] = AutoModel.from_pretrained(image_model_name)
            self.modality_dims['image'] = self.modality_encoders['image'].config.hidden_size
            self.image_processor = AutoImageProcessor.from_pretrained(image_model_name)
        except Exception as e:
            logger.warning(f"Failed to load image model: {e}")
        
        # Audio encoder
        try:
            self.modality_encoders['audio'] = AutoModel.from_pretrained(audio_model_name)
            self.modality_dims['audio'] = self.modality_encoders['audio'].config.hidden_size
            self.audio_processor = AutoProcessor.from_pretrained(audio_model_name)
        except Exception as e:
            logger.warning(f"Failed to load audio model: {e}")
        
        # Text encoder
        try:
            self.modality_encoders['text'] = AutoModel.from_pretrained(text_model_name)
            self.modality_dims['text'] = self.modality_encoders['text'].config.hidden_size
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        except Exception as e:
            logger.warning(f"Failed to load text model: {e}")
        
        # Video encoder
        try:
            self.modality_encoders['video'] = AutoModel.from_pretrained(video_model_name)
            self.modality_dims['video'] = self.modality_encoders['video'].config.hidden_size
            self.video_processor = AutoProcessor.from_pretrained(video_model_name)
        except Exception as e:
            logger.warning(f"Failed to load video model: {e}")
    
    def encode_modality(self, modality: str, data: Any) -> torch.Tensor:
        """Encode data using the appropriate transformer model."""
        if modality not in self.modality_encoders:
            raise ValueError(f"Modality {modality} not supported")
        
        model = self.modality_encoders[modality]
        
        try:
            if modality == 'image':
                inputs = self.image_processor(data, return_tensors="pt")
            elif modality == 'audio':
                inputs = self.audio_processor(data, return_tensors="pt", sampling_rate=16000)
            elif modality == 'text':
                inputs = self.text_tokenizer(data, truncation=True, padding=True, return_tensors="pt")
            elif modality == 'video':
                inputs = self.video_processor(data, return_tensors="pt")
            else:
                raise ValueError(f"Unknown modality: {modality}")
            
            with torch.no_grad():
                if hasattr(model, 'output_hidden_states'):
                    outputs = model(**inputs, output_hidden_states=True)
                    # Use last hidden state
                    features = outputs.last_hidden_state
                    # Global average pooling
                    features = features.mean(dim=1)
                else:
                    outputs = model(**inputs)
                    features = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
                    features = features.mean(dim=1)
            
            return features
            
        except Exception as e:
            logger.error(f"Error encoding {modality}: {e}")
            raise
    
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass through multimodal model.
        
        Args:
            inputs: Dictionary containing data for each modality
                - 'image': PIL Image or tensor
                - 'audio': Audio waveform or tensor
                - 'text': String or list of strings
                - 'video': Video frames or tensor
        
        Returns:
            Dictionary with predictions and intermediate features
        """
        features = {}
        available_modalities = []
        
        # Encode each available modality
        for modality, data in inputs.items():
            if modality in self.modality_encoders and data is not None:
                try:
                    features[modality] = self.encode_modality(modality, data)
                    available_modalities.append(modality)
                except Exception as e:
                    logger.warning(f"Failed to encode {modality}: {e}")
        
        if not available_modalities:
            return {
                "error": "No valid modalities provided",
                "prediction": None,
                "confidence": 0.0
            }
        
        # Fuse modalities
        if len(available_modalities) == 1:
            # Single modality
            fused_features = features[available_modalities[0]]
        else:
            # Multi-modal fusion
            feature_list = [features[mod] for mod in available_modalities]
            
            if self.fusion_strategy == "attention":
                fused_features = self.fusion(feature_list)
            elif self.fusion_strategy == "adaptive":
                fused_features = self.fusion(feature_list)
            else:  # concat
                fused_features = torch.cat(feature_list, dim=-1)
        
        # Classification
        logits = self.classifier(fused_features)
        probs = F.softmax(logits, dim=-1)
        
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs.max().item()
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probs.tolist(),
            "features": {k: v.cpu().numpy().tolist() for k, v in features.items()},
            "available_modalities": available_modalities,
            "fusion_strategy": self.fusion_strategy
        }


# Convenience function for multimodal analysis
def analyze_multimodal_deepfake(
    image=None,
    audio=None,
    text=None,
    video=None,
    fusion_strategy="adaptive"
) -> Dict[str, Any]:
    """Quick multimodal deepfake analysis."""
    detector = MultimodalDeepfakeDetector(fusion_strategy=fusion_strategy)
    
    inputs = {}
    if image is not None:
        inputs['image'] = image
    if audio is not None:
        inputs['audio'] = audio
    if text is not None:
        inputs['text'] = text
    if video is not None:
        inputs['video'] = video
    
    return detector(inputs)
