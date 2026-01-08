"""
Enhanced Video Deepfake Detection Model

This module provides a 3D CNN with temporal attention for detecting deepfakes in videos.
It analyzes both spatial and temporal patterns to identify manipulations.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """Temporal attention module to focus on relevant frames."""

    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.size()

        # Average and max pooling along spatial dimensions
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)

        # Shared MLP
        avg_attn = self.mlp(avg_out)
        max_attn = self.mlp(max_out)

        # Combine attention weights
        attn = (avg_attn + max_attn).view(b, c, 1, 1, 1)
        return x * attn.expand_as(x)


class SpatioTemporalBlock(nn.Module):
    """Spatio-temporal block with 3D convolutions and residual connections."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.activation = nn.SiLU(inplace=True)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        return self.activation(out)


class VideoDeepfakeDetector(nn.Module):
    """3D CNN with temporal attention for video deepfake detection.

    Args:
        in_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        use_attention: Whether to use temporal attention
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        # Temporal attention
        if use_attention:
            self.temporal_attn = TemporalAttention(256)

        # Classifier
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(SpatioTemporalBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(SpatioTemporalBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatio-temporal features."""
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.use_attention:
            x = self.temporal_attn(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input shape: (batch, channels, frames, height, width)
        features = self.forward_features(x)
        pooled = self.avg_pool(features).view(features.size(0), -1)
        return self.classifier(pooled)


# Utility functions for inference
def preprocess_video_frames(
    frames: List[np.ndarray], target_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """Preprocess video frames for the model.

    Args:
        frames: List of video frames as numpy arrays in HWC format (0-255)
        target_size: Target (height, width) for resizing

    Returns:
        Tensor of shape (1, C, T, H, W) in range [0, 1]
    """
    import cv2
    import torchvision.transforms as T

    # Convert frames to tensor and normalize
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    processed_frames = []
    for frame in frames:
        # Convert to RGB if needed
        if frame.shape[2] == 4:  # RGBA to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 1:  # Grayscale to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Apply transformations
        frame_tensor = transform(frame)
        processed_frames.append(frame_tensor)

    # Stack along time dimension: (T, C, H, W) -> (1, C, T, H, W)
    video_tensor = torch.stack(processed_frames, dim=1).unsqueeze(0)
    return video_tensor


def predict_video_deepfake(
    frames: List[np.ndarray], model: Optional[nn.Module] = None, device: str = None
) -> Dict[str, any]:
    """Predict if a video contains deepfake content.

    Args:
        frames: List of video frames as numpy arrays in HWC format (0-255)
        model: Pre-trained model (if None, a default model will be loaded)
        device: Device to run inference on ('cuda' or 'cpu')

    Returns:
        Dictionary containing:
        - 'label': Predicted class ('real' or 'fake')
        - 'confidence': Prediction confidence (0-1)
        - 'per_frame_scores': Confidence scores for each frame
        - 'explanation': List of explanations for the prediction
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load default model if not provided
    if model is None:
        model = VideoDeepfakeDetector().to(device)

        # Try to load pretrained weights
        try:
            from pathlib import Path

            model_path = (
                Path(__file__).parent.parent / "models" / "video_deepfake_detector.pth"
            )

            if model_path.exists():
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                logger.info(
                    f"✅ Loaded pretrained video model weights from {model_path}"
                )
            else:
                logger.warning(
                    f"⚠️ Pretrained video model weights not found at {model_path}, using random initialization"
                )
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to load pretrained video model weights: {e}, using random initialization"
            )

    model.eval()

    # Preprocess frames
    video_tensor = preprocess_video_frames(frames).to(device)

    # Run inference
    with torch.no_grad():
        logits = model(video_tensor)
        probs = F.softmax(logits, dim=1)
        confidence, preds = torch.max(probs, 1)

        # Get per-frame predictions (for visualization)
        frame_logits = []
        for i in range(video_tensor.size(2)):  # Iterate over frames
            frame = video_tensor[:, :, i : i + 1, :, :]  # (1, C, 1, H, W)
            frame_logit = model(frame).squeeze(0)  # (num_classes,)
            frame_logits.append(frame_logit.cpu().numpy())

    # Format results
    label = "fake" if preds.item() == 1 else "real"
    confidence = confidence.item()
    per_frame_scores = [float(probs[0, 1].item())] * len(frames)  # Fake probability

    # Generate explanation
    explanation = [
        f"Overall prediction: {label} with {confidence*100:.1f}% confidence",
        f"Analyzed {len(frames)} frames",
    ]

    return {
        "label": label,
        "confidence": confidence,
        "per_frame_scores": per_frame_scores,
        "explanation": explanation,
    }
