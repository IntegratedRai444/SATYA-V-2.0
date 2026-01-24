"""
Temporal Models for Video Analysis
Implements 3D CNNs and temporal attention mechanisms for deepfake video detection.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLSTM(nn.Module):
    """LSTM-based temporal model for video sequence analysis."""

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 128),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq, hidden*2)

        # Use last hidden state from both directions
        # h_n shape: (num_layers*2, batch, hidden)
        forward_hidden = h_n[-2, :, :]  # Last layer forward
        backward_hidden = h_n[-1, :, :]  # Last layer backward
        hidden = torch.cat(
            [forward_hidden, backward_hidden], dim=1
        )  # (batch, hidden*2)

        # Classification
        logits = self.fc(hidden)

        return logits


class TemporalAttention(nn.Module):
    """Temporal attention mechanism to focus on important frames."""

    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, features)
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        weighted = torch.sum(x * attention_weights, dim=1)  # (batch_size, features)
        return weighted, attention_weights.squeeze(-1)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for video analysis."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        # 3D convolution layers
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        # Temporal attention
        self.temporal_attention = TemporalAttention(256 * 7 * 7)

        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x shape: (batch_size, channels, frames, height, width)
        batch_size = x.size(0)

        # 3D convolutions
        x = self.conv1(x)  # (batch_size, 64, frames, 56, 56)
        x = self.conv2(x)  # (batch_size, 128, frames/2, 28, 28)
        x = self.conv3(x)  # (batch_size, 256, frames/4, 14, 14)

        # Reshape for attention
        x = x.permute(0, 2, 1, 3, 4)  # (batch_size, frames/4, 256, 14, 14)
        x = x.contiguous().view(
            batch_size, x.size(1), -1
        )  # (batch_size, frames/4, 256*14*14)

        # Apply temporal attention
        features, attention_weights = self.temporal_attention(
            x
        )  # (batch_size, 256*14*14)

        # Classification
        logits = self.fc(features)

        return {
            "logits": logits,
            "attention_weights": attention_weights,
            "features": features,
        }


class VideoDeepfakeDetector(nn.Module):
    """End-to-end video deepfake detection model."""

    def __init__(self, backbone: str = "r3d", num_classes: int = 2):
        super().__init__()
        self.backbone = self._build_backbone(backbone)
        self.temporal_net = TemporalConvNet()
        self.classifier = nn.Linear(512 + 256 * 7 * 7, num_classes)

    def _build_backbone(self, name: str) -> nn.Module:
        """Build 2D CNN backbone for frame feature extraction."""
        if name == "resnet18":
            model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
            return torch.nn.Sequential(*list(model.children())[:-1])
        elif name == "efficientnet":
            model = torch.hub.load(
                "NVIDIA/DeepLearningExamples:torchhub",
                "nvidia_efficientnet_b0",
                pretrained=True,
            )
            return torch.nn.Sequential(*list(model.children())[:-1])
        else:
            raise ValueError(f"Unsupported backbone: {name}")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, frames, height, width)
        """
        batch_size, channels, num_frames, h, w = x.size()

        # Extract frame features using 2D backbone
        frame_features = []
        for t in range(num_frames):
            frame = x[:, :, t, :, :]  # (batch_size, channels, h, w)
            feat = self.backbone(frame)  # (batch_size, features, h', w')
            frame_features.append(feat)

        # Stack frame features
        frame_features = torch.stack(
            frame_features, dim=2
        )  # (batch_size, features, num_frames, h', w')

        # Process with temporal network
        temporal_output = self.temporal_net(frame_features)

        # Combine features
        combined = torch.cat(
            [
                temporal_output["features"],
                frame_features.mean(dim=[2, 3, 4]),  # Global average pooling
            ],
            dim=1,
        )

        # Final classification
        logits = self.classifier(combined)

        return {
            "logits": logits,
            "attention_weights": temporal_output["attention_weights"],
            "frame_predictions": F.softmax(logits, dim=1),
        }


def load_video_model(
    checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> VideoDeepfakeDetector:
    """Load a pretrained video deepfake detection model."""
    model = VideoDeepfakeDetector()
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model
