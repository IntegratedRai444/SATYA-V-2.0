"""
Tests for the video deepfake detection model.
"""
import os
# Add parent directory to path
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).parent.parent))

from models.video_model import (SpatioTemporalBlock, TemporalAttention,
                                VideoDeepfakeDetector, predict_video_deepfake,
                                preprocess_video_frames)

# Test data configuration
BATCH_SIZE = 2
NUM_FRAMES = 16
CHANNELS = 3
HEIGHT, WIDTH = 112, 112


# Fixtures
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def test_video_batch():
    return torch.randn(BATCH_SIZE, CHANNELS, NUM_FRAMES, HEIGHT, WIDTH)


@pytest.fixture
def test_frames():
    # Create random RGB frames (0-255)
    return [np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8) for _ in range(10)]


# Test classes
class TestTemporalAttention:
    def test_initialization(self, device):
        attn = TemporalAttention(64).to(device)
        assert isinstance(attn.avg_pool, torch.nn.AdaptiveAvgPool3d)
        assert isinstance(attn.max_pool, torch.nn.AdaptiveMaxPool3d)
        assert len(attn.mlp) == 4  # 2 Linear, ReLU, Sigmoid

    def test_forward_pass(self, device):
        attn = TemporalAttention(64).to(device)
        x = torch.randn(2, 64, 16, 14, 14, device=device)
        out = attn(x)
        assert out.shape == x.shape


class TestSpatioTemporalBlock:
    def test_initialization(self, device):
        block = SpatioTemporalBlock(32, 64).to(device)
        assert isinstance(block.conv1, torch.nn.Conv3d)
        assert block.conv1.in_channels == 32
        assert block.conv1.out_channels == 64

    def test_forward_pass(self, device):
        block = SpatioTemporalBlock(32, 32).to(device)
        x = torch.randn(2, 32, 16, 14, 14, device=device)
        out = block(x)
        assert out.shape == x.shape

    def test_downsampling(self, device):
        block = SpatioTemporalBlock(32, 64, stride=2).to(device)
        x = torch.randn(2, 32, 16, 28, 28, device=device)
        out = block(x)
        assert out.shape == (2, 64, 8, 14, 14)


class TestVideoDeepfakeDetector:
    @pytest.fixture
    def model(self, device):
        model = VideoDeepfakeDetector(
            in_channels=CHANNELS, num_classes=2, use_attention=True
        ).to(device)
        return model

    def test_initialization(self, model):
        assert isinstance(model, VideoDeepfakeDetector)
        assert model.training is True
        assert model.use_attention is True

    def test_forward_pass(self, model, test_video_batch, device):
        test_video_batch = test_video_batch.to(device)
        output = model(test_video_batch)
        assert output.shape == (BATCH_SIZE, 2)

    def test_feature_extraction(self, model, test_video_batch, device):
        test_video_batch = test_video_batch.to(device)
        features = model.forward_features(test_video_batch)
        assert features.dim() == 5  # (batch, channels, frames, h, w)

    def test_training_step(self, model, test_video_batch, device):
        model.train()
        test_video_batch = test_video_batch.to(device)
        labels = torch.randint(0, 2, (BATCH_SIZE,), device=device)

        # Forward pass
        outputs = model(test_video_batch)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        # Backward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check gradients
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)


class TestPreprocessing:
    def test_preprocess_video_frames(self, test_frames):
        video_tensor = preprocess_video_frames(test_frames)
        assert isinstance(video_tensor, torch.Tensor)
        assert video_tensor.dim() == 5  # (1, C, T, H, W)
        assert video_tensor.shape[1] == 3  # RGB channels
        assert video_tensor.shape[2] == len(test_frames)  # Number of frames

    def test_preprocess_video_frames_grayscale(self):
        # Test with grayscale input
        frames = [
            np.random.randint(0, 256, (240, 320), dtype=np.uint8) for _ in range(5)
        ]
        video_tensor = preprocess_video_frames(frames)
        assert video_tensor.shape[1] == 3  # Should be converted to RGB

    def test_preprocess_video_frames_rgba(self):
        # Test with RGBA input
        frames = [
            np.random.randint(0, 256, (240, 320, 4), dtype=np.uint8) for _ in range(5)
        ]
        video_tensor = preprocess_video_frames(frames)
        assert video_tensor.shape[1] == 3  # Should be converted to RGB


class TestInference:
    def test_predict_video_deepfake(self, test_frames, device):
        # Test with default model
        result = predict_video_deepfake(test_frames, device=device)

        # Check output format
        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result
        assert "per_frame_scores" in result
        assert "explanation" in result

        # Type checks
        assert isinstance(result["label"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["per_frame_scores"], list)
        assert len(result["per_frame_scores"]) == len(test_frames)
        assert all(isinstance(x, float) for x in result["per_frame_scores"])
        assert all(0 <= x <= 1 for x in result["per_frame_scores"])

    def test_predict_with_custom_model(self, test_frames, device):
        # Create a small test model
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(3, 16, kernel_size=3, padding=1)
                self.pool = torch.nn.AdaptiveAvgPool3d(1)
                self.fc = torch.nn.Linear(16, 2)

            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
                return self.fc(x)

        model = TestModel().to(device)
        result = predict_video_deepfake(test_frames, model=model, device=device)
        assert result["label"] in ["real", "fake"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
