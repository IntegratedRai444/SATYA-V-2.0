"""
Tests for the audio deepfake detection model.
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

from models.audio_model import (AudioDeepfakeDetector, AudioFeatureExtractor,
                                ResidualBlock)

# Test data configuration
TEST_AUDIO_SHAPE = (1, 64, 128)  # (channels, height, width)
BATCH_SIZE = 4
NUM_CLASSES = 2


# Fixtures
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def test_audio_batch():
    return torch.randn(BATCH_SIZE, *TEST_AUDIO_SHAPE)


@pytest.fixture
def test_labels():
    return torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))


# Test classes
class TestResidualBlock:
    def test_initialization(self, device):
        block = ResidualBlock(32, 64).to(device)
        assert isinstance(block.conv1, torch.nn.Conv2d)
        assert block.conv1.in_channels == 32
        assert block.conv1.out_channels == 64

    def test_forward_pass(self, device):
        block = ResidualBlock(32, 32).to(device)
        x = torch.randn(2, 32, 16, 16, device=device)
        out = block(x)
        assert out.shape == x.shape

    def test_downsampling(self, device):
        block = ResidualBlock(32, 64, stride=2).to(device)
        x = torch.randn(2, 32, 16, 16, device=device)
        out = block(x)
        assert out.shape == (2, 64, 8, 8)


class TestAudioFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return AudioFeatureExtractor()

    def test_feature_extraction(self, extractor):
        # Create dummy audio data (1 second of audio at 16kHz)
        audio = np.random.randn(16000)
        features = extractor.extract_features(audio)

        # Check that all expected features are present
        expected_features = ["mfcc", "mel_spec", "chroma", "contrast", "zcr", "rms"]
        for feat in expected_features:
            assert feat in features
            assert isinstance(features[feat], np.ndarray)
            assert features[feat].size > 0


class TestAudioDeepfakeDetector:
    @pytest.fixture
    def model(self, device):
        model = AudioDeepfakeDetector(
            input_size=TEST_AUDIO_SHAPE[1:],  # Remove channel dimension
            num_classes=NUM_CLASSES,
            use_attention=True,
        ).to(device)
        return model

    def test_initialization(self, model, device):
        assert isinstance(model, AudioDeepfakeDetector)
        assert model.training is True

    def test_forward_pass(self, model, test_audio_batch, device):
        test_audio_batch = test_audio_batch.to(device)
        model = model.to(device)

        # Test with batch
        output = model(test_audio_batch)
        assert output.shape == (BATCH_SIZE, NUM_CLASSES)

        # Test with single sample
        single_sample = test_audio_batch[0].unsqueeze(0)
        output_single = model(single_sample)
        assert output_single.shape == (1, NUM_CLASSES)

    def test_attention_mechanism(self, device):
        model = AudioDeepfakeDetector(
            input_size=TEST_AUDIO_SHAPE[1:], num_classes=NUM_CLASSES, use_attention=True
        ).to(device)

        # Check attention module exists
        assert hasattr(model, "attention")
        assert model.attention is not None

        # Test with attention disabled
        model_no_attn = AudioDeepfakeDetector(
            input_size=TEST_AUDIO_SHAPE[1:],
            num_classes=NUM_CLASSES,
            use_attention=False,
        ).to(device)
        assert model_no_attn.attention is None

    def test_training_step(self, model, test_audio_batch, test_labels, device):
        model.train()
        test_audio_batch = test_audio_batch.to(device)
        test_labels = test_labels.to(device)

        # Forward pass
        output = model(test_audio_batch)
        loss = torch.nn.functional.cross_entropy(output, test_labels)

        # Backward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that parameters were updated
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)

    def test_model_save_load(self, model, tmp_path, device):
        # Save model
        model_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), model_path)

        # Create new model and load weights
        new_model = AudioDeepfakeDetector(
            input_size=TEST_AUDIO_SHAPE[1:], num_classes=NUM_CLASSES
        ).to(device)
        new_model.load_state_dict(torch.load(model_path))

        # Check that models have same parameters
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2, atol=1e-6)


# Integration test with DataLoader
class TestModelIntegration:
    def test_dataloader_integration(self, model, test_audio_batch, test_labels, device):
        # Create dataset and dataloader
        dataset = TensorDataset(test_audio_batch, test_labels)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        model = model.to(device)
        model.train()

        # Test training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            output = model(batch_x)
            loss = criterion(output, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check output shape
            assert output.shape == (batch_x.size(0), NUM_CLASSES)
            break  # Just test one batch


# Test model with different input sizes
@pytest.mark.parametrize("height,width", [(64, 128), (128, 256), (32, 64)])
def test_model_with_different_sizes(height, width, device):
    model = AudioDeepfakeDetector(
        input_size=(height, width), num_classes=NUM_CLASSES
    ).to(device)

    # Test with batch
    test_input = torch.randn(2, 1, height, width, device=device)
    output = model(test_input)
    assert output.shape == (2, NUM_CLASSES)


# Test model with different batch sizes
@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_model_with_different_batch_sizes(batch_size, device):
    model = AudioDeepfakeDetector(
        input_size=TEST_AUDIO_SHAPE[1:], num_classes=NUM_CLASSES
    ).to(device)

    test_input = torch.randn(batch_size, *TEST_AUDIO_SHAPE, device=device)
    output = model(test_input)
    assert output.shape == (batch_size, NUM_CLASSES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
