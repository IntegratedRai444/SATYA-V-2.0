"""
Enhanced Audio Analysis for Deepfake Detection
Implements advanced audio processing and feature extraction.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Handles audio preprocessing and feature extraction."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 128,
        n_mels: int = 64,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Mel spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

        # MFCC transform
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=20,
            melkwargs={"n_fft": n_fft, "n_mels": n_mels, "hop_length": hop_length},
        )

    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and resample if necessary."""
        try:
            # Load audio with soundfile (faster than torchaudio for some formats)
            waveform, sr = sf.read(file_path, dtype="float32")

            # Convert to mono if stereo
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)

            # Convert to tensor
            waveform = torch.from_numpy(waveform)

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.sample_rate
                )
                waveform = resampler(waveform)

            return waveform, self.sample_rate

        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise

    def extract_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract various audio features."""
        features = {}

        # Ensure waveform is 2D: (channels, samples)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)

        # Mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        log_mel = torch.log(mel_spec + 1e-6)  # Add small value to avoid log(0)

        # MFCC
        mfcc = self.mfcc(waveform)

        # Spectral features
        spectral_centroid = self._spectral_centroid(waveform)
        spectral_bandwidth = self._spectral_bandwidth(waveform)
        spectral_contrast = self._spectral_contrast(waveform)

        # Temporal features
        zero_crossing_rate = self._zero_crossing_rate(waveform)
        rms_energy = self._rms_energy(waveform)

        # Pitch and harmonic features
        pitch = self._pitch_tracking(waveform.numpy())

        # Voice activity detection
        vad = self._voice_activity_detection(waveform)

        # Compile features
        features.update(
            {
                "log_mel": log_mel,
                "mfcc": mfcc,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "spectral_contrast": spectral_contrast,
                "zero_crossing_rate": zero_crossing_rate,
                "rms_energy": rms_energy,
                "pitch": pitch,
                "vad": vad,
            }
        )

        return features

    def _spectral_centroid(self, waveform: torch.Tensor) -> torch.Tensor:
        """Calculate spectral centroid."""
        spec = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True
        ).abs()
        freqs = torch.linspace(
            0, self.sample_rate // 2, spec.shape[-2], device=waveform.device
        )
        centroid = torch.sum(freqs.unsqueeze(-1) * spec, dim=-2) / (
            torch.sum(spec, dim=-2) + 1e-6
        )
        return centroid

    def _spectral_bandwidth(self, waveform: torch.Tensor) -> torch.Tensor:
        """Calculate spectral bandwidth."""
        spec = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True
        ).abs()
        centroid = self._spectral_centroid(waveform).unsqueeze(-2)
        freqs = torch.linspace(
            0, self.sample_rate // 2, spec.shape[-2], device=waveform.device
        ).unsqueeze(-1)
        bandwidth = torch.sum((freqs - centroid).pow(2) * spec, dim=-2) / (
            torch.sum(spec, dim=-2) + 1e-6
        )
        return bandwidth

    def _spectral_contrast(
        self, waveform: torch.Tensor, n_bands: int = 6
    ) -> torch.Tensor:
        """Calculate spectral contrast."""
        # Convert to numpy for librosa (more efficient for this operation)
        waveform_np = waveform.squeeze().numpy()
        contrast = librosa.feature.spectral_contrast(
            y=waveform_np,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_bands=n_bands,
        )
        return torch.from_numpy(contrast).float()

    def _zero_crossing_rate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Calculate zero crossing rate."""
        return torch.tensor(
            [
                librosa.feature.zero_crossing_rate(
                    w.squeeze().numpy(),
                    frame_length=self.n_fft,
                    hop_length=self.hop_length,
                )
                for w in waveform
            ]
        ).float()

    def _rms_energy(self, waveform: torch.Tensor) -> torch.Tensor:
        """Calculate RMS energy."""
        return torch.tensor(
            [
                librosa.feature.rms(
                    y=w.squeeze().numpy(),
                    frame_length=self.n_fft,
                    hop_length=self.hop_length,
                )
                for w in waveform
            ]
        ).float()

    def _pitch_tracking(self, waveform: np.ndarray) -> torch.Tensor:
        """Extract pitch using Pyin algorithm."""
        try:
            # Use librosa's pyin for pitch tracking
            f0, voiced_flag, _ = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=self.sample_rate,
                frame_length=self.n_fft,
                hop_length=self.hop_length,
            )
            return torch.from_numpy(f0).float()
        except Exception as e:
            logger.warning(f"Pitch tracking failed: {e}")
            return torch.zeros(1, waveform.shape[0] // self.hop_length + 1)

    def _voice_activity_detection(
        self, waveform: torch.Tensor, threshold: float = 0.5
    ) -> torch.Tensor:
        """Simple voice activity detection."""
        # Convert to numpy for librosa
        waveform_np = waveform.squeeze().numpy()

        # Use librosa's VAD
        intervals = librosa.effects.split(
            waveform_np,
            top_db=30,  # Adjust based on your needs
            frame_length=self.n_fft,
            hop_length=self.hop_length,
        )

        # Create VAD mask
        vad_mask = np.zeros_like(waveform_np, dtype=np.float32)
        for start, end in intervals:
            vad_mask[start:end] = 1.0

        return torch.from_numpy(vad_mask).float()


class AudioDeepfakeDetector(nn.Module):
    """Neural network for audio deepfake detection."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        # CNN for spectrogram processing
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256 * 4,  # Adjust based on your input size after conv layers
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 128),  # 2 * hidden_size (bidirectional)
            nn.Tanh(),
            nn.Linear(128, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),  # 2 * hidden_size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x shape: (batch, 1, n_mels, time)

        # CNN layers
        x = self.conv1(x)  # (batch, 64, n_mels//2, time//2)
        x = self.conv2(x)  # (batch, 128, n_mels//4, time//4)
        x = self.conv3(x)  # (batch, 256, n_mels//8, time//8)

        # Reshape for LSTM: (batch, time, features)
        batch_size, channels, height, time_steps = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, height)
        x = x.contiguous().view(
            batch_size, time_steps, -1
        )  # (batch, time, channels*height)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, 2*hidden_size)

        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, time, 1)
        context = torch.sum(
            attention_weights * lstm_out, dim=1
        )  # (batch, 2*hidden_size)

        # Classification
        logits = self.classifier(context)  # (batch, num_classes)

        return {
            "logits": logits,
            "attention_weights": attention_weights.squeeze(-1),
            "features": context,
        }


def detect_audio_deepfake(
    audio_path: str,
    model: Optional[nn.Module] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Detect deepfake audio using the provided model.

    Args:
        audio_path: Path to the audio file
        model: Pre-trained model (if None, will use default)
        device: Device to run inference on ('cuda' or 'cpu')
        threshold: Decision threshold for binary classification

    Returns:
        Dictionary containing detection results
    """
    start_time = time.time()

    # Initialize preprocessor
    preprocessor = AudioPreprocessor()

    try:
        # Load and preprocess audio
        waveform, _ = preprocessor.load_audio(audio_path)
        features = preprocessor.extract_features(waveform.unsqueeze(0))

        # If no model provided, use a simple rule-based approach
        if model is None:
            return {
                "success": True,
                "is_deepfake": False,
                "confidence": 0.0,  # No confidence when no model provided
                "features": {k: v.shape for k, v in features.items()},
                "processing_time": time.time() - start_time,
                "message": "No model provided, using default features",
            }

        # Prepare input for model
        log_mel = features["log_mel"].unsqueeze(0).to(device)  # Add batch dim

        # Run inference
        with torch.no_grad():
            outputs = model(log_mel)
            probs = F.softmax(outputs["logits"], dim=1)
            fake_prob = probs[0, 1].item()  # Assuming class 1 is 'fake'

        return {
            "success": True,
            "is_deepfake": fake_prob > threshold,
            "confidence": fake_prob,
            "processing_time": time.time() - start_time,
            "attention_weights": outputs["attention_weights"].cpu().numpy(),
            "features": {k: v.cpu().numpy() for k, v in features.items()},
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time,
        }


def train_audio_model(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_save_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Train the audio deepfake detection model.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        model_save_path: Path to save the best model

    Returns:
        Dictionary containing training history
    """
    # Initialize model, loss, and optimizer
    model = AudioDeepfakeDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Training history
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        ):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs["logits"], targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = outputs["logits"].max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Batch {batch_idx+1}/{len(train_loader)}: Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%"
                )

        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Save metrics
        train_loss /= len(train_loader)
        train_acc = 100.0 * correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_loss < best_val_loss and model_save_path:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                model_save_path,
            )
            print(f"Model saved to {model_save_path}")

        print(
            f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

    return history


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Validate the model on the validation set."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs["logits"], targets)

            # Statistics
            val_loss += loss.item()
            _, predicted = outputs["logits"].max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc
