import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """Extracts audio features for deepfake detection."""

    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract multiple audio features."""
        features = {}

        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        # Extract chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Extract spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Extract zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )

        # Extract RMS energy
        rms = librosa.feature.rms(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )

        return {
            "mfcc": mfcc,
            "mel_spec": mel_spec,
            "chroma": chroma,
            "contrast": contrast,
            "zcr": zcr,
            "rms": rms,
        }


class AudioDeepfakeDetector(nn.Module):
    """Enhanced neural network for audio deepfake detection with improved architecture.

    Features:
    - Residual connections for better gradient flow
    - Squeeze-and-Excitation blocks for channel attention
    - Swish activation for better performance
    - Adaptive pooling for variable input sizes
    - Dropout and batch normalization for regularization

    Args:
        input_size: Expected input size (height, width)
        num_classes: Number of output classes (default: 2 for real/fake)
        dropout_rate: Dropout rate for regularization (default: 0.3)
        use_attention: Whether to use channel attention (default: True)
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (64, 128),
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_attention = use_attention

        # Enhanced feature extraction with residual blocks
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks with increasing channels
        self.layer1 = self._make_residual_block(32, 64, 2)
        self.layer2 = self._make_residual_block(64, 128, 2, stride=2)
        self.layer3 = self._make_residual_block(128, 256, 2, stride=2)

        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(256, 16, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(16, 256, kernel_size=1),
                nn.Sigmoid(),
            )

        # Adaptive pooling and classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
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

    def _make_residual_block(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """Create a residual block with optional downsampling"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor"""
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.use_attention:
            attention_weights = self.attention(x)
            x = x * attention_weights

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

        features = self.forward_features(x)
        pooled = self.adaptive_pool(features).view(features.size(0), -1)
        return self.classifier(pooled)


class ResidualBlock(nn.Module):
    """Residual block with optional downsampling"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
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

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def _get_conv_output_size(self, input_size: Tuple[int, int]) -> int:
        """Calculate the size of the flattened features after convolutions."""
        x = torch.zeros(1, 1, *input_size)
        x = self.features(x)
        return int(torch.prod(torch.tensor(x.size()[1:])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, height, width) or (batch_size, height, width)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        # Extract features
        x = self.features(x)

        # Flatten the features
        x = x.view(x.size(0), -1)

        # Classify
        x = self.classifier(x)

        return x

    def get_feature_extractor(self) -> nn.Module:
        """Get the feature extraction part of the model."""
        return self.features

    def get_classifier(self) -> nn.Module:
        """Get the classifier part of the model."""
        return self.classifier


class AudioDeepfakeDetectorAPI:
    """API for audio deepfake detection with training and inference support.

    This class provides a high-level interface for training and using the audio deepfake
    detection model. It supports both training from scratch and fine-tuning pretrained models,
    as well as inference with various optimizations (ONNX, TensorRT).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = None,
        input_size: Tuple[int, int] = (64, 128),
        num_classes: int = 2,
    ):
        """Initialize the detector.

        Args:
            model_path: Path to a pretrained model checkpoint
            device: Device to run the model on ('cuda' or 'cpu')
            input_size: Expected input size (height, width)
            num_classes: Number of output classes
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.num_classes = num_classes
        self.feature_extractor = AudioFeatureExtractor()

        # Initialize model
        self.model = AudioDeepfakeDetector(
            input_size=input_size, num_classes=num_classes
        ).to(self.device)

        # Load pretrained weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        # Configure ONNX and TensorRT
        self.use_onnx = False
        self.use_tensorrt = False
        self.ort_session = None

        # Training configuration
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Configure batch processing
        self.batch_size = 32
        self.max_workers = 4

        logger.info(f"Initialized AudioDeepfakeDetector on device: {self.device}")

    def train_epoch(
        self, dataloader: torch.utils.data.DataLoader, epoch: int, num_epochs: int
    ) -> Tuple[float, float]:
        """Train the model for one epoch.

        Args:
            dataloader: DataLoader for training data
            epoch: Current epoch number
            num_epochs: Total number of epochs

        Returns:
            Tuple of (average loss, accuracy) for the epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"loss": running_loss / total, "acc": 100.0 * correct / total}
            )

        # Step the learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Validate the model on the validation set.

        Args:
            dataloader: DataLoader for validation data

        Returns:
            Tuple of (average loss, accuracy) on the validation set
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validating", ncols=100):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(dataloader.dataset)
        val_acc = 100.0 * correct / total

        return val_loss, val_acc

    def configure_optimizers(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        scheduler_patience: int = 5,
        scheduler_factor: float = 0.1,
    ):
        """Configure the optimizer and learning rate scheduler.

        Args:
            learning_rate: Initial learning rate
            weight_decay: Weight decay for L2 regularization
            scheduler_patience: Patience for learning rate reduction
            scheduler_factor: Factor by which to reduce the learning rate
        """
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",  # Monitor validation accuracy
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def save_model(self, path: str):
        """Save the model to disk.

        Args:
            path: Path to save the model checkpoint
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
                if self.optimizer
                else None,
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler
                else None,
                "input_size": self.input_size,
                "num_classes": self.num_classes,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a model from disk.

        Args:
            path: Path to the model checkpoint
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if (
            self.optimizer
            and "optimizer_state_dict" in checkpoint
            and checkpoint["optimizer_state_dict"] is not None
        ):
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if (
            self.scheduler
            and "scheduler_state_dict" in checkpoint
            and checkpoint["scheduler_state_dict"] is not None
        ):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Loaded model from {path}")

    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load the model from disk or initialize a new one."""
        model = AudioDeepfakeDetector()

        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
                logger.info("Using randomly initialized model")
        else:
            logger.info("No model path provided, using randomly initialized model")

        return model.to(self.device)

    def preprocess_audio(
        self, audio_data: Union[bytes, np.ndarray, str], sr: int = None
    ) -> np.ndarray:
        """Preprocess audio data for inference."""
        try:
            # Load audio if path is provided
            if isinstance(audio_data, str):
                if not os.path.exists(audio_data):
                    raise FileNotFoundError(f"Audio file not found: {audio_data}")
                y, _ = librosa.load(audio_data, sr=self.feature_extractor.sr)
            # Convert bytes to numpy array
            elif isinstance(audio_data, bytes):
                y, _ = librosa.load(
                    io.BytesIO(audio_data), sr=self.feature_extractor.sr
                )
            # Use numpy array directly
            elif isinstance(audio_data, np.ndarray):
                y = audio_data
                if sr and sr != self.feature_extractor.sr:
                    y = librosa.resample(
                        y, orig_sr=sr, target_sr=self.feature_extractor.sr
                    )
            else:
                raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

            # Extract features
            features = self.feature_extractor.extract_features(y)

            # Stack features to create a single feature tensor
            feature_tensor = np.vstack(
                [
                    features["mfcc"],
                    features["mel_spec"],
                    features["chroma"],
                    features["contrast"],
                    features["zcr"],
                    features["rms"],
                ]
            )

            # Normalize features
            feature_tensor = (feature_tensor - np.mean(feature_tensor)) / (
                np.std(feature_tensor) + 1e-8
            )

            return feature_tensor

        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise

    def predict(
        self,
        audio_data: Union[bytes, np.ndarray, str, List[Union[bytes, np.ndarray, str]]],
        batch_size: int = None,
    ) -> List[Dict[str, Any]]:
        """Predict if audio is real or fake."""
        # Handle single input
        if not isinstance(audio_data, list):
            audio_data = [audio_data]

        # Process in batches
        batch_size = batch_size or self.batch_size
        results = []

        for i in range(0, len(audio_data), batch_size):
            batch = audio_data[i : i + batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)

        return results if len(results) > 1 else results[0]

    def _process_batch(
        self, audio_batch: List[Union[bytes, np.ndarray, str]]
    ) -> List[Dict[str, Any]]:
        """Process a batch of audio samples."""
        try:
            # Preprocess batch in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                features = list(executor.map(self.preprocess_audio, audio_batch))

            # Convert to tensor
            features_tensor = torch.tensor(np.array(features), dtype=torch.float32).to(
                self.device
            )

            # Run inference
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probs = F.softmax(outputs, dim=1)
                fake_probs = probs[:, 1].cpu().numpy()

            # Format results
            batch_results = []
            for i, fake_prob in enumerate(fake_probs):
                confidence = float(fake_prob if fake_prob > 0.5 else 1.0 - fake_prob)
                label = "FAKE" if fake_prob > 0.5 else "REAL"

                batch_results.append(
                    {
                        "label": label,
                        "confidence": confidence * 100,
                        "fake_probability": float(fake_prob),
                        "real_probability": float(1.0 - fake_prob),
                        "explanation": self._generate_explanation(fake_prob),
                    }
                )

            return batch_results

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Return error results for the failed batch
            return [
                {"label": "ERROR", "confidence": 0.0, "error": str(e)}
                for _ in audio_batch
            ]

    def _generate_explanation(self, fake_prob: float) -> str:
        """Generate a human-readable explanation of the prediction."""
        if fake_prob > 0.8:
            return "Highly likely to be a deepfake audio (confidence > 80%)"
        elif fake_prob > 0.6:
            return "Likely to be a deepfake audio (confidence > 60%)"
        elif fake_prob > 0.4:
            return "Uncertain, but leaning towards deepfake (confidence 40-60%)"
        elif fake_prob > 0.2:
            return "Likely to be real audio (confidence > 60%)"
        else:
            return "Highly likely to be real audio (confidence > 80%)"

    def save_model(self, path: str):
        """Save the model to disk."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")

    def optimize_for_inference(
        self, use_onnx: bool = True, use_tensorrt: bool = False, output_path: str = None
    ):
        """Optimize the model for inference."""
        if use_onnx:
            self._convert_to_onnx(output_path)

        if use_tensorrt and use_onnx:
            self._optimize_with_tensorrt(output_path)

    def _convert_to_onnx(self, output_dir: str = None):
        """Convert the model to ONNX format."""
        try:
            import onnx
            import onnxruntime as ort

            if output_dir is None:
                output_dir = "models"

            os.makedirs(output_dir, exist_ok=True)
            onnx_path = os.path.join(output_dir, "audio_deepfake_detector.onnx")

            # Create dummy input
            dummy_input = torch.randn(1, 1, 64, 128).to(self.device)

            # Export the model
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            # Verify ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # Create ONNX Runtime session
            self.ort_session = ort.InferenceSession(onnx_path)
            self.use_onnx = True

            logger.info(f"Model successfully exported to ONNX: {onnx_path}")

        except ImportError:
            logger.warning("ONNX not available, skipping ONNX export")
        except Exception as e:
            logger.error(f"Failed to convert to ONNX: {e}")

    def _optimize_with_tensorrt(self, output_dir: str = None):
        """Optimize the ONNX model with TensorRT."""
        try:
            import tensorrt as trt

            if output_dir is None:
                output_dir = "models"

            os.makedirs(output_dir, exist_ok=True)
            onnx_path = os.path.join(output_dir, "audio_deepfake_detector.onnx")
            engine_path = os.path.join(output_dir, "audio_deepfake_detector.trt")

            logger = trt.Logger(trt.Logger.INFO)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)

            # Parse ONNX model
            with open(onnx_path, "rb") as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logger.log(trt.Logger.ERROR, parser.get_error(error).desc())
                    raise RuntimeError("Failed to parse ONNX model")

            # Build configuration
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB

            # Build engine
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")

            # Save engine
            with open(engine_path, "wb") as f:
                f.write(engine.serialize())

            logger.info(f"Successfully optimized model with TensorRT: {engine_path}")

        except ImportError:
            logger.warning("TensorRT not available, skipping optimization")
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")


# Backward compatibility
def predict_audio_deepfake(audio_bytes):
    """Legacy function for backward compatibility."""
    detector = AudioDeepfakeDetectorAPI()
    result = detector.predict(audio_bytes)
    return result["label"], result["confidence"], [result["explanation"]]


def batch_predict_audio_deepfake(audio_data_list):
    """Batch prediction for multiple audio samples."""
    detector = AudioDeepfakeDetectorAPI()
    return detector.predict(audio_data_list)
