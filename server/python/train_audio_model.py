#!/usr/bin/env python3
"""
Audio Deepfake Detection Model Training Script

This script handles the training and validation of the audio deepfake detection model.
It includes data loading, training loop, validation, and model saving.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.audio_model import AudioDeepfakeDetectorAPI, AudioFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    """Dataset for loading and processing audio files for training."""
    
    def __init__(self, real_dir: str, fake_dir: str, feature_extractor: AudioFeatureExtractor,
                 max_duration: float = 4.0, sample_rate: int = 16000, augment: bool = False):
        """
        Initialize the dataset.
        
        Args:
            real_dir: Directory containing real audio files
            fake_dir: Directory containing fake audio files
            feature_extractor: Feature extractor instance
            max_duration: Maximum duration in seconds (audio will be trimmed/padded)
            sample_rate: Target sample rate
            augment: Whether to apply data augmentation
        """
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.augment = augment
        
        # Load file paths
        self.real_files = self._load_audio_files(real_dir)
        self.fake_files = self._load_audio_files(fake_dir)
        
        # Create labels (0 for real, 1 for fake)
        self.files = self.real_files + self.fake_files
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)
        
        logger.info(f"Loaded {len(self.real_files)} real and {len(self.fake_files)} fake audio samples")
    
    def _load_audio_files(self, directory: str) -> List[str]:
        """Load audio file paths from directory."""
        if not os.path.isdir(directory):
            logger.warning(f"Directory not found: {directory}")
            return []
            
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            audio_files.extend(Path(directory).rglob(ext))
        
        return [str(f) for f in audio_files]
    
    def _load_audio(self, file_path: str) -> np.ndarray:
        """Load and preprocess audio file."""
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            # Ensure audio is the correct length
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            elif len(audio) < self.max_samples:
                audio = np.pad(audio, (0, max(0, self.max_samples - len(audio))), 'constant')
            
            # Apply data augmentation
            if self.augment:
                audio = self._augment_audio(audio)
                
            return audio
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return np.zeros(self.max_samples)
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply data augmentation to audio."""
        # Add small random noise
        if np.random.random() < 0.3:  # 30% chance
            noise = np.random.normal(0, 0.001, audio.shape)
            audio = audio + noise
        
        # Random gain
        if np.random.random() < 0.3:  # 30% chance
            gain = np.random.uniform(0.9, 1.1)
            audio = audio * gain
            
        # Random time stretching
        if np.random.random() < 0.2:  # 20% chance
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            elif len(audio) < self.max_samples:
                audio = np.pad(audio, (0, max(0, self.max_samples - len(audio))), 'constant')
        
        return audio
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and process a single audio file."""
        file_path = self.files[idx]
        label = self.labels[idx]
        
        # Load and preprocess audio
        audio = self._load_audio(file_path)
        
        # Extract features
        features = self.feature_extractor.extract_features(audio)
        
        # Convert to tensor (using MFCCs as primary feature)
        features_tensor = torch.from_numpy(features['mfcc']).float()
        
        # Add channel dimension (1 for grayscale)
        features_tensor = features_tensor.unsqueeze(0)
        
        return features_tensor, label


def create_data_loaders(
    real_dir: str,
    fake_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    num_workers: int = 4,
    max_duration: float = 4.0,
    sample_rate: int = 16000
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing."""
    # Initialize feature extractor
    feature_extractor = AudioFeatureExtractor(sr=sample_rate)
    
    # Create full dataset
    full_dataset = AudioDataset(
        real_dir=real_dir,
        fake_dir=fake_dir,
        feature_extractor=feature_extractor,
        max_duration=max_duration,
        sample_rate=sample_rate,
        augment=True
    )
    
    # Split into train, val, test
    dataset_size = len(full_dataset)
    test_size = int(dataset_size * test_split)
    val_size = int((dataset_size - test_size) * val_split)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Created data loaders with {len(train_loader)} training, {len(val_loader)} validation, and {len(test_loader)} test batches")
    
    return train_loader, val_loader, test_loader


def train_model(
    model: AudioDeepfakeDetectorAPI,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 10,
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs"
) -> Dict[str, List[float]]:
    """Train the model with early stopping and checkpointing."""
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(log_dir, f"audio_detector_{timestamp}"))
    
    # Configure optimizer and learning rate scheduler
    model.configure_optimizers(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler_patience=patience // 2,
        scheduler_factor=0.5
    )
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_acc = model.train_epoch(train_loader, epoch, num_epochs)
        
        # Validate
        val_loss, val_acc = model.validate(val_loader)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        logger.info(
            f"Epoch {epoch+1}/{num_epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            
            # Save the best model
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            model.save_model(best_model_path)
            logger.info(f"New best model saved with val_acc: {val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        model.save_model(checkpoint_path)
        
        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping after {epoch+1} epochs without improvement")
            break
    
    # Close TensorBoard writer
    writer.close()
    
    return history


def evaluate_model(
    model: AudioDeepfakeDetectorAPI,
    test_loader: DataLoader
) -> Dict[str, float]:
    """Evaluate the model on the test set."""
    # Evaluate on test set
    test_loss, test_acc = model.validate(test_loader)
    
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Audio Deepfake Detection Model')
    
    # Data arguments
    parser.add_argument('--real-dir', type=str, required=True,
                        help='Directory containing real audio files')
    parser.add_argument('--fake-dir', type=str, required=True,
                        help='Directory containing fake audio files')
    
    # Model arguments
    parser.add_argument('--input-size', type=int, nargs=2, default=[64, 128],
                        help='Input size (height, width) for the model')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of output classes')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    
    # Data processing arguments
    parser.add_argument('--max-duration', type=float, default=4.0,
                        help='Maximum duration of audio in seconds')
    parser.add_argument('--sample-rate', type=int, default=16000,
                        help='Target sample rate for audio')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to save TensorBoard logs')
    
    # Device arguments
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training (cuda or cpu)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        batch_size=args.batch_size,
        max_duration=args.max_duration,
        sample_rate=args.sample_rate
    )
    
    # Initialize model
    model = AudioDeepfakeDetectorAPI(
        model_path=None,  # Start with random weights
        device=device,
        input_size=tuple(args.input_size),
        num_classes=args.num_classes
    )
    
    # Train the model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Load the best model
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_model(best_model_path)
        logger.info(f"Loaded best model from {best_model_path}")
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final Test Accuracy: {test_metrics['test_acc']:.2f}%")


if __name__ == "__main__":
    main()
