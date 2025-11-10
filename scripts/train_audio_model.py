"""
Audio Model Training Script
Trains and evaluates the audio deepfake detection model.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from server.python.models.audio_enhanced import (
    AudioPreprocessor, 
    AudioDeepfakeDetector,
    train_audio_model
)

class AudioDataset(Dataset):
    """Audio dataset for deepfake detection."""
    
    def __init__(self, data_dir: str, split: str = 'train', max_samples: int = None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing 'real' and 'fake' subdirectories
            split: 'train', 'val', or 'test'
            max_samples: Maximum number of samples to load (for testing)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.preprocessor = AudioPreprocessor()
        
        # Load real and fake audio paths
        self.real_files = list((self.data_dir / 'real').glob('*.wav'))
        self.fake_files = list((self.data_dir / 'fake').glob('*.wav'))
        
        # Limit samples for testing
        if max_samples:
            self.real_files = self.real_files[:max_samples//2]
            self.fake_files = self.fake_files[:max_samples//2]
        
        # Create labels (0 for real, 1 for fake)
        self.files = self.real_files + self.fake_files
        self.labels = [0] * len(self.real_files) + [1] * len(self.fake_files)
        
        # Shuffle the dataset
        indices = np.random.permutation(len(self.files))
        self.files = [self.files[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        # Split into train/val/test (80/10/10)
        train_size = int(0.8 * len(self.files))
        val_size = int(0.1 * len(self.files))
        
        if split == 'train':
            self.files = self.files[:train_size]
            self.labels = self.labels[:train_size]
        elif split == 'val':
            self.files = self.files[train_size:train_size + val_size]
            self.labels = self.labels[train_size:train_size + val_size]
        else:  # test
            self.files = self.files[train_size + val_size:]
            self.labels = self.labels[train_size + val_size:]
        
        print(f'Loaded {len(self.files)} {split} samples ({len(self.real_files)} real, {len(self.fake_files)} fake)')
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample.
        
        Returns:
            tuple: (log_mel_spectrogram, label)
        """
        try:
            # Load and preprocess audio
            waveform, _ = self.preprocessor.load_audio(str(self.files[idx]))
            features = self.preprocessor.extract_features(waveform.unsqueeze(0))
            
            # Get log mel spectrogram and add channel dimension
            log_mel = features['log_mel'].unsqueeze(0)  # (1, n_mels, time)
            
            return log_mel, self.labels[idx]
            
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            # Return a random tensor of the same shape
            return torch.zeros(1, 64, 100), 0  # Default shape, adjust if needed

def get_data_loaders(
    data_dir: str, 
    batch_size: int = 16,
    num_workers: int = 4,
    max_samples: int = None
) -> tuple:
    """
    Get data loaders for training, validation, and testing.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = AudioDataset(data_dir, 'train', max_samples)
    val_dataset = AudioDataset(data_dir, 'val', max_samples)
    test_dataset = AudioDataset(data_dir, 'test', max_samples)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def main():
    parser = argparse.ArgumentParser(description='Train Audio Deepfake Detection Model')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing real/ and fake/ subdirectories')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of worker processes for data loading')
    parser.add_argument('--output-dir', type=str, default='saved_models',
                       help='Directory to save the trained model')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to use (for testing)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get data loaders
    print('Loading datasets...')
    train_loader, val_loader, test_loader = get_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples
    )
    
    # Initialize model
    model = AudioDeepfakeDetector().to(device)
    
    # Model save path with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = os.path.join(args.output_dir, f'audio_deepfake_detector_{timestamp}.pth')
    
    # Train the model
    print('Starting training...')
    history = train_audio_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        model_save_path=model_save_path
    )
    
    # Save training history
    history_path = os.path.join(args.output_dir, f'training_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'Training complete! Model saved to {model_save_path}')
    print(f'Training history saved to {history_path}')
    
    # Evaluate on test set
    print('\nEvaluating on test set...')
    test_loss, test_acc = validate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

if __name__ == '__main__':
    main()
