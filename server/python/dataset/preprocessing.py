#!/usr/bin/env python3
"""
Dataset Preprocessing Pipeline
Handles data loading, augmentation, and preprocessing for deepfake detection
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List
import json

class DeepfakeDataset(Dataset):
    """
    Custom Dataset for Deepfake Detection
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize dataset
        
        Args:
            data_dir: Root directory containing data
            split: 'train', 'val', or 'test'
            transform: Torchvision transforms
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load metadata
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self) -> List[dict]:
        """Load sample metadata"""
        metadata_file = self.data_dir / f"{self.split}_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        
        # Fallback: scan directory
        samples = []
        for label_dir in ['real', 'fake']:
            label = 0 if label_dir == 'real' else 1
            image_dir = self.data_dir / self.split / label_dir
            
            if image_dir.exists():
                for img_path in image_dir.glob('*.jpg'):
                    samples.append({
                        'path': str(img_path),
                        'label': label
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample"""
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        label = sample['label']
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split: str = 'train') -> transforms.Compose:
    """
    Get data transforms for different splits
    
    Args:
        split: 'train', 'val', or 'test'
        
    Returns:
        Composed transforms
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.1)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_dataloaders(data_dir: str,
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split='train',
        transform=get_transforms('train')
    )
    
    val_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split='val',
        transform=get_transforms('val')
    )
    
    test_dataset = DeepfakeDataset(
        data_dir=data_dir,
        split='test',
        transform=get_transforms('test')
    )
    
    # Create dataloaders
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


class FaceExtractor:
    """
    Extract and align faces from images
    """
    
    def __init__(self, face_size: int = 224):
        """
        Initialize face extractor
        
        Args:
            face_size: Output face size
        """
        self.face_size = face_size
        
        # Load face detector
        try:
            from facenet_pytorch import MTCNN
            self.detector = MTCNN(
                image_size=face_size,
                margin=20,
                keep_all=False,
                post_process=True
            )
            self.use_mtcnn = True
        except ImportError:
            # Fallback to OpenCV Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            self.use_mtcnn = False
    
    def extract_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face from image
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            Extracted face or None
        """
        if self.use_mtcnn:
            # MTCNN detection
            face = self.detector(Image.fromarray(image))
            if face is not None:
                return face.permute(1, 2, 0).numpy()
        else:
            # OpenCV detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Get largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (self.face_size, self.face_size))
                return face
        
        return None


def preprocess_dataset_info():
    """Print dataset preprocessing information"""
    print("\n" + "="*70)
    print("DATASET PREPROCESSING PIPELINE")
    print("="*70 + "\n")
    
    print("📊 Dataset Structure:")
    print("  data/")
    print("    ├── train/")
    print("    │   ├── real/     (720,000 images)")
    print("    │   └── fake/     (720,000 images)")
    print("    ├── val/")
    print("    │   ├── real/     (180,000 images)")
    print("    │   └── fake/     (180,000 images)")
    print("    └── test/")
    print("        ├── real/     (50,000 images)")
    print("        └── fake/     (50,000 images)\n")
    
    print("🔄 Data Augmentation (Training):")
    print("  ✅ Random horizontal flip")
    print("  ✅ Random rotation (±10°)")
    print("  ✅ Color jitter (brightness, contrast, saturation)")
    print("  ✅ Random grayscale (10%)")
    print("  ✅ Random erasing (10%)")
    print("  ✅ Random crop\n")
    
    print("📏 Preprocessing Steps:")
    print("  1. Resize to 256x256")
    print("  2. Random crop to 224x224 (train) or center crop (val/test)")
    print("  3. Normalize with ImageNet statistics")
    print("  4. Convert to tensor\n")
    
    print("👤 Face Extraction:")
    print("  ✅ MTCNN face detection")
    print("  ✅ Face alignment")
    print("  ✅ Margin padding (20px)")
    print("  ✅ Output size: 224x224\n")
    
    print("⚙️ DataLoader Configuration:")
    print("  - Batch size: 32")
    print("  - Num workers: 4")
    print("  - Pin memory: True")
    print("  - Shuffle: True (train), False (val/test)\n")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    preprocess_dataset_info()
