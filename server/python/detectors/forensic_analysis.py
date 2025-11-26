"""
Forensic Analysis Module for Image Deepfake Detection
Implements ELA, PRNU, and additional forensic techniques
"""

import numpy as np
import cv2
from PIL import Image
import io
import logging
from typing import Dict, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def analyze_ela(image_path: str, quality: int = 90) -> Dict:
    """
    Error Level Analysis (ELA) - Detects compression artifacts
    
    Args:
        image_path: Path to image
        quality: JPEG quality for resave (default 90%)
        
    Returns:
        Dictionary with ELA score and artifact map
    """
    try:
        # Load original image
        original = Image.open(image_path).convert('RGB')
        
        # Resave at specified quality
        buffer = io.BytesIO()
        original.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)
        
        # Convert to numpy
        orig_np = np.array(original, dtype=np.float32)
        resaved_np = np.array(resaved, dtype=np.float32)
        
        # Calculate difference
        ela_diff = np.abs(orig_np - resaved_np)
        
        # Enhance for visualization
        ela_enhanced = np.clip(ela_diff * 10, 0, 255).astype(np.uint8)
        
        # Calculate score based on difference magnitude
        mean_diff = np.mean(ela_diff)
        std_diff = np.std(ela_diff)
        max_diff = np.max(ela_diff)
        
        # Scoring logic:
        # - Low difference (< 5) = likely authentic (high score)
        # - High difference (> 20) = likely manipulated (low score)
        # - Moderate difference = uncertain
        if mean_diff < 5:
            score = 0.9
        elif mean_diff > 20:
            score = 0.3
        else:
            score = 0.7 - (mean_diff - 5) / 30  # Linear interpolation
        
        return {
            'score': float(np.clip(score, 0, 1)),
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'max_difference': float(max_diff),
            'method': 'error_level_analysis',
            'suspicious_regions': int(np.sum(ela_diff > 15))  # Count high-diff pixels
        }
        
    except Exception as e:
        logger.error(f"ELA analysis failed: {e}")
        return {'score': 0.5, 'error': str(e)}


def analyze_prnu(image_np: np.ndarray) -> Dict:
    """
    PRNU (Photo Response Non-Uniformity) Analysis
    Detects camera sensor noise fingerprint inconsistencies
    
    Args:
        image_np: Image as numpy array
        
    Returns:
        Dictionary with PRNU analysis results
    """
    try:
        # Convert to grayscale
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image_np.astype(np.float32)
        
        # Apply Wiener filter to extract noise residual
        # Simple approximation: use median filter as denoised version
        denoised = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
        noise_residual = gray - denoised
        
        # Analyze noise characteristics
        noise_std = np.std(noise_residual)
        noise_mean = np.mean(np.abs(noise_residual))
        
        # Divide image into blocks and analyze noise consistency
        block_size = 64
        h, w = gray.shape
        block_stds = []
        
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block_noise = noise_residual[i:i+block_size, j:j+block_size]
                block_stds.append(np.std(block_noise))
        
        # Consistent noise = authentic, inconsistent = manipulated
        noise_variance = np.var(block_stds)
        
        # Scoring: lower variance = more consistent = more authentic
        if noise_variance < 0.5:
            score = 0.9
        elif noise_variance > 2.0:
            score = 0.4
        else:
            score = 0.9 - (noise_variance - 0.5) * 0.33
        
        return {
            'score': float(np.clip(score, 0, 1)),
            'noise_std': float(noise_std),
            'noise_mean': float(noise_mean),
            'noise_variance': float(noise_variance),
            'method': 'prnu_analysis',
            'consistent': noise_variance < 1.0
        }
        
    except Exception as e:
        logger.error(f"PRNU analysis failed: {e}")
        return {'score': 0.5, 'error': str(e)}


class MesoNet(nn.Module):
    """
    MesoNet: Lightweight CNN for deepfake detection
    Based on "MesoNet: a Compact Facial Video Forgery Detection Network"
    """
    
    def __init__(self, num_classes=2):
        super(MesoNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(8, 8, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(8)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(8, 16, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(4, 4)
        
        # Fully connected layers
        # Assuming input size 256x256, after pooling: 256/2/2/2/4 = 8
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # Conv blocks
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        
        return x


def create_mesonet(pretrained=False):
    """
    Create MesoNet model
    
    Args:
        pretrained: Whether to load pretrained weights (if available)
        
    Returns:
        MesoNet model
    """
    model = MesoNet(num_classes=2)
    
    if pretrained:
        try:
            from pathlib import Path
            import torch
            
            # Try to load from models directory
            model_path = Path(__file__).parent.parent / 'models' / 'mesonet_pretrained.pth'
            
            if model_path.exists():
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info(f"✅ Loaded pretrained MesoNet weights from {model_path}")
            else:
                logger.warning(f"⚠️ Pretrained MesoNet weights not found at {model_path}, using random initialization")
        except Exception as e:
            logger.warning(f"⚠️ Failed to load pretrained MesoNet weights: {e}, using random initialization")
    
    return model
