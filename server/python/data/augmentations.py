"""
Advanced data augmentation techniques for deepfake detection.
Implements CutMix, MixUp, Random Erasing, and Test-Time Augmentation (TTA).
"""
import random
from typing import Tuple, Dict, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as TF


def rand_bbox(size: Tuple[int, int, int], lam: float) -> Tuple[int, int, int, int, int, int]:
    """Generate random bounding box for CutMix.
    
    Args:
        size: Tuple of (C, H, W) for the input tensor
        lam: Lambda value for CutMix (controls the size of the cutout)
    """
    W = size[2]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutMix:
    """CutMix augmentation.
    
    Reference: https://arxiv.org/abs/1905.04899
    """
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Args:
            alpha: Controls the beta distribution for sampling the mix ratio
            prob: Probability of applying CutMix
        """
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, float, int, int]:
        """
        Args:
            batch: Tuple of (images, labels) where images is (B, C, H, W) tensor
            
        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lam)
        """
        images, labels = batch
        batch_size = images.size(0)
        
        if random.random() > self.prob or batch_size < 2:
            return images, labels, torch.ones(batch_size), torch.zeros(batch_size, dtype=torch.long), -1
        
        # Generate random permutation of batch indices
        indices = torch.randperm(batch_size, device=images.device)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Generate random bounding box
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = shuffled_images[:, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        return mixed_images, labels, shuffled_labels, lam, indices


class MixUp:
    """MixUp augmentation.
    
    Reference: https://arxiv.org/abs/1710.09412
    """
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        """
        Args:
            alpha: Controls the beta distribution for sampling the mix ratio
            prob: Probability of applying MixUp
        """
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor, float, int]:
        """
        Args:
            batch: Tuple of (images, labels) where images is (B, C, H, W) tensor
            
        Returns:
            Tuple of (mixed_images, labels_a, labels_b, lam, indices)
        """
        images, labels = batch
        batch_size = images.size(0)
        
        if random.random() > self.prob or batch_size < 2:
            return images, labels, torch.zeros(batch_size, dtype=torch.long), 1.0, -1
        
        # Generate random permutation of batch indices
        indices = torch.randperm(batch_size, device=images.device)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Apply MixUp
        mixed_images = lam * images + (1 - lam) * shuffled_images
        
        return mixed_images, labels, shuffled_labels, lam, indices


class RandomErasing:
    """Random Erasing augmentation.
    
    Reference: https://arxiv.org/abs/1708.04896
    """
    def __init__(self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33), 
                 ratio: Tuple[float, float] = (0.3, 3.3), value: float = 0, inplace: bool = False):
        """
        Args:
            p: Probability of applying random erasing
            scale: Range of proportion of erased area
            ratio: Range of aspect ratio of erased area
            value: Erasing value (can be a single value or a tuple for RGB)
            inplace: Whether to do this operation in-place
        """
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace
        
    def __call__(self, img: Tensor) -> Tensor:
        """
        Args:
            img: Tensor image of size (C, H, W) to be erased.
            
        Returns:
            Erased Tensor image.
        """
        if random.random() > self.p:
            return img
            
        if not self.inplace:
            img = img.clone()
            
        c, h, w = img.size()
        
        # Sample scale and ratio
        area = h * w
        target_area = random.uniform(*self.scale) * area
        aspect_ratio = random.uniform(*self.ratio)
        
        # Calculate erasing size
        w_erase = int(round(np.sqrt(target_area * aspect_ratio)))
        h_erase = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if w_erase > w or h_erase > h:
            return img
            
        # Sample position
        top = random.randint(0, h - h_erase)
        left = random.randint(0, w - w_erase)
        
        # Apply erasing
        if isinstance(self.value, (int, float)):
            img[:, top:top + h_erase, left:left + w_erase] = self.value
        else:
            img[:, top:top + h_erase, left:left + w_erase] = torch.tensor(
                self.value, dtype=img.dtype, device=img.device).view(-1, 1, 1)
                
        return img


class TestTimeAugmentation:
    """Test-Time Augmentation for model inference."""
    
    def __init__(self, model: torch.nn.Module, n_augmentations: int = 5, 
                 softmax: bool = True, device: str = 'cuda'):
        """
        Args:
            model: The model to use for inference
            n_augmentations: Number of augmentations to apply
            softmax: Whether to apply softmax to model outputs
            device: Device to run inference on
        """
        self.model = model
        self.n_augmentations = n_augmentations
        self.softmax = softmax
        self.device = device
        
        # Define the augmentations to apply
        self.augmentations = [
            # Original
            lambda x: x,
            # Horizontal flip
            lambda x: TF.hflip(x),
            # Rotation (small angles)
            lambda x: TF.rotate(x, 5),
            lambda x: TF.rotate(x, -5),
            # Color jitter
            lambda x: TF.adjust_brightness(x, 1.2),
            lambda x: TF.adjust_contrast(x, 1.2),
        ]
        
    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply test-time augmentation and average predictions.
        
        Args:
            x: Input tensor of shape (1, C, H, W)
            
        Returns:
            Averaged predictions
        """
        self.model.eval()
        
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # Get predictions for each augmentation
        with torch.no_grad():
            # Get base prediction
            outputs = self.model(x)
            if self.softmax:
                outputs = F.softmax(outputs, dim=1)
            
            # Initialize sum of predictions
            pred_sum = outputs.clone()
            
            # Apply augmentations and accumulate predictions
            for _ in range(self.n_augmentations - 1):
                # Select a random augmentation
                aug_fn = random.choice(self.augmentations[1:])  # Skip original
                x_aug = aug_fn(x)
                
                # Get prediction for augmented input
                output = self.model(x_aug)
                if self.softmax:
                    output = F.softmax(output, dim=1)
                
                # If augmentation was a flip, we need to flip the predictions back
                if aug_fn == self.augmentations[1]:  # Horizontal flip
                    output = output.flip(1)  # Flip class predictions
                
                pred_sum += output
        
        # Average predictions
        return pred_sum / self.n_augmentations
