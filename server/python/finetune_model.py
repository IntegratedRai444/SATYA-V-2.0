"""
Model Fine-tuning Script for SatyaAI

This script handles fine-tuning deepfake detection models on custom datasets.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from PIL import Image
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import augmentations
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.augmentations import CutMix, MixUp, RandomErasing, TestTimeAugmentation

class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection with advanced augmentations."""
    
    def __init__(self, data_dir, transform=None, is_train=True, use_augmentations=True):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied.
            is_train (bool): Whether this is training data (affects augmentations).
            use_augmentations (bool): Whether to apply advanced augmentations.
        """
        self.data_dir = Path(data_dir)
        self.is_train = is_train
        self.use_augmentations = use_augmentations
        
        # Get all image files with support for various extensions
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        self.real_imgs = []
        self.fake_imgs = []
        
        real_dir = self.data_dir / 'real'
        fake_dir = self.data_dir / 'fake'
        
        if real_dir.exists():
            for ext in image_extensions:
                self.real_imgs.extend(real_dir.glob(ext))
                self.real_imgs.extend(real_dir.glob(ext.upper()))
                
        if fake_dir.exists():
            for ext in image_extensions:
                self.fake_imgs.extend(fake_dir.glob(ext))
                self.fake_imgs.extend(fake_dir.glob(ext.upper()))
        
        # Create labels (0 for real, 1 for fake)
        self.samples = ([(str(p), 0) for p in self.real_imgs] + 
                       [(str(p), 1) for p in self.fake_imgs])
        
        # Set default transforms if none provided
        self.transform = transform or self._get_default_transforms()
        
        # Initialize advanced augmentations
        self.cutmix = CutMix(alpha=1.0, prob=0.5) if is_train and use_augmentations else None
        self.mixup = MixUp(alpha=0.2, prob=0.5) if is_train and use_augmentations else None
    
    def _get_default_transforms(self):
        """Get default transformations based on training/validation mode."""
        if self.is_train:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),
                RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
            ])
        else:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Try to load with PIL first
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                # Fallback to OpenCV if PIL fails
                import cv2
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            # Apply transformations
            if self.transform:
                image = self.transform(image)
                
            # Convert label to tensor
            label = torch.tensor(label, dtype=torch.long)
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            # Return a random image if loading fails
            dummy_img = torch.randn(3, 256, 256)
            return dummy_img, torch.tensor(0, dtype=torch.long)

class DeepfakeModel(nn.Module):
    """Deepfake detection model based on EfficientNet."""
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # Load pre-trained EfficientNet
        self.base_model = models.efficientnet_b7(pretrained=pretrained)
        
        # Replace the classifier
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=num_features, out_features=num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

def train_epoch(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    device='cuda',
    cutmix: Optional[CutMix] = None,
    mixup: Optional[MixUp] = None,
    grad_clip: float = 1.0
):
    """Train model for one epoch with optional CutMix and MixUp."""
    model.train()
    running_loss = 0.0
    running_ce_loss = 0.0
    running_reg_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply CutMix if available
        if cutmix and random.random() < 0.5:
            inputs, labels_a, labels_b, lam, _ = cutmix((inputs, labels))
            outputs = model(inputs)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        # Apply MixUp if available
        elif mixup and random.random() < 0.5:
            inputs, labels_a, labels_b, lam, _ = mixup((inputs, labels))
            outputs = model(inputs)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            # Standard forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
        # For CutMix/MixUp, we need to adjust the accuracy calculation
        if cutmix and random.random() < 0.5:
            correct += (lam * predicted.eq(labels_a.data).cpu().sum().float() +
                       (1 - lam) * predicted.eq(labels_b.data).cpu().sum().float())
        elif mixup and random.random() < 0.5:
            correct += (lam * predicted.eq(labels_a.data).cpu().sum().float() +
                       (1 - lam) * predicted.eq(labels_b.data).cpu().sum().float())
        else:
            correct += predicted.eq(labels.data).cpu().sum()
        
        # Update running loss
        running_loss += loss.item() * inputs.size(0)
        
        # Log progress
        if batch_idx % 50 == 0:
            logger.info(f'Batch {batch_idx}/{len(train_loader)}: Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def train_model(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler=None,
    num_epochs=50,
    device='cuda',
    early_stopping_patience=10,
    checkpoint_dir='checkpoints',
    use_amp=True,
    grad_clip=1.0,
    use_cutmix=True,
    use_mixup=True
):
    """Train the model with advanced features."""
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize variables
    best_val_acc = 0.0
    epochs_without_improvement = 0
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # Initialize advanced augmentations
    cutmix = CutMix(alpha=1.0, prob=0.5) if use_cutmix else None
    mixup = MixUp(alpha=0.2, prob=0.5) if use_mixup else None
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f'\nEpoch {epoch+1}/{num_epochs}')
        logger.info('-' * 30)
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            cutmix=cutmix, mixup=mixup, grad_clip=grad_clip
        )
        
        # Validate
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
        # Step the learning rate scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None:
            scheduler.step()
        
        # Print statistics
        logger.info(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        
        # Check for improvement
        if val_acc > best_val_acc:
            logger.info('Validation accuracy improved. Saving model...')
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, best_model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            logger.info(f'No improvement for {epochs_without_improvement} epochs')
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                logger.info(f'Early stopping after {epoch+1} epochs')
                break
    
    # Load best model weights
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded best model with val acc: {checkpoint["val_acc"]:.2f}%')
    
    return model

def validate_model(
    model, 
    val_loader, 
    criterion, 
    device='cuda',
    use_tta: bool = True,
    tta_steps: int = 5
):
    """Validate the model with optional test-time augmentation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize TTA if enabled
    tta = TestTimeAugmentation(model, n_augmentations=tta_steps) if use_tta else None
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass with or without TTA
            if tta and use_tta:
                outputs = tta(inputs)
            else:
                outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate metrics
    val_loss = running_loss / len(val_loader.dataset)
    val_acc = 100.0 * correct / total
    
    return val_loss, val_acc

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train deepfake detection model')
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, default='data/train',
                       help='Directory containing training data')
    parser.add_argument('--val_dir', type=str, default='data/val',
                       help='Directory containing validation data')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='efficientnet_b7',
                       choices=['efficientnet_b7', 'resnet50', 'resnet101'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    
    # Augmentation parameters
    parser.add_argument('--use_cutmix', action='store_true',
                       help='Use CutMix augmentation')
    parser.add_argument('--use_mixup', action='store_true',
                       help='Use MixUp augmentation')
    parser.add_argument('--use_tta', action='store_true',
                       help='Use test-time augmentation during validation')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

def seed_everything(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    # Set device
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Configuration
    config = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs,
        'train_dir': args.train_dir,
        'val_dir': args.val_dir,
        'checkpoint_dir': args.checkpoint_dir,
        'model_name': args.model_name,
        'pretrained': args.pretrained,
        'use_cutmix': args.use_cutmix,
        'use_mixup': args.use_mixup,
        'use_tta': args.use_tta,
        'grad_clip': args.grad_clip,
        'device': device
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        config['train_dir'],
        is_train=True,
        use_augmentations=True
    )
    
    val_dataset = DeepfakeDataset(
        config['val_dir'],
        is_train=False,
        use_augmentations=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Required for CutMix/MixUp
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize model
    model = DeepfakeModel(num_classes=2, pretrained=config['pretrained'])
    model = model.to(device)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        amsgrad=True
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Train the model
    logger.info('Starting training...')
    logger.info(f'Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples')
    logger.info(f'Using model: {config["model_name"]}')
    logger.info(f'Using device: {device}')
    logger.info(f'Batch size: {config["batch_size"]}')
    logger.info(f'Learning rate: {config["learning_rate"]}')
    logger.info(f'Using CutMix: {config["use_cutmix"]}')
    logger.info(f'Using MixUp: {config["use_mixup"]}')
    logger.info(f'Using TTA: {config["use_tta"]}')
    
    try:
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config['num_epochs'],
            device=device,
            early_stopping_patience=15,
            checkpoint_dir=config['checkpoint_dir'],
            use_amp=True,
            grad_clip=config['grad_clip'],
            use_cutmix=config['use_cutmix'],
            use_mixup=config['use_mixup']
        )
        
        # Save the final model
        final_model_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, final_model_path)
        logger.info(f'Final model saved to {final_model_path}')
        
    except KeyboardInterrupt:
        logger.info('Training interrupted. Saving model...')
        final_model_path = os.path.join(config['checkpoint_dir'], 'interrupted_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, final_model_path)
        logger.info(f'Interrupted model saved to {final_model_path}')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f'Error in main(): {str(e)}', exc_info=True)
        raise
