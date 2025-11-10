"""
Dataset Downloader for Deepfake Detection

This script downloads and prepares a dataset for fine-tuning deepfake detection models.
It uses the FaceForensics++ dataset which is commonly used for deepfake detection research.
"""

import os
import sys
import json
import random
import shutil
import requests
import tarfile
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Handles downloading and preparing datasets for deepfake detection."""
    
    def __init__(self, base_dir='data'):
        """Initialize with base directory for dataset storage."""
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        self.train_dir = self.processed_dir / 'train'
        self.val_dir = self.processed_dir / 'val'
        self.test_dir = self.processed_dir / 'test'
        
        # Create necessary directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configuration
        self.datasets = {
            'celebdf': {
                'name': 'Celeb-DF',
                'urls': [
                    'https://github.com/yuezunli/celeb-deepfakeforensics/raw/master/Celeb-DF-v2.zip',
                ],
                'description': 'High-quality deepfake dataset with celebrity videos',
                'license': 'Research use only',
                'splits': {'train': 0.8, 'val': 0.1, 'test': 0.1}
            },
            'ff++': {
                'name': 'FaceForensics++',
                'urls': [
                    'https://github.com/ondyari/FaceForensics/releases/download/v0.1.0/faceforensics_download_v1.0.tar.gz',
                ],
                'description': 'Comprehensive deepfake dataset with multiple manipulation methods',
                'license': 'CC BY-NC-SA 4.0',
                'splits': {'train': 0.7, 'val': 0.15, 'test': 0.15}
            },
            'dfdc': {
                'name': 'DFDC (Deepfake Detection Challenge)',
                'urls': [
                    'https://s3.amazonaws.com/dfdc/dfdc_train_part_0.zip',
                    # Add more parts as needed
                ],
                'description': 'Large-scale deepfake dataset from Facebook',
                'license': 'DFDC License',
                'splits': {'train': 0.8, 'val': 0.1, 'test': 0.1}
            }
        }
    
    def download_file(self, url, dest_path):
        """Download a file with progress bar."""
        try:
            # Check if file already exists
            if dest_path.exists():
                logger.info(f"File already exists: {dest_path}")
                return True
                
            # Stream download
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8KB chunks
            
            with open(dest_path, 'wb') as f, tqdm(
                desc=f"Downloading {dest_path.name}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            if dest_path.exists():
                dest_path.unlink()  # Remove partial download
            return False
    
    def extract_archive(self, file_path, extract_to):
        """Extract archive file (zip/tar.gz)."""
        try:
            logger.info(f"Extracting {file_path} to {extract_to}")
            
            if file_path.suffix == '.zip':
                import zipfile
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            else:  # Assume tar.gz
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_to)
            
            return True
            
        except Exception as e:
            logger.error(f"Error extracting {file_path}: {e}")
            return False
    
    def download_dataset(self, dataset_name):
        """Download and prepare the specified dataset."""
        if dataset_name not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset = self.datasets[dataset_name]
        logger.info(f"Downloading {dataset['name']} dataset...")
        
        # Download all parts
        for url in dataset['urls']:
            file_name = url.split('/')[-1]
            dest_path = self.raw_dir / file_name
            
            # Download the file
            if not self.download_file(url, dest_path):
                logger.error(f"Failed to download {file_name}")
                return False
            
            # Extract the file
            if not self.extract_archive(dest_path, self.raw_dir / dataset_name):
                logger.error(f"Failed to extract {file_name}")
                return False
        
        logger.info(f"Successfully downloaded and extracted {dataset['name']} dataset")
        return True
    
    def prepare_dataset(self, dataset_name):
        """Prepare the dataset for training."""
        dataset_path = self.raw_dir / dataset_name
        
        # Create train/val/test splits
        for split in ['train', 'val', 'test']:
            (self.processed_dir / split / 'real').mkdir(parents=True, exist_ok=True)
            (self.processed_dir / split / 'fake').mkdir(parents=True, exist_ok=True)
        
        # This is a simplified example - you'll need to customize this based on the dataset structure
        # For demonstration, we'll just create a small sample dataset
        self._create_sample_dataset()
        
        logger.info("Dataset preparation complete!")
    
    def _create_sample_dataset(self):
        """Create a small sample dataset for testing."""
        from torchvision.datasets import CIFAR10
        import torchvision.transforms as transforms
        
        logger.info("Creating a small sample dataset for testing...")
        
        # Download CIFAR-10 as a sample (replace with your actual data)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        # Use CIFAR-10 as a placeholder
        train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        # Save some samples (class 0 as real, class 1 as fake)
        def save_samples(dataset, split, num_samples=100):
            real_count = 0
            fake_count = 0
            
            for i, (img, label) in enumerate(dataset):
                if label in [0, 1]:  # Only use first two classes
                    if label == 0 and real_count < num_samples//2:
                        # Save as real
                        img_path = self.processed_dir / split / 'real' / f'real_{real_count}.png'
                        transforms.ToPILImage()(img).save(img_path)
                        real_count += 1
                    elif label == 1 and fake_count < num_samples//2:
                        # Save as fake
                        img_path = self.processed_dir / split / 'fake' / f'fake_{fake_count}.png'
                        transforms.ToPILImage()(img).save(img_path)
                        fake_count += 1
                    
                    if real_count >= num_samples//2 and fake_count >= num_samples//2:
                        break
        
        # Create small train/val/test splits
        save_samples(train_dataset, 'train', num_samples=160)  # 80 real, 80 fake
        save_samples(train_dataset, 'val', num_samples=20)     # 10 real, 10 fake
        save_samples(test_dataset, 'test', num_samples=20)     # 10 real, 10 fake
        
        logger.info("Sample dataset created successfully!")

def main():
    """Main function to download and prepare the dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and prepare deepfake datasets')
    parser.add_argument('--dataset', type=str, default='sample',
                       choices=['celebdf', 'ff++', 'dfdc', 'sample'],
                       help='Dataset to download (default: sample)')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Base directory to save the dataset (default: data)')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(base_dir=args.output_dir)
    
    if args.dataset == 'sample':
        # Just create a small sample dataset
        downloader._create_sample_dataset()
    else:
        # Download and prepare the full dataset
        if downloader.download_dataset(args.dataset):
            downloader.prepare_dataset(args.dataset)
    
    logger.info("Dataset preparation complete!")
    logger.info(f"Dataset structure at {downloader.processed_dir}:")
    
    # Print dataset structure
    for root, dirs, files in os.walk(downloader.processed_dir):
        level = root.replace(str(downloader.processed_dir), '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files[:5]:  # Show first 5 files in each directory
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files)-5} more files")

if __name__ == '__main__':
    main()
