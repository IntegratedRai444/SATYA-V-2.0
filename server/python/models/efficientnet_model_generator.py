"""
Enhanced EfficientNet-B7 Model Generator
Creates and configures proper EfficientNet-B7 models for deepfake detection
Can generate models with different configurations and training states
"""

import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class EfficientNetModelGenerator:
    """
    Enhanced generator for EfficientNet-B7 deepfake detection models
    """
    
    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or Path("e:/SATYA-V-2.0/models")
        self.model_configs = self._load_model_configs()
    
    def _load_model_configs(self) -> Dict[str, Any]:
        """Load model configurations"""
        return {
            'imagenet_pretrained': {
                'description': 'EfficientNet-B7 with ImageNet pretrained weights',
                'classifier_layers': [512, 2],
                'dropout_rates': [0.4, 0.3],
                'freeze_backbone': False,
                'use_batchnorm': True
            },
            'deepfake_optimized': {
                'description': 'EfficientNet-B7 optimized for deepfake detection',
                'classifier_layers': [1024, 512, 256, 2],
                'dropout_rates': [0.5, 0.4, 0.3, 0.2],
                'freeze_backbone': True,
                'use_batchnorm': True
            },
            'lightweight': {
                'description': 'Lighter EfficientNet-B7 for faster inference',
                'classifier_layers': [256, 2],
                'dropout_rates': [0.3, 0.2],
                'freeze_backbone': True,
                'use_batchnorm': False
            }
        }
    
    def create_model(self, config_name: str = 'imagenet_pretrained') -> nn.Module:
        """
        Create EfficientNet-B7 model with specified configuration
        
        Args:
            config_name: Configuration to use
            
        Returns:
            Configured model
        """
        if config_name not in self.model_configs:
            raise ValueError(f"Unknown config: {config_name}")
        
        config = self.model_configs[config_name]
        
        # Load base model
        model = models.efficientnet_b7(weights='IMAGENET1K_V1')
        
        # Freeze backbone if specified
        if config['freeze_backbone']:
            for param in model.features.parameters():
                param.requires_grad = False
        
        # Build custom classifier
        classifier_layers = []
        in_features = model.classifier[1].in_features
        
        for i, (out_features, dropout_rate) in enumerate(zip(
            config['classifier_layers'][:-1], 
            config['dropout_rates'][:-1]
        )):
            classifier_layers.extend([
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, out_features)
            ])
            
            if config['use_batchnorm']:
                classifier_layers.append(nn.BatchNorm1d(out_features))
            
            classifier_layers.append(nn.ReLU(inplace=True))
            in_features = out_features
        
        # Final layer (no activation, no dropout)
        classifier_layers.append(
            nn.Linear(in_features, config['classifier_layers'][-1])
        )
        
        # Replace classifier
        model.classifier[1] = nn.Sequential(*classifier_layers)
        
        logger.info(f"Created EfficientNet-B7 with config: {config_name}")
        return model
    
    def save_model(self, model: nn.Module, config_name: str, 
                   save_path: Optional[Path] = None) -> Path:
        """
        Save model with metadata
        
        Args:
            model: Model to save
            config_name: Configuration name used
            save_path: Custom save path
            
        Returns:
            Path where model was saved
        """
        if save_path is None:
            save_path = self.models_dir / "dfdc_efficientnet_b7" / "model.pth"
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        metadata = {
            'model_config': self.model_configs[config_name],
            'architecture': 'efficientnet_b7',
            'config_name': config_name,
            'num_classes': 2,
            'input_size': 224,
            'created_at': str(Path().absolute()),
            'training_info': {
                'pretrained': 'imagenet1k',
                'fine_tuned': False,
                'purpose': 'deepfake_detection',
                'version': '1.0.0'
            },
            'model_state_dict': model.state_dict()
        }
        
        # Save with metadata
        torch.save(metadata, save_path)
        logger.info(f"Model saved to {save_path}")
        
        # Save config info
        config_path = save_path.parent / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return save_path
    
    def generate_and_save(self, config_name: str = 'imagenet_pretrained') -> Path:
        """
        Generate and save model in one step
        
        Args:
            config_name: Configuration to use
            
        Returns:
            Path where model was saved
        """
        model = self.create_model(config_name)
        return self.save_model(model, config_name)
    
    def test_model(self, model_path: Path) -> bool:
        """
        Test if saved model loads and works correctly
        
        Args:
            model_path: Path to saved model
            
        Returns:
            True if model works correctly
        """
        try:
            # Load model
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Recreate model architecture
            config_name = checkpoint.get('config_name', 'imagenet_pretrained')
            model = self.create_model(config_name)
            
            # Load weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Test inference
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            
            # Verify output shape
            if output.shape == (1, 2):
                logger.info("✅ Model test passed")
                return True
            else:
                logger.error(f"❌ Wrong output shape: {output.shape}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Model test failed: {e}")
            return False

def main():
    """Main function to generate models"""
    generator = EfficientNetModelGenerator()
    
    print("Available configurations:")
    for name, config in generator.model_configs.items():
        print(f"  - {name}: {config['description']}")
    
    # Generate default model
    print("\nGenerating EfficientNet-B7 model...")
    model_path = generator.generate_and_save('imagenet_pretrained')
    
    # Test the model
    print("Testing generated model...")
    if generator.test_model(model_path):
        print("✅ Model generation and test successful!")
    else:
        print("❌ Model generation failed!")

if __name__ == "__main__":
    main()
