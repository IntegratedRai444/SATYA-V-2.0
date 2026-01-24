"""
Video Model Weights Generator for SatyaAI
Creates properly initialized video deepfake detection model weights.
"""

import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.video_model import VideoDeepfakeDetector
from models.temporal_models import TemporalConvNet, TemporalLSTM

logger = logging.getLogger(__name__)

def generate_video_model_weights() -> Dict[str, Any]:
    """Generate and save all video model weights."""
    
    models_dir = Path(__file__).resolve().parents[3] / "models"
    video_dir = models_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'success': True,
        'generated_models': [],
        'errors': []
    }
    
    try:
        # 1. Generate main 3D CNN video model weights
        logger.info("🎥 Generating 3D CNN video model weights...")
        video_model = VideoDeepfakeDetector(
            in_channels=3,
            num_classes=2,
            dropout_rate=0.3,
            use_attention=True
        )
        
        # Initialize with proper weights
        video_model._initialize_weights()
        
        # Save main video model
        main_model_path = video_dir / "model.pth"
        torch.save(video_model.state_dict(), main_model_path)
        results['generated_models'].append(str(main_model_path))
        logger.info(f"✅ Saved main video model: {main_model_path}")
        
        # Save alternative name for video_model.py compatibility
        alt_model_path = models_dir / "video_deepfake_detector.pth"
        torch.save(video_model.state_dict(), alt_model_path)
        results['generated_models'].append(str(alt_model_path))
        logger.info(f"✅ Saved alternative video model: {alt_model_path}")
        
    except Exception as e:
        error_msg = f"Failed to generate 3D CNN weights: {e}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
        results['success'] = False
    
    try:
        # 2. Generate Temporal LSTM weights
        logger.info("🧠 Generating Temporal LSTM weights...")
        temporal_lstm = TemporalLSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            num_classes=2,
            dropout=0.3
        )
        
        # Initialize LSTM weights
        for name, param in temporal_lstm.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        
        # Save Temporal LSTM
        lstm_path = video_dir / "temporal_lstm.pth"
        torch.save(temporal_lstm.state_dict(), lstm_path)
        results['generated_models'].append(str(lstm_path))
        logger.info(f"✅ Saved Temporal LSTM: {lstm_path}")
        
    except Exception as e:
        error_msg = f"Failed to generate Temporal LSTM weights: {e}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
        # Don't fail the entire operation for LSTM
    
    try:
        # 3. Generate Temporal ConvNet weights
        logger.info("🌊 Generating Temporal ConvNet weights...")
        temporal_convnet = TemporalConvNet(in_channels=3, num_classes=2)
        
        # Initialize ConvNet weights
        temporal_convnet._initialize_weights()
        
        # Save Temporal ConvNet
        convnet_path = video_dir / "temporal_3dcnn.pth"
        torch.save(temporal_convnet.state_dict(), convnet_path)
        results['generated_models'].append(str(convnet_path))
        logger.info(f"✅ Saved Temporal ConvNet: {convnet_path}")
        
    except Exception as e:
        error_msg = f"Failed to generate Temporal ConvNet weights: {e}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
        # Don't fail the entire operation for ConvNet
    
    return results

def validate_generated_weights() -> bool:
    """Validate that generated weights can be loaded correctly."""
    
    try:
        models_dir = Path(__file__).resolve().parents[3] / "models"
        video_dir = models_dir / "video"
        
        # Test main video model
        main_model_path = video_dir / "model.pth"
        if main_model_path.exists():
            model = VideoDeepfakeDetector()
            state_dict = torch.load(main_model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info("✅ Main video model weights validated")
        
        # Test alternative path
        alt_model_path = models_dir / "video_deepfake_detector.pth"
        if alt_model_path.exists():
            model = VideoDeepfakeDetector()
            state_dict = torch.load(alt_model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info("✅ Alternative video model weights validated")
        
        # Test Temporal LSTM
        lstm_path = video_dir / "temporal_lstm.pth"
        if lstm_path.exists():
            lstm_model = TemporalLSTM()
            state_dict = torch.load(lstm_path, map_location='cpu')
            lstm_model.load_state_dict(state_dict)
            logger.info("✅ Temporal LSTM weights validated")
        
        # Test Temporal ConvNet
        convnet_path = video_dir / "temporal_3dcnn.pth"
        if convnet_path.exists():
            convnet_model = TemporalConvNet()
            state_dict = torch.load(convnet_path, map_location='cpu')
            convnet_model.load_state_dict(state_dict)
            logger.info("✅ Temporal ConvNet weights validated")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Weight validation failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🚀 Starting Video Model Weights Generation...")
    print("=" * 60)
    
    # Generate weights
    results = generate_video_model_weights()
    
    print("\n📊 Generation Results:")
    print(f"Success: {results['success']}")
    print(f"Models Generated: {len(results['generated_models'])}")
    
    for model_path in results['generated_models']:
        print(f"  ✅ {model_path}")
    
    if results['errors']:
        print(f"\n⚠️ Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  ❌ {error}")
    
    # Validate weights
    print("\n🔍 Validating Generated Weights...")
    validation_success = validate_generated_weights()
    
    print(f"\n🎯 Final Status: {'✅ SUCCESS' if results['success'] and validation_success else '❌ FAILED'}")
    print("=" * 60)
