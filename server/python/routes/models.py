"""
Models Status Routes for SatyaAI Python Backend
Provides model status and availability information
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

class ModelStatus(BaseModel):
    available: bool
    weights: str
    device: str
    model_name: str
    version: str
    confidence_threshold: float
    last_loaded: str = None

class ModelsStatusResponse(BaseModel):
    success: bool
    models: Dict[str, ModelStatus]
    system_info: Dict[str, Any]

@router.get("/status", response_model=ModelsStatusResponse)
async def get_models_status():
    """
    Get the status of all ML models
    Returns availability, weights location, device info, and model details
    """
    try:
        import torch
        from model_loader import ensure_models_available
        
        # Get model availability status
        model_status_result = ensure_models_available()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Build detailed model status
        models_status = {}
        
        # Image model status
        models_status['image'] = ModelStatus(
            available=model_status_result['models'].get('image', {}).get('available', False),
            weights=model_status_result['models'].get('image', {}).get('weights', 'missing'),
            device=device,
            model_name='EfficientNet-B7',
            version='1.0.0',
            confidence_threshold=0.7
        )
        
        # Audio model status
        models_status['audio'] = ModelStatus(
            available=model_status_result['models'].get('audio', {}).get('available', False),
            weights=model_status_result['models'].get('audio', {}).get('weights', 'missing'),
            device=device,
            model_name='Wav2Vec2-Large',
            version='1.0.0',
            confidence_threshold=0.6
        )
        
        # Video model status
        models_status['video'] = ModelStatus(
            available=model_status_result['models'].get('video', {}).get('available', False),
            weights=model_status_result['models'].get('video', {}).get('weights', 'missing'),
            device=device,
            model_name='X3D-XS',
            version='1.0.0',
            confidence_threshold=0.65
        )
        
        # System information
        system_info = {
            'device': device,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'strict_mode': model_status_result.get('strict_mode', False),
            'models_dir': os.getenv('MODEL_DIR', 'models'),
            'total_models': len([m for m in model_status_result['models'].values() if m.get('available', False)])
        }
        
        return ModelsStatusResponse(
            success=True,
            models=models_status,
            system_info=system_info
        )
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        
        # Return error status
        models_status = {
            'image': ModelStatus(
                available=False,
                weights='error',
                device='cpu',
                model_name='EfficientNet-B7',
                version='1.0.0',
                confidence_threshold=0.7
            ),
            'audio': ModelStatus(
                available=False,
                weights='error',
                device='cpu',
                model_name='Wav2Vec2-Large',
                version='1.0.0',
                confidence_threshold=0.6
            ),
            'video': ModelStatus(
                available=False,
                weights='error',
                device='cpu',
                model_name='X3D-XS',
                version='1.0.0',
                confidence_threshold=0.65
            )
        }
        
        return ModelsStatusResponse(
            success=False,
            models=models_status,
            system_info={
                'device': 'cpu',
                'error': str(e)
            }
        )

@router.get("/list")
async def list_available_models():
    """
    List all available models with basic information
    """
    try:
        models_dir = Path(os.getenv('MODEL_DIR', 'models'))
        available_models = []
        
        # Check for image models
        image_models = [
            models_dir / "dfdc_efficientnet_b7" / "model.pth",
            models_dir / "xception" / "model.pth"
        ]
        
        for model_path in image_models:
            if model_path.exists():
                available_models.append({
                    'type': 'image',
                    'name': model_path.parent.name,
                    'path': str(model_path),
                    'size': model_path.stat().st_size if model_path.exists() else 0
                })
        
        # Check for audio models
        audio_model = models_dir / "audio" / "model.pth"
        if audio_model.exists():
            available_models.append({
                'type': 'audio',
                'name': 'audio_model',
                'path': str(audio_model),
                'size': audio_model.stat().st_size
            })
        
        # Check for video models
        video_model = models_dir / "video" / "model.pth"
        if video_model.exists():
            available_models.append({
                'type': 'video',
                'name': 'video_model',
                'path': str(video_model),
                'size': video_model.stat().st_size
            })
        
        return {
            'success': True,
            'models': available_models,
            'total': len(available_models)
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
