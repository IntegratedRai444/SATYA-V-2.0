"""
Model Preloader Service
Preloads ML models for faster first requests
"""

import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelPreloader:
    """Service for preloading ML models to improve first-request latency"""
    
    def __init__(self):
        self.preloaded_models: Dict[str, any] = {}
        self.preload_status: Dict[str, str] = {}
        self.preload_times: Dict[str, float] = {}
        
    async def preload_all_models(self) -> Dict[str, bool]:
        """Preload all available ML models"""
        results = {}
        
        # Preload text models only (fastest)
        results['text'] = await self._preload_text_models()
        
        # Skip heavy models for faster startup
        logger.info("📦 Skipping heavy model preloading (image, video, audio)")
        logger.info("🤖 Heavy models will load on first use")
        
        return {'image': False, 'video': False, 'audio': False, 'text': results.get('text', False)}
    
    async def _preload_image_models(self) -> bool:
        """Preload image detection models"""
        try:
            start_time = datetime.now()
            
            # Try to import and initialize image models
            from detectors.image_detector import get_detector_by_type
            detector = get_detector_by_type('image')
            
            # Check if models are loaded
            if hasattr(detector, 'model') and detector.model is not None:
                self.preloaded_models['image'] = detector
                self.preload_status['image'] = 'success'
                self.preload_times['image'] = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Image models preloaded in {self.preload_times['image']:.2f}s")
                return True
            else:
                self.preload_status['image'] = 'failed'
                logger.warning("⚠️ Image models failed to preload")
                return False
                
        except Exception as e:
            self.preload_status['image'] = 'error'
            logger.error(f"❌ Failed to preload image models: {e}")
            return False
    
    async def _preload_video_models(self) -> bool:
        """Preload video detection models"""
        try:
            start_time = datetime.now()
            
            from detectors.video_detector import VideoDetector
            detector = get_detector_by_type('video')
            
            if hasattr(detector, 'model') and detector.model is not None:
                self.preloaded_models['video'] = detector
                self.preload_status['video'] = 'success'
                self.preload_times['video'] = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Video models preloaded in {self.preload_times['video']:.2f}s")
                return True
            else:
                self.preload_status['video'] = 'failed'
                logger.warning("⚠️ Video models failed to preload")
                return False
                
        except Exception as e:
            self.preload_status['video'] = 'error'
            logger.error(f"❌ Failed to preload video models: {e}")
            return False
    
    async def _preload_audio_models(self) -> bool:
        """Preload audio detection models"""
        try:
            start_time = datetime.now()
            
            from detectors.audio_detector import AudioDetector
            detector = get_detector_by_type('audio')
            
            if hasattr(detector, 'model') and detector.model is not None:
                self.preloaded_models['audio'] = detector
                self.preload_status['audio'] = 'success'
                self.preload_times['audio'] = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Audio models preloaded in {self.preload_times['audio']:.2f}s")
                return True
            else:
                self.preload_status['audio'] = 'failed'
                logger.warning("⚠️ Audio models failed to preload")
                return False
                
        except Exception as e:
            self.preload_status['audio'] = 'error'
            logger.error(f"❌ Failed to preload audio models: {e}")
            return False
    
    async def _preload_text_models(self) -> bool:
        """Preload text detection models"""
        try:
            start_time = datetime.now()
            
            from detectors.text_nlp_detector import TextNLPDetector
            detector = get_detector_by_type('text')
            
            if hasattr(detector, 'model') and detector.model is not None:
                self.preloaded_models['text'] = detector
                self.preload_status['text'] = 'success'
                self.preload_times['text'] = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Text models preloaded in {self.preload_times['text']:.2f}s")
                return True
            else:
                self.preload_status['text'] = 'failed'
                logger.warning("⚠️ Text models failed to preload")
                return False
                
        except Exception as e:
            self.preload_status['text'] = 'error'
            logger.error(f"❌ Failed to preload text models: {e}")
            return False
    
    def get_preloaded_model(self, modality: str) -> Optional[any]:
        """Get a preloaded model by modality"""
        return self.preloaded_models.get(modality)
    
    def get_preload_status(self) -> Dict[str, str]:
        """Get the preload status of all models"""
        return self.preload_status.copy()
    
    def get_preload_summary(self) -> Dict[str, any]:
        """Get a summary of preload performance"""
        return {
            'total_models': len(self.preloaded_models),
            'status': self.preload_status,
            'times': self.preload_times,
            'total_time': sum(self.preload_times.values())
        }

# Global preloader instance
model_preloader = ModelPreloader()
