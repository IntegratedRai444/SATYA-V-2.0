"""
Model Preloader Service
Preloads ML models for faster first requests with memory-safe loading
"""

import logging
import asyncio
import psutil
import gc
import os
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# 🔥 PRODUCTION: Memory thresholds raised to 98% for image model loading
MAX_IMAGE_MODEL_MEMORY_PERCENT = 98
MIN_MEMORY_FOR_IMAGE_MODEL_GB = 2  # Minimum 2GB required for any attempt

class ModelPreloader:
    """Service for preloading ML models to improve first-request latency"""
    
    def __init__(self):
        self.preloaded_models: Dict[str, any] = {}
        self.preload_status: Dict[str, str] = {}
        self.preload_times: Dict[str, float] = {}
        
    async def preload_all_models(self) -> Dict[str, bool]:
        """Preload all available ML models with memory-safe loading"""
        results = {}
        
        # Check available memory first
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024**3)
        memory_percent = memory_info.percent
        
        logger.info(f"🧠 Memory check: {available_memory_gb:.1f}GB available ({memory_percent:.1f}% used)")
        
        # 🔥 PRODUCTION: Check for force override flag
        force_image_model = os.getenv('FORCE_IMAGE_MODEL', 'false').lower() == 'true'
        if force_image_model:
            logger.warning("🚀 FORCE_IMAGE_MODEL=true - Ignoring memory thresholds for image model")
        
        # 🔥 PRODUCTION: Load order fixed - Image models FIRST to prevent fragmentation
        # 1. Image models (highest priority, load first)
        image_attempted = False
        if force_image_model or memory_percent < MAX_IMAGE_MODEL_MEMORY_PERCENT:
            if available_memory_gb >= MIN_MEMORY_FOR_IMAGE_MODEL_GB or force_image_model:
                logger.info("🖼️ Loading image models FIRST (prevent fragmentation)")
                image_attempted = True
                results['image'] = await self._preload_image_models_lightweight()
            else:
                logger.warning(f"⚠️ Image model skipped - insufficient memory: {available_memory_gb:.1f}GB < {MIN_MEMORY_FOR_IMAGE_MODEL_GB}GB")
                results['image'] = False
        else:
            logger.warning(f"⚠️ Image model skipped - memory usage too high: {memory_percent:.1f}% > {MAX_IMAGE_MODEL_MEMORY_PERCENT}%")
            results['image'] = False
        
        # 2. Text models (always load, lightweight)
        logger.info("📝 Loading text models")
        results['text'] = await self._preload_text_models_lightweight()
        
        # 3. Optional detectors (load last)
        if available_memory_gb > 6:  # Only if sufficient memory
            logger.info("🎥 Loading video models")
            results['video'] = await self._preload_video_models_lightweight()
            logger.info("🎵 Loading audio models")
            results['audio'] = await self._preload_audio_models_lightweight()
        else:
            logger.info("⚠️ Skipping heavy models - insufficient memory")
            results['video'] = False
            results['audio'] = False
        
        # Force garbage collection
        gc.collect()
        
        return results
    
    async def _preload_image_models_lightweight(self) -> bool:
        """Preload image detection models with memory safety - ALWAYS ATTEMPT"""
        try:
            start_time = datetime.now()
            
            # 🔥 PRODUCTION: Safe memory guards
            import torch
            torch.set_num_threads(2)  # Limit to 2 threads
            
            # 🔥 PRODUCTION: Always attempt image model load - no silent skips
            memory_before = psutil.virtual_memory().percent
            logger.info(f"🖼️ Image model load attempt - memory: {memory_before:.1f}%")
            
            try:
                from services.detector_singleton import DetectorSingleton
                detector = DetectorSingleton()
                
                # 🔥 PRODUCTION: Use torch.no_grad() for memory efficiency
                with torch.no_grad():
                    # Initialize image detector
                    if hasattr(detector, 'initialize_image_detector'):
                        await detector.initialize_image_detector()
                    else:
                        # Fallback: try direct import
                        from detectors.image_detector import ImageDetector
                        detector.image_detector = ImageDetector()
                
                memory_after = psutil.virtual_memory().percent
                memory_increase = memory_after - memory_before
                
                # Check if model loaded successfully
                if hasattr(detector, 'image_detector') and detector.image_detector is not None:
                    self.preloaded_models['image'] = detector.image_detector
                    self.preload_status['image'] = 'success'
                    self.preload_times['image'] = (datetime.now() - start_time).total_seconds()
                    logger.info(f"✅ Image models loaded in {self.preload_times['image']:.2f}s (memory +{memory_increase:.1f}%)")
                    return True
                else:
                    self.preload_status['image'] = 'failed'
                    logger.error("❌ Image model load failed - detector is None")
                    return False
                    
            except MemoryError as e:
                # 🔥 PRODUCTION: Only fallback on real MemoryError
                logger.error(f"❌ Image model OOM - MemoryError: {e}")
                self.preload_status['image'] = 'memory_error'
                return False
            except Exception as e:
                # 🔥 PRODUCTION: Log all other failures explicitly
                logger.error(f"❌ Image model load failed: {type(e).__name__}: {e}")
                self.preload_status['image'] = 'error'
                return False
                
        except Exception as e:
            self.preload_status['image'] = 'initialization_error'
            logger.error(f"❌ Image preloader initialization failed: {e}")
            return False
    
    async def _preload_video_models_lightweight(self) -> bool:
        """Preload video detection models with memory safety"""
        try:
            start_time = datetime.now()
            
            # Video models are heavy, only load if sufficient memory
            memory_info = psutil.virtual_memory()
            if memory_info.available < (6 * 1024**3):  # Less than 6GB available
                logger.warning("⚠️ Insufficient memory for video models, skipping")
                self.preload_status['video'] = 'skipped_memory'
                return False
            
            try:
                from services.detector_singleton import DetectorSingleton
                detector = DetectorSingleton()
                
                memory_before = psutil.virtual_memory().percent
                
                # Initialize video detector
                if hasattr(detector, 'initialize_video_detector'):
                    await detector.initialize_video_detector()
                else:
                    # Fallback: try direct import
                    from detectors.video_detector import VideoDetector
                    detector.video_detector = VideoDetector()
                
                memory_after = psutil.virtual_memory().percent
                memory_increase = memory_after - memory_before
                
                if hasattr(detector, 'video_detector') and detector.video_detector is not None:
                    self.preloaded_models['video'] = detector.video_detector
                    self.preload_status['video'] = 'success'
                    self.preload_times['video'] = (datetime.now() - start_time).total_seconds()
                    logger.info(f"✅ Video models preloaded in {self.preload_times['video']:.2f}s (memory +{memory_increase:.1f}%)")
                    return True
                else:
                    self.preload_status['video'] = 'failed'
                    logger.warning("⚠️ Video models failed to preload")
                    return False
                    
            except MemoryError as e:
                logger.error(f"❌ Out of memory loading video models: {e}")
                self.preload_status['video'] = 'memory_error'
                return False
            except Exception as e:
                logger.error(f"❌ Failed to preload video models: {e}")
                self.preload_status['video'] = 'error'
                return False
                
        except Exception as e:
            self.preload_status['video'] = 'error'
            logger.error(f"❌ Video preloader initialization failed: {e}")
            return False
    
    async def _preload_audio_models_lightweight(self) -> bool:
        """Preload audio detection models with memory safety"""
        try:
            start_time = datetime.now()
            
            try:
                from services.detector_singleton import DetectorSingleton
                detector = DetectorSingleton()
                
                memory_before = psutil.virtual_memory().percent
                
                # Initialize audio detector
                if hasattr(detector, 'initialize_audio_detector'):
                    await detector.initialize_audio_detector()
                else:
                    # Fallback: try direct import
                    from detectors.audio_detector import AudioDetector
                    detector.audio_detector = AudioDetector()
                
                memory_after = psutil.virtual_memory().percent
                memory_increase = memory_after - memory_before
                
                if hasattr(detector, 'audio_detector') and detector.audio_detector is not None:
                    self.preloaded_models['audio'] = detector.audio_detector
                    self.preload_status['audio'] = 'success'
                    self.preload_times['audio'] = (datetime.now() - start_time).total_seconds()
                    logger.info(f"✅ Audio models preloaded in {self.preload_times['audio']:.2f}s (memory +{memory_increase:.1f}%)")
                    return True
                else:
                    self.preload_status['audio'] = 'failed'
                    logger.warning("⚠️ Audio models failed to preload")
                    return False
                    
            except MemoryError as e:
                logger.error(f"❌ Out of memory loading audio models: {e}")
                self.preload_status['audio'] = 'memory_error'
                return False
            except Exception as e:
                logger.error(f"❌ Failed to preload audio models: {e}")
                self.preload_status['audio'] = 'error'
                return False
                
        except Exception as e:
            self.preload_status['audio'] = 'error'
            logger.error(f"❌ Audio preloader initialization failed: {e}")
            return False
    
    async def _preload_text_models_lightweight(self) -> bool:
        """Preload text detection models (always lightweight)"""
        try:
            start_time = datetime.now()
            
            try:
                from services.detector_singleton import DetectorSingleton
                detector = DetectorSingleton()
                
                memory_before = psutil.virtual_memory().percent
                
                # Initialize text detector
                if hasattr(detector, 'initialize_text_detector'):
                    await detector.initialize_text_detector()
                else:
                    # Fallback: try direct import
                    from detectors.text_nlp_detector import TextNLPDetector
                    detector.text_detector = TextNLPDetector()
                
                memory_after = psutil.virtual_memory().percent
                memory_increase = memory_after - memory_before
                
                if hasattr(detector, 'text_detector') and detector.text_detector is not None:
                    self.preloaded_models['text'] = detector.text_detector
                    self.preload_status['text'] = 'success'
                    self.preload_times['text'] = (datetime.now() - start_time).total_seconds()
                    logger.info(f"✅ Text models preloaded in {self.preload_times['text']:.2f}s (memory +{memory_increase:.1f}%)")
                    return True
                else:
                    self.preload_status['text'] = 'failed'
                    logger.warning("⚠️ Text models failed to preload")
                    return False
                    
            except MemoryError as e:
                logger.error(f"❌ Out of memory loading text models: {e}")
                self.preload_status['text'] = 'memory_error'
                return False
            except Exception as e:
                logger.error(f"❌ Failed to preload text models: {e}")
                self.preload_status['text'] = 'error'
                return False
                
        except Exception as e:
            self.preload_status['text'] = 'error'
            logger.error(f"❌ Text preloader initialization failed: {e}")
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
