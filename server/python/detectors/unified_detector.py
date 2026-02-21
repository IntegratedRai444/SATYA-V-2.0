"""
Unified Deepfake Detector Interface
===================================

Provides a standardized interface for all deepfake detection modalities
with consistent error handling, performance monitoring, and result formatting.

This unifies:
- Enhanced model loading with HuggingFace fallbacks
- Comprehensive forensic analysis 
- Multi-modal fusion capabilities
- Production-ready features
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Supported detection modalities"""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


class DetectionResult:
    """Standardized detection result format"""
    
    def __init__(self, modality: ModalityType, success: bool = True):
        self.modality = modality
        self.success = success
        self.authenticity = "UNCERTAIN"
        self.confidence = 0.5
        self.analysis_date = datetime.utcnow().isoformat()
        self.processing_time = 0.0
        self.key_findings = []
        self.technical_details = {}
        self.ml_analysis = {}
        self.forensic_analysis = {}
        self.error = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {
            "modality": self.modality.value,
            "success": self.success,
            "authenticity": self.authenticity,
            "confidence": self.confidence,
            "analysis_date": self.analysis_date,
            "processing_time_seconds": self.processing_time,
            "key_findings": self.key_findings,
            "technical_details": self.technical_details,
            "ml_analysis": self.ml_analysis,
            "forensic_analysis": self.forensic_analysis
        }
        
        if self.error:
            result["error"] = self.error
            
        return result


class UnifiedDetector:
    """
    Unified interface for all deepfake detection modalities.
    
    This class provides a consistent interface for image, video, audio,
    and multi-modal deepfake detection with standardized error handling,
    performance monitoring, and result formatting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the unified detector.
        
        Args:
            config: Configuration dictionary with model paths and settings
        """
        self.config = config or {}
        self.model_path = self.config.get("MODEL_PATH", "models")
        self.enable_gpu = self.config.get("ENABLE_GPU", torch.cuda.is_available())
        self.enable_forensics = self.config.get("ENABLE_FORENSICS", True)
        self.enable_multimodal = self.config.get("ENABLE_MULTIMODAL", True)
        
        # Initialize detectors
        self.image_detector = None
        self.video_detector = None
        self.audio_detector = None
        self.multimodal_engine = None
        
        # Performance tracking
        self.performance_metrics = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time": 0.0,
            "modality_stats": {
                "image": {"count": 0, "avg_time": 0.0},
                "video": {"count": 0, "avg_time": 0.0},
                "audio": {"count": 0, "avg_time": 0.0},
                "multimodal": {"count": 0, "avg_time": 0.0}
            }
        }
        
        # Initialize components
        self._initialize_detectors()
        
    def _initialize_detectors(self) -> None:
        """Initialize all detector components using singleton pattern"""
        try:
            # Use singleton pattern for all detectors
            from services.detector_singleton import DetectorSingleton
            detector_singleton = DetectorSingleton()
            
            # Initialize image detector
            try:
                self.image_detector = detector_singleton.get_detector('image', {'enable_gpu': self.enable_gpu})
                logger.info("✅ Image detector initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize image detector: {e}")
                
            # Initialize video detector
            try:
                video_config = {
                    "enable_gpu": self.enable_gpu,
                    "use_optimization": True,
                    "use_enhanced_model": True,
                    "temporal_window": 30,
                    "frame_sample_rate": 2,
                    "max_frames": 100
                }
                self.video_detector = detector_singleton.get_detector('video', video_config)
                logger.info("✅ Video detector initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize video detector: {e}")
                
            # Initialize audio detector
            try:
                audio_config = {
                    "enable_gpu": self.enable_gpu
                }
                self.audio_detector = detector_singleton.get_detector('audio', audio_config)
                logger.info("✅ Audio detector initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize audio detector: {e}")
                
            # Initialize multi-modal engine
            if self.enable_multimodal:
                try:
                    from satyaai_core import SatyaAICore
                    multimodal_config = {
                        "MODEL_PATH": self.model_path,
                        "ENABLE_GPU": self.enable_gpu,
                        "ENABLE_SENTINEL": True,
                        "MAX_WORKERS": 4,
                        "CACHE_RESULTS": True
                    }
                    self.multimodal_engine = SatyaAICore(multimodal_config)
                    logger.info("✅ Multi-modal engine initialized")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize multi-modal engine: {e}")
                    
        except Exception as e:
            logger.error(f"Detector initialization failed: {e}")
            
    def detect_image(self, image: Union[np.ndarray, bytes, Image.Image], 
                    analyze_forensics: Optional[bool] = None) -> DetectionResult:
        """
        Detect deepfakes in an image.
        
        Args:
            image: Input image as numpy array, bytes, or PIL Image
            analyze_forensics: Whether to perform forensic analysis (overrides config)
            
        Returns:
            DetectionResult with analysis findings
        """
        result = DetectionResult(ModalityType.IMAGE)
        start_time = time.time()
        
        try:
            if not self.image_detector:
                raise RuntimeError("Image detector not available")
                
            # Convert input to numpy array if needed
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
                image = np.array(image)
            elif isinstance(image, Image.Image):
                image = np.array(image)
                
            # Determine if forensics should be enabled
            forensics_enabled = analyze_forensics if analyze_forensics is not None else self.enable_forensics
            
            # Perform analysis
            analysis_result = self.image_detector.analyze(
                image=image,
                detect_faces=True,
                analyze_forensics=forensics_enabled,
                return_face_data=True
            )
            
            # Extract results
            if analysis_result.get("success", False):
                result.authenticity = analysis_result.get("authenticity", "UNCERTAIN")
                result.confidence = analysis_result.get("confidence", 0.5)
                result.key_findings = analysis_result.get("key_findings", [])
                result.technical_details = analysis_result.get("technical_details", {})
                result.ml_analysis = analysis_result.get("ml_analysis", {})
                result.forensic_analysis = analysis_result.get("forensic_analysis", {})
                
                # Add image-specific details
                result.technical_details.update({
                    "modality": "image",
                    "forensics_enabled": forensics_enabled,
                    "face_detection": "face_analysis" in analysis_result
                })
                
            else:
                result.success = False
                result.error = analysis_result.get("error", "Unknown error")
                
        except Exception as e:
            logger.error(f"Image detection failed: {e}")
            result.success = False
            result.error = str(e)
            
        finally:
            result.processing_time = time.time() - start_time
            self._update_performance_metrics(ModalityType.IMAGE, result.success, result.processing_time)
            
        return result
        
    def detect_video(self, video: Union[bytes, str], 
                    analyze_frames: Optional[int] = None) -> DetectionResult:
        """
        Detect deepfakes in a video.
        
        Args:
            video: Input video as bytes or file path
            analyze_frames: Number of frames to analyze (None for auto)
            
        Returns:
            DetectionResult with analysis findings
        """
        result = DetectionResult(ModalityType.VIDEO)
        start_time = time.time()
        
        try:
            if not self.video_detector:
                raise RuntimeError("Video detector not available")
                
            # Perform video analysis
            analysis_result = self.video_detector.analyze(
                video=video,
                max_frames=analyze_frames
            )
            
            # Extract results
            if analysis_result.get("success", False):
                result.authenticity = analysis_result.get("authenticity", "UNCERTAIN")
                result.confidence = analysis_result.get("confidence", 0.5)
                result.key_findings = analysis_result.get("key_findings", [])
                result.technical_details = analysis_result.get("technical_details", {})
                result.ml_analysis = analysis_result.get("ml_analysis", {})
                
                # Add video-specific details
                result.technical_details.update({
                    "modality": "video",
                    "frames_analyzed": analysis_result.get("frames_analyzed", 0),
                    "duration_seconds": analysis_result.get("duration", 0.0)
                })
                
            else:
                result.success = False
                result.error = analysis_result.get("error", "Unknown error")
                
        except Exception as e:
            logger.error(f"Video detection failed: {e}")
            result.success = False
            result.error = str(e)
            
        finally:
            result.processing_time = time.time() - start_time
            self._update_performance_metrics(ModalityType.VIDEO, result.success, result.processing_time)
            
        return result
        
    def detect_audio(self, audio: Union[bytes, str]) -> DetectionResult:
        """
        Detect deepfakes in audio.
        
        Args:
            audio: Input audio as bytes or file path
            
        Returns:
            DetectionResult with analysis findings
        """
        result = DetectionResult(ModalityType.AUDIO)
        start_time = time.time()
        
        try:
            if not self.audio_detector:
                raise RuntimeError("Audio detector not available")
                
            # Perform audio analysis
            analysis_result = self.audio_detector.analyze(audio=audio)
            
            # Extract results
            if analysis_result.get("success", False):
                result.authenticity = analysis_result.get("authenticity", "UNCERTAIN")
                result.confidence = analysis_result.get("confidence", 0.5)
                result.key_findings = analysis_result.get("key_findings", [])
                result.technical_details = analysis_result.get("technical_details", {})
                result.ml_analysis = analysis_result.get("ml_analysis", {})
                
                # Add audio-specific details
                result.technical_details.update({
                    "modality": "audio",
                    "duration_seconds": analysis_result.get("duration", 0.0),
                    "sample_rate": analysis_result.get("sample_rate", 0)
                })
                
            else:
                result.success = False
                result.error = analysis_result.get("error", "Unknown error")
                
        except Exception as e:
            logger.error(f"Audio detection failed: {e}")
            result.success = False
            result.error = str(e)
            
        finally:
            result.processing_time = time.time() - start_time
            self._update_performance_metrics(ModalityType.AUDIO, result.success, result.processing_time)
            
        return result
        
    def detect_multimodal(self, image: Optional[Union[np.ndarray, bytes, Image.Image]] = None,
                         video: Optional[Union[bytes, str]] = None,
                         audio: Optional[Union[bytes, str]] = None) -> DetectionResult:
        """
        Detect deepfakes using multi-modal analysis.
        
        Args:
            image: Optional image input
            video: Optional video input
            audio: Optional audio input
            
        Returns:
            DetectionResult with fused analysis findings
        """
        result = DetectionResult(ModalityType.MULTIMODAL)
        start_time = time.time()
        
        try:
            if not self.multimodal_engine:
                raise RuntimeError("Multi-modal engine not available")
                
            # Convert inputs to bytes if needed
            image_bytes = None
            video_bytes = None
            audio_bytes = None
            
            if image:
                if isinstance(image, (np.ndarray, Image.Image)):
                    if isinstance(image, np.ndarray):
                        image_pil = Image.fromarray(image)
                    else:
                        image_pil = image
                    import io
                    img_buffer = io.BytesIO()
                    image_pil.save(img_buffer, format='PNG')
                    image_bytes = img_buffer.getvalue()
                else:
                    image_bytes = image
                    
            if video and isinstance(video, str):
                with open(video, 'rb') as f:
                    video_bytes = f.read()
            else:
                video_bytes = video
                
            if audio and isinstance(audio, str):
                with open(audio, 'rb') as f:
                    audio_bytes = f.read()
            else:
                audio_bytes = audio
                
            # Perform multi-modal analysis
            analysis_result = self.multimodal_engine.analyze_multimodal(
                image_buffer=image_bytes,
                video_buffer=video_bytes,
                audio_buffer=audio_bytes
            )
            
            # Extract results
            if analysis_result.get("success", True):
                result.authenticity = analysis_result.get("authenticity", "UNCERTAIN")
                result.confidence = analysis_result.get("confidence", 0.5)
                result.key_findings = analysis_result.get("key_findings", [])
                result.technical_details = analysis_result.get("technical_details", {})
                result.ml_analysis = analysis_result.get("ml_analysis", {})
                result.forensic_analysis = analysis_result.get("forensic_analysis", {})
                
                # Add multi-modal specific details
                result.technical_details.update({
                    "modality": "multimodal",
                    "modalities_analyzed": analysis_result.get("modalities_analyzed", {}),
                    "fusion_engine": "satyaai_core"
                })
                
            else:
                result.success = False
                result.error = analysis_result.get("error", "Unknown error")
                
        except Exception as e:
            logger.error(f"Multi-modal detection failed: {e}")
            result.success = False
            result.error = str(e)
            
        finally:
            result.processing_time = time.time() - start_time
            self._update_performance_metrics(ModalityType.MULTIMODAL, result.success, result.processing_time)
            
        return result
        
    def _update_performance_metrics(self, modality: ModalityType, success: bool, processing_time: float) -> None:
        """Update performance tracking metrics"""
        self.performance_metrics["total_analyses"] += 1
        
        if success:
            self.performance_metrics["successful_analyses"] += 1
        else:
            self.performance_metrics["failed_analyses"] += 1
            
        # Update modality-specific stats
        modality_key = modality.value
        modality_stats = self.performance_metrics["modality_stats"][modality_key]
        modality_stats["count"] += 1
        
        # Update average processing time
        current_avg = modality_stats["avg_time"]
        count = modality_stats["count"]
        modality_stats["avg_time"] = (current_avg * (count - 1) + processing_time) / count
        
        # Update overall average
        total_count = self.performance_metrics["total_analyses"]
        current_overall_avg = self.performance_metrics["average_processing_time"]
        self.performance_metrics["average_processing_time"] = (
            (current_overall_avg * (total_count - 1) + processing_time) / total_count
        )
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.performance_metrics,
            "success_rate": (
                self.performance_metrics["successful_analyses"] / 
                max(self.performance_metrics["total_analyses"], 1)
            ),
            "available_detectors": {
                "image": self.image_detector is not None,
                "video": self.video_detector is not None,
                "audio": self.audio_detector is not None,
                "multimodal": self.multimodal_engine is not None
            }
        }
        
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about available detectors"""
        info = {
            "unified_detector_version": "2.0.0",
            "config": self.config,
            "available_detectors": {},
            "performance_metrics": self.get_performance_metrics()
        }
        
        # Get info from each detector
        if self.image_detector:
            info["available_detectors"]["image"] = {
                "type": "ImageDetector",
                "forensics_available": True,
                "face_detection": True
            }
            
        if self.video_detector:
            info["available_detectors"]["video"] = {
                "type": "VideoDetector",
                "frame_analysis": True
            }
            
        if self.audio_detector:
            info["available_detectors"]["audio"] = {
                "type": "AudioDetector",
                "spectral_analysis": True
            }
            
        if self.multimodal_engine:
            info["available_detectors"]["multimodal"] = {
                "type": "SatyaAICore",
                "fusion_engine": True,
                "cross_modal_correlation": True
            }
            
        return info


# Global instance for easy access
_unified_detector_instance = None


def get_unified_detector(config: Optional[Dict[str, Any]] = None) -> UnifiedDetector:
    """
    Get or create the global unified detector instance.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        UnifiedDetector instance
    """
    global _unified_detector_instance
    
    if _unified_detector_instance is None:
        _unified_detector_instance = UnifiedDetector(config)
        
    return _unified_detector_instance
