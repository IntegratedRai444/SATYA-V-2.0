"""
SatyaAI Core Module
==================

Main interface for the deepfake detection system with multi-modal analysis capabilities.

This module provides the core functionality for detecting deepfakes across different
media types (image, video, audio) and combines their results using a fusion engine.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar,
                    Union)

# Import type hints for better IDE support
if True:  # For better code folding
    from pathlib import Path
    from typing import (Any, Awaitable, Callable, Dict, Generic, List,
                        Optional, Tuple, TypeVar, Union)

    import numpy as np
    import torch
    from PIL import Image


# Import detectors with proper error handling
class DetectorType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


# Lazy imports with better error handling
class LazyImport:
    """Lazy import utility to handle optional dependencies."""

    def __init__(self, module_name: str, package: Optional[str] = None):
        self.module_name = module_name
        self.package = package
        self._module = None

    def __getattr__(self, name: str) -> Any:
        if self._module is None:
            try:
                self._module = __import__(self.module_name, fromlist=[""])
                if self.package and hasattr(self._module, self.package):
                    self._module = getattr(self._module, self.package)
            except ImportError as e:
                raise ImportError(
                    f"Required package '{self.module_name}' is not installed. "
                    f"Please install it with 'pip install {self.module_name}'"
                ) from e
        return getattr(self._module, name)


# Try to import optional dependencies
try:
    from detectors.image_detector import ImageDetector

    IMAGE_DETECTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ImageDetector not available: {e}")
    ImageDetector = None
    IMAGE_DETECTOR_AVAILABLE = False

try:
    from detectors.video_detector import VideoDetector

    VIDEO_DETECTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"VideoDetector not available: {e}")
    VideoDetector = None
    VIDEO_DETECTOR_AVAILABLE = False

try:
    from detectors.audio_detector import AudioDetector

    AUDIO_DETECTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AudioDetector not available: {e}")
    AudioDetector = None
    AUDIO_DETECTOR_AVAILABLE = False

try:
    from detectors.fusion_engine import FusionEngine

    FUSION_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"FusionEngine not available: {e}")
    FusionEngine = None
    FUSION_ENGINE_AVAILABLE = False

# Import Sentinel components
try:
    from .reasoning_engine import (Conclusion, ConfidenceLevel, EvidenceType,
                                   ReasoningEngine)
    from .sentinel_agent import AnalysisRequest, AnalysisType, SentinelAgent

    SENTINEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Sentinel components not available: {e}")
    SENTINEL_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Type aliases
MediaBuffer = Union[bytes, str, os.PathLike]
AnalysisResult = Dict[str, Any]
DetectionResult = Dict[str, Any]
ConfigDict = Dict[str, Any]


class AnalysisError(Exception):
    """Base exception for analysis errors."""

    pass


class DetectorNotAvailableError(AnalysisError):
    """Raised when a required detector is not available."""

    pass


class InvalidMediaError(AnalysisError):
    """Raised when the provided media is invalid or corrupted."""

    pass


class SatyaAICore:
    """
    Main SatyaAI detection engine that coordinates all modalities.

    This class serves as the central component for deepfake detection, providing
    a unified interface for analyzing images, videos, and audio. It manages the
    initialization of specialized detectors and handles the fusion of their results.

    Attributes:
        config (ConfigDict): Configuration settings for the detector
        model_path (str): Path to the directory containing model files
        enable_gpu (bool): Whether to use GPU acceleration if available
        image_detector (Optional[ImageDetector]): Instance of ImageDetector
        video_detector (Optional[VideoDetector]): Instance of VideoDetector
        audio_detector (Optional[AudioDetector]): Instance of AudioDetector
        fusion_engine (Optional[FusionEngine]): Instance of FusionEngine
        sentinel_agent (Optional[SentinelAgent]): Instance of SentinelAgent for advanced analysis
    """

    def __init__(self, config: Optional[ConfigDict] = None):
        """
        Initialize SatyaAI with configuration.

        Args:
            config: Configuration dictionary with model paths and settings.
                   If None, uses default configuration.

        Raises:
            RuntimeError: If required dependencies are missing
            OSError: If model files cannot be loaded
            ValueError: If configuration is invalid
        """
        # Set up default configuration
        self.config: ConfigDict = {
            "MODEL_PATH": "./models",
            "ENABLE_GPU": False,
            "ENABLE_SENTINEL": SENTINEL_AVAILABLE,
            "LOG_LEVEL": "INFO",
            "MAX_WORKERS": 4,
            "CACHE_RESULTS": True,
            **(config or {}),
        }

        # Configure logging
        log_level = getattr(
            logging, self.config.get("LOG_LEVEL", "INFO").upper(), logging.INFO
        )
        logging.basicConfig(level=log_level)

        self.model_path: str = os.path.abspath(self.config["MODEL_PATH"])
        self.enable_gpu: bool = self.config["ENABLE_GPU"]
        self._initialized: bool = False

        # Initialize detectors
        self.image_detector: Optional[ImageDetector] = None
        self.video_detector: Optional[VideoDetector] = None
        self.audio_detector: Optional[AudioDetector] = None
        self.fusion_engine: Optional[FusionEngine] = None
        self.sentinel_agent: Optional[SentinelAgent] = None
        self.reasoning_engine: Optional[ReasoningEngine] = None

        # Result cache
        self._cache: Dict[str, Any] = {}

        # Initialize components
        try:
            self._initialize_components()
            self._initialized = True
            logger.info("SatyaAI Core initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize SatyaAI Core: {e}", exc_info=True)
            raise

    def _initialize_components(self) -> None:
        """
        Initialize all detection components.

        This method initializes the individual detectors (image, video, audio) and the
        fusion engine. It handles any initialization errors gracefully, allowing the
        system to operate in a degraded mode if some components fail to load.

        Raises:
            RuntimeError: If no detectors could be loaded
        """
        initialized_components = []

        # Initialize image detector
        if IMAGE_DETECTOR_AVAILABLE:
            try:
                self.image_detector = ImageDetector(self.model_path, self.enable_gpu)
                initialized_components.append("image_detector")
                logger.info("Image detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize image detector: {e}", exc_info=True)

        # Initialize video detector
        if VIDEO_DETECTOR_AVAILABLE:
            try:
                self.video_detector = VideoDetector(self.model_path, self.enable_gpu)
                initialized_components.append("video_detector")
                logger.info("Video detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize video detector: {e}", exc_info=True)

        # Initialize audio detector
        if AUDIO_DETECTOR_AVAILABLE:
            try:
                self.audio_detector = AudioDetector(self.model_path, self.enable_gpu)
                initialized_components.append("audio_detector")
                logger.info("Audio detector initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize audio detector: {e}", exc_info=True)

        # Initialize fusion engine
        if FUSION_ENGINE_AVAILABLE:
            try:
                self.fusion_engine = FusionEngine()
                initialized_components.append("fusion_engine")
                logger.info("Fusion engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize fusion engine: {e}", exc_info=True)

        # Initialize Sentinel components if available
        if SENTINEL_AVAILABLE and self.config.get("ENABLE_SENTINEL", True):
            try:
                self.reasoning_engine = ReasoningEngine()
                self.sentinel_agent = SentinelAgent(config=self.config)
                initialized_components.extend(["reasoning_engine", "sentinel_agent"])
                logger.info("Sentinel components initialized successfully")
            except Exception as e:
                logger.error(
                    f"Failed to initialize Sentinel components: {e}", exc_info=True
                )

        if not initialized_components:
            raise RuntimeError("Failed to initialize any detection components")

        logger.info(f"Initialized components: {', '.join(initialized_components)}")

    async def analyze_image(self, image_buffer: bytes, **kwargs) -> Dict[str, Any]:
        """
        Analyze an image for deepfake detection.

        This method delegates all image analysis to the SentinelAgent, which in turn
        uses the centralized deepfake_classifier for ML inference.

        Args:
            image_buffer: Image data as bytes
            **kwargs: Additional arguments for the analysis
                - confidence_threshold: Minimum confidence score (0-1)
                - return_evidence: Whether to include detailed evidence
                - timeout: Maximum time to wait for analysis (seconds)

        Returns:
            Dictionary containing analysis results with the following keys:
            - success (bool): Whether the analysis was successful
            - is_deepfake (bool): Whether the image is a deepfake
            - confidence (float): Confidence score (0-1)
            - model_info (dict): Information about the model used
            - evidence (list, optional): Detailed evidence if return_evidence=True

        Raises:
            RuntimeError: If analysis fails for any reason
            ValueError: If input is invalid
        """
        # Input validation
        if not isinstance(image_buffer, bytes) or len(image_buffer) == 0:
            raise ValueError("image_buffer must be a non-empty bytes object")

        # Check if SentinelAgent is available
        if not hasattr(self, "sentinel_agent") or self.sentinel_agent is None:
            raise RuntimeError("SentinelAgent is not available")

        # Check if deepfake model is available
        from ..models.deepfake_classifier import (get_model_info,
                                                  is_model_available)

        if not is_model_available():
            raise RuntimeError("Deepfake detection model is not available")

        try:
            # Create analysis request
            from .sentinel_agent import AnalysisRequest, AnalysisType

            request = AnalysisRequest(
                analysis_type="image",
                content=content,
                metadata={
                    "source": image_path,
                    "analyze_faces": analyze_faces,
                    "analyze_forensics": analyze_forensics,
                },
            )

            # Run analysis and return results
            result = asyncio.run(self.sentinel_agent.analyze(request))
            return result.dict()

        except Exception as e:
            logger.error(f"Image analysis failed: {str(e)}", exc_info=True)
            return self._create_error_result(
                f"Image analysis failed: {str(e)}",
                error_type="analysis_error",
                details={"original_error": str(e)},
            )

    def analyze_video(self, video_buffer: bytes) -> Dict[str, Any]:
        """
        Analyze a video for deepfake detection.

        Args:
            video_buffer: Video data as bytes

        Returns:
            Analysis result dictionary
        """
        try:
            if not self.video_detector:
                raise DetectorNotAvailableError("Video detector is not available")

            logger.info("Starting video analysis")
            result = self.video_detector.analyze(video_buffer)

            # Add metadata
            result.update(
                {
                    "analysis_date": datetime.now().isoformat(),
                    "case_id": f"vid-{int(datetime.now().timestamp())}",
                    "server_version": "2.0.0",
                }
            )

            logger.info(
                f"Video analysis completed: {result.get('authenticity', 'Unknown')}"
            )
            return result

        except ValueError as e:
            logger.error(f"Invalid video data: {e}")
            raise InvalidMediaError(f"Invalid video data: {str(e)}")
        except Exception as e:
            logger.error(f"Video analysis failed: {e}", exc_info=True)
            raise AnalysisError(f"Video analysis failed: {str(e)}")

    def analyze_audio(self, audio_buffer: bytes) -> Dict[str, Any]:
        """
        Analyze audio for deepfake detection.

        Args:
            audio_buffer: Audio data as bytes

        Returns:
            Analysis result dictionary
        """
        try:
            if not self.audio_detector:
                raise DetectorNotAvailableError("Audio detector is not available")

            logger.info("Starting audio analysis")
            result = self.audio_detector.analyze(audio_buffer)

            # Add metadata
            result.update(
                {
                    "analysis_date": datetime.now().isoformat(),
                    "case_id": f"aud-{int(datetime.now().timestamp())}",
                    "server_version": "2.0.0",
                }
            )

            logger.info(
                f"Audio analysis completed: {result.get('authenticity', 'Unknown')}"
            )
            return result

        except ValueError as e:
            logger.error(f"Invalid audio data: {e}")
            raise InvalidMediaError(f"Invalid audio data: {str(e)}")
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}", exc_info=True)
            raise AnalysisError(f"Audio analysis failed: {str(e)}")

    def analyze_multimodal(
        self,
        image_buffer: Optional[bytes] = None,
        audio_buffer: Optional[bytes] = None,
        video_buffer: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Perform multimodal analysis combining multiple media types.

        Args:
            image_buffer: Optional image data
            audio_buffer: Optional audio data
            video_buffer: Optional video data

        Returns:
            Unified analysis result
        """
        try:
            if not self.fusion_engine:
                raise DetectorNotAvailableError("Fusion engine is not available")

            logger.info("Starting multimodal analysis")

            # Analyze each modality
            results = {}

            if image_buffer:
                results["image"] = self.analyze_image(image_buffer)

            if video_buffer:
                results["video"] = self.analyze_video(video_buffer)

            if audio_buffer:
                results["audio"] = self.analyze_audio(audio_buffer)

            if not results:
                raise AnalysisError("No valid media provided for analysis")

            # Fuse results
            fused_result = self.fusion_engine.fuse(results)

            logger.info(
                f"Multimodal analysis completed: {fused_result.get('authenticity', 'Unknown')}"
            )
            return fused_result

        except Exception as e:
            logger.error(f"Multimodal analysis failed: {e}", exc_info=True)
            raise AnalysisError(f"Multimodal analysis failed: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models and their status.

        Returns:
            Model information dictionary
        """
        return {
            "image_detector": {
                "available": self.image_detector is not None,
                "models_loaded": self.image_detector.models_loaded
                if self.image_detector
                else False,
            },
            "video_detector": {
                "available": self.video_detector is not None,
                "models_loaded": self.video_detector.models_loaded
                if self.video_detector
                else False,
            },
            "audio_detector": {
                "available": self.audio_detector is not None,
                "models_loaded": self.audio_detector.models_loaded
                if self.audio_detector
                else False,
            },
            "fusion_engine": {"available": self.fusion_engine is not None},
        }

    def _generate_cache_key(self, analysis_type: str, data: Any) -> str:
        """
        Generate a cache key for analysis results.

        Args:
            analysis_type: Type of analysis (e.g., 'image', 'video', 'audio')
            data: Input data to analyze (used to generate a hash)

        Returns:
            A unique string key for caching
        """
        import hashlib

        if isinstance(data, (str, os.PathLike)):
            # Use file path and modification time for files
            try:
                mtime = os.path.getmtime(data)
                key_data = f"{os.path.abspath(data)}:{mtime}"
            except (OSError, TypeError):
                key_data = str(data)
        elif isinstance(data, bytes):
            # Use hash for binary data
            key_data = hashlib.sha256(data).hexdigest()
        else:
            # Fallback to string representation
            key_data = str(data)

        return f"{analysis_type}:{hashlib.md5(key_data.encode()).hexdigest()}"

    def _create_error_result(
        self,
        error_message: str,
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a standardized error result.

        Args:
            error_message: Human-readable error message
            error_type: Type of error (e.g., 'validation', 'processing', 'timeout')
            details: Additional error details

        Returns:
            A dictionary with error information and metadata
        """
        error_type = error_type or "unknown_error"
        error_id = f"err-{int(datetime.now().timestamp())}-{os.urandom(2).hex()}"

        result = {
            "success": False,
            "authenticity": "ANALYSIS FAILED",
            "confidence": 0.0,
            "analysis_date": datetime.now().isoformat(),
            "case_id": error_id,
            "key_findings": [f"Error: {error_message}"],
            "error": error_message,
            "error_type": error_type,
            "error_id": error_id,
            "details": {
                "analysis_type": "error",
                "error_type": error_type,
                "timestamp": datetime.now().isoformat(),
                **(details or {}),
            },
        }

        logger.error(
            f"Analysis error [{error_type}]: {error_message}",
            extra={"error_id": error_id, "details": details},
        )

        return result


# Global instance container with thread safety and resource management
class SatyaAIContainer:
    """
    Thread-safe singleton container for SatyaAI Core instance.

    This class ensures that only one instance of SatyaAICore exists per process
    and provides thread-safe access to it. It also handles proper cleanup of
    resources when the instance is no longer needed.
    """

    _instance: ClassVar[Optional["SatyaAICore"]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _initialized: ClassVar[bool] = False
    _initialization_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def get_instance(cls, config: Optional[ConfigDict] = None) -> "SatyaAICore":
        """
        Get or create the global SatyaAI instance in a thread-safe manner.

        Args:
            config: Configuration dictionary (only used on first call)

        Returns:
            SatyaAI instance

        Raises:
            RuntimeError: If initialization fails
        """
        async with cls._lock:
            if cls._instance is None:
                # Use default config if none provided
                if config is None:
                    config = {
                        "MODEL_PATH": os.environ.get("MODEL_PATH", "./models"),
                        "ENABLE_GPU": os.environ.get("ENABLE_GPU", "False").lower()
                        == "true",
                        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO"),
                        "ENABLE_SENTINEL": os.environ.get(
                            "ENABLE_SENTINEL", "True"
                        ).lower()
                        == "true",
                        "MAX_WORKERS": int(os.environ.get("MAX_WORKERS", "4")),
                        "CACHE_RESULTS": os.environ.get("CACHE_RESULTS", "True").lower()
                        == "true",
                    }

                # Initialize the instance
                cls._instance = SatyaAICore(config)

                # Mark as initialized
                cls._initialized = True

                # Register cleanup on program exit
                import atexit

                atexit.register(cls.cleanup)

                logger.info("Created new SatyaAI Core instance")

            return cls._instance

    @classmethod
    async def reset(cls) -> None:
        """
        Reset the global instance and clean up resources.

        This is primarily useful for testing or when you need to force a reload
        of all models and configurations.
        """
        async with cls._lock:
            if cls._instance is not None:
                # Clean up resources
                await cls.cleanup()
                cls._instance = None
                cls._initialized = False
                logger.info("Reset SatyaAI Core instance")

    @classmethod
    def cleanup(cls) -> None:
        """Clean up resources used by the global instance."""
        if cls._instance is not None:
            # Perform any necessary cleanup
            if (
                hasattr(cls._instance, "sentinel_agent")
                and cls._instance.sentinel_agent
            ):
                try:
                    asyncio.create_task(cls._instance.sentinel_agent.shutdown())
                except Exception as e:
                    logger.warning(f"Error during Sentinel agent shutdown: {e}")

            # Clear cache
            if hasattr(cls._instance, "_cache"):
                cls._instance._cache.clear()

            logger.info("Cleaned up SatyaAI Core resources")


async def get_satyaai_instance(config: Optional[ConfigDict] = None) -> SatyaAICore:
    """
    Get or create the global SatyaAI instance.

    This is the recommended way to access the SatyaAI Core functionality.
    It ensures that only one instance exists per process and handles proper
    initialization and cleanup.

    Args:
        config: Configuration dictionary (only used on first call).
               If None, uses environment variables or defaults.

    Returns:
        SatyaAI instance

    Example:
        >>> # Basic usage
        >>> satya = await get_satyaai_instance()
        >>>
        >>> # With custom configuration
        >>> config = {
        ...     'MODEL_PATH': '/path/to/models',
        ...     'ENABLE_GPU': True,
        ...     'LOG_LEVEL': 'DEBUG'
        ... }
        >>> satya = await get_satyaai_instance(config)
    """
    return await SatyaAIContainer.get_instance(config)


async def reset_satyaai_instance() -> None:
    """
    Reset the global SatyaAI instance.

    This will clean up all resources and force a reload of all models
    the next time get_satyaai_instance() is called.

    Note:
        This is primarily useful for testing or when you need to force
        a reload of the configuration and models.
    """
    await SatyaAIContainer.reset()


# Backward compatibility for synchronous code
_sync_instance = None
_sync_lock = asyncio.Lock()


def get_sync_satyaai_instance(config: Optional[ConfigDict] = None) -> SatyaAICore:
    """
    Synchronous version of get_satyaai_instance.

    Warning:
        This is provided for backward compatibility only. New code should use
        the async version (get_satyaai_instance) with asyncio.

    Args:
        config: Configuration dictionary (only used on first call)

    Returns:
        SatyaAI instance
    """
    global _sync_instance

    if _sync_instance is None:
        # Create a new event loop if we're not in one
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, use it
            return loop.run_until_complete(get_satyaai_instance(config))
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                _sync_instance = loop.run_until_complete(get_satyaai_instance(config))
                return _sync_instance
            finally:
                loop.close()

    return _sync_instance
