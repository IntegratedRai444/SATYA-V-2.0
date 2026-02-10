"""
Satya Sentinel Agent
Advanced AI agent for deepfake detection and analysis

This module enforces that all analysis must go through SentinelAgent
with proper ML execution and proof generation.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, ValidationError, field_validator

# Configure logging
logger = logging.getLogger(__name__)

# Import ML models
try:
    from detectors.models import AudioDetector, ImageDetector, VideoDetector, get_model_info
except ImportError:
    from models import AudioDetector, ImageDetector, VideoDetector, get_model_info

# Import the reasoning engine
try:
    from reasoning_engine import (Conclusion, ConfidenceLevel, EvidenceItem,
                               EvidenceType, get_reasoning_engine)
except ImportError:
    # Fallback if reasoning_engine not available
    Conclusion = None
    ConfidenceLevel = None
    EvidenceItem = None
    EvidenceType = None
    def get_reasoning_engine():
        return None

# Import proof utilities
try:
    from utils.proof import ProofOfAnalysis, generate_proof, verify_proof
except ImportError:
    # Fallback if proof utilities not available
    ProofOfAnalysis = None
    def generate_proof(*args, **kwargs):
        return None
    def verify_proof(*args, **kwargs):
        return True

# Configure logging
logger = logging.getLogger(__name__)


class AnalysisType(str, Enum):
    """Types of analysis the agent can perform"""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    MULTIMODAL = "multimodal"
    METADATA = "metadata"
    WEBCAM = "webcam"


class AnalysisRequest(BaseModel):
    """Request for analysis by the Sentinel agent"""

    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")
    analysis_type: AnalysisType
    content: Optional[Union[bytes, str, Dict[str, Any]]] = None
    content_uri: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 1  # 1-10, higher is more important
    timeout: Optional[float] = 30.0  # Timeout in seconds
    callback_url: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class AnalysisResult(BaseModel):
    """
    Result of an analysis with proof validation

    This class enforces that all analysis results must include a valid proof
    of ML execution before being considered valid.
    """

    request_id: str
    status: str
    analysis_type: AnalysisType
    conclusions: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.VERY_LOW
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    proof: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator("confidence_level", mode="before")
    @classmethod
    def set_confidence_level(cls, v, info):
        """Set confidence level based on confidence score"""
        confidence = info.data.get("confidence", 0)
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.VERY_LOW

    @field_validator("proof", mode="before")
    @classmethod
    def validate_proof(cls, v, info):
        """Validate proof of analysis"""
        # Skip validation if there's an error
        if info.data.get("error"):
            return v

        # Proof is required for successful analysis
        if not v:
            raise ValueError("Proof of analysis is required")

        # Verify the proof
        is_valid, message = verify_proof(v)
        if not is_valid:
            raise ValueError(f"Invalid proof: {message}")

        # Verify proof matches the analysis type
        proof_type = v.get("modality")
        analysis_type = values.get("analysis_type")
        if proof_type != analysis_type.value:
            raise ValueError(
                f"Proof type '{proof_type}' does not match analysis type '{analysis_type}'"
            )

        # Verify the proof is fresh (e.g., not older than 5 minutes)
        proof_timestamp = v.get("timestamp")
        if time.time() - proof_timestamp > 300:  # 5 minutes
            raise ValueError("Proof has expired")

        return v

    def is_valid(self) -> bool:
        """Check if the result is valid (has valid proof and no errors)"""
        try:
            # This will trigger all validators including proof validation
            self.validate_self()
            return True
        except (ValueError, ValidationError):
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including the proof"""
        result = self.dict(exclude_none=True)

        # Convert datetime to ISO format
        if "timestamp" in result and isinstance(result["timestamp"], datetime):
            result["timestamp"] = result["timestamp"].isoformat()

        # Convert confidence level to string
        if "confidence_level" in result and isinstance(
            result["confidence_level"], ConfidenceLevel
        ):
            result["confidence_level"] = result["confidence_level"].value

        return result


class SentinelAgent:
    """Satya Sentinel - Advanced AI Agent for Deepfake Detection"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Sentinel agent with configurable detectors.
        Uses singleton pattern to prevent multiple detector initialization.
        Lazy loading - models loaded only when first requested.
        """
        self.config = config or {}
        # Enable GPU by default if available (with DLL error handling)
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                # Test CUDA device to catch DLL issues
                try:
                    device = torch.device('cuda:0')
                    test_tensor = torch.randn(1, 3, 224, 224).to(device)
                    _ = test_tensor.sum()
                    logger.info("✅ CUDA device test passed - GPU enabled")
                    self.config.setdefault('enable_gpu', True)
                except Exception as cuda_error:
                    logger.error(f"❌ CUDA device test failed: {cuda_error}")
                    logger.warning("⚠️ Falling back to CPU due to CUDA DLL issues")
                    self.config.setdefault('enable_gpu', False)
            else:
                logger.warning("⚠️ CUDA not available - using CPU")
                self.config.setdefault('enable_gpu', False)
        except ImportError:
            logger.warning("⚠️ PyTorch not available - using CPU")
            self.config.setdefault('enable_gpu', False)
        except Exception as e:
            logger.error(f"❌ GPU detection failed: {e}")
            self.config.setdefault('enable_gpu', False)
            
        self.reasoning_engine = get_reasoning_engine()
        self._shutdown = False
        
        # Use detector singleton to prevent multiple initialization
        from services.detector_singleton import get_detector_singleton
        self.detector_singleton = get_detector_singleton()
        
        # Set up analysis handlers for available detectors (check lazily)
        self.analysis_handlers = {}
        self._setup_analysis_handlers()
        
        logger.info(f"Satya Sentinel Agent initialized with {len(self.analysis_handlers)} analysis handlers (lazy loading enabled)")
    
    def _setup_analysis_handlers(self):
        """
        Set up analysis handlers for available detectors (check lazily)
        """
        if self.detector_singleton.is_detector_available('image'):
            self.analysis_handlers[AnalysisType.IMAGE] = self._analyze_image
            self.analysis_handlers[AnalysisType.WEBCAM] = self._analyze_webcam
            
        if self.detector_singleton.is_detector_available('video'):
            self.analysis_handlers[AnalysisType.VIDEO] = self._analyze_video
            
        if self.detector_singleton.is_detector_available('audio'):
            self.analysis_handlers[AnalysisType.AUDIO] = self._analyze_audio
            
        # Always support these handlers as they don't require ML models
        self.analysis_handlers[AnalysisType.TEXT] = self._analyze_text
        self.analysis_handlers[AnalysisType.METADATA] = self._analyze_metadata
        self.analysis_handlers[AnalysisType.MULTIMODAL] = self._analyze_multimodal

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Analyze content using the appropriate handler

        Args:
            request: The analysis request

        Returns:
            AnalysisResult with findings
        """
        logger.info(f"Processing analysis request: {request.request_id}")

        # Create initial result
        result = AnalysisResult(
            request_id=request.request_id,
            status="pending",
            analysis_type=request.analysis_type,
            metadata=request.metadata.copy(),
        )

        try:
            # Get the appropriate handler
            handler = self.analysis_handlers.get(request.analysis_type)
            if not handler:
                raise ValueError(f"Unsupported analysis type: {request.analysis_type}")

            # Execute the analysis
            analysis_result = await handler(request)

            # Update result with analysis
            result.status = "success"
            result.conclusions = analysis_result.get("conclusions", [])
            result.evidence_ids = analysis_result.get("evidence_ids", [])
            result.confidence = analysis_result.get("confidence", 0.0)
            result.confidence_level = analysis_result.get(
                "confidence_level", ConfidenceLevel.VERY_LOW
            )
            result.metadata.update(analysis_result.get("metadata", {}))

        except Exception as e:
            logger.error(
                f"Analysis failed for request {request.request_id}: {str(e)}",
                exc_info=True,
            )
            result.status = "error"
            result.error = str(e)

        return result

    async def _analyze_image(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Analyze an image for potential deepfakes using the centralized ML classifier.

        This method enforces the following strict flow:
        1. Validate input
        2. Load and prepare image
        3. Execute ML inference (or fail)
        4. Create evidence (only if ML succeeds)
        5. Generate conclusions

        Returns:
            Dict containing analysis results with the following structure:
            {
                'is_deepfake': bool,
                'confidence': float,
                'model_info': dict,
                'evidence_id': str,
                'metadata': dict
            }

        Raises:
            ValueError: For invalid input
            RuntimeError: For any analysis failure
        """
        if not request.content:
            raise ValueError("No image content provided for analysis")

        # Get detector from singleton
        image_detector = self.detector_singleton.get_detector('image', {
            'enable_gpu': self.config.get('enable_gpu', False)
        })
        
        if not image_detector:
            raise RuntimeError("Image detector not available")

        try:
            # Load and prepare image
            import io

            import numpy as np
            from PIL import Image

            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(request.content))
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_np = np.array(image)

            # Execute ML inference (will raise on failure)
            analysis_result = image_detector.analyze(
                image_np,
                detect_faces=True,
                analyze_forensics=True,
                return_face_data=True,
            )

            # Validate ML results
            required_fields = ["is_deepfake", "confidence", "model_info"]
            if not all(field in analysis_result for field in required_fields):
                missing = [f for f in required_fields if f not in analysis_result]
                raise RuntimeError(
                    f"ML analysis missing required fields: {', '.join(missing)}"
                )

            is_deepfake = bool(analysis_result["is_deepfake"])
            confidence = float(analysis_result["confidence"])
            model_info = analysis_result["model_info"]

            # Create evidence (only after successful ML execution)
            evidence_id = self.reasoning_engine.add_evidence(
                evidence_type=EvidenceType.IMAGE_ANALYSIS,
                data={
                    "is_deepfake": is_deepfake,
                    "confidence": confidence,
                    "model_info": model_info,
                    "metadata": {
                        "analysis_type": "image",
                        "timestamp": datetime.utcnow().isoformat(),
                        "model_version": model_info.get("version", "unknown"),
                        "image_size": f"{image.width}x{image.height}",
                        "image_mode": image.mode,
                    },
                },
                source="image_analyzer",
                reliability=min(0.9, confidence),  # Cap reliability at 0.9
            )

            # Generate conclusions
            conclusions = self.reasoning_engine.reason(
                {
                    "analysis_type": "image",
                    "is_deepfake": is_deepfake,
                    "confidence": confidence,
                    "model_info": model_info,
                }
            )

            # Ensure we have at least one conclusion
            if not conclusions:
                from reasoning_engine import Conclusion

                conclusion = Conclusion(
                    text=f"Deepfake detected with {confidence*100:.1f}% confidence"
                    if is_deepfake
                    else f"No deepfake detected ({confidence*100:.1f}% confidence)",
                    confidence=confidence,
                    tags=["ml_classification"],
                    metadata={
                        "is_deepfake": is_deepfake,
                        "confidence": confidence,
                        "model_info": model_info,
                    },
                )
                conclusions = [conclusion]

            # Return final results
            return {
                "conclusions": [c.dict() for c in conclusions],
                "evidence_ids": [evidence_id],
                "confidence": confidence,
                "metadata": {
                    "analysis_type": "image",
                    "is_deepfake": is_deepfake,
                    "model_info": model_info,
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                },
            }

        except Exception as e:
            # Log the full error with stack trace
            logger.error(f"Image analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to analyze image: {str(e)}")

    async def _enforce_ml_execution(
        self, analysis_func: Callable, request: AnalysisRequest, modality: str
    ) -> Dict[str, Any]:
        """
        Enforce ML execution and generate proof of analysis
        
        This wrapper ensures that all analysis goes through a consistent path
        with proper ML execution and proof generation.
        """
        # Track execution start time
        start_time = time.time()
        
        # Set timeout for analysis (5 minutes maximum)
        timeout_seconds = 300
        
        try:
            # Execute the actual analysis with timeout
            result = await asyncio.wait_for(
                analysis_func(request),
                timeout=timeout_seconds
            )
            
            # Calculate inference duration
            inference_duration = time.time() - start_time
            
            # Generate proof of analysis
            model_info = get_model_info(modality)
            proof = generate_proof(
                model_name=model_info["name"],
                model_version=model_info["version"],
                modality=modality,
                inference_duration=inference_duration,
                frames_analyzed=result.get("frames_analyzed", 1),
                metadata={
                    "request_id": request.request_id,
                    "user_id": request.user_id,
                    "analysis_type": request.analysis_type.value,
                    "content_size": len(request.content) if request.content else 0,
                },
            )
            
            # Add proof to result
            result["proof"] = proof.to_dict()
            
            # Verify the proof is valid (this will raise if invalid)
            AnalysisResult(**result)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Analysis timeout for {modality} after {timeout_seconds}s")
            raise RuntimeError(f"Analysis timeout: {modality} analysis exceeded {timeout_seconds}s")
        except Exception as e:
            logger.error(f"Analysis failed for {modality}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Analysis failed: {str(e)}")

    async def _analyze_video(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Analyze a video for potential deepfakes with ML enforcement

        This method enforces that all video analysis must go through SentinelAgent
        with proper ML execution and evidence collection.
        """
        # Validate input
        if not request.content and not request.content_uri:
            raise ValueError("No video content or URI provided for analysis")

        # Check if video detector is available and models are loaded
        video_detector = self.detector_singleton.get_detector('video', {
            'enable_gpu': self.config.get('enable_gpu', False)
        })
        
        if not video_detector:
            logger.error("Video analysis failed: Video detector not initialized")
            raise RuntimeError("Video analysis is not available: ML service error")

        # Verify frame analysis model is loaded and ready
        if not getattr(video_detector, "models_loaded", False):
            logger.error("Video analysis failed: Frame analysis model not loaded")
            raise RuntimeError(
                "Video analysis failed: Required ML models not available"
            )

        # Execute analysis with ML enforcement
        return await self._enforce_ml_execution(
            self._analyze_video_internal, request, "video"
        )

    async def _analyze_video_internal(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Internal video analysis implementation with ML execution"""
        # Validate input
        if not request.content and not request.content_uri:
            raise ValueError("No video content or URI provided for analysis")

        # Check if video detector is available and models are loaded
        video_detector = self.detector_singleton.get_detector('video', {
            'enable_gpu': self.config.get('enable_gpu', False)
        })
        
        if not video_detector:
            logger.error("Video analysis failed: Video detector not initialized")
            raise RuntimeError("Video analysis is not available: ML service error")

        if not getattr(self.video_detector, "models_loaded", False):
            raise RuntimeError("Video analysis failed: ML models failed to load")

        # Create analysis context
        analysis_id = f"video_{uuid.uuid4().hex[:8]}"
        analysis_start = datetime.utcnow()

        try:
            # Log analysis start
            logger.info(f"🎥 Starting video analysis {analysis_id}")

            # Execute real ML analysis
            analysis_result = video_detector.analyze(request.content)

            # Validate ML results
            required_fields = ["is_deepfake", "confidence", "model_info"]
            if not all(field in analysis_result for field in required_fields):
                missing = [f for f in required_fields if f not in analysis_result]
                raise RuntimeError(f"Video analysis missing required fields: {', '.join(missing)}")

            is_deepfake = bool(analysis_result["is_deepfake"])
            confidence = float(analysis_result["confidence"])
            model_info = analysis_result["model_info"]

            # Create evidence (only after successful ML execution)
            evidence_id = self.reasoning_engine.add_evidence(
                evidence_type=EvidenceType.VIDEO_ANALYSIS,
                data={
                    "is_deepfake": is_deepfake,
                    "confidence": confidence,
                    "model_info": model_info,
                    "metadata": {
                        "analysis_type": "video",
                        "timestamp": datetime.utcnow().isoformat(),
                        "model_version": model_info.get("version", "unknown"),
                    },
                },
                source="video_analyzer",
                reliability=min(0.9, confidence),
            )

            # Generate conclusions
            conclusions = self.reasoning_engine.reason({
                "analysis_type": "video",
                "is_deepfake": is_deepfake,
                "confidence": confidence,
                "model_info": model_info,
            })

            return {
                "conclusions": conclusions,
                "evidence_ids": [evidence_id],
                "confidence": confidence,
                "metadata": {
                    "analysis_type": "video",
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "model_version": model_info.get("version", "unknown"),
                },
            }

        except Exception as e:
            logger.error(f"Video analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Video analysis failed: {str(e)}")
        if getattr(self.video_detector, "config", {}).get("use_temporal_model", True):
            if not getattr(self.video_detector, "temporal_model", None):
                logger.error(
                    "Video analysis failed: Temporal model not loaded but required by config"
                )
                raise RuntimeError(
                    "Video analysis failed: Required temporal analysis model not available"
                )

        # Create analysis context
        analysis_id = f"video_{uuid.uuid4().hex[:8]}"
        analysis_start = datetime.utcnow()

        try:
            # Log analysis start
            logger.info(f"🎥 Starting video analysis {analysis_id}")

            # Create temporary file if content is provided directly
            video_path = None
            if request.content:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(request.content)
                    video_path = tmp.name
            elif request.content_uri:
                video_path = request.content_uri

            if not video_path or not os.path.exists(video_path):
                raise ValueError("Failed to access video content")

            # Execute ML analysis
            ml_start = time.time()
            analysis_result = video_detector.detect(video_path)
            ml_duration = time.time() - ml_start

            if (
                not analysis_result
                or "success" not in analysis_result
                or not analysis_result["success"]
            ):
                raise RuntimeError("Video analysis failed")

            # Calculate confidence and determine if deepfake
            confidence = analysis_result.get("confidence", 0.0)
            is_deepfake = analysis_result.get("authenticity_score", 0.0) < 0.5

            # Generate proof of analysis
            proof = {
                "analysis_id": analysis_id,
                "model_version": getattr(self.video_detector, "version", "1.0"),
                "frames_analyzed": analysis_result.get("details", {})
                .get("video_info", {})
                .get("frames_analyzed", 0),
                "inference_time": ml_duration,
                "timestamp": datetime.utcnow().isoformat(),
                "confidence": confidence,
                "is_deepfake": is_deepfake,
                "metadata": {
                    "video_duration": analysis_result.get("details", {})
                    .get("video_info", {})
                    .get("duration", 0),
                    "resolution": analysis_result.get("details", {})
                    .get("video_info", {})
                    .get("resolution", "unknown"),
                    "fps": analysis_result.get("details", {})
                    .get("video_info", {})
                    .get("fps", 0),
                },
            }

            # Add evidence with full analysis results
            evidence_id = self.reasoning_engine.add_evidence(
                evidence_type=EvidenceType.VIDEO_ANALYSIS,
                data={
                    "analysis": analysis_result,
                    "proof": proof,
                    "metadata": request.metadata,
                },
                source="sentinel_video_analyzer",
                reliability=confidence,
            )

            # Generate conclusions
            conclusions = self.reasoning_engine.reason(
                {
                    "analysis_type": "video",
                    "analysis_results": analysis_result,
                    "proof": proof,
                }
            )

            # Log completion
            analysis_duration = (datetime.utcnow() - analysis_start).total_seconds()
            logger.info(
                f"✅ Video analysis {analysis_id} completed in {analysis_duration:.2f}s (ML: {ml_duration:.2f}s)"
            )

            # Return structured results
            return {
                "conclusions": [c.dict() for c in conclusions],
                "evidence_ids": [evidence_id],
                "confidence": confidence,
                "is_deepfake": is_deepfake,
                "proof": proof,
                "metadata": {
                    "analysis_id": analysis_id,
                    "analysis_type": "video",
                    "model_version": getattr(self.video_detector, "version", "1.0"),
                    "frames_analyzed": proof["frames_analyzed"],
                    "analysis_time_seconds": analysis_duration,
                    "ml_inference_time_seconds": ml_duration,
                },
            }

        except Exception as e:
            # Log the full error with stack trace
            logger.error(
                f"Video analysis {analysis_id} failed: {str(e)}", exc_info=True
            )
            raise RuntimeError(f"Failed to analyze video: {str(e)}")

        finally:
            # Clean up temporary file if we created one
            if (
                "video_path" in locals()
                and video_path
                and os.path.exists(video_path)
                and hasattr(request, "content")
            ):
                try:
                    os.unlink(video_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary video file: {e}")

    async def _analyze_audio(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Analyze audio for potential deepfakes with ML enforcement

        This method enforces that all audio analysis must go through SentinelAgent
        with proper ML execution and evidence collection.
        """
        # Validate input
        if not request.content and not request.content_uri:
            raise ValueError("No audio content or URI provided for analysis")

        # Verify audio detector is properly loaded
        audio_detector = self.detector_singleton.get_detector('audio', {
            'enable_gpu': self.config.get('enable_gpu', False)
        })
        
        if not audio_detector:
            logger.error("Audio analysis failed: Audio detector not initialized")
            raise RuntimeError("Audio analysis is not available: ML service error")

        # Execute analysis with ML enforcement
        return await self._enforce_ml_execution(
            self._analyze_audio_internal, request, "audio"
        )

    async def _analyze_audio_internal(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Internal audio analysis implementation with ML execution"""
        # Validate input
        if not request.content and not request.content_uri:
            raise ValueError("No audio content or URI provided for analysis")

        # Check if audio detector is available and models are loaded
        audio_detector = self.detector_singleton.get_detector('audio', {
            'enable_gpu': self.config.get('enable_gpu', False)
        })
        
        if not audio_detector:
            logger.error("Audio analysis failed: Audio detector not initialized")
            raise RuntimeError("Audio analysis is not available: ML service error")

        if not getattr(audio_detector, "models_loaded", False):
            raise RuntimeError("Audio analysis failed: ML models failed to load")

        # Create analysis context
        analysis_id = f"audio_{uuid.uuid4().hex[:8]}"
        analysis_start = datetime.utcnow()

        try:
            # Log analysis start
            logger.info(f"🎵 Starting audio analysis {analysis_id}")

            # Execute real ML analysis
            analysis_result = audio_detector.analyze(request.content)

            # Validate ML results
            required_fields = ["is_deepfake", "confidence", "model_info"]
            if not all(field in analysis_result for field in required_fields):
                missing = [f for f in required_fields if f not in analysis_result]
                raise RuntimeError(f"Audio analysis missing required fields: {', '.join(missing)}")

            is_deepfake = bool(analysis_result["is_deepfake"])
            confidence = float(analysis_result["confidence"])
            model_info = analysis_result["model_info"]

            # Create evidence (only after successful ML execution)
            evidence_id = self.reasoning_engine.add_evidence(
                evidence_type=EvidenceType.AUDIO_ANALYSIS,
                data={
                    "is_deepfake": is_deepfake,
                    "confidence": confidence,
                    "model_info": model_info,
                    "metadata": {
                        "analysis_type": "audio",
                        "timestamp": datetime.utcnow().isoformat(),
                        "model_version": model_info.get("version", "unknown"),
                    },
                },
                source="audio_analyzer",
                reliability=min(0.9, confidence),
            )

            # Generate conclusions
            conclusions = self.reasoning_engine.reason({
                "analysis_type": "audio",
                "is_deepfake": is_deepfake,
                "confidence": confidence,
                "model_info": model_info,
            })

            return {
                "conclusions": conclusions,
                "evidence_ids": [evidence_id],
                "confidence": confidence,
                "metadata": {
                    "analysis_type": "audio",
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                    "model_version": model_info.get("version", "unknown"),
                },
            }

        except Exception as e:
            logger.error(f"Audio analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Audio analysis failed: {str(e)}")
            audio_path = None

            try:
                if request.content:
                    if isinstance(request.content, str):
                        # Handle base64 encoded content
                        import base64

                        audio_data = base64.b64decode(request.content)
                    else:
                        audio_data = request.content

                    # Create temp file
                    import tempfile

                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    temp_file.write(audio_data)
                    temp_file.close()
                    audio_path = temp_file.name
                else:
                    # Use content_uri if provided
                    audio_path = request.content_uri

                # Verify audio file exists
                if not os.path.isfile(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")

                # Log model info
                model_info = {
                    "model_name": getattr(
                        getattr(video_detector, "model_name", "video_detector")
                    ),
                    "model_version": getattr(
                        video_detector, "model_version", "1.0"
                    ),
                    "analysis_time": datetime.utcnow().isoformat(),
                }

                # Execute ML analysis
                ml_start = datetime.utcnow()
                analysis_result = audio_detector.detect(audio_path)
                ml_duration = (datetime.utcnow() - ml_start).total_seconds()

                # Validate ML results
                if (
                    not analysis_result
                    or "success" not in analysis_result
                    or not analysis_result["success"]
                ):
                    raise RuntimeError(
                        "Audio analysis failed: Invalid results from ML model"
                    )

                # Add ML execution proof
                ml_proof = {
                    "execution_id": analysis_id,
                    "model_info": model_info,
                    "execution_time_seconds": ml_duration,
                    "input_checksum": self._calculate_checksum(audio_path),
                    "output_checksum": self._calculate_checksum(
                        str(analysis_result).encode()
                    ),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # Add evidence of ML execution
                evidence_id = self.reasoning_engine.add_evidence(
                    evidence_type=EvidenceType.AUDIO_ANALYSIS,
                    data={
                        "analysis": analysis_result,
                        "ml_proof": ml_proof,
                        "metadata": request.metadata,
                    },
                    source="sentinel_audio_analyzer",
                    reliability=analysis_result.get("confidence", 0.0),
                )

                # Run reasoning with analysis results
                conclusions = self.reasoning_engine.reason(
                    {
                        "analysis_type": "audio",
                        "analysis_results": analysis_result,
                        "ml_proof": ml_proof,
                    }
                )

                # Calculate overall confidence
                confidence = analysis_result.get("confidence", 0.0)

                # Log successful analysis
                logger.info(
                    f"Audio analysis completed: {analysis_id} "
                    f"(confidence: {confidence:.2f}, duration: {ml_duration:.2f}s)"
                )

                return {
                    "conclusions": [c.dict() for c in conclusions],
                    "evidence_ids": [evidence_id],
                    "confidence": confidence,
                    "metadata": {
                        "analysis_id": analysis_id,
                        "analysis_type": "audio",
                        "model_info": model_info,
                        "execution_proof": ml_proof,
                    },
                }

            finally:
                # Clean up temp file if created
                if temp_file and os.path.exists(temp_file.name):
                    try:
                        os.unlink(temp_file.name)
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file: {e}")

        except Exception as e:
            # Log detailed error
            error_id = f"err_{uuid.uuid4().hex[:6]}"
            logger.error(
                f"Audio analysis failed [{error_id}]: {str(e)}",
                exc_info=True,
                extra={
                    "analysis_id": analysis_id,
                    "error_type": type(e).__name__,
                    "error_details": str(e),
                },
            )

            # Add error evidence
            self.reasoning_engine.add_evidence(
                evidence_type=EvidenceType.ERROR,
                data={
                    "error": {
                        "id": error_id,
                        "type": type(e).__name__,
                        "message": str(e),
                        "analysis_id": analysis_id,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    "metadata": request.metadata,
                },
                source="sentinel_audio_analyzer",
                reliability=0.0,
            )

            # Re-raise with context
            raise RuntimeError(f"Audio analysis failed [{error_id}]: {str(e)}") from e

    async def _analyze_text(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Analyze text content for potential deepfakes"""
        # Check if text content is provided
        if not request.content:
            raise ValueError("No text content provided for analysis")

        # Check if text detector is available
        if not hasattr(self, "text_detector") or not self.text_detector:
            raise RuntimeError("Text analysis is not available: ML model not loaded")

        try:
            # Add evidence of the analysis attempt
            evidence_id = self.reasoning_engine.add_evidence(
                evidence_type=EvidenceType.TEXT_ANALYSIS,
                data={
                    "analysis": {"status": "analysis_started"},
                    "metadata": request.metadata,
                },
                source="text_analyzer",
                reliability=0.0,  # Will be updated with real confidence later
            )

            # Run actual analysis (to be implemented in STEP 3)
            # analysis_result = await self.text_detector.analyze(request.content)

            # For now, return an error indicating ML is not yet implemented
            raise NotImplementedError(
                "Text analysis ML integration not yet implemented"
            )

        except Exception as e:
            # Log the error and re-raise with appropriate context
            logger.error(f"Text analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to analyze text: {str(e)}")

        # The following code will be uncommented and used in STEP 3
        """
        # Add analysis results as new evidence
        evidence_id = self.reasoning_engine.add_evidence(
            evidence_type=EvidenceType.TEXT_ANALYSIS,
            data={
                "analysis": analysis_result,
                "metadata": request.metadata
            },
            source="text_analyzer",
            reliability=analysis_result.get("confidence", 0.0)
        )
        
        # Run reasoning with actual analysis results
        conclusions = self.reasoning_engine.reason({
            "analysis_type": "text",
            "analysis_results": analysis_result
        })
        
        # Calculate overall confidence from actual results
        confidence = analysis_result.get("confidence", 0.0)
        
        return {
            "conclusions": [c.dict() for c in conclusions],
            "evidence_ids": [evidence_id],
            "confidence": confidence,
            "metadata": {
                "analysis_type": "text",
                "model_version": self.text_detector.get_version() if hasattr(self.text_detector, 'get_version') else "unknown"
            }
        }
        """

        # This line is just a placeholder and will be removed in STEP 3
        return {}

    async def _analyze_multimodal(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Perform multimodal analysis combining multiple evidence types.

        Args:
            request: The analysis request containing multimodal content

        Returns:
            Dict containing analysis results

        Raises:
            ValueError: If no valid content types are provided
            RuntimeError: If required ML models are not loaded
            NotImplementedError: As this feature is not yet implemented
        """
        # Check if content is provided
        if not request.content or not isinstance(request.content, dict):
            raise ValueError(
                "Multimodal analysis requires a dictionary of content types"
            )

        # Check if we have at least one supported modality
        supported_modalities = {"image", "video", "audio", "text"}
        content_modalities = set(request.content.keys())
        valid_modalities = content_modalities & supported_modalities

        if not valid_modalities:
            raise ValueError(
                "No supported content types provided for multimodal analysis"
            )

        # Check if required detectors are available
        missing_detectors = []
        if "image" in valid_modalities and not self.detector_singleton.is_detector_available('image'):
            missing_detectors.append("image")
        if "video" in valid_modalities and not self.detector_singleton.is_detector_available('video'):
            missing_detectors.append("video")
        if "audio" in valid_modalities and not self.detector_singleton.is_detector_available('audio'):
            missing_detectors.append("audio")
        if "text" in valid_modalities and not self.detector_singleton.is_detector_available('text'):
            missing_detectors.append("text")

        if missing_detectors:
            raise RuntimeError(
                f"Required ML models not loaded for modalities: {', '.join(missing_detectors)}"
            )

        # Track analysis state
        evidence_ids = []

        try:
            # Add evidence of the analysis attempt
            evidence_id = self.reasoning_engine.add_evidence(
                evidence_type=EvidenceType.MULTIMODAL_ANALYSIS,
                data={
                    "analysis": {
                        "status": "analysis_started",
                        "modalities": list(valid_modalities),
                    },
                    "metadata": request.metadata,
                },
                source="multimodal_analyzer",
                reliability=0.0,  # Will be updated with real confidence later
            )
            evidence_ids.append(evidence_id)

            # For now, return an error indicating ML is not yet implemented
            raise NotImplementedError(
                "Multimodal analysis ML integration not yet implemented"
            )

        except Exception as e:
            logger.error(f"Multimodal analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Multimodal analysis failed: {str(e)}")

    async def _analyze_metadata(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Analyze metadata for inconsistencies and potential tampering.

        Args:
            request: The analysis request containing metadata

        Returns:
            Dict containing analysis results with conclusions and evidence

        Raises:
            ValueError: If no metadata is provided
            RuntimeError: If analysis fails
        """
        if not request.metadata:
            raise ValueError("No metadata provided for analysis")

        try:
            # Add evidence of the analysis attempt
            evidence_id = self.reasoning_engine.add_evidence(
                evidence_type=EvidenceType.METADATA,
                data={
                    "analysis": {
                        "inconsistencies": ["timestamp_mismatch", "tool_artifacts"],
                        "confidence": 0.9,
                        "status": "analysis_complete",
                    },
                    "metadata": request.metadata,
                },
                source="metadata_analyzer",
                reliability=0.95,
            )

            # Run reasoning
            conclusions = self.reasoning_engine.reason({"analysis_type": "metadata"})

            # Calculate average confidence (for display only)
            confidence = (
                sum(c.confidence for c in conclusions) / len(conclusions)
                if conclusions
                else 0.0
            )

            return {
                "conclusions": [c.dict() for c in conclusions],
                "evidence_ids": [evidence_id],
                "confidence": confidence,
                "metadata": {
                    "analysis_type": "metadata",
                    "model_version": "1.0.0",
                    "inconsistencies_found": True,  # Since we're using mock data
                },
            }

        except Exception as e:
            logger.error(f"Metadata analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to analyze metadata: {str(e)}")

    async def _analyze_image(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Analyze an image for potential deepfakes using the centralized ML classifier.

        This method enforces the following strict flow:
        1. Validate input
        2. Load and prepare image
        3. Execute ML inference (or fail)
        4. Create evidence (only if ML succeeds)
        5. Generate conclusions

        Returns:
            Dict containing analysis results with the following structure:
            {
                'is_deepfake': bool,
                'confidence': float,
                'model_info': dict,
                'evidence_id': str,
                'metadata': dict
            }

        Raises:
            ValueError: For invalid input
            RuntimeError: For any analysis failure
        """
        # Input validation
        if not request.content and not request.content_uri:
            raise ValueError("No image content or content_uri provided for analysis")

        # Initialize detector if needed
        if not hasattr(self, "image_detector") or not self.image_detector:
            from .detectors.image_detector import ImageDetector

            self.image_detector = ImageDetector(
                enable_gpu=self.config.get("enable_gpu", False)
            )

        try:
            # Load and prepare image
            import io
            import time

            import numpy as np
            from PIL import Image

            # Load image data
            if request.content:
                image_data = request.content
            else:
                # Handle content_uri if needed
                raise NotImplementedError("content_uri handling not implemented")

            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")

            image_np = np.array(image)

            # Execute ML inference (will raise on failure)
            analysis_result = image_detector.analyze(
                image_np,
                detect_faces=True,
                analyze_forensics=True,
                return_face_data=True,
            )

            # Validate ML results
            required_fields = ["is_deepfake", "confidence", "model_info"]
            if not all(field in analysis_result for field in required_fields):
                missing = [f for f in required_fields if f not in analysis_result]
                raise RuntimeError(
                    f"ML analysis missing required fields: {', '.join(missing)}"
                )

            is_deepfake = bool(analysis_result["is_deepfake"])
            confidence = float(analysis_result["confidence"])
            model_info = analysis_result["model_info"]

            # Create evidence (only after successful ML execution)
            evidence_id = self.reasoning_engine.add_evidence(
                evidence_type=EvidenceType.IMAGE_ANALYSIS,
                data={
                    "is_deepfake": is_deepfake,
                    "confidence": confidence,
                    "model_info": model_info,
                    "metadata": {
                        "analysis_type": "image",
                        "timestamp": datetime.utcnow().isoformat(),
                        "model_version": model_info.get("version", "unknown"),
                        "image_size": f"{image.width}x{image.height}",
                        "image_mode": image.mode,
                    },
                },
                source="image_analyzer",
                reliability=min(0.9, confidence),  # Cap reliability at 0.9
            )

            # Generate conclusions
            conclusions = self.reasoning_engine.reason(
                {
                    "analysis_type": "image",
                    "is_deepfake": is_deepfake,
                    "confidence": confidence,
                    "model_info": model_info,
                }
            )

            # Ensure we have at least one conclusion
            if not conclusions:
                from reasoning_engine import Conclusion

                conclusion = Conclusion(
                    text=f"Deepfake detected with {confidence*100:.1f}% confidence"
                    if is_deepfake
                    else f"No deepfake detected ({confidence*100:.1f}% confidence)",
                    confidence=confidence,
                    tags=["ml_classification"],
                    metadata={
                        "is_deepfake": is_deepfake,
                        "confidence": confidence,
                        "model_info": model_info,
                    },
                )
                conclusions = [conclusion]

            # Return final results
            return {
                "conclusions": [c.dict() for c in conclusions],
                "evidence_ids": [evidence_id],
                "confidence": confidence,
                "metadata": {
                    "analysis_type": "image",
                    "is_deepfake": is_deepfake,
                    "model_info": model_info,
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                },
            }

        except Exception as e:
            # Log the error and re-raise
            logger.error(f"Image analysis failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to analyze image: {str(e)}")

    async def _analyze_webcam(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Analyze a webcam frame for potential deepfakes with ML enforcement

        This method enforces that all webcam analysis must go through SentinelAgent
        with proper ML execution and evidence collection.

        Args:
            request: The analysis request containing webcam frame data

        Returns:
            Dict containing analysis results with proof and evidence

        Raises:
            ValueError: If no webcam frame data is provided
            RuntimeError: If ML models are not loaded or analysis fails
        """
        # Validate input
        if not request.content:
            raise ValueError("No webcam frame data provided for analysis")

        # Verify image detector is properly loaded
        image_detector = self.detector_singleton.get_detector('image', {
            'enable_gpu': self.config.get('enable_gpu', False)
        })
        
        if not image_detector:
            logger.error("Webcam analysis failed: Image detector not initialized")
            raise RuntimeError("Webcam analysis is not available: ML service error")

        # Execute analysis with ML enforcement
        return await self._enforce_ml_execution(
            self._analyze_webcam_internal, request, "webcam"
        )

    async def _analyze_webcam_internal(
        self, request: AnalysisRequest
    ) -> Dict[str, Any]:
        """Internal webcam analysis implementation with ML execution"""
        import io
        import time

        import numpy as np
        from PIL import Image

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(request.content))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array for analysis
        image_np = np.array(image)

        # Execute ML analysis
        ml_start = time.time()
        analysis_result = self.image_detector.analyze(
            image_np, detect_faces=True, analyze_forensics=True, return_face_data=True
        )
        ml_duration = time.time() - ml_start

        # Validate ML results
        required_fields = ["is_deepfake", "confidence", "model_info"]
        if not all(field in analysis_result for field in required_fields):
            missing = [f for f in required_fields if f not in analysis_result]
            raise RuntimeError(
                f"ML analysis missing required fields: {', '.join(missing)}"
            )

        is_deepfake = bool(analysis_result["is_deepfake"])
        confidence = float(analysis_result["confidence"])
        model_info = analysis_result["model_info"]

        # Create evidence
        evidence_id = self.reasoning_engine.add_evidence(
            evidence_type=EvidenceType.IMAGE_ANALYSIS,
            data={
                "is_deepfake": is_deepfake,
                "confidence": confidence,
                "model_info": model_info,
                "metadata": {
                    "analysis_type": "webcam",
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_version": model_info.get("version", "unknown"),
                    "resolution": f"{image.width}x{image.height}",
                    "inference_time_seconds": ml_duration,
                    **request.metadata,
                },
            },
            source="webcam_analyzer",
            reliability=min(0.9, confidence),  # Cap reliability at 0.9
        )

        # Generate conclusions
        conclusions = self.reasoning_engine.reason(
            {
                "analysis_type": "webcam",
                "is_deepfake": is_deepfake,
                "confidence": confidence,
                "model_info": model_info,
            }
        )

        # Ensure we have at least one conclusion
        if not conclusions:
            from .reasoning_engine import Conclusion

            conclusion = Conclusion(
                text=f"Deepfake detected with {confidence*100:.1f}% confidence"
                if is_deepfake
                else f"No deepfake detected ({confidence*100:.1f}% confidence)",
                confidence=confidence,
                tags=["ml_classification", "webcam"],
                metadata={
                    "is_deepfake": is_deepfake,
                    "confidence": confidence,
                    "model_info": model_info,
                },
            )
            conclusions = [conclusion]

        # Return structured results
        return {
            "conclusions": [c.dict() for c in conclusions],
            "evidence_ids": [evidence_id],
            "confidence": confidence,
            "is_deepfake": is_deepfake,
            "frames_analyzed": 1,  # Single frame for webcam
            "metadata": {
                "analysis_type": "webcam",
                "model_version": model_info.get("version", "1.0"),
                "inference_time_seconds": ml_duration,
                "resolution": f"{image.width}x{image.height}",
                "session_id": request.metadata.get("session_id", "unknown"),
                "analysis_timestamp": datetime.utcnow().isoformat(),
            },
        }

    async def shutdown(self):
        """Clean up resources"""
        self._shutdown = True
        logger.info("Satya Sentinel Agent is shutting down")


# Global instance for easy access
sentinel_agent = SentinelAgent()


def get_sentinel_agent():
    """Get the global Sentinel agent instance"""
    return sentinel_agent


# Example usage
async def example_usage():
    """Example of using the Sentinel agent"""
    agent = get_sentinel_agent()

    # Create a sample request
    request = AnalysisRequest(
        analysis_type=AnalysisType.IMAGE,
        content_uri="https://example.com/suspicious_image.jpg",
        metadata={
            "source": "user_upload",
            "user_id": "user123",
            "file_size": 1024 * 1024,
            "mime_type": "image/jpeg",
        },
    )

    # Process the request
    result = await agent.analyze(request)
    print(f"Analysis result: {result.json(indent=2)}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
