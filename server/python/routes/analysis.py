"""
Analysis Routes
Handle analysis history and statistics with database integration
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional
import base64
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, field_validator

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.analysis import AnalysisResult
from services.database import get_db_manager
from sentinel_agent import AnalysisRequest, AnalysisType

logger = logging.getLogger(__name__)

router = APIRouter()

# Unified Request/Response models
class UnifiedAnalysisRequest(BaseModel):
    """Standardized request format for all media types"""
    media_type: str  # "image", "video", or "audio"
    data_base64: str  # base64 encoded file content
    mime_type: str
    filename: str
    
    @field_validator('media_type')
    @classmethod
    def validate_media_type(cls, v):
        allowed_types = ['image', 'video', 'audio']
        if v not in allowed_types:
            raise ValueError(f"media_type must be one of {allowed_types}")
        return v
    
    @field_validator('data_base64')
    @classmethod
    def validate_base64(cls, v):
        if not v:
            raise ValueError("data_base64 cannot be empty")
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("data_base64 must be valid base64")
        return v
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v):
        if not v:
            raise ValueError("filename cannot be empty")
        if len(v) > 255:
            raise ValueError("filename cannot exceed 255 characters")
        return v

class UnifiedAnalysisResponse(BaseModel):
    """Standardized response format for all media types"""
    success: bool
    media_type: str
    fake_score: float  # 0.0 - 1.0
    label: str  # "Deepfake", "Authentic", or "Unknown"
    processing_time: float  # milliseconds
    error: str | None = None
    # Optional detailed fields
    model_name: str | None = None
    model_version: str | None = None
    confidence: float | None = None
    proof: dict | None = None
    metadata: dict | None = None
    
    @field_validator('fake_score')
    @classmethod
    def validate_fake_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("fake_score must be between 0.0 and 1.0")
        return v
    
    @field_validator('label')
    @classmethod
    def validate_label(cls, v: str) -> str:
        allowed_labels = ['Deepfake', 'Authentic', 'Unknown']
        if v not in allowed_labels:
            raise ValueError(f"label must be one of {allowed_labels}")
        return v
    
    @field_validator('processing_time')
    @classmethod
    def validate_processing_time(cls, v: float) -> float:
        if v < 0:
            raise ValueError("processing_time cannot be negative")
        return v

# Legacy models for backward compatibility
class AnalysisRequestModel(BaseModel):
    image: str  # base64 encoded
    mimeType: str
    jobId: str
    filename: str

class AnalysisResponse(BaseModel):
    status: str
    is_deepfake: bool
    confidence: float
    model_name: str
    model_version: str
    summary: dict
    proof: dict

@router.post("/analyze/{media_type}")
async def analyze_unified_media(
    media_type: str,
    file: UploadFile = File(...),
    job_id: str = Form(...),
    user_id: str = Form(None),
    request: Request = None
):
    """Unified media analysis endpoint - returns inference-only payload"""
    start_time = datetime.utcnow()
    logger.info(f"[INFERENCE REQUEST] {media_type} analysis for job_id: {job_id}")
    
    try:
        # Validate media type
        if media_type not in ["image", "video", "audio"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid media type: {media_type}"
            )
            
        # Read file content
        content = await file.read()
        
        # Create analysis request
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType[media_type.upper()],
            content=content,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "job_id": job_id,
                "user_id": user_id,
            },
        )
        
        # Process with ML model - inference only
        if not hasattr(request.app.state, "sentinel_agent") or request.app.state.sentinel_agent is None:
            raise HTTPException(
                status_code=503,
                detail="ML service unavailable"
            )
        
        # Execute inference with timeout
        import asyncio
        result = await asyncio.wait_for(
            request.app.state.sentinel_agent.analyze(analysis_request),
            timeout=300.0  # 5 minutes
        )
        
        if not result or not hasattr(result, "conclusions") or not result.conclusions:
            raise HTTPException(
                status_code=500,
                detail="No valid inference results"
            )
        
        # Get the primary conclusion
        primary_conclusion = max(
            result.conclusions, key=lambda c: (c.confidence, c.severity)
        )
        
        # Extract model info
        model_info = {}
        if result.conclusions:
            model_info = result.conclusions[0].metadata.get("model_info", {})
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Return inference-only payload - Node owns job lifecycle
        return {
            "confidence": float(primary_conclusion.confidence),
            "is_deepfake": primary_conclusion.is_deepfake,
            "model_name": model_info.get("name", f"SatyaAI-{media_type.capitalize()}"),
            "model_version": model_info.get("version", "1.0.0"),
            "analysis_data": {
                "processing_time": processing_time,
                "evidence_id": result.evidence_ids[0] if hasattr(result, "evidence_ids") and result.evidence_ids else file.filename,
                "media_type": media_type,
                "inference_timestamp": datetime.utcnow().isoformat(),
            },
            "proof": getattr(result, 'proof', {}),
        }
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Analysis timeout"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[INFERENCE ERROR] {media_type} analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

@router.post("/analyze/video")
async def analyze_video_unified(request: Request, data: UnifiedAnalysisRequest):
    """
    Unified video analysis endpoint - accepts standardized JSON format
    """
    start_time = datetime.utcnow()
    logger.info(f"[REQUEST RECEIVED] Video analysis for {data.filename}")
    
    try:
        # Validate ML service availability
        if not hasattr(request.app.state, "sentinel_agent"):
            logger.error("[MODEL INFERENCE FAILED] ML models not initialized")
            return UnifiedAnalysisResponse(
                success=False,
                media_type="video",
                fake_score=0.0,
                label="Unknown",
                processing_time=0,
                error="Analysis service unavailable - ML models not initialized"
            )

        # Decode base64 video
        try:
            logger.info("[FILE DECODED] Decoding base64 video data")
            video_data = base64.b64decode(data.data_base64)
        except Exception as e:
            logger.error(f"[FILE DECODED] Invalid base64 data: {str(e)}")
            return UnifiedAnalysisResponse(
                success=False,
                media_type="video",
                fake_score=0.0,
                label="Unknown",
                processing_time=0,
                error=f"Invalid video data: {str(e)}"
            )

        # Create analysis request
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.VIDEO,
            content=video_data,
            metadata={
                "filename": data.filename,
                "mime_type": data.mime_type,
                "request_id": f"req_{uuid.uuid4().hex[:8]}",
            },
        )

        # Execute analysis with timeout guard
        logger.info("[MODEL INFERENCE START] Starting video analysis")
        
        # Check if ML service is available
        if not hasattr(request.app.state, "sentinel_agent") or request.app.state.sentinel_agent is None:
            logger.error("[ML SERVICE UNAVAILABLE] SentinelAgent not initialized")
            return UnifiedAnalysisResponse(
                success=False,
                media_type="video",
                fake_score=0.0,
                label="Unknown",
                processing_time=0,
                error="ML service temporarily unavailable - please try again later"
            )
        
        try:
            # Add timeout wrapper for ML inference (5 minutes max)
            import asyncio
            result = await asyncio.wait_for(
                request.app.state.sentinel_agent.analyze(analysis_request),
                timeout=300.0  # 5 minutes
            )
            logger.info("[MODEL INFERENCE DONE] Analysis completed")
        except asyncio.TimeoutError:
            logger.error("[MODEL INFERENCE TIMEOUT] Video analysis timed out after 5 minutes")
            return UnifiedAnalysisResponse(
                success=False,
                media_type="video",
                fake_score=0.0,
                label="Unknown",
                processing_time=300000,  # 5 minutes in ms
                error="Video analysis timeout - file too large or complex. Please try with a smaller file."
            )

        if not result or not hasattr(result, "conclusions") or not result.conclusions:
            logger.error("[MODEL INFERENCE FAILED] No valid analysis results")
            return UnifiedAnalysisResponse(
                success=False,
                media_type="video",
                fake_score=0.0,
                label="Unknown",
                processing_time=0,
                error="Failed to analyze video: No valid analysis results"
            )

        # Get the primary conclusion
        primary_conclusion = max(
            result.conclusions, key=lambda c: (c.confidence, c.severity)
        )

        # Extract model info
        model_info = {}
        if result.conclusions:
            model_info = result.conclusions[0].metadata.get("model_info", {})

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        response = UnifiedAnalysisResponse(
            success=True,
            media_type="video",
            fake_score=float(primary_conclusion.confidence),
            label="Deepfake" if primary_conclusion.is_deepfake else "Authentic",
            processing_time=processing_time,
            model_name=model_info.get("name", "SatyaAI-Video"),
            model_version=model_info.get("version", "1.0.0"),
            confidence=float(primary_conclusion.confidence),
            proof=getattr(result, 'proof', {}),
            metadata={
                "processing_time": getattr(result, 'processing_time', 0),
                "frames_analyzed": getattr(result, 'frames_analyzed', 0),
                "evidence_id": result.evidence_ids[0] if hasattr(result, "evidence_ids") and result.evidence_ids else data.filename,
            }
        )

        logger.info(f"[RESPONSE SENT] Video analysis completed successfully")
        return response

    except Exception as e:
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"[MODEL INFERENCE FAILED] Video analysis failed: {str(e)}", exc_info=True)
        return UnifiedAnalysisResponse(
            success=False,
            media_type="video",
            fake_score=0.0,
            label="Unknown",
            processing_time=processing_time,
            error=f"Failed to process video: {str(e)}"
        )

@router.post("/analyze/audio")
async def analyze_audio_unified(request: Request, data: UnifiedAnalysisRequest):
    """
    Unified audio analysis endpoint - accepts standardized JSON format
    """
    start_time = datetime.utcnow()
    logger.info(f"[REQUEST RECEIVED] Audio analysis for {data.filename}")
    
    try:
        # Validate ML service availability
        if not hasattr(request.app.state, "sentinel_agent"):
            logger.error("[MODEL INFERENCE FAILED] ML models not initialized")
            return UnifiedAnalysisResponse(
                success=False,
                media_type="audio",
                fake_score=0.0,
                label="Unknown",
                processing_time=0,
                error="Analysis service unavailable - ML models not initialized"
            )

        # Decode base64 audio
        try:
            logger.info("[FILE DECODED] Decoding base64 audio data")
            audio_data = base64.b64decode(data.data_base64)
        except Exception as e:
            logger.error(f"[FILE DECODED] Invalid base64 data: {str(e)}")
            return UnifiedAnalysisResponse(
                success=False,
                media_type="audio",
                fake_score=0.0,
                label="Unknown",
                processing_time=0,
                error=f"Invalid audio data: {str(e)}"
            )

        # Create analysis request
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.AUDIO,
            content=audio_data,
            metadata={
                "filename": data.filename,
                "mime_type": data.mime_type,
                "request_id": f"req_{uuid.uuid4().hex[:8]}",
            },
        )

        # Execute analysis with timeout guard
        logger.info("[MODEL INFERENCE START] Starting audio analysis")
        
        # Check if ML service is available
        if not hasattr(request.app.state, "sentinel_agent") or request.app.state.sentinel_agent is None:
            logger.error("[ML SERVICE UNAVAILABLE] SentinelAgent not initialized")
            return UnifiedAnalysisResponse(
                success=False,
                media_type="audio",
                fake_score=0.0,
                label="Unknown",
                processing_time=0,
                error="ML service temporarily unavailable - please try again later"
            )
        
        try:
            # Add timeout wrapper for ML inference (5 minutes max)
            import asyncio
            result = await asyncio.wait_for(
                request.app.state.sentinel_agent.analyze(analysis_request),
                timeout=300.0  # 5 minutes
            )
            logger.info("[MODEL INFERENCE DONE] Analysis completed")
        except asyncio.TimeoutError:
            logger.error("[MODEL INFERENCE TIMEOUT] Audio analysis timed out after 5 minutes")
            return UnifiedAnalysisResponse(
                success=False,
                media_type="audio",
                fake_score=0.0,
                label="Unknown",
                processing_time=300000,  # 5 minutes in ms
                error="Audio analysis timeout - file too large or complex. Please try with a smaller file."
            )

        if not result or not hasattr(result, "conclusions") or not result.conclusions:
            logger.error("[MODEL INFERENCE FAILED] No valid analysis results")
            return UnifiedAnalysisResponse(
                success=False,
                media_type="audio",
                fake_score=0.0,
                label="Unknown",
                processing_time=0,
                error="Failed to analyze audio: No valid analysis results"
            )

        # Get the primary conclusion
        primary_conclusion = max(
            result.conclusions, key=lambda c: (c.confidence, c.severity)
        )

        # Extract model info
        model_info = {}
        if result.conclusions:
            model_info = result.conclusions[0].metadata.get("model_info", {})

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        response = UnifiedAnalysisResponse(
            success=True,
            media_type="audio",
            fake_score=float(primary_conclusion.confidence),
            label="Deepfake" if primary_conclusion.is_deepfake else "Authentic",
            processing_time=processing_time,
            model_name=model_info.get("name", "SatyaAI-Audio"),
            model_version=model_info.get("version", "1.0.0"),
            confidence=float(primary_conclusion.confidence),
            proof=getattr(result, 'proof', {}),
            metadata={
                "processing_time": getattr(result, 'processing_time', 0),
                "evidence_id": result.evidence_ids[0] if hasattr(result, "evidence_ids") and result.evidence_ids else data.filename,
            }
        )

        logger.info(f"[RESPONSE SENT] Audio analysis completed successfully")
        return response

    except Exception as e:
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"[MODEL INFERENCE FAILED] Audio analysis failed: {str(e)}", exc_info=True)
        return UnifiedAnalysisResponse(
            success=False,
            media_type="audio",
            fake_score=0.0,
            label="Unknown",
            processing_time=processing_time,
            error=f"Failed to process audio: {str(e)}"
        )

@router.post("/image")
async def analyze_image(request: Request, data: AnalysisRequestModel):
    """
    Analyze an image for deepfake detection
    """
    try:
        if not hasattr(request.app.state, "sentinel_agent"):
            raise HTTPException(
                status_code=503, 
                detail="Analysis service unavailable - ML models not initialized"
            )

        # Decode base64 image
        try:
            image_data = base64.b64decode(data.image)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid image data: {str(e)}"
            )

        # Create analysis request
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.IMAGE,
            content=image_data,
            metadata={
                "filename": data.filename,
                "mime_type": data.mimeType,
                "job_id": data.jobId,
                "request_id": f"req_{uuid.uuid4().hex[:8]}",
            },
        )

        # Execute analysis through SentinelAgent with timeout guard
        try:
            import asyncio
            result = await asyncio.wait_for(
                request.app.state.sentinel_agent.analyze(analysis_request),
                timeout=300.0  # 5 minutes
            )
        except asyncio.TimeoutError:
            logger.error("[MODEL INFERENCE TIMEOUT] Image analysis timed out after 5 minutes")
            raise HTTPException(
                status_code=408,
                detail="Image analysis timeout - file too large or complex. Please try with a smaller file."
            )

        if not result or not hasattr(result, "conclusions") or not result.conclusions:
            raise HTTPException(
                status_code=500,
                detail="Failed to analyze image: No valid analysis results",
            )

        # Get the primary conclusion (highest confidence or most severe)
        primary_conclusion = max(
            result.conclusions, key=lambda c: (c.confidence, c.severity)
        )

        # Extract model info from the first available conclusion
        model_info = {}
        if result.conclusions:
            model_info = result.conclusions[0].metadata.get("model_info", {})

        # Format the response to match what Node.js expects
        response = {
            "status": "success",
            "is_deepfake": primary_conclusion.is_deepfake,
            "confidence": float(primary_conclusion.confidence),
            "model_name": model_info.get("name", "SatyaAI-Image"),
            "model_version": model_info.get("version", "1.0.0"),
            "summary": {
                "isDeepfake": primary_conclusion.is_deepfake,
                "modelInfo": model_info,
                "processingTime": getattr(result, 'processing_time', 0),
                "evidenceId": result.evidence_ids[0] if hasattr(result, "evidence_ids") and result.evidence_ids else data.jobId,
            },
            "proof": {
                "model_name": model_info.get("name", "SatyaAI-Image"),
                "model_version": model_info.get("version", "1.0.0"),
                "modality": "image",
                "timestamp": datetime.utcnow().isoformat(),
                "inference_duration": getattr(result, 'processing_time', 0),
                "frames_analyzed": 1,
                "signature": getattr(result, 'signature', f"sig_{uuid.uuid4().hex[:16]}"),
                "metadata": {
                    "request_id": f"req_{uuid.uuid4().hex[:8]}",
                    "user_id": "anonymous",
                    "analysis_type": "image",
                    "content_size": len(image_data),
                },
            }
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process image: {str(e)}"
        )

@router.post("/video")
async def analyze_video(request: Request, data: AnalysisRequestModel):
    """
    Analyze a video for deepfake detection
    """
    try:
        if not hasattr(request.app.state, "sentinel_agent"):
            raise HTTPException(
                status_code=503, 
                detail="Analysis service unavailable - ML models not initialized"
            )

        # Decode base64 video
        try:
            video_data = base64.b64decode(data.image)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid video data: {str(e)}"
            )

        # Create analysis request
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.VIDEO,
            content=video_data,
            metadata={
                "filename": data.filename,
                "mime_type": data.mimeType,
                "job_id": data.jobId,
                "request_id": f"req_{uuid.uuid4().hex[:8]}",
            },
        )

        # Execute analysis through SentinelAgent with timeout guard
        try:
            import asyncio
            result = await asyncio.wait_for(
                request.app.state.sentinel_agent.analyze(analysis_request),
                timeout=300.0  # 5 minutes
            )
        except asyncio.TimeoutError:
            logger.error("[MODEL INFERENCE TIMEOUT] Video analysis timed out after 5 minutes")
            raise HTTPException(
                status_code=408,
                detail="Video analysis timeout - file too large or complex. Please try with a smaller file."
            )

        if not result or not hasattr(result, "conclusions") or not result.conclusions:
            raise HTTPException(
                status_code=500,
                detail="Failed to analyze video: No valid analysis results",
            )

        # Get the primary conclusion
        primary_conclusion = max(
            result.conclusions, key=lambda c: (c.confidence, c.severity)
        )

        # Extract model info
        model_info = {}
        if result.conclusions:
            model_info = result.conclusions[0].metadata.get("model_info", {})

        # Format response
        response = {
            "status": "success",
            "is_deepfake": primary_conclusion.is_deepfake,
            "confidence": float(primary_conclusion.confidence),
            "model_name": model_info.get("name", "SatyaAI-Video"),
            "model_version": model_info.get("version", "1.0.0"),
            "summary": {
                "isDeepfake": primary_conclusion.is_deepfake,
                "modelInfo": model_info,
                "processingTime": getattr(result, 'processing_time', 0),
                "evidenceId": result.evidence_ids[0] if hasattr(result, "evidence_ids") and result.evidence_ids else data.jobId,
            },
            "proof": {
                "model_name": model_info.get("name", "SatyaAI-Video"),
                "model_version": model_info.get("version", "1.0.0"),
                "modality": "video",
                "timestamp": datetime.utcnow().isoformat(),
                "inference_duration": getattr(result, 'processing_time', 0),
                "frames_analyzed": getattr(result, 'frames_analyzed', 0),
                "signature": getattr(result, 'signature', f"sig_{uuid.uuid4().hex[:16]}"),
                "metadata": {
                    "request_id": f"req_{uuid.uuid4().hex[:8]}",
                    "user_id": "anonymous",
                    "analysis_type": "video",
                    "content_size": len(video_data),
                },
            }
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process video: {str(e)}"
        )

@router.post("/audio")
async def analyze_audio(request: Request, data: AnalysisRequestModel):
    """
    Analyze audio for deepfake detection
    """
    try:
        if not hasattr(request.app.state, "sentinel_agent"):
            raise HTTPException(
                status_code=503, 
                detail="Analysis service unavailable - ML models not initialized"
            )

        # Decode base64 audio
        try:
            audio_data = base64.b64decode(data.image)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid audio data: {str(e)}"
            )

        # Create analysis request
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.AUDIO,
            content=audio_data,
            metadata={
                "filename": data.filename,
                "mime_type": data.mimeType,
                "job_id": data.jobId,
                "request_id": f"req_{uuid.uuid4().hex[:8]}",
            },
        )

        # Execute analysis through SentinelAgent with timeout guard
        try:
            import asyncio
            result = await asyncio.wait_for(
                request.app.state.sentinel_agent.analyze(analysis_request),
                timeout=300.0  # 5 minutes
            )
        except asyncio.TimeoutError:
            logger.error("[MODEL INFERENCE TIMEOUT] Audio analysis timed out after 5 minutes")
            raise HTTPException(
                status_code=408,
                detail="Audio analysis timeout - file too large or complex. Please try with a smaller file."
            )

        if not result or not hasattr(result, "conclusions") or not result.conclusions:
            raise HTTPException(
                status_code=500,
                detail="Failed to analyze audio: No valid analysis results",
            )

        # Get the primary conclusion
        primary_conclusion = max(
            result.conclusions, key=lambda c: (c.confidence, c.severity)
        )

        # Extract model info
        model_info = {}
        if result.conclusions:
            model_info = result.conclusions[0].metadata.get("model_info", {})

        # Format response
        response = {
            "status": "success",
            "is_deepfake": primary_conclusion.is_deepfake,
            "confidence": float(primary_conclusion.confidence),
            "model_name": model_info.get("name", "SatyaAI-Audio"),
            "model_version": model_info.get("version", "1.0.0"),
            "summary": {
                "isDeepfake": primary_conclusion.is_deepfake,
                "modelInfo": model_info,
                "processingTime": getattr(result, 'processing_time', 0),
                "evidenceId": result.evidence_ids[0] if hasattr(result, "evidence_ids") and result.evidence_ids else data.jobId,
            },
            "proof": {
                "model_name": model_info.get("name", "SatyaAI-Audio"),
                "model_version": model_info.get("version", "1.0.0"),
                "modality": "audio",
                "timestamp": datetime.utcnow().isoformat(),
                "inference_duration": getattr(result, 'processing_time', 0),
                "frames_analyzed": 1,
                "signature": getattr(result, 'signature', f"sig_{uuid.uuid4().hex[:16]}"),
                "metadata": {
                    "request_id": f"req_{uuid.uuid4().hex[:8]}",
                    "user_id": "anonymous",
                    "analysis_type": "audio",
                    "content_size": len(audio_data),
                },
            }
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process audio: {str(e)}"
        )


@router.get("/history")
async def get_analysis_history(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: Optional[int] = None,
):
    """
    Get analysis history
    """
    try:
        db = get_db_manager()

        # If user_id provided, get user's history
        if user_id:
            analyses = db.get_user_analyses(user_id, limit, offset)
        else:
            # Otherwise get recent global analyses (for admin/demo)
            analyses = db.get_recent_analyses(limit)

        # Convert to list of dicts
        results = []
        for a in analyses:
            results.append(
                {
                    "id": a.file_id,
                    "filename": a.file_name,
                    "type": a.file_type,
                    "score": a.authenticity_score,
                    "label": a.label,
                    "confidence": a.confidence,
                    "timestamp": a.created_at,
                    "details": a.details,
                }
            )

        return {
            "success": True,
            "count": len(results),
            "limit": limit,
            "offset": offset,
            "history": results,
        }

    except Exception as e:
        logger.error(f"Failed to get history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.get("/stats")
async def get_analysis_stats():
    """
    Get overall analysis statistics
    """
    try:
        db = get_db_manager()
        stats = db.get_analysis_stats()

        return {"success": True, "stats": stats}

    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/{file_id}")
async def get_analysis_detail(file_id: str):
    """
    Get detailed analysis result by ID
    """
    try:
        db = get_db_manager()
        analysis = db.get_analysis_by_file_id(file_id)

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return {
            "success": True,
            "id": analysis.file_id,
            "filename": analysis.file_name,
            "type": analysis.file_type,
            "score": analysis.authenticity_score,
            "label": analysis.label,
            "confidence": analysis.confidence,
            "timestamp": analysis.created_at,
            "details": analysis.details,
            "processing_time": analysis.processing_time,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis detail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get analysis: {str(e)}")

@router.get("/health")
async def health_check(request: Request):
    """
    Health check endpoint that validates ML models are loaded and functional
    """
    try:
        logger.info("[HEALTH CHECK] Starting model validation")
        
        # Check if SentinelAgent is available
        if not hasattr(request.app.state, "sentinel_agent"):
            logger.error("[HEALTH CHECK] SentinelAgent not initialized")
            return {
                "status": "unhealthy",
                "error": "ML models not initialized",
                "models_loaded": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        sentinel_agent = request.app.state.sentinel_agent
        
        # Check model availability
        model_status = {
            "image": hasattr(sentinel_agent, "image_detector") and bool(sentinel_agent.image_detector),
            "video": hasattr(sentinel_agent, "video_detector") and bool(sentinel_agent.video_detector),
            "audio": hasattr(sentinel_agent, "audio_detector") and bool(sentinel_agent.audio_detector),
        }
        
        # Check if all models are loaded (not just any)
        models_loaded = all(model_status.values())
        
        if not models_loaded:
            logger.error(f"[HEALTH CHECK] Not all models loaded: {model_status}")
            return {
                "status": "unhealthy",
                "error": "Not all ML models loaded",
                "models_loaded": False,
                "model_status": model_status,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        logger.info(f"[HEALTH CHECK] Models loaded: {model_status}")
        
        return {
            "status": "healthy",
            "models_loaded": True,
            "model_status": model_status,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "SatyaAI Analysis Service"
        }
        
    except Exception as e:
        logger.error(f"[HEALTH CHECK] Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": f"Health check failed: {str(e)}",
            "models_loaded": False,
            "timestamp": datetime.utcnow().isoformat()
        }
