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
from pydantic import BaseModel

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.analysis import AnalysisResult
from services.database import get_db_manager
from sentinel_agent import AnalysisRequest, AnalysisType

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response models
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

        # Execute analysis through SentinelAgent
        result = await request.app.state.sentinel_agent.analyze(analysis_request)

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

        # Execute analysis through SentinelAgent
        result = await request.app.state.sentinel_agent.analyze(analysis_request)

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

        # Execute analysis through SentinelAgent
        result = await request.app.state.sentinel_agent.analyze(analysis_request)

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
