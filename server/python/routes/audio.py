"""
Audio Upload and Analysis Route
Dedicated route for audio processing with ML analysis
"""

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import (APIRouter, File, HTTPException, Request,
                     UploadFile, status)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from sentinel_agent import AnalysisRequest, AnalysisType

logger = logging.getLogger(__name__)

router = APIRouter()

# Upload directory
UPLOAD_DIR = Path("uploads/audio")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# File size limit
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50 MB

# Allowed file types
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/wav", "audio/ogg", "audio/m4a", "audio/mp3", "audio/webm"}


# Models
class AudioAnalysisResponse(BaseModel):
    success: bool
    file_id: str
    filename: str
    size: int
    content_type: str
    analysis_id: str
    timestamp: str
    analysis: dict


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
    request_id: str
    timestamp: str


@router.post(
    "/",
    response_model=AudioAnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyze audio file for deepfake detection",
    description="Upload and analyze an audio file for potential deepfake content."
)
async def upload_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to analyze"),
):
    """
    Upload and analyze audio through SentinelAgent with ML enforcement.

    This endpoint processes audio files for deepfake detection.
    - Rate limiting (optional - can be added later)
    - Strict input validation
    - Secure file handling
    - No authentication required (handled by Node.js)
    """
    try:
        # Validate file type
        if file.content_type not in ALLOWED_AUDIO_TYPES:
            error_msg = f"Unsupported media type: {file.content_type}"
            raise HTTPException(
                status_code=415,
                detail=error_msg,
            )

        # Read file content with size limit
        content = await file.read(MAX_AUDIO_SIZE + 1)
        file_size = len(content)
        if file_size > MAX_AUDIO_SIZE:
            error_msg = f"File too large: {file_size} bytes (max: {MAX_AUDIO_SIZE} bytes)"
            raise HTTPException(
                status_code=413,
                detail=error_msg,
            )

        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        safe_filename = f"{file_id}{file_extension}"
        file_path = UPLOAD_DIR / safe_filename

        # Save file
        with open(file_path, "wb") as f:
            f.write(content)

        # Create analysis request
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.AUDIO,
            content=content,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file_size,
            },
        )

        # Process through SentinelAgent
        result = await request.app.state.sentinel_agent.analyze(analysis_request)

        if not result or not hasattr(result, "conclusions") or not result.conclusions:
            error_msg = "Failed to analyze audio: No valid analysis results"
            raise HTTPException(
                status_code=500,
                detail=error_msg,
            )

        # Get the primary conclusion (highest confidence or most severe)
        primary_conclusion = max(
            result.conclusions, key=lambda c: (c.confidence, c.severity)
        )

        # Extract model info from the first available conclusion
        model_info = {}
        if result.conclusions:
            model_info = result.conclusions[0].metadata.get("model_info", {})

        # Return response
        return AudioAnalysisResponse(
            success=True,
            file_id=file_id,
            filename=file.filename,
            size=file_size,
            content_type=file.content_type,
            analysis_id=result.analysis_id,
            timestamp=datetime.utcnow().isoformat(),
            analysis={
                "is_deepfake": primary_conclusion.is_deepfake,
                "confidence": float(primary_conclusion.confidence),
                "model_info": model_info,
                "timestamp": datetime.utcnow().isoformat(),
                "evidence_id": result.evidence_ids[0]
                if hasattr(result, "evidence_ids") and result.evidence_ids
                else file_id,
            },
        )

    except HTTPException:
        raise

    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        raise HTTPException(
            status_code=500,
            detail=error_msg,
        )


@router.get("/{file_id}")
async def get_audio(file_id: str):
    """Get analyzed audio file by ID"""
    try:
        # Resolve file path (search for any file with given UUID)
        matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
        if not matches:
            raise HTTPException(status_code=404, detail="Audio file not found")
        file_path = matches[0]
        
        # Determine content type
        content_type = "audio/mpeg"  # Default
        if file_path.suffix.lower() in ['.mp3']:
            content_type = "audio/mpeg"
        elif file_path.suffix.lower() in ['.wav']:
            content_type = "audio/wav"
        elif file_path.suffix.lower() in ['.ogg']:
            content_type = "audio/ogg"
        
        return FileResponse(
            path=file_path,
            media_type=content_type,
            filename=file_path.name,
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Audio file not found")
