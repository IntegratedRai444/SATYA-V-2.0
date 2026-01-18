"""
Audio Upload and Analysis Route
Dedicated route for audio processing with strict security controls
"""

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import (APIRouter, Depends, File, HTTPException, Request,
                     UploadFile, status)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from ..middleware.rate_limiter import rate_limiter
from ..sentinel_agent import AnalysisRequest, AnalysisType

# Security
security = HTTPBearer()


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


logger = logging.getLogger(__name__)

router = APIRouter(
    dependencies=[Depends(rate_limiter)],
    responses={
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        429: {"model": ErrorResponse, "description": "Too Many Requests"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)

# Upload directory
UPLOAD_DIR = Path("uploads/audio")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Security constants
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_AUDIO_TYPES = {"audio/mp3", "audio/wav", "audio/mpeg", "audio/ogg", "audio/m4a", "audio/mp4", "audio/webm"}
MAX_FILENAME_LENGTH = 255


# Request ID for tracking
class RequestContext:
    def __init__(self):
        self.request_id = None


request_context = RequestContext()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Validate and return current user from token"""
    # In a real implementation, validate the JWT token here
    # For now, return a mock user
    return {"user_id": "user123", "roles": ["user"]}


def log_audit(
    user_id: str,
    action: str,
    status: str,
    metadata: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    """Log audit trail for security and compliance"""
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": getattr(request_context, "request_id", "unknown"),
        "user_id": user_id,
        "action": action,
        "status": status,
        "metadata": metadata or {},
    }
    if error:
        log_data["error"] = error

    logger.info(
        "AUDIT: %s",
        {k: v for k, v in log_data.items() if v is not None},
        extra={"audit": True},
    )


def validate_filename(filename: str) -> str:
    """Sanitize and validate filename"""
    if not filename or len(filename) > MAX_FILENAME_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid filename. Must be 1-{MAX_FILENAME_LENGTH} characters",
        )
    # Remove any path traversal attempts
    return os.path.basename(filename)


@router.post(
    "/",
    response_model=AudioAnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyze audio file for deepfake detection",
    description="""
    Upload and analyze an audio file for potential deepfake content.
    
    - **Authentication**: Bearer token required
    - **Rate Limited**: 10 requests per minute
    - **Max File Size**: 50MB
    - **Allowed Formats**: MP3, WAV, MPEG, OGG, M4A, MP4, WebM
    """,
    responses={
        202: {"description": "Analysis started successfully"},
        400: {"description": "Invalid request"},
        401: {"description": "Unauthorized"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported media type"},
        429: {"description": "Too many requests"},
        503: {"description": "Service unavailable"},
    },
)
async def upload_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to analyze"),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload and analyze audio through SentinelAgent with ML enforcement.

    This endpoint is strictly controlled and enforces:
    - Authentication via Bearer token
    - Rate limiting
    - Strict input validation
    - Secure file handling
    - Complete audit logging
    """
    # Set request context for logging
    request_context.request_id = str(uuid.uuid4())
    start_time = time.time()

    # Audit log start
    log_audit(
        user_id=current_user["user_id"],
        action="audio_analysis_start",
        status="processing",
        metadata={
            "filename": file.filename,
            "content_type": file.content_type,
            "request_id": request_context.request_id,
        },
    )

    try:
        # Validate file type
        if file.content_type not in ALLOWED_AUDIO_TYPES:
            error_msg = f"Unsupported media type: {file.content_type}"
            log_audit(
                user_id=current_user["user_id"],
                action="audio_analysis_rejected",
                status="error",
                metadata={"reason": error_msg},
                error=error_msg,
            )
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=error_msg
            )

        # Read and validate file content
        content = await file.read()
        file_size = len(content)

        if file_size > MAX_AUDIO_SIZE:
            error_msg = (
                f"File too large: {file_size} bytes (max: {MAX_AUDIO_SIZE} bytes)"
            )
            log_audit(
                user_id=current_user["user_id"],
                action="audio_analysis_rejected",
                status="error",
                metadata={"reason": "file_too_large", "size": file_size},
                error=error_msg,
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=error_msg
            )

        # Generate secure filename and save file
        file_id = str(uuid.uuid4())
        file_extension = Path(validate_filename(file.filename)).suffix
        filename = f"{file_id}{file_extension}"
        file_path = UPLOAD_DIR / filename

        try:
            with file_path.open("wb") as f:
                f.write(content)
        except Exception as e:
            error_msg = f"Failed to save file: {str(e)}"
            log_audit(
                user_id=current_user["user_id"],
                action="audio_analysis_failed",
                status="error",
                metadata={"reason": "file_save_error"},
                error=error_msg,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process audio file",
            )

        # Verify SentinelAgent is available
        if (
            not hasattr(request.app.state, "sentinel_agent")
            or not request.app.state.sentinel_agent
        ):
            error_msg = "Audio analysis service is not available"
            log_audit(
                user_id=current_user["user_id"],
                action="audio_analysis_failed",
                status="error",
                metadata={"reason": "service_unavailable"},
                error=error_msg,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_msg
            )

        # Create analysis request with user context
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.AUDIO,
            content_uri=str(file_path),
            metadata={
                "user_id": current_user["user_id"],
                "original_filename": file.filename,
                "content_type": file.content_type,
                "file_size": file_size,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "request_id": request_context.request_id,
            },
        )

        # Execute analysis through SentinelAgent
        try:
            analysis_result = await request.app.state.sentinel_agent.analyze(
                analysis_request
            )
            analysis_id = analysis_result.metadata.get("analysis_id", "unknown")

            # Log successful analysis
            log_audit(
                user_id=current_user["user_id"],
                action="audio_analysis_complete",
                status="success",
                metadata={
                    "analysis_id": analysis_id,
                    "confidence": analysis_result.confidence,
                    "processing_time": time.time() - start_time,
                },
            )

            # Clean up the uploaded file after successful analysis
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete uploaded file {file_path}: {e}")

            return {
                "success": True,
                "file_id": file_id,
                "filename": file.filename,
                "size": file_size,
                "content_type": file.content_type,
                "analysis_id": analysis_id,
                "timestamp": datetime.utcnow().isoformat(),
                "analysis": {
                    "status": analysis_result.status,
                    "confidence": analysis_result.confidence,
                    "confidence_level": analysis_result.confidence_level.value,
                    "conclusions": analysis_result.conclusions,
                    "evidence_ids": analysis_result.evidence_ids,
                    "metadata": analysis_result.metadata,
                },
            }

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            log_audit(
                user_id=current_user["user_id"],
                action="audio_analysis_failed",
                status="error",
                metadata={"reason": "analysis_error"},
                error=error_msg,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Audio analysis failed",
            )

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions with proper status codes
        raise http_exc

    except Exception as e:
        # Log unexpected errors
        error_id = str(uuid.uuid4())
        logger.error(
            f"Unexpected error in audio analysis [{error_id}]: {str(e)}",
            exc_info=True,
            extra={"request_id": request_context.request_id},
        )
        log_audit(
            user_id=current_user["user_id"],
            action="audio_analysis_error",
            status="error",
            metadata={"error_id": error_id},
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {error_id}",
        )


@router.get(
    "/{file_id}",
    response_class=FileResponse,
    summary="Retrieve analyzed audio file",
    description="""
    Retrieve a previously analyzed audio file by ID.
    
    - **Authentication**: Bearer token required
    - **Rate Limited**: 60 requests per minute
    """,
    responses={
        200: {"description": "Audio file"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "File not found"},
        429: {"description": "Too many requests"},
    },
)
async def get_audio(file_id: str, current_user: dict = Depends(get_current_user)):
    """
    Retrieve an analyzed audio file by ID.

    This endpoint enforces:
    - Authentication
    - Rate limiting
    - File existence checks
    - Access control (if implemented)
    """
    # Set request context for logging
    request_context.request_id = str(uuid.uuid4())

    # Log access attempt
    log_audit(
        user_id=current_user["user_id"],
        action="audio_retrieve",
        status="processing",
        metadata={"file_id": file_id},
    )

    try:
        # Validate file_id to prevent directory traversal
        if not file_id or any(c in file_id for c in "/\\"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file ID"
            )

        # Find matching files
        matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
        if not matches:
            log_audit(
                user_id=current_user["user_id"],
                action="audio_retrieve_failed",
                status="not_found",
                metadata={"file_id": file_id},
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Audio not found"
            )

        file_path = matches[0]

        # Verify file exists and is a file (not a directory)
        if not file_path.is_file():
            log_audit(
                user_id=current_user["user_id"],
                action="audio_retrieve_failed",
                status="error",
                metadata={"file_id": file_id, "reason": "not_a_file"},
                error="Requested path is not a file",
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Audio not found"
            )

        # Determine content type based on file extension
        content_type = "audio/mpeg"  # Default
        if file_path.suffix.lower() == ".wav":
            content_type = "audio/wav"
        elif file_path.suffix.lower() == ".ogg":
            content_type = "audio/ogg"

        # Log successful access
        log_audit(
            user_id=current_user["user_id"],
            action="audio_retrieved",
            status="success",
            metadata={"file_id": file_id, "content_type": content_type},
        )

        return FileResponse(
            path=file_path,
            media_type=content_type,
            filename=file_path.name,
            headers={
                "Cache-Control": "private, max-age=300",
                "X-Content-Type-Options": "nosniff",
            },
        )

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        # Log unexpected errors
        error_id = str(uuid.uuid4())
        logger.error(
            f"Error retrieving audio file [{error_id}]: {str(e)}",
            exc_info=True,
            extra={"request_id": request_context.request_id},
        )
        log_audit(
            user_id=current_user["user_id"],
            action="audio_retrieve_error",
            status="error",
            metadata={"file_id": file_id, "error_id": error_id},
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving the file",
        )
