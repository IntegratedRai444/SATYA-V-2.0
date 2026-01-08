"""
Webcam Capture and Analysis Route
Dedicated route for webcam/real-time capture with ML enforcement
"""

import base64
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

from ..auth import User, get_current_user
from ..sentinel_agent import AnalysisRequest, AnalysisType, get_sentinel_agent

logger = logging.getLogger(__name__)

router = APIRouter()

# Rate limiting and session tracking
WEBCAM_SESSIONS: Dict[str, Dict[str, Any]] = {}
MAX_CONCURRENT_SESSIONS = 10
RATE_LIMIT = 30  # Max requests per minute

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class WebcamCapture(BaseModel):
    """Webcam frame capture request model"""

    image_data: str = Field(..., description="Base64 encoded image data")
    format: str = Field("jpeg", description="Image format (jpeg, png)")
    session_id: Optional[str] = Field(
        None, description="Optional session ID for tracking"
    )


def validate_webcam_session(session_id: str, user_id: str) -> None:
    """Validate webcam session and apply rate limiting"""
    current_time = time.time()

    # Clean up old sessions
    expired_sessions = [
        sid
        for sid, session in WEBCAM_SESSIONS.items()
        if current_time - session["last_active"] > 300  # 5 minute timeout
    ]
    for sid in expired_sessions:
        WEBCAM_SESSIONS.pop(sid, None)

    # Check session limits
    user_sessions = [
        sid for sid, session in WEBCAM_SESSIONS.items() if session["user_id"] == user_id
    ]

    if len(user_sessions) >= MAX_CONCURRENT_SESSIONS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Maximum concurrent webcam sessions ({MAX_CONCURRENT_SESSIONS}) exceeded",
        )

    # Initialize or update session
    if session_id not in WEBCAM_SESSIONS:
        WEBCAM_SESSIONS[session_id] = {
            "user_id": user_id,
            "start_time": current_time,
            "request_count": 0,
            "last_active": current_time,
            "frames_analyzed": 0,
            "last_frame_time": 0,
        }

    # Check rate limiting
    session = WEBCAM_SESSIONS[session_id]
    time_since_last_frame = current_time - session["last_frame_time"]

    if time_since_last_frame < (60.0 / RATE_LIMIT):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {RATE_LIMIT} requests per minute",
        )

    # Update session state
    session["last_active"] = current_time
    session["last_frame_time"] = current_time
    session["request_count"] += 1
    session["frames_analyzed"] += 1


@router.post("/capture")
async def capture_and_analyze(
    request: Request,
    capture: WebcamCapture,
    current_user: User = Depends(get_current_user),
):
    """
    Analyze webcam capture with ML enforcement

    This endpoint enforces:
    - Authentication
    - Rate limiting
    - Session management
    - ML analysis via SentinelAgent
    - Proof of analysis

    - **image_data**: Base64 encoded image data (required)
    - **format**: Image format (jpeg, png, default: jpeg)
    - **session_id**: Optional session ID for tracking (auto-generated if not provided)
    """
    try:
        # Generate or validate session ID
        session_id = capture.session_id or f"webcam_{uuid.uuid4().hex[:16]}"

        # Apply rate limiting and session validation
        validate_webcam_session(session_id, current_user.user_id)

        # Decode base64 image
        try:
            if "," in capture.image_data:  # Handle data URL format
                capture.image_data = capture.image_data.split(",", 1)[1]
            image_bytes = base64.b64decode(capture.image_data)
        except Exception as e:
            logger.error(f"Failed to decode image data: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image data"
            )

        # Create analysis request
        analysis_request = AnalysisRequest(
            analysis_type=AnalysisType.WEBCAM,
            content=image_bytes,
            metadata={
                "source": "webcam",
                "user_id": current_user.user_id,
                "session_id": session_id,
                "format": capture.format,
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        # Get Sentinel agent and analyze
        sentinel_agent = get_sentinel_agent()
        analysis_result = await sentinel_agent.analyze(analysis_request)

        # Verify proof of analysis
        if not analysis_result.proof:
            logger.error("No proof of analysis generated")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Analysis failed: No proof generated",
            )

        # Log successful analysis
        logger.info(
            f"Webcam analysis completed - "
            f"Session: {session_id}, "
            f"User: {current_user.user_id}, "
            f"Deepfake: {analysis_result.is_deepfake}, "
            f"Confidence: {analysis_result.confidence:.2f}"
        )

        # Return results with proof
        return {
            "success": True,
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": analysis_result.dict(),
            "proof": analysis_result.proof,
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Webcam analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@router.get("/status")
async def webcam_status(current_user: User = Depends(get_current_user)):
    """
    Check webcam capture status and rate limits

    Returns:
        Dict with status information including rate limits and session info
    """
    user_sessions = [
        session
        for session in WEBCAM_SESSIONS.values()
        if session["user_id"] == current_user.user_id
    ]

    return {
        "success": True,
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat(),
        "rate_limit": RATE_LIMIT,
        "concurrent_sessions": len(user_sessions),
        "max_concurrent_sessions": MAX_CONCURRENT_SESSIONS,
        "sessions": [
            {
                "session_id": sid,
                "active_seconds": int(time.time() - s["start_time"]),
                "frames_analyzed": s["frames_analyzed"],
                "last_active": datetime.fromtimestamp(s["last_active"]).isoformat(),
            }
            for sid, s in WEBCAM_SESSIONS.items()
            if s["user_id"] == current_user.user_id
        ],
    }
