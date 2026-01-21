"""
Video Upload and Analysis Route
Dedicated route for video processing
"""

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.responses import FileResponse

from sentinel_agent import AnalysisRequest, AnalysisType, SentinelAgent
from utils.rate_limiter import check_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter()

# Upload directory
UPLOAD_DIR = Path("uploads/videos")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# File size limit
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100 MB

# Allowed file types
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/mkv"}


@router.post("/")
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    analyze: bool = Form(True),
):
    """
    Upload and analyze a video

    - **file**: Video file to upload (max 100MB)
    - **analyze**: Whether to run ML analysis (default: True)

    Returns analysis results with proof of ML execution
    """
    # Rate limiting (optional - can be added later)
    # rate_limit = f"video_upload:{request.client.host}"
    # if not await check_rate_limit(rate_limit, limit=5, period=60):
    #     raise HTTPException(
    #         status_code=429, detail="Rate limit exceeded. Please try again later."
    #     )

    try:
        # Validate file type
        if file.content_type not in ALLOWED_VIDEO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_VIDEO_TYPES)}",
            )

        # Read file content with size limit
        content = await file.read(MAX_VIDEO_SIZE + 1)
        if len(content) > MAX_VIDEO_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max: {MAX_VIDEO_SIZE / (1024*1024):.1f} MB",
            )

        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{file_id}{file_extension}"
        file_path = UPLOAD_DIR / filename

        # Save file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as f:
            f.write(content)

        logger.info(
            f"Video uploaded: {filename} ({len(content)} bytes) by user {current_user.get('id', 'unknown')}"
        )

        # Log the analysis request
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        logger.info(f"Starting video analysis {request_id} for {filename}")

        # Analyze if requested
        analysis_result = None
        if analyze:
            if not hasattr(request.app.state, "sentinel_agent"):
                raise HTTPException(
                    status_code=503, detail="Analysis service unavailable"
                )

            try:
                # Create analysis request
                analysis_request = AnalysisRequest(
                    analysis_type=AnalysisType.VIDEO,
                    content=content,
                    metadata={
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "user_id": current_user.get("id", "anonymous"),
                        "client_ip": request.client.host,
                        "request_id": request_id,
                    },
                )

                # Execute analysis through SentinelAgent
                analysis_result = await request.app.state.sentinel_agent.analyze(
                    analysis_request
                )

                # Verify proof of analysis with strict validation
                if not analysis_result or "proof" not in analysis_result:
                    logger.error(
                        f"Analysis failed - no proof generated for {request_id}"
                    )
                    raise HTTPException(
                        status_code=500, detail="Analysis failed - no proof generated"
                    )

                # Extract and validate proof
                proof = analysis_result.get("proof", {})
                required_proof_fields = [
                    "analysis_id",
                    "model_version",
                    "frames_analyzed",
                    "inference_time",
                    "timestamp",
                    "confidence",
                ]

                # Validate all required proof fields exist
                missing_fields = [
                    field for field in required_proof_fields if field not in proof
                ]
                if missing_fields:
                    logger.error(
                        f"Invalid proof - missing fields: {missing_fields} for {request_id}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"Invalid analysis proof - missing required fields: {', '.join(missing_fields)}",
                    )

                # Validate inference time is positive
                if (
                    not isinstance(proof.get("inference_time"), (int, float))
                    or proof["inference_time"] <= 0
                ):
                    logger.error(
                        f"Invalid proof - invalid inference time for {request_id}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="Invalid analysis proof - invalid inference time",
                    )

                # Validate model version is present and non-empty
                if not proof.get("model_version"):
                    logger.error(
                        f"Invalid proof - missing model version for {request_id}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="Invalid analysis proof - missing model version",
                    )

                # Log successful analysis
                logger.info(
                    f"✅ Video analysis completed for {request_id} - "
                    f"Model: {proof['model_version']}, "
                    f"Frames: {proof['frames_analyzed']}, "
                    f"Inference: {proof['inference_time']:.2f}s, "
                    f"Authenticity: {not analysis_result.get('is_deepfake', True)} "
                    f"(Confidence: {analysis_result.get('confidence', 0)*100:.1f}%)"
                )

            except Exception as e:
                logger.error(
                    f"Video analysis failed for {request_id}: {str(e)}", exc_info=True
                )
                raise HTTPException(
                    status_code=500, detail=f"Analysis failed: {str(e)}"
                )

        # Return results with proof of analysis
        return {
            "success": True,
            "request_id": request_id,
            "file_id": file_id,
            "filename": file.filename,
            "size": len(content),
            "content_type": file.content_type,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": analysis_result,
            "proof": analysis_result.pop("proof", None) if analysis_result else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{file_id}")
async def get_video(file_id: str):
    """Get video by ID"""
    try:
        try:
            # Resolve the file path (search for any file with the given UUID)
            matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
            if not matches:
                raise HTTPException(status_code=404, detail="Video not found")
            file_path = matches[0]
            return FileResponse(
                path=file_path,
                media_type="video/mp4",
                filename=file_path.name,
            )
        except Exception:
            raise HTTPException(status_code=404, detail="Video not found")
    except Exception as e:
        raise HTTPException(status_code=404, detail="Video not found")
