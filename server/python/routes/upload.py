"""
Upload Routes
Handle file uploads with direct ML integration and database storage
"""

import logging
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Request,
                     UploadFile)
from fastapi.responses import JSONResponse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from services.database import get_db_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# File size limits (in bytes) - Match Node.js backend
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_VIDEO_SIZE = 50 * 1024 * 1024   # 50 MB (matches Node.js)
MAX_AUDIO_SIZE = 50 * 1024 * 1024   # 50 MB

# Allowed file types
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/mov", "video/mkv"}
ALLOWED_AUDIO_TYPES = {"audio/mp3", "audio/wav", "audio/mpeg", "audio/ogg"}


@router.post("/image")
async def upload_image(request: Request, file: UploadFile = File(...)):
    """
    Legacy image upload endpoint - redirects to /api/analysis/image
    """
    from urllib.parse import urlencode

    from fastapi.responses import RedirectResponse

    # Redirect to the main analysis endpoint
    redirect_url = request.url_for("upload_image_v2")
    return await upload_image_v2(request, file)


@router.post("/video")
async def upload_video(
    request: Request, file: UploadFile = File(...), analyze: bool = Form(True)
):
    """
    Upload and analyze a video
    """
    try:
        # Validate file type
        if file.content_type not in ALLOWED_VIDEO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_VIDEO_TYPES)}",
            )

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Validate file size
        if file_size > MAX_VIDEO_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_VIDEO_SIZE / (1024*1024):.1f} MB",
            )

        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{file_id}{file_extension}"

        # Save file
        upload_path = UPLOAD_DIR / "videos"
        upload_path.mkdir(exist_ok=True)
        file_path = upload_path / filename

        with file_path.open("wb") as f:
            f.write(content)

        logger.info(f"Video uploaded: {filename} ({file_size} bytes)")

        # Database integration
        db = get_db_manager()

        # Save file metadata
        db.save_file_metadata(
            file_id=file_id,
            original_filename=file.filename,
            stored_filename=filename,
            file_path=str(file_path),
            file_type="video",
            file_size=file_size,
        )

        # Analyze if requested
        analysis_result = None
        if analyze and hasattr(request.app.state, "video_detector"):
            logger.info(f"Analyzing video: {filename}")
            detector = request.app.state.video_detector
            analysis_result = detector.detect(str(file_path))

            # Return analysis result for Node to handle
            # Database persistence is now handled by Node.js
            if analysis_result:
                logger.info(f"📤 Video analysis result for {filename} returned to Node for persistence")

        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "saved_as": filename,
            "size": file_size,
            "content_type": file.content_type,
            "path": str(file_path),
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": analysis_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/audio")
async def upload_audio(
    request: Request, file: UploadFile = File(...), analyze: bool = Form(True)
):
    """
    Upload and analyze audio
    """
    try:
        # Validate file type
        if file.content_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_AUDIO_TYPES)}",
            )

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Validate file size
        if file_size > MAX_AUDIO_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_AUDIO_SIZE / (1024*1024):.1f} MB",
            )

        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{file_id}{file_extension}"

        # Save file
        upload_path = UPLOAD_DIR / "audio"
        upload_path.mkdir(exist_ok=True)
        file_path = upload_path / filename

        with file_path.open("wb") as f:
            f.write(content)

        logger.info(f"Audio uploaded: {filename} ({file_size} bytes)")

        # Database integration
        db = get_db_manager()

        # Save file metadata
        db.save_file_metadata(
            file_id=file_id,
            original_filename=file.filename,
            stored_filename=filename,
            file_path=str(file_path),
            file_type="audio",
            file_size=file_size,
        )

        # Analyze if requested
        analysis_result = None
        if analyze and hasattr(request.app.state, "audio_detector"):
            logger.info(f"Analyzing audio: {filename}")
            detector = request.app.state.audio_detector
            analysis_result = detector.detect(str(file_path))

            # Return analysis result for Node to handle
            # Database persistence is now handled by Node.js
            if analysis_result:
                logger.info(f"📤 Audio analysis result for {filename} returned to Node for persistence")

        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "saved_as": filename,
            "size": file_size,
            "content_type": file.content_type,
            "path": str(file_path),
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": analysis_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/text")
async def upload_text(
    request: Request, text: str = Form(...), analyze: bool = Form(True)
):
    """
    Analyze text for AI-generated content
    """
    try:
        # Analyze if requested
        analysis_result = None
        if analyze and hasattr(request.app.state, "text_nlp_detector"):
            logger.info(f"Analyzing text ({len(text)} characters)")
            detector = request.app.state.text_nlp_detector
            analysis_result = detector.detect(text)

            # Database integration
            db = get_db_manager()

            # Generate ID for text analysis
            file_id = str(uuid.uuid4())

            # Return analysis result for Node to handle
            # Database persistence is now handled by Node.js
            if analysis_result:
                logger.info(f"📤 Text analysis result returned to Node for persistence")

        return {
            "success": True,
            "text_length": len(text),
            "word_count": len(text.split()),
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": analysis_result,
        }

    except Exception as e:
        logger.error(f"Text analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
