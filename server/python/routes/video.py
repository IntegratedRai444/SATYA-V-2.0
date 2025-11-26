"""
Video Upload and Analysis Route
Dedicated route for video processing
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
from datetime import datetime
import logging

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
    analyze: bool = Form(True)
):
    """
    Upload and analyze a video
    
    - **file**: Video file to upload
    - **analyze**: Whether to run ML analysis (default: True)
    """
    try:
        # Validate file type
        if file.content_type not in ALLOWED_VIDEO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_VIDEO_TYPES)}"
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Validate file size
        if file_size > MAX_VIDEO_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max: {MAX_VIDEO_SIZE / (1024*1024):.1f} MB"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{file_id}{file_extension}"
        file_path = UPLOAD_DIR / filename
        
        # Save file
        with file_path.open("wb") as f:
            f.write(content)
        
        logger.info(f"Video uploaded: {filename} ({file_size} bytes)")
        
        # Analyze if requested
        analysis_result = None
        if analyze and hasattr(request.app.state, 'video_detector'):
            logger.info(f"Analyzing video: {filename}")
            detector = request.app.state.video_detector
            analysis_result = detector.detect(str(file_path))
        
        return {
            "success": True,
            "file_id": file_id,
            "filename": file.filename,
            "saved_as": filename,
            "size": file_size,
            "content_type": file.content_type,
            "path": str(file_path),
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": analysis_result
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