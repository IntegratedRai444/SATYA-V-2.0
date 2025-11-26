"""
Audio Upload and Analysis Route
Dedicated route for audio processing
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
UPLOAD_DIR = Path("uploads/audio")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# File size limit
MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50 MB

# Allowed file types
ALLOWED_AUDIO_TYPES = {"audio/mp3", "audio/wav", "audio/mpeg", "audio/ogg"}


@router.post("/")
async def upload_audio(
    request: Request,
    file: UploadFile = File(...),
    analyze: bool = Form(True)
):
    """
    Upload and analyze audio
    
    - **file**: Audio file to upload
    - **analyze**: Whether to run ML analysis (default: True)
    """
    try:
        # Validate file type
        if file.content_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_AUDIO_TYPES)}"
            )
        
        # Read file content
        content = await file.read()
        file_size = len(content)
        
        # Validate file size
        if file_size > MAX_AUDIO_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max: {MAX_AUDIO_SIZE / (1024*1024):.1f} MB"
            )
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{file_id}{file_extension}"
        file_path = UPLOAD_DIR / filename
        
        # Save file
        with file_path.open("wb") as f:
            f.write(content)
        
        logger.info(f"Audio uploaded: {filename} ({file_size} bytes)")
        
        # Analyze if requested
        analysis_result = None
        if analyze and hasattr(request.app.state, 'audio_detector'):
            logger.info(f"Analyzing audio: {filename}")
            detector = request.app.state.audio_detector
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
        logger.error(f"Audio upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{file_id}")
async def get_audio(file_id: str):
    """Get audio by ID"""
    try:
        matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
        if not matches:
            raise HTTPException(status_code=404, detail="Audio not found")
        file_path = matches[0]
        return FileResponse(
            path=file_path,
            media_type="audio/mpeg",
            filename=file_path.name,
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Audio not found")